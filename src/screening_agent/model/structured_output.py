"""Helpers for resilient structured-output generation across chat backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Generic, Protocol, TypeVar

from langchain.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from screening_agent.audit import emit_custom_debug_event
from screening_agent.model.control_models import StructuredOutputInvoker

StructuredPayloadT = TypeVar("StructuredPayloadT", bound=BaseModel)


class StructuredOutputCapableModel(Protocol):
    """Protocol for models that support both plain and structured invocation."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Generate a plain AI response for the supplied messages.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Plain AI message output.
        """

        ...

    def with_structured_output(
        self,
        schema: type[StructuredPayloadT],
    ) -> StructuredOutputInvoker[StructuredPayloadT]:
        """Return a native structured-output wrapper.

        Args:
            schema: Desired Pydantic payload type.

        Returns:
            A native structured-output invoker.
        """

        ...


class StructuredOutputRetryError(ValueError):
    """Raised when structured-output generation exhausts all allowed attempts."""


@dataclass(frozen=True)
class ResilientStructuredOutputInvoker(Generic[StructuredPayloadT]):
    """Structured-output wrapper with a manual JSON fallback path.

    This wrapper first tries provider-native structured output and, if that
    fails, performs bounded retry attempts using explicit JSON-only repair
    instructions. Only JSON parsing is allowed as a fallback; no heuristic
    semantic parsing is used.
    """

    model: StructuredOutputCapableModel
    schema: type[StructuredPayloadT]
    max_attempts: int = 3

    def invoke(self, messages: list[Any]) -> StructuredPayloadT:
        """Return a structured payload for the supplied messages.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Parsed payload matching `schema`.

        Raises:
            ValueError: If both native structured output and fallback parsing fail.
        """

        failure_messages: list[str] = []

        for attempt_index in range(self.max_attempts):
            attempt_number = attempt_index + 1
            strategy = "native_structured_output" if attempt_index == 0 else "json_repair"
            try:
                if attempt_index == 0:
                    emit_custom_debug_event(
                        "structured_output_attempt",
                        schema_name=self.schema.__name__,
                        payload={
                            "attempt_number": attempt_number,
                            "max_attempts": self.max_attempts,
                            "strategy": strategy,
                            "messages": messages,
                        },
                    )
                    native_result = self.model.with_structured_output(self.schema).invoke(messages)
                    if isinstance(native_result, self.schema):
                        emit_custom_debug_event(
                            "structured_output_success",
                            schema_name=self.schema.__name__,
                            payload={
                                "attempt_number": attempt_number,
                                "strategy": strategy,
                                "parsed_payload": native_result.model_dump(),
                            },
                        )
                        return native_result
                    parsed_native_result = self.schema.model_validate(native_result)
                    emit_custom_debug_event(
                        "structured_output_success",
                        schema_name=self.schema.__name__,
                        payload={
                            "attempt_number": attempt_number,
                            "strategy": strategy,
                            "parsed_payload": parsed_native_result.model_dump(),
                        },
                    )
                    return parsed_native_result

                fallback_messages = _augment_messages_with_json_instruction(
                    messages,
                    self.schema,
                    failure_messages,
                )
                emit_custom_debug_event(
                    "structured_output_attempt",
                    schema_name=self.schema.__name__,
                    payload={
                        "attempt_number": attempt_number,
                        "max_attempts": self.max_attempts,
                        "strategy": strategy,
                        "messages": fallback_messages,
                    },
                )
                fallback_message = self.model.invoke(fallback_messages)
                fallback_text = _coerce_message_content(fallback_message.content)
                emit_custom_debug_event(
                    "structured_output_response",
                    schema_name=self.schema.__name__,
                    payload={
                        "attempt_number": attempt_number,
                        "strategy": strategy,
                        "response": fallback_message,
                        "response_text": fallback_text,
                    },
                )
                for candidate_json in _iter_json_candidates(fallback_text):
                    try:
                        parsed_fallback_result = self.schema.model_validate_json(candidate_json)
                        emit_custom_debug_event(
                            "structured_output_success",
                            schema_name=self.schema.__name__,
                            payload={
                                "attempt_number": attempt_number,
                                "strategy": strategy,
                                "parsed_payload": parsed_fallback_result.model_dump(),
                            },
                        )
                        return parsed_fallback_result
                    except Exception as parse_error:
                        failure_messages.append(
                            f"Attempt {attempt_number} JSON validation failed: {parse_error}",
                        )
                failure_messages.append(
                    f"Attempt {attempt_number} did not return valid JSON: {fallback_text!r}",
                )
                emit_custom_debug_event(
                    "structured_output_attempt_failure",
                    schema_name=self.schema.__name__,
                    payload={
                        "attempt_number": attempt_number,
                        "strategy": strategy,
                        "error": failure_messages[-1],
                    },
                )
            except Exception as error:
                failure_messages.append(
                    f"Attempt {attempt_number} failed: {type(error).__name__}: {error}",
                )
                emit_custom_debug_event(
                    "structured_output_attempt_failure",
                    schema_name=self.schema.__name__,
                    payload={
                        "attempt_number": attempt_number,
                        "strategy": strategy,
                        "error": failure_messages[-1],
                    },
                )

        emit_custom_debug_event(
            "structured_output_exhausted",
            schema_name=self.schema.__name__,
            payload={
                "max_attempts": self.max_attempts,
                "failures": failure_messages,
            },
        )
        raise StructuredOutputRetryError(
            "Unable to produce structured output for "
            f"{self.schema.__name__} after {self.max_attempts} attempts. "
            f"Failures: {' | '.join(failure_messages)}"
        )


def _augment_messages_with_json_instruction(
    messages: list[Any],
    schema: type[StructuredPayloadT],
    failure_messages: list[str],
) -> list[Any]:
    """Merge existing system guidance with a JSON-only repair instruction.

    Args:
        messages: Original invocation messages.
        schema: Target Pydantic schema.

    Returns:
        Updated message list with a combined system instruction.
    """

    combined_system_parts = [_coerce_message_content(message.content) for message in messages if isinstance(message, SystemMessage)]
    combined_system_parts.append(_build_json_instruction(schema, failure_messages))
    non_system_messages = [message for message in messages if not isinstance(message, SystemMessage)]
    return [
        SystemMessage(content="\n\n".join(part for part in combined_system_parts if part.strip())),
        *non_system_messages,
    ]


def _build_json_instruction(
    schema: type[StructuredPayloadT],
    failure_messages: list[str],
) -> str:
    """Build a deterministic JSON-only instruction for fallback calls.

    Args:
        schema: Target Pydantic schema.
        failure_messages: Previous validation or parsing failures.

    Returns:
        System-level instruction containing the JSON schema.
    """

    schema_payload = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    failure_summary = "\n".join(f"- {message}" for message in failure_messages[-3:])
    return (
        "You must repair the previous response and return only one valid JSON object. "
        "Do not add markdown fences, explanations, labels, or prefixes.\n\n"
        + (
            f"Previous failures to correct:\n{failure_summary}\n\n"
            if failure_summary
            else ""
        )
        +
        f"JSON schema:\n{schema_payload}"
    )


def _iter_json_candidates(raw_text: str) -> list[str]:
    """Yield candidate JSON strings extracted from a plain-text response.

    Args:
        raw_text: Plain-text model response.

    Returns:
        Candidate JSON fragments ordered from most to least direct.
    """

    stripped_text = raw_text.strip()
    if not stripped_text:
        return []

    candidates: list[str] = [stripped_text]
    fenced_matches = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", stripped_text, re.DOTALL | re.IGNORECASE)
    ]
    candidates.extend(fragment for fragment in fenced_matches if fragment)

    json_decoder = json.JSONDecoder()
    for start_index, character in enumerate(stripped_text):
        if character != "{":
            continue
        try:
            _, end_index = json_decoder.raw_decode(stripped_text[start_index:])
        except json.JSONDecodeError:
            continue
        candidates.append(stripped_text[start_index : start_index + end_index])
        break

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _coerce_message_content(content: object) -> str:
    """Normalize LangChain message content into plain text.

    Args:
        content: Arbitrary message payload.

    Returns:
        Plain-text representation of the message content.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for block in content:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                fragments.append(str(block.get("text", "")))
        return "\n".join(fragment for fragment in fragments if fragment).strip()
    return str(content)