"""Helpers for resilient structured-output generation across chat backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Generic, Protocol, TypeVar

from langchain.messages import AIMessage, SystemMessage
from pydantic import BaseModel

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


@dataclass(frozen=True)
class ResilientStructuredOutputInvoker(Generic[StructuredPayloadT]):
    """Structured-output wrapper with a manual JSON fallback path.

    Native structured-output support varies across OpenAI-compatible providers.
    This wrapper first tries the provider-native implementation and, when that
    fails, retries with an explicit JSON-only instruction and parses the text
    response locally.
    """

    model: StructuredOutputCapableModel
    schema: type[StructuredPayloadT]
    text_parser: Callable[[str], StructuredPayloadT | None] | None = None

    def invoke(self, messages: list[Any]) -> StructuredPayloadT:
        """Return a structured payload for the supplied messages.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Parsed payload matching `schema`.

        Raises:
            ValueError: If both native structured output and fallback parsing fail.
        """

        try:
            native_result = self.model.with_structured_output(self.schema).invoke(messages)
            if isinstance(native_result, self.schema):
                return native_result
            return self.schema.model_validate(native_result)
        except Exception as native_error:  # pragma: no cover - exercised through fallback tests.
            fallback_message = self.model.invoke(
                _augment_messages_with_json_instruction(messages, self.schema),
            )
            fallback_text = _coerce_message_content(fallback_message.content)
            parsed_result = self._parse_fallback_text(fallback_text)
            if parsed_result is not None:
                return parsed_result
            raise ValueError(
                "Unable to parse structured output for "
                f"{self.schema.__name__}. Native error: {native_error}. "
                f"Fallback content: {fallback_text!r}"
            ) from native_error

    def _parse_fallback_text(self, raw_text: str) -> StructuredPayloadT | None:
        """Parse manual fallback text into the target schema.

        Args:
            raw_text: Plain-text content returned by the fallback model call.

        Returns:
            Parsed structured payload when successful, otherwise `None`.
        """

        for candidate_json in _iter_json_candidates(raw_text):
            try:
                return self.schema.model_validate_json(candidate_json)
            except Exception:
                continue

        if self.text_parser is not None:
            return self.text_parser(raw_text)
        return None


def _augment_messages_with_json_instruction(
    messages: list[Any],
    schema: type[StructuredPayloadT],
) -> list[Any]:
    """Merge existing system guidance with a JSON-only repair instruction.

    Args:
        messages: Original invocation messages.
        schema: Target Pydantic schema.

    Returns:
        Updated message list with a combined system instruction.
    """

    combined_system_parts = [_coerce_message_content(message.content) for message in messages if isinstance(message, SystemMessage)]
    combined_system_parts.append(_build_json_instruction(schema))
    non_system_messages = [message for message in messages if not isinstance(message, SystemMessage)]
    return [
        SystemMessage(content="\n\n".join(part for part in combined_system_parts if part.strip())),
        *non_system_messages,
    ]


def _build_json_instruction(schema: type[StructuredPayloadT]) -> str:
    """Build a deterministic JSON-only instruction for fallback calls.

    Args:
        schema: Target Pydantic schema.

    Returns:
        System-level instruction containing the JSON schema.
    """

    schema_payload = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Return only a valid JSON object that satisfies the schema below. "
        "Do not add markdown fences, explanations, or prefixes.\n\n"
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