"""Final response node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import json
import re

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import get_last_human_message_text
from screening_agent.graph.state import (
    AssistantState,
    build_active_patient_header,
    create_audit_event,
    mask_security_number,
)
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import FINAL_ANSWER_SYSTEM_PROMPT
from screening_agent.tools.specialist_tool import ClinicalScreeningOutput

_DEFAULT_RESPONSE_BODY = "I could not produce a response for this turn."
_PATIENT_LOOKUP_SUMMARY_INSTRUCTION = (
    "This turn only loaded a patient record. Confirm that the patient context is active "
    "and include a concise descriptive summary of the loaded record using "
    "`state_snapshot.active_patient.clinical_context`. Do not perform symptom analysis, infer new "
    "diagnoses, recommend exams, or add details that are not present in the record."
)
_VIDEO_PATIENT_CORRELATION_INSTRUCTION = (
    "If the latest turn asks about correlation, comparison, or compatibility between "
    "the video evidence and the active patient's history, use `state_snapshot.active_patient` "
    "as the only source for the patient-history side and the video fields as the only "
    "source for the video-evidence side. State uncertainty when the relationship is only "
    "compatible rather than directly established."
)
_SECURITY_NUMBER_PATTERN = re.compile(r"\b\d{8}\b")


def build_finalize_response_node(
    control_model: ControlModel,
) -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that composes the final assistant answer.

    Args:
        control_model: Chat model used to render the final user-facing answer.

    Returns:
        A LangGraph node callable.
    """

    def finalize_response(state: AssistantState) -> dict[str, object]:
        """Compose the final user-facing answer.

        Args:
            state: Current graph state.

        Returns:
            State update with the final response and appended AI message.
        """

        specialist_output = _parse_specialist_output(state)
        header = _resolve_active_patient_header(state, specialist_output=specialist_output)
        draft_response = _resolve_draft_response(state, specialist_output=specialist_output)
        latest_user_message = _resolve_latest_user_message(state)
        prompt_payload = _build_final_answer_context_payload(
            state=state,
            latest_user_message=latest_user_message,
            draft_response=draft_response,
            specialist_output=specialist_output,
            header=header,
            response_instruction=_resolve_response_instruction(state),
        )
        messages = _build_final_answer_messages(prompt_payload)
        emit_custom_debug_event(
            "final_answer_prompt",
            node_name="final_answer",
            payload={"messages": messages},
        )

        try:
            response = control_model.invoke(messages)
            response_text = getattr(response, "text", None)
            if not isinstance(response_text, str) and response_text is not None and not callable(response_text):
                response_text = str(response_text)
            if not isinstance(response_text, str) or not response_text.strip():
                response_text = str(response.content).strip()
            if not response_text:
                raise ValueError("Final-answer model returned empty content.")
            response_text = _mask_security_numbers_in_text(response_text)
            final_response = _normalize_final_response(
                response_text,
                header=header,
            )
            final_message = AIMessage(content=final_response)
            emit_custom_debug_event(
                "final_answer_response",
                node_name="final_answer",
                payload={
                    "response": response,
                    "normalized_response": final_response,
                },
            )
            event = create_audit_event(
                event_type="final_answer",
                status="success",
                node_name="final_answer",
                detail="Generated the final assistant answer with the control model.",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            final_response = (
                "I could not generate the final response with the current model configuration. "
                f"Technical detail: {type(exc).__name__}."
            )
            final_message = AIMessage(content=final_response)
            emit_custom_debug_event(
                "final_answer_failure",
                node_name="final_answer",
                payload={
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            event = create_audit_event(
                event_type="final_answer",
                status="error",
                node_name="final_answer",
                detail=(
                    "Final-answer model failed; returned a technical error instead of a fabricated answer. "
                    f"Error: {type(exc).__name__}: {exc}"
                ),
            )

        emit_console_audit(event)
        emit_custom_debug_event(
            "final_answer_completed",
            node_name="final_answer",
            payload={"final_response": final_response},
        )
        return {
            "last_response": final_response,
            "messages": [final_message],
            "audit_events": [event],
        }

    return finalize_response


def _resolve_active_patient_header(
    state: AssistantState,
    *,
    specialist_output: ClinicalScreeningOutput | None,
) -> str | None:
    """Resolve the header shown above the final response when needed.

    Args:
        state: Current graph state.
        specialist_output: Parsed specialist output, if available.

    Returns:
        Optional active-patient header.
    """

    active_patient = state.get("active_patient")
    if active_patient is None:
        return None

    router_intent = state.get("router_intent")
    if specialist_output is not None:
        return build_active_patient_header(active_patient)
    if router_intent in {"video_interpretation", "video_qa", "video_symptom_analysis"}:
        return build_active_patient_header(active_patient)
    if router_intent == "patient_lookup" and state.get("patient_lookup_status") == "loaded":
        return build_active_patient_header(active_patient)
    return None


def _resolve_latest_user_message(state: AssistantState) -> str:
    """Return the latest user message for language preservation.

    Args:
        state: Current graph state.

    Returns:
        Latest human message text, if available.
    """

    try:
        return get_last_human_message_text(state.get("messages", []))
    except ValueError:
        return ""


def _resolve_draft_response(
    state: AssistantState,
    *,
    specialist_output: ClinicalScreeningOutput | None,
) -> str:
    """Resolve the draft response used by the final-answer model.

    Args:
        state: Current graph state.
        specialist_output: Parsed specialist output, if available.

    Returns:
        Draft response text.
    """

    if specialist_output is not None:
        return ""
    turn_outcome_draft = _draft_from_turn_outcome(state.get("turn_outcome"))
    if turn_outcome_draft is not None:
        return turn_outcome_draft
    video_interpretation = state.get("video_interpretation")
    if isinstance(video_interpretation, str) and video_interpretation.strip():
        return video_interpretation.strip()

    patient_lookup_status = state.get("patient_lookup_status")
    if (
        state.get("router_intent") == "patient_lookup"
        and patient_lookup_status == "loaded"
        and state.get("active_patient") is not None
    ):
        return (
            "Patient context loaded successfully. Use `state_snapshot.active_patient` to summarize "
            "the loaded record."
        )

    last_response = state.get("last_response")
    if isinstance(last_response, str) and last_response.strip():
        return last_response.strip()

    if patient_lookup_status == "loaded":
        return "Patient context loaded successfully."
    if patient_lookup_status == "selection_required":
        return "Multiple patients were found. Please choose one of the numbered options."
    if patient_lookup_status == "not_found":
        return "No patient was found for the provided identifier."
    return _DEFAULT_RESPONSE_BODY


def _draft_from_turn_outcome(turn_outcome: object) -> str | None:
    """Build a concise operational draft from a structured turn outcome."""

    if not isinstance(turn_outcome, dict):
        return None
    outcome_type = turn_outcome.get("type")
    outcome_drafts = {
        "video_upload_confirmation_requested": (
            "A video is needed to answer this request. Ask whether the user wants to upload "
            "a video file now or provide a local path with `video_path=...`."
        ),
        "video_upload_confirmation_unclear": (
            "The answer did not clearly confirm or decline video upload. Ask the user to reply "
            "yes to upload a video, no to continue without it, or provide `video_path=...`."
        ),
        "video_upload_confirmed": (
            "The user confirmed they want to provide a video. Ask them to upload one video file now."
        ),
        "video_upload_still_needed": (
            "The graph is still waiting for the video file or a local path before continuing."
        ),
        "video_upload_declined": (
            "The user declined to provide a video. Explain that video-based analysis cannot continue "
            "without a video file or local path."
        ),
        "video_upload_timeout": (
            "No video file was received before the upload timeout. Explain that the user can try again "
            "or send a local path with `video_path=...`."
        ),
        "video_upload_cancelled": (
            "The video upload was cancelled. Explain that the user can try again or provide a local path."
        ),
        "video_clinical_extraction_failed": (
            "Structured clinical context could not be extracted safely from the video. Explain the technical "
            "failure without inventing clinical findings."
        ),
    }
    if outcome_type == "processing_error":
        detail = turn_outcome.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    draft = outcome_drafts.get(str(outcome_type))
    return draft if isinstance(draft, str) else None


def _parse_specialist_output(state: AssistantState) -> ClinicalScreeningOutput | None:
    """Parse the stored specialist JSON payload.

    Args:
        state: Current graph state.

    Returns:
        Parsed specialist output, if valid and relevant for this turn.
    """

    specialist_output_json = state.get("specialist_output_json")
    if not isinstance(specialist_output_json, str) or not specialist_output_json.strip():
        return None
    try:
        return ClinicalScreeningOutput.model_validate_json(specialist_output_json)
    except Exception:
        return None


def _resolve_response_instruction(state: AssistantState) -> str | None:
    """Return an optional final-answer instruction for the current flow."""

    turn_outcome = state.get("turn_outcome")
    if isinstance(turn_outcome, dict) and turn_outcome.get("type") == "processing_error":
        return None

    if (
        state.get("router_intent") == "patient_lookup"
        and state.get("patient_lookup_status") == "loaded"
        and state.get("active_patient") is not None
    ):
        return _PATIENT_LOOKUP_SUMMARY_INSTRUCTION
    if (
        state.get("router_intent") in {"video_interpretation", "video_qa", "video_symptom_analysis"}
        and state.get("active_patient") is not None
    ):
        return _VIDEO_PATIENT_CORRELATION_INSTRUCTION
    return None


def _build_final_answer_context_payload(
    *,
    state: AssistantState,
    latest_user_message: str,
    draft_response: str,
    specialist_output: ClinicalScreeningOutput | None,
    header: str | None,
    response_instruction: str | None,
) -> dict[str, object]:
    """Build the complete sanitized context passed to the final-answer model.

    Args:
        state: Current graph state.
        latest_user_message: Latest user request text.
        draft_response: Draft response prepared by earlier nodes.
        specialist_output: Parsed specialist output, if available.
        header: Optional active-patient header.
        response_instruction: Optional flow-specific response instruction.

    Returns:
        JSON-serializable payload for the final-answer prompt.
    """

    conversation_history = _normalize_conversation_history(state.get("messages", []))
    state_snapshot = _build_state_snapshot(state, conversation_history=conversation_history)
    turn_outcome = state.get("turn_outcome")
    is_processing_error = (
        isinstance(turn_outcome, dict)
        and turn_outcome.get("type") == "processing_error"
    )
    context_notes = []
    if is_processing_error:
        context_notes.append(
            "turn_outcome is processing_error; previous analysis fields may be stale and "
            "must not be presented as current findings."
        )
    if state.get("active_patient") is not None:
        context_notes.append(
            "An active patient exists in state_snapshot.active_patient; do not claim that "
            "patient history is unavailable."
        )
    if isinstance(state.get("specialist_output_json"), str) and str(state.get("specialist_output_json")).strip():
        context_notes.append(
            "A symptom specialist result exists in state_snapshot.specialist_output_json and "
            "derived_context.specialist_output."
        )

    return {
        "final_answer_context": {
            "latest_user_message": _sanitize_json_value(latest_user_message),
            "response_task": _build_response_task(is_processing_error=is_processing_error),
            "conversation_history": conversation_history,
            "state_snapshot": state_snapshot,
            "derived_context": {
                "active_patient_header": _sanitize_json_value(header),
                "draft_response": _sanitize_json_value(draft_response),
                "response_instruction": _sanitize_json_value(response_instruction)
                if isinstance(response_instruction, str) and response_instruction.strip()
                else None,
                "specialist_output": (
                    _sanitize_json_value(specialist_output.model_dump())
                    if specialist_output is not None
                    else None
                ),
                "video_clinical_context": _parse_json_context(
                    state.get("video_clinical_context_json")
                ),
            },
            "context_notes": context_notes,
        },
    }


def _build_response_task(*, is_processing_error: bool) -> str:
    """Build the task instruction embedded in the final-answer payload."""

    if is_processing_error:
        return (
            "Answer the latest user message using conversation_history and state_snapshot, "
            "but because the current turn has a processing_error, explain the operational "
            "failure clearly and do not reuse stale patient, video, or specialist evidence "
            "as if it had just been produced."
        )
    return (
        "Answer the latest user message using all available information in conversation_history "
        "and state_snapshot. Prefer state facts over assumptions. Consider the active patient, "
        "clinical history, specialist output, video analysis, video interpretation, and "
        "video clinical context whenever they are present."
    )


def _build_state_snapshot(
    state: AssistantState,
    *,
    conversation_history: list[dict[str, object]],
) -> dict[str, object]:
    """Return a sanitized JSON-safe snapshot containing every state key."""

    snapshot: dict[str, object] = {}
    for key, value in state.items():
        key_text = str(key)
        if key_text == "messages":
            snapshot[key_text] = conversation_history
        else:
            snapshot[key_text] = _sanitize_json_value(value)
    if "messages" not in snapshot:
        snapshot["messages"] = conversation_history
    return snapshot


def _normalize_conversation_history(messages: object) -> list[dict[str, object]]:
    """Normalize graph messages into JSON-safe role/content/tool records."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        return []
    normalized_messages: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        normalized_messages.append(_normalize_message(message, index=index))
    return normalized_messages


def _normalize_message(message: object, *, index: int) -> dict[str, object]:
    """Normalize one LangChain message or message-like object."""

    message_type = str(getattr(message, "type", message.__class__.__name__))
    entry: dict[str, object] = {
        "index": index,
        "type": _sanitize_json_value(message_type),
        "role": _message_role(message_type),
        "content": _sanitize_json_value(getattr(message, "content", str(message))),
    }

    for attribute in ("name", "id", "tool_call_id"):
        value = getattr(message, attribute, None)
        if value:
            entry[attribute] = _sanitize_json_value(value)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = _sanitize_json_value(tool_calls)

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        entry["invalid_tool_calls"] = _sanitize_json_value(invalid_tool_calls)

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, Mapping) and additional_kwargs:
        entry["additional_kwargs"] = _sanitize_json_value(additional_kwargs)

    return entry


def _message_role(message_type: str) -> str:
    """Map LangChain message types to prompt-facing conversation roles."""

    normalized_type = message_type.lower()
    role_by_type = {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return role_by_type.get(normalized_type, normalized_type)


def _parse_json_context(value: object) -> object:
    """Parse a JSON context string when possible, otherwise return a sanitized value."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _sanitize_json_value(json.loads(value))
    except json.JSONDecodeError:
        return _sanitize_json_value(value)


def _sanitize_json_value(value: object) -> object:
    """Convert arbitrary state values into JSON-safe sanitized data."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _mask_security_numbers_in_text(value)
    if isinstance(value, Mapping):
        sanitized_mapping: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "security_number":
                sanitized_mapping[key_text] = mask_security_number(str(item))
            else:
                sanitized_mapping[key_text] = _sanitize_json_value(item)
        return sanitized_mapping
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [_sanitize_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _sanitize_json_value(model_dump())
    return _mask_security_numbers_in_text(str(value))


def _mask_security_numbers_in_text(text: str) -> str:
    """Mask 8-digit fictional security numbers in free text."""

    return _SECURITY_NUMBER_PATTERN.sub(
        lambda match: mask_security_number(match.group(0)),
        text,
    )


def _build_final_answer_messages(payload: dict[str, object]) -> list[object]:
    """Build the messages sent to the final-answer model.

    Args:
        payload: Structured final-answer payload.

    Returns:
        LangChain-compatible message list.
    """

    return [
        SystemMessage(content=FINAL_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


def _normalize_final_response(
    response_text: str,
    *,
    header: str | None,
) -> str:
    """Ensure mandatory response sections are present in the final answer.

    Args:
        response_text: Model-generated final answer text.
        header: Optional active-patient header.

    Returns:
        Final answer with mandatory sections guaranteed.
    """

    normalized_response = response_text.strip()
    if header and header not in normalized_response:
        normalized_response = f"{header}\n\n{normalized_response}".strip()
    return normalized_response
