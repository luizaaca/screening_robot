"""Final response node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable
import json

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import coerce_message_text
from screening_agent.graph.state import AssistantState, build_active_patient_header, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import FINAL_ANSWER_SYSTEM_PROMPT

CLINICAL_DISCLAIMER = (
    "Clinical screening support only. This assistant does not replace professional "
    "medical evaluation, diagnosis, or emergency care."
)
_DEFAULT_RESPONSE_BODY = "I do not have a response yet."


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

        header = _resolve_active_patient_header(state)
        prompt_payload = _build_final_answer_payload(state, header=header)
        fallback_response = _assemble_fallback_final_response(state, header=header)
        messages = _build_final_answer_messages(prompt_payload)
        emit_custom_debug_event(
            "final_answer_prompt",
            node_name="final_answer",
            payload={"messages": messages},
        )

        try:
            response = control_model.invoke(messages)
            response_text = coerce_message_text(response.content).strip()
            if not response_text:
                raise ValueError("Final-answer model returned empty content.")
            final_response = _normalize_final_response(
                response_text,
                header=header,
                include_disclaimer=bool(state.get("response_requires_disclaimer")),
            )
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
            final_response = fallback_response
            emit_custom_debug_event(
                "final_answer_fallback",
                node_name="final_answer",
                payload={
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_response": fallback_response,
                },
            )
            event = create_audit_event(
                event_type="final_answer",
                status="warning",
                node_name="final_answer",
                detail=(
                    "Final-answer model failed; fell back to deterministic response assembly. "
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
            "active_patient_header": header,
            "last_response": final_response,
            "messages": [AIMessage(content=final_response)],
            "audit_events": [event],
        }

    return finalize_response


def _resolve_active_patient_header(state: AssistantState) -> str | None:
    """Resolve the header shown above the final response when a patient is active.

    Args:
        state: Current graph state.

    Returns:
        Existing or recomputed patient header.
    """

    active_patient_header = state.get("active_patient_header")
    if isinstance(active_patient_header, str) and active_patient_header.strip():
        return active_patient_header.strip()
    active_patient = state.get("active_patient")
    if active_patient:
        return build_active_patient_header(active_patient)
    return None


def _assemble_fallback_final_response(
    state: AssistantState,
    *,
    header: str | None,
) -> str:
    """Assemble the deterministic final response used as a safety fallback.

    Args:
        state: Current graph state.
        header: Optional active-patient header.

    Returns:
        Deterministically assembled final response.
    """

    response_body = (state.get("response_body") or _DEFAULT_RESPONSE_BODY).strip()
    response_sections = [section for section in [header, response_body] if section]
    if state.get("response_requires_disclaimer"):
        response_sections.append(CLINICAL_DISCLAIMER)
    return "\n\n".join(response_sections)


def _build_final_answer_payload(
    state: AssistantState,
    *,
    header: str | None,
) -> dict[str, object]:
    """Build the structured payload passed to the final-answer model.

    Args:
        state: Current graph state.
        header: Optional active-patient header.

    Returns:
        JSON-serializable payload for the final-answer prompt.
    """

    return {
        "active_patient_header": header,
        "response_body": (state.get("response_body") or _DEFAULT_RESPONSE_BODY).strip(),
        "response_requires_disclaimer": bool(state.get("response_requires_disclaimer")),
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }


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
    include_disclaimer: bool,
) -> str:
    """Ensure mandatory response sections are present in the final answer.

    Args:
        response_text: Model-generated final answer text.
        header: Optional active-patient header.
        include_disclaimer: Whether the clinical disclaimer must be appended.

    Returns:
        Final answer with mandatory sections guaranteed.
    """

    normalized_response = response_text.strip()
    if header and header not in normalized_response:
        normalized_response = f"{header}\n\n{normalized_response}".strip()
    if include_disclaimer and CLINICAL_DISCLAIMER not in normalized_response:
        normalized_response = f"{normalized_response}\n\n{CLINICAL_DISCLAIMER}".strip()
    return normalized_response
