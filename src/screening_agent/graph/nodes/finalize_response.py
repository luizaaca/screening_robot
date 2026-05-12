"""Final response node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable
import json

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import get_last_human_message_text
from screening_agent.graph.state import AssistantState, build_active_patient_header, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import FINAL_ANSWER_SYSTEM_PROMPT
from screening_agent.tools.specialist_tool import ClinicalScreeningOutput

CLINICAL_DISCLAIMER = (
    "Clinical screening support only. This assistant does not replace professional "
    "medical evaluation, diagnosis, or emergency care."
)
_DEFAULT_RESPONSE_BODY = "I could not produce a response for this turn."


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
        prompt_payload = _build_final_answer_payload(
            latest_user_message=latest_user_message,
            draft_response=draft_response,
            specialist_output=specialist_output,
            header=header,
        )
        fallback_response = _assemble_fallback_final_response(
            header=header,
            draft_response=draft_response,
            specialist_output=specialist_output,
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
            final_response = _normalize_final_response(
                response_text,
                header=header,
                include_disclaimer=specialist_output is not None,
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
            final_response = fallback_response
            final_message = AIMessage(content=final_response)
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
    last_response = state.get("last_response")
    if isinstance(last_response, str) and last_response.strip():
        return last_response.strip()

    patient_lookup_status = state.get("patient_lookup_status")
    if patient_lookup_status == "loaded":
        return "Patient context loaded successfully."
    if patient_lookup_status == "selection_required":
        return "Multiple patients were found. Please choose one of the numbered options."
    if patient_lookup_status == "not_found":
        return "No patient was found for the provided identifier."
    return _DEFAULT_RESPONSE_BODY


def _parse_specialist_output(state: AssistantState) -> ClinicalScreeningOutput | None:
    """Parse the stored specialist JSON payload.

    Args:
        state: Current graph state.

    Returns:
        Parsed specialist output, if valid and relevant for this turn.
    """

    if state.get("router_intent") not in {"symptom_analysis", "patient_lookup_then_analysis"}:
        return None
    specialist_output_json = state.get("specialist_output_json")
    if not isinstance(specialist_output_json, str) or not specialist_output_json.strip():
        return None
    try:
        return ClinicalScreeningOutput.model_validate_json(specialist_output_json)
    except Exception:
        return None


def _assemble_fallback_final_response(
    *,
    header: str | None,
    draft_response: str,
    specialist_output: ClinicalScreeningOutput | None,
) -> str:
    """Assemble the deterministic final response used as a safety fallback.

    Args:
        header: Optional active-patient header.
        draft_response: Draft response prepared by earlier nodes.
        specialist_output: Parsed specialist output, if available.

    Returns:
        Deterministically assembled final response.
    """

    if specialist_output is None:
        response_sections = [section for section in [header, draft_response or _DEFAULT_RESPONSE_BODY] if section]
        return "\n\n".join(response_sections)

    if specialist_output.support_status == "inconclusive":
        body = (
            "The available information is inconclusive. Candidate conditions to consider: "
            f"{', '.join(specialist_output.candidate_diseases)}."
        )
    else:
        body = (
            "Most likely conditions to consider: "
            f"{', '.join(specialist_output.candidate_diseases)}."
        )

    exams_line = (
        "Recommended exams/tests: "
        f"{', '.join(specialist_output.recommended_exams_tests)}."
    )
    sections = [section for section in [header, body, exams_line, CLINICAL_DISCLAIMER] if section]
    return "\n\n".join(sections)


def _build_final_answer_payload(
    *,
    latest_user_message: str,
    draft_response: str,
    specialist_output: ClinicalScreeningOutput | None,
    header: str | None,
) -> dict[str, object]:
    """Build the structured payload passed to the final-answer model.

    Args:
        latest_user_message: Latest user request text.
        draft_response: Draft response prepared by earlier nodes.
        specialist_output: Parsed specialist output, if available.
        header: Optional active-patient header.

    Returns:
        JSON-serializable payload for the final-answer prompt.
    """

    return {
        "active_patient_header": header,
        "latest_user_message": latest_user_message,
        "draft_response": draft_response,
        "specialist_output": (
            specialist_output.model_dump() if specialist_output is not None else None
        ),
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
