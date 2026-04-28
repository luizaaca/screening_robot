"""Symptom-analysis node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import get_last_human_message_text
from screening_agent.graph.state import AssistantState, ClinicalAnalysisResult, create_audit_event
from screening_agent.model.clinical_backend import ClinicalBackend, ClinicalAnalysisPayload



def build_symptom_analysis_node(
    clinical_backend: ClinicalBackend,
) -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that runs structured clinical analysis.

    Args:
        clinical_backend: Backend responsible for clinical reasoning.

    Returns:
        A LangGraph node callable.
    """

    def symptom_analysis(state: AssistantState) -> dict[str, object]:
        """Analyze the user's clinical request.

        Args:
            state: Current graph state.

        Returns:
            State update with structured analysis and response text.
        """

        latest_user_message = get_last_human_message_text(state.get("messages", []))
        emit_custom_debug_event(
            "symptom_analysis_request",
            node_name="symptom_analysis",
            backend_name=getattr(clinical_backend, "backend_name", "unknown"),
            payload={
                "user_message": latest_user_message,
                "active_patient": state.get("active_patient"),
            },
        )
        try:
            analysis_payload = clinical_backend.analyze(
                user_message=latest_user_message,
                active_patient=state.get("active_patient"),
            )
            analysis_result = _to_state_analysis_result(
                analysis_payload,
                backend_name=clinical_backend.backend_name,
            )
            event = create_audit_event(
                event_type="symptom_analysis",
                status="success",
                node_name="symptom_analysis",
                detail=(
                    "Generated structured clinical analysis with status "
                    f"{analysis_result['status']}."
                ),
            )
            emit_custom_debug_event(
                "symptom_analysis_result",
                node_name="symptom_analysis",
                backend_name=clinical_backend.backend_name,
                payload={"analysis_result": analysis_result},
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            analysis_result = {
                "status": "insufficient_information",
                "primary_hypothesis": None,
                "differential_hypotheses": [],
                "recommended_exams": [],
                "reasoning_summary": f"Clinical backend failure: {exc}",
                "safety_notes": [
                    "Review the backend configuration before relying on the screening summary.",
                ],
                "user_response": (
                    "I could not generate a safe structured screening summary with the current clinical backend. "
                    "Please review the configuration and try again before relying on this workflow."
                ),
                "backend_name": getattr(clinical_backend, "backend_name", "unknown"),
            }
            event = create_audit_event(
                event_type="symptom_analysis",
                status="error",
                node_name="symptom_analysis",
                detail=f"Clinical backend execution failed: {exc}",
            )
            emit_custom_debug_event(
                "symptom_analysis_failure",
                node_name="symptom_analysis",
                backend_name=getattr(clinical_backend, "backend_name", "unknown"),
                payload={"error": str(exc)},
            )
        emit_console_audit(event)
        return {
            "analysis_result": analysis_result,
            "response_kind": "analysis",
            "response_body": analysis_result["user_response"],
            "response_requires_disclaimer": True,
            "audit_events": [event],
        }

    return symptom_analysis



def _to_state_analysis_result(
    payload: ClinicalAnalysisPayload,
    *,
    backend_name: str,
) -> ClinicalAnalysisResult:
    """Convert backend payload into the state representation.

    Args:
        payload: Backend payload returned by the clinical model.
        backend_name: Name of the runtime backend.

    Returns:
        State-compatible analysis result.
    """

    normalized_status = payload.status
    if normalized_status not in {"analysis_ready", "insufficient_information", "urgent_attention"}:
        normalized_status = "insufficient_information"
    return {
        "status": normalized_status,
        "primary_hypothesis": payload.primary_hypothesis,
        "differential_hypotheses": payload.differential_hypotheses,
        "recommended_exams": payload.recommended_exams,
        "reasoning_summary": payload.reasoning_summary,
        "safety_notes": payload.safety_notes,
        "user_response": payload.user_response,
        "backend_name": backend_name,
    }
