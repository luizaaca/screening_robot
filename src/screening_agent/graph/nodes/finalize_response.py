"""Final response node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import AIMessage

from screening_agent.audit import emit_console_audit
from screening_agent.graph.state import AssistantState, build_active_patient_header, create_audit_event

CLINICAL_DISCLAIMER = (
    "Clinical screening support only. This assistant does not replace professional "
    "medical evaluation, diagnosis, or emergency care."
)



def build_finalize_response_node() -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that composes the final assistant answer.

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

        response_body = (state.get("response_body") or "I do not have a response yet.").strip()
        active_patient = state.get("active_patient")
        header = build_active_patient_header(active_patient) if active_patient else None
        response_sections = [section for section in [header, response_body] if section]
        if state.get("response_requires_disclaimer"):
            response_sections.append(CLINICAL_DISCLAIMER)
        final_response = "\n\n".join(response_sections)
        event = create_audit_event(
            event_type="finalize_response",
            status="success",
            node_name="finalize_response",
            detail="Composed final assistant response.",
        )
        emit_console_audit(event)
        return {
            "active_patient_header": header,
            "last_response": final_response,
            "messages": [AIMessage(content=final_response)],
            "audit_events": [event],
        }

    return finalize_response
