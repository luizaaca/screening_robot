"""Clear-active-patient node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import SystemMessage

from screening_agent.audit import emit_console_audit
from screening_agent.graph.message_utils import get_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import CLEAR_ACTIVE_PATIENT_SYSTEM_PROMPT



def build_clear_active_patient_node(control_model: ControlModel) -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that clears the active patient context.

    Args:
        control_model: Chat model used to generate the confirmation message.

    Returns:
        A LangGraph node callable.
    """

    def clear_active_patient(state: AssistantState) -> dict[str, object]:
        """Clear the active patient context and confirm the action.

        Args:
            state: Current graph state.

        Returns:
            State update clearing patient-related fields.
        """

        active_patient = state.get("active_patient")
        prompt = "\n\n".join(
            [
                CLEAR_ACTIVE_PATIENT_SYSTEM_PROMPT,
                (
                    f"Active patient before clearing: {active_patient['full_name']}"
                    if active_patient
                    else "Active patient before clearing: none"
                ),
            ],
        )
        response = control_model.invoke(
            [SystemMessage(content=prompt), *state.get("messages", [])[-4:]],
        )
        response_text = get_message_text(response)
        event = create_audit_event(
            event_type="clear_active_patient",
            status="success",
            node_name="clear_active_patient",
            detail="Cleared the active patient context.",
        )
        emit_console_audit(event)
        return {
            "active_patient": None,
            "patient_lookup_status": None,
            "patient_lookup_candidates": [],
            "specialist_output_json": None,
            "last_response": response_text,
            "audit_events": [event],
        }

    return clear_active_patient
