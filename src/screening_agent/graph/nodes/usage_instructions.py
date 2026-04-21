"""Usage-instructions node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import SystemMessage

from screening_agent.audit import emit_console_audit
from screening_agent.graph.message_utils import coerce_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import USAGE_INSTRUCTIONS_SYSTEM_PROMPT



def build_usage_instructions_node(control_model: ControlModel) -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that explains how the assistant should be used.

    Args:
        control_model: Chat model used to generate a concise answer.

    Returns:
        A LangGraph node callable.
    """

    def usage_instructions(state: AssistantState) -> dict[str, object]:
        """Generate user-facing usage instructions.

        Args:
            state: Current graph state.

        Returns:
            State update with the user-facing response body.
        """

        response = control_model.invoke(
            [
                SystemMessage(content=USAGE_INSTRUCTIONS_SYSTEM_PROMPT),
                *state.get("messages", [])[-4:],
            ],
        )
        response_text = coerce_message_text(response.content)
        event = create_audit_event(
            event_type="usage_instructions",
            status="success",
            node_name="usage_instructions",
            detail="Generated usage instructions response.",
        )
        emit_console_audit(event)
        return {
            "response_kind": "usage",
            "response_body": response_text,
            "response_requires_disclaimer": False,
            "audit_events": [event],
        }

    return usage_instructions
