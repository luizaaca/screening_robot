"""Invalid-request node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import SystemMessage

from screening_agent.audit import emit_console_audit
from screening_agent.graph.message_utils import coerce_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import INVALID_REQUEST_SYSTEM_PROMPT



def build_invalid_request_node(control_model: ControlModel) -> Callable[[AssistantState], dict[str, object]]:
    """Build the node that handles out-of-scope requests.

    Args:
        control_model: Chat model used to generate the restricted response.

    Returns:
        A LangGraph node callable.
    """

    def invalid_request(state: AssistantState) -> dict[str, object]:
        """Generate a restricted out-of-scope answer.

        Args:
            state: Current graph state.

        Returns:
            State update with the user-facing response body.
        """

        response = control_model.invoke(
            [
                SystemMessage(content=INVALID_REQUEST_SYSTEM_PROMPT),
                *state.get("messages", [])[-4:],
            ],
        )
        response_text = coerce_message_text(response.content)
        event = create_audit_event(
            event_type="invalid_request",
            status="success",
            node_name="invalid_request",
            detail="Generated restricted out-of-scope response.",
        )
        emit_console_audit(event)
        return {
            "response_kind": "invalid",
            "response_body": response_text,
            "response_requires_disclaimer": False,
            "audit_events": [event],
        }

    return invalid_request
