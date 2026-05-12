"""Fail-closed node for processing errors in the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import AIMessage

from screening_agent.audit import emit_console_audit
from screening_agent.graph.state import AssistantState, create_audit_event

PROCESSING_ERROR_RESPONSE = (
    "I could not complete this request safely after repeated structured-processing attempts. "
    "Please restate the request or review the model configuration before relying on this workflow."
)


def build_processing_error_node() -> Callable[[AssistantState], dict[str, object]]:
    """Build the deterministic fail-closed response node.

    Returns:
        A LangGraph node callable.
    """

    def processing_error(state: AssistantState) -> dict[str, object]:
        """Return a professional fail-closed response.

        Args:
            state: Current graph state.

        Returns:
            State update with a deterministic processing-error response.
        """

        detail = state.get("processing_error_detail") or PROCESSING_ERROR_RESPONSE
        event = create_audit_event(
            event_type="processing_error",
            status="error",
            node_name="processing_error",
            detail=detail,
        )
        emit_console_audit(event)
        return {
            "last_response": detail,
            "specialist_output_json": None,
            "messages": [AIMessage(content=detail)],
            "audit_events": [event],
        }

    return processing_error