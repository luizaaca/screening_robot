"""Router node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from langchain.messages import SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.state import AssistantState, RouteIntent, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.model.structured_output import ResilientStructuredOutputInvoker, StructuredOutputRetryError
from screening_agent.prompts import ROUTER_SYSTEM_PROMPT


class RouteDecision(BaseModel):
    """Structured router output used by the root graph."""

    intent: RouteIntent = Field(description="Chosen workflow intent.")
    rationale: str = Field(description="Short explanation for the routing choice.")


def build_router_node(
    control_model: ControlModel,
) -> Callable[[AssistantState], Command[Literal[
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "clear_active_patient",
    "invalid_request",
    "processing_error",
]]]:
    """Build the structured router node.

    Args:
        control_model: Tool-capable chat model used for routing decisions.

    Returns:
        A LangGraph node callable.
    """

    structured_router = ResilientStructuredOutputInvoker(
        model=control_model,
        schema=RouteDecision,
        max_attempts=3,
    )

    def route_request(
        state: AssistantState,
    ) -> Command[Literal[
        "usage_instructions",
        "patient_lookup",
        "symptom_analysis",
        "clear_active_patient",
        "invalid_request",
        "processing_error",
    ]]:
        """Choose the next workflow node from the latest user turn.

        Args:
            state: Current graph state.

        Returns:
            A `Command` with updated router metadata and the target node.
        """

        router_messages = [
            SystemMessage(content=_build_router_prompt(state)),
            *state.get("messages", [])[-6:],
        ]
        emit_custom_debug_event(
            "router_prompt",
            node_name="router",
            payload={"messages": router_messages},
        )
        try:
            decision = structured_router.invoke(router_messages)
        except StructuredOutputRetryError as error:
            emit_custom_debug_event(
                "router_failure",
                node_name="router",
                payload={"error": str(error)},
            )
            event = create_audit_event(
                event_type="routing",
                status="error",
                node_name="router",
                detail=(
                    "Routing failed closed after repeated structured-output attempts. "
                    f"Error: {error}"
                ),
            )
            emit_console_audit(event)
            return Command(
                update={
                    "processing_error_detail": (
                        "I could not safely classify the request after repeated structured-output attempts."
                    ),
                    "audit_events": [event],
                },
                goto="processing_error",
            )
        goto = _map_intent_to_node(decision.intent)
        emit_custom_debug_event(
            "router_result",
            node_name="router",
            payload={
                "decision": decision.model_dump(),
                "goto": goto,
            },
        )
        event = create_audit_event(
            event_type="routing",
            status="success",
            node_name="router",
            detail=f"Routed request to {goto}: {decision.rationale}",
        )
        emit_console_audit(event)
        return Command(
            update={
                "router_intent": decision.intent,
                "router_rationale": decision.rationale,
                "processing_error_detail": None,
                "audit_events": [event],
            },
            goto=goto,
        )

    return route_request



def _build_router_prompt(state: AssistantState) -> str:
    """Augment the static router prompt with live session context.

    Args:
        state: Current graph state.

    Returns:
        Full router prompt with session summary.
    """

    active_patient = state.get("active_patient")
    candidates = state.get("patient_lookup_candidates", [])
    session_lines = [
        ROUTER_SYSTEM_PROMPT,
        "",
        "Session context:",
        f"- Active patient loaded: {'yes' if active_patient else 'no'}",
        f"- Pending patient candidates: {len(candidates)}",
        f"- Last lookup status: {state.get('patient_lookup_status')}",
    ]
    if active_patient:
        session_lines.append(f"- Active patient name: {active_patient['full_name']}")
    return "\n".join(session_lines)



def _map_intent_to_node(
    intent: RouteIntent,
) -> Literal[
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "clear_active_patient",
    "invalid_request",
]:
    """Map an intent to the actual node name used in the root graph.

    Args:
        intent: Structured router intent.

    Returns:
        Concrete node name.
    """

    if intent in {"patient_lookup", "patient_lookup_then_analysis"}:
        return "patient_lookup"
    return cast(
        Literal[
            "usage_instructions",
            "patient_lookup",
            "symptom_analysis",
            "clear_active_patient",
            "invalid_request",
        ],
        intent,
    )
