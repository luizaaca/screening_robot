"""Symptom-analysis node for the screening assistant graph."""

from __future__ import annotations

from typing import Any
import json

from langchain.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import get_last_human_message_text, get_last_tool_message, get_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import CLINICAL_ANALYSIS_SYSTEM_PROMPT
from screening_agent.tools.specialist_tool import (
    ClinicalScreeningOutput,
    SpecialistInvoker,
    build_symptom_specialist_tool,
)

_TOOL_EXECUTION_REPAIR_MESSAGE = (
    "Error: The specialist tool call could not be completed. Review the tool arguments and try again."
)
_SPECIALIST_FAILURE_RESPONSE = (
    "I could not generate a structured screening result safely with the current specialist configuration. "
    "Please review the model setup before relying on this workflow."
)


def build_symptom_analysis_node(
    control_model: ControlModel,
    specialist_invoker: SpecialistInvoker,
) -> Any:
    """Build the specialist tool-calling subgraph.

    Args:
        control_model: Tool-capable chat model used to call the specialist tool.
        specialist_invoker: Structured specialist invoker.

    Returns:
        A compiled LangGraph subgraph.
    """

    specialist_tool = build_symptom_specialist_tool(specialist_invoker)
    tool_model = control_model.bind_tools([specialist_tool])

    def specialist_agent(state: AssistantState) -> dict[str, object]:
        """Request a specialist tool call from the control model.

        Args:
            state: Current graph state.

        Returns:
            AI message update that should contain a specialist tool call.
        """

        latest_user_message = get_last_human_message_text(state.get("messages", []))
        active_patient = state.get("active_patient")
        active_patient_context = (
            active_patient["clinical_context"]
            if active_patient is not None
            else "No active patient context was loaded."
        )
        prompt = "\n\n".join(
            [
                CLINICAL_ANALYSIS_SYSTEM_PROMPT,
                "Workflow rules:",
                "- Call the `run_symptom_specialist` tool exactly once.",
                "- Pass the latest user symptom request as `clinical_request`.",
                "- Pass the active patient context string as `active_patient_context`.",
                "- Do not answer directly before the tool call.",
                "",
                "Resolved active patient context:",
                active_patient_context,
                "",
                "Latest user message:",
                latest_user_message,
            ],
        )
        specialist_messages = [SystemMessage(content=prompt), *state.get("messages", [])[-6:]]
        emit_custom_debug_event(
            "symptom_analysis_prompt",
            node_name="symptom_analysis_agent",
            payload={"messages": specialist_messages},
        )
        response = tool_model.invoke(specialist_messages)
        emit_custom_debug_event(
            "symptom_analysis_tool_request",
            node_name="symptom_analysis_agent",
            payload={"response": response},
        )
        return {"messages": [response]}

    def capture_specialist_output(state: AssistantState) -> dict[str, object]:
        """Validate the specialist tool result and store its JSON payload.

        Args:
            state: Current graph state.

        Returns:
            State update with stored specialist JSON or a fail-closed message.
        """

        try:
            tool_message = get_last_tool_message(state.get("messages", []))
        except ValueError:
            event = create_audit_event(
                event_type="symptom_analysis",
                status="error",
                node_name="capture_specialist_output",
                detail="The specialist flow ended without a tool result.",
            )
            emit_console_audit(event)
            return {
                "specialist_output_json": None,
                "last_response": _SPECIALIST_FAILURE_RESPONSE,
                "audit_events": [event],
            }

        raw_tool_text = get_message_text(tool_message).strip()
        if raw_tool_text.lower().startswith("error:"):
            event = create_audit_event(
                event_type="symptom_analysis",
                status="error",
                node_name="capture_specialist_output",
                detail=f"Specialist tool execution failed: {raw_tool_text}",
            )
            emit_console_audit(event)
            return {
                "specialist_output_json": None,
                "last_response": _SPECIALIST_FAILURE_RESPONSE,
                "audit_events": [event],
            }

        payload = ClinicalScreeningOutput.model_validate_json(raw_tool_text)
        specialist_output_json = json.dumps(payload.model_dump(), ensure_ascii=False)
        event = create_audit_event(
            event_type="symptom_analysis",
            status="success",
            node_name="capture_specialist_output",
            detail=(
                "Captured specialist output with support status "
                f"{payload.support_status}."
            ),
        )
        emit_custom_debug_event(
            "symptom_analysis_result",
            node_name="capture_specialist_output",
            payload={"specialist_output": payload.model_dump()},
        )
        emit_console_audit(event)
        return {
            "specialist_output_json": specialist_output_json,
            "last_response": None,
            "audit_events": [event],
        }

    builder = StateGraph(AssistantState)
    builder.add_node("symptom_analysis_agent", specialist_agent)
    builder.add_node(
        "symptom_specialist_tool",
        ToolNode([specialist_tool], handle_tool_errors=_TOOL_EXECUTION_REPAIR_MESSAGE),
    )
    builder.add_node("capture_specialist_output", capture_specialist_output)
    builder.add_edge(START, "symptom_analysis_agent")
    builder.add_conditional_edges(
        "symptom_analysis_agent",
        tools_condition,
        {
            "tools": "symptom_specialist_tool",
            "__end__": "capture_specialist_output",
        },
    )
    builder.add_edge("symptom_specialist_tool", "capture_specialist_output")
    builder.add_edge("capture_specialist_output", END)
    return builder.compile()
