"""Patient lookup subgraph for tool-calling retrieval flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.data import PatientRepository
from screening_agent.graph.message_utils import get_last_ai_message_text, get_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import PATIENT_LOOKUP_SYSTEM_PROMPT
from screening_agent.tools import build_patient_lookup_tools

MAX_LOOKUP_TOOL_REPAIR_ATTEMPTS = 3
TOOL_EXECUTION_REPAIR_MESSAGE = (
    "Error: The patient-lookup tool call could not be completed. "
    "Review the tool arguments and try again."
)
LOOKUP_PROCESSING_ERROR_RESPONSE = (
    "I could not complete the patient lookup safely after repeated tool-execution failures. "
    "Please restate the patient identifier or review the control-model configuration before relying on this workflow."
)



def build_patient_lookup_subgraph(
    control_model: ControlModel,
    repository: PatientRepository,
) -> Any:
    """Build the patient lookup tool-calling subgraph.

    Args:
        control_model: Tool-capable chat model.
        repository: Repository used by lookup tools.

    Returns:
        A compiled LangGraph subgraph.
    """

    tools = build_patient_lookup_tools(repository)
    lookup_model = control_model.bind_tools(tools)

    def patient_lookup_agent(state: AssistantState) -> dict[str, object]:
        """Run the lookup LLM that decides when to call patient tools.

        Args:
            state: Current graph state.

        Returns:
            A message update with the model response.
        """

        lookup_messages = [
            SystemMessage(content=_build_lookup_prompt(state)),
            *state.get("messages", []),
        ]
        emit_custom_debug_event(
            "patient_lookup_prompt",
            node_name="patient_lookup_agent",
            payload={"messages": lookup_messages},
        )
        response = lookup_model.invoke(lookup_messages)
        emit_custom_debug_event(
            "patient_lookup_response",
            node_name="patient_lookup_agent",
            payload={"response": response},
        )
        return {"messages": [response]}

    def evaluate_tool_results(
        state: AssistantState,
    ) -> Command[Literal["patient_lookup_agent", "lookup_processing_error"]]:
        """Inspect tool results and enforce a bounded repair loop.

        Args:
            state: Current graph state.

        Returns:
            Command routing back to the agent or to a fail-closed subgraph exit.
        """

        retry_count = state.get("patient_lookup_retry_count", 0)
        if _last_tool_message_is_execution_error(state):
            retry_count += 1
        else:
            retry_count = 0

        if retry_count >= MAX_LOOKUP_TOOL_REPAIR_ATTEMPTS:
            emit_custom_debug_event(
                "patient_lookup_tool_repair_exhausted",
                node_name="evaluate_tool_results",
                payload={"retry_count": retry_count},
            )
            event = create_audit_event(
                event_type="patient_lookup_tool_repair",
                status="error",
                node_name="evaluate_tool_results",
                detail="Patient lookup failed closed after repeated tool-execution errors.",
            )
            emit_console_audit(event)
            return Command(
                update={
                    "patient_lookup_retry_count": retry_count,
                    "processing_error_detail": LOOKUP_PROCESSING_ERROR_RESPONSE,
                    "audit_events": [event],
                },
                goto="lookup_processing_error",
            )

        return Command(
            update={"patient_lookup_retry_count": retry_count},
            goto="patient_lookup_agent",
        )

    def capture_lookup_response(state: AssistantState) -> dict[str, object]:
        """Capture the final lookup answer produced by the agent.

        Args:
            state: Current graph state.

        Returns:
            State update with the lookup response body.
        """

        try:
            response_text = get_last_ai_message_text(state.get("messages", []))
        except ValueError:
            response_text = _default_lookup_response(state)
        if not response_text.strip():
            response_text = _default_lookup_response(state)
        event = create_audit_event(
            event_type="patient_lookup_response",
            status="success",
            node_name="capture_lookup_response",
            detail="Captured the lookup response from the patient lookup subgraph.",
        )
        emit_console_audit(event)
        return {
            "last_response": response_text,
            "specialist_output_json": None,
            "patient_lookup_retry_count": 0,
            "processing_error_detail": None,
            "audit_events": [event],
        }

    def lookup_processing_error(state: AssistantState) -> dict[str, object]:
        """Exit the lookup flow with a deterministic fail-closed response.

        Args:
            state: Current graph state.

        Returns:
            State update with the fail-closed patient-lookup message.
        """

        event = create_audit_event(
            event_type="patient_lookup_processing_error",
            status="error",
            node_name="lookup_processing_error",
            detail=state.get("processing_error_detail") or LOOKUP_PROCESSING_ERROR_RESPONSE,
        )
        emit_console_audit(event)
        return {
            "last_response": LOOKUP_PROCESSING_ERROR_RESPONSE,
            "specialist_output_json": None,
            "messages": [AIMessage(content=LOOKUP_PROCESSING_ERROR_RESPONSE)],
            "patient_lookup_retry_count": 0,
            "audit_events": [event],
        }

    builder = StateGraph(AssistantState)
    builder.add_node("patient_lookup_agent", patient_lookup_agent)
    builder.add_node(
        "patient_lookup_tools",
        ToolNode(tools, handle_tool_errors=TOOL_EXECUTION_REPAIR_MESSAGE),
    )
    builder.add_node("evaluate_tool_results", evaluate_tool_results)
    builder.add_node("capture_lookup_response", capture_lookup_response)
    builder.add_node("lookup_processing_error", lookup_processing_error)
    builder.add_edge(START, "patient_lookup_agent")
    builder.add_conditional_edges(
        "patient_lookup_agent",
        tools_condition,
        {
            "tools": "patient_lookup_tools",
            "__end__": "capture_lookup_response",
        },
    )
    builder.add_edge("patient_lookup_tools", "evaluate_tool_results")
    builder.add_edge("capture_lookup_response", END)
    builder.add_edge("lookup_processing_error", END)
    return builder.compile()



def _build_lookup_prompt(state: AssistantState) -> str:
    """Augment the lookup prompt with live session context.

    Args:
        state: Current graph state.

    Returns:
        Full lookup system prompt.
    """

    active_patient = state.get("active_patient")
    candidates = state.get("patient_lookup_candidates", [])
    prompt_lines = [
        PATIENT_LOOKUP_SYSTEM_PROMPT,
        "",
        "Session context:",
        f"- Active patient loaded: {'yes' if active_patient else 'no'}",
        f"- Pending candidates: {len(candidates)}",
        f"- Last lookup status: {state.get('patient_lookup_status')}",
    ]
    if active_patient:
        prompt_lines.append(f"- Active patient name: {active_patient['full_name']}")
    if candidates:
        prompt_lines.append("- A numbered candidate list is already available in state.")
    return "\n".join(prompt_lines)



def _default_lookup_response(state: AssistantState) -> str:
    """Generate a deterministic fallback for lookup flows.

    Args:
        state: Current graph state.

    Returns:
        Fallback response text.
    """

    status = state.get("patient_lookup_status")
    if status == "loaded" and state.get("active_patient"):
        return "Patient context loaded successfully."
    if status == "selection_required":
        return "Multiple patients were found. Please choose one of the numbered options."
    if status == "not_found":
        return "No patient was found for the provided identifier."
    return "Patient lookup completed."



def _last_tool_message_is_execution_error(state: AssistantState) -> bool:
    """Return whether the latest tool message represents an execution error.

    Args:
        state: Current graph state.

    Returns:
        `True` when the latest message is a tool error emitted by `ToolNode`.
    """

    messages = state.get("messages", [])
    if not messages:
        return False
    last_message = messages[-1]
    if not isinstance(last_message, ToolMessage):
        return False
    return get_message_text(last_message).strip().lower().startswith("error:")
