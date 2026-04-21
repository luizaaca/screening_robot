"""Patient lookup subgraph for tool-calling retrieval flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from screening_agent.audit import emit_console_audit
from screening_agent.data import PatientRepository
from screening_agent.graph.message_utils import get_last_ai_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import PATIENT_LOOKUP_SYSTEM_PROMPT
from screening_agent.tools import build_patient_lookup_tools, execute_patient_lookup_tool_call



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

        response = lookup_model.invoke(
            [
                SystemMessage(content=_build_lookup_prompt(state)),
                *state.get("messages", []),
            ],
        )
        return {"messages": [response]}

    def patient_lookup_router(
        state: AssistantState,
    ) -> Literal["patient_lookup_tools", "capture_lookup_response"]:
        """Decide whether the lookup agent produced tool calls or a final answer.

        Args:
            state: Current graph state.

        Returns:
            Next subgraph node name.
        """

        last_message = state.get("messages", [])[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "patient_lookup_tools"
        return "capture_lookup_response"

    def patient_lookup_tools(state: AssistantState) -> dict[str, object]:
        """Execute emitted patient-lookup tool calls and aggregate their updates.

        Args:
            state: Current graph state.

        Returns:
            Aggregated state updates from the executed tool calls.
        """

        last_message = state.get("messages", [])[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        aggregate_update: dict[str, object] = {}
        current_state = state
        for tool_call in last_message.tool_calls:
            tool_update = execute_patient_lookup_tool_call(repository, current_state, tool_call)
            aggregate_update = _merge_state_updates(aggregate_update, tool_update)
            current_state = {**current_state, **aggregate_update}
        return aggregate_update

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
            "response_kind": "lookup",
            "response_body": response_text,
            "response_requires_disclaimer": False,
            "audit_events": [event],
        }

    builder = StateGraph(AssistantState)
    builder.add_node("patient_lookup_agent", patient_lookup_agent)
    builder.add_node("patient_lookup_tools", patient_lookup_tools)
    builder.add_node("capture_lookup_response", capture_lookup_response)
    builder.add_edge(START, "patient_lookup_agent")
    builder.add_conditional_edges("patient_lookup_agent", patient_lookup_router)
    builder.add_edge("patient_lookup_tools", "patient_lookup_agent")
    builder.add_edge("capture_lookup_response", END)
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


def _merge_state_updates(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    """Merge two tool-update dictionaries preserving appended list fields.

    Args:
        left: Existing aggregated update.
        right: New tool update.

    Returns:
        Merged update dictionary.
    """

    merged = dict(left)
    for key, value in right.items():
        if key in {"messages", "audit_events"}:
            merged[key] = [*list(merged.get(key, [])), *list(value)]
        else:
            merged[key] = value
    return merged
