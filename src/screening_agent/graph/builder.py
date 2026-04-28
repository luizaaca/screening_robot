"""Root graph builder for the screening assistant."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from screening_agent.audit import emit_console_audit
from screening_agent.config import AppSettings
from screening_agent.data import PatientRepository
from screening_agent.graph.nodes import (
    build_clear_active_patient_node,
    build_finalize_response_node,
    build_invalid_request_node,
    build_processing_error_node,
    build_router_node,
    build_symptom_analysis_node,
    build_usage_instructions_node,
)
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.graph.subgraphs import build_patient_lookup_subgraph
from screening_agent.model import ClinicalBackend, create_clinical_backend, create_control_model
from screening_agent.model.control_models import ControlModel



def build_screening_graph(
    *,
    control_model: ControlModel,
    clinical_backend: ClinicalBackend,
    repository: PatientRepository,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the root LangGraph workflow.

    Args:
        control_model: Tool-capable chat model for router and control nodes.
        clinical_backend: Backend used for symptom analysis.
        repository: Patient repository used by lookup tools.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph workflow.
    """

    builder = StateGraph(AssistantState)
    builder.add_node("router", build_router_node(control_model))
    builder.add_node("usage_instructions", build_usage_instructions_node(control_model))
    builder.add_node("patient_lookup", build_patient_lookup_subgraph(control_model, repository))
    builder.add_node("route_after_lookup", _route_after_lookup)
    builder.add_node("symptom_analysis", build_symptom_analysis_node(clinical_backend))
    builder.add_node("clear_active_patient", build_clear_active_patient_node(control_model))
    builder.add_node("invalid_request", build_invalid_request_node(control_model))
    builder.add_node("processing_error", build_processing_error_node())
    builder.add_node("finalize_response", build_finalize_response_node())

    builder.add_edge(START, "router")
    builder.add_edge("usage_instructions", "finalize_response")
    builder.add_edge("patient_lookup", "route_after_lookup")
    builder.add_edge("symptom_analysis", "finalize_response")
    builder.add_edge("clear_active_patient", "finalize_response")
    builder.add_edge("invalid_request", "finalize_response")
    builder.add_edge("processing_error", "finalize_response")
    builder.add_edge("finalize_response", END)

    compiled_graph = builder.compile(checkpointer=checkpointer)
    return compiled_graph



def build_default_graph(settings: AppSettings | None = None) -> Any:
    """Build the graph using environment-driven configuration.

    Args:
        settings: Optional pre-built application settings.

    Returns:
        A compiled LangGraph workflow ready to invoke.
    """

    resolved_settings = settings or AppSettings.from_env()
    repository = PatientRepository(resolved_settings.patient_database_path)
    repository.initialize_database()
    return build_screening_graph(
        control_model=create_control_model(resolved_settings),
        clinical_backend=create_clinical_backend(resolved_settings),
        repository=repository,
        checkpointer=InMemorySaver() if resolved_settings.use_in_memory_checkpointer else None,
    )



def _route_after_lookup(
    state: AssistantState,
) -> Command[Literal["symptom_analysis", "finalize_response"]]:
    """Route after patient lookup depending on the original user intent.

    Args:
        state: Current graph state.

    Returns:
        A command targeting either clinical analysis or final response composition.
    """

    should_continue_to_analysis = (
        state.get("router_intent") == "patient_lookup_then_analysis"
        and state.get("patient_lookup_status") == "loaded"
    )
    goto: Literal["symptom_analysis", "finalize_response"]
    if should_continue_to_analysis:
        goto = "symptom_analysis"
        detail = "Lookup succeeded in a combined request; continuing to symptom analysis."
    else:
        goto = "finalize_response"
        detail = "Lookup flow will finalize without symptom analysis."
    event = create_audit_event(
        event_type="route_after_lookup",
        status="success",
        node_name="route_after_lookup",
        detail=detail,
    )
    emit_console_audit(event)
    return Command(update={"audit_events": [event]}, goto=goto)
