"""Root graph builder for the screening assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from screening_agent.audit import emit_console_audit
from screening_agent.config import AppSettings, VideoPipelineSettings
from screening_agent.data import PatientRepository
from screening_agent.graph.nodes import (
    VideoProcessor,
    build_clear_active_patient_node,
    build_finalize_response_node,
    build_invalid_request_node,
    build_processing_error_node,
    build_router_node,
    build_symptom_analysis_node,
    build_usage_instructions_node,
    build_video_analysis_node,
    build_video_clinical_extraction_node,
    build_video_interpretation_node,
)
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.graph.subgraphs import build_patient_lookup_subgraph
from screening_agent.model import create_control_model, create_specialist_invoker, create_video_analyst_model
from screening_agent.model.control_models import ControlModel
from screening_agent.tools.specialist_tool import SpecialistInvoker



def build_screening_graph(
    *,
    control_model: ControlModel,
    specialist_invoker: SpecialistInvoker,
    repository: PatientRepository,
    video_analyst_model: ControlModel | None = None,
    video_pipeline_settings: VideoPipelineSettings | None = None,
    video_processor: VideoProcessor | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the root LangGraph workflow.

    Args:
        control_model: Tool-capable chat model for router and control nodes.
        specialist_invoker: Backend-specific structured specialist invoker.
        repository: Patient repository used by lookup tools.
        video_analyst_model: Optional chat model for video QA.
        video_pipeline_settings: Optional settings for video processing.
        video_processor: Optional video processor injected for tests.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph workflow.
    """

    builder: Any = StateGraph(AssistantState)
    resolved_video_pipeline_settings = (
        video_pipeline_settings
        if video_pipeline_settings is not None
        else VideoPipelineSettings(output_dir=Path("outputs/videos"))
    )
    resolved_video_analyst_model = video_analyst_model or control_model
    builder.add_node("router", build_router_node(control_model))
    builder.add_node("usage_instructions", build_usage_instructions_node(control_model))
    builder.add_node("patient_lookup", build_patient_lookup_subgraph(control_model, repository))
    builder.add_node("route_after_lookup", _route_after_lookup)
    builder.add_node("symptom_analysis", build_symptom_analysis_node(control_model, specialist_invoker))
    builder.add_node(
        "video_analysis",
        build_video_analysis_node(
            resolved_video_pipeline_settings,
            video_processor=video_processor,
        ),
    )
    builder.add_node(
        "video_interpretation",
        build_video_interpretation_node(
            resolved_video_analyst_model,
            resolved_video_pipeline_settings,
            video_processor=video_processor,
        ),
    )
    builder.add_node(
        "video_clinical_extraction",
        build_video_clinical_extraction_node(
            resolved_video_analyst_model,
            resolved_video_pipeline_settings,
            video_processor=video_processor,
        ),
    )
    builder.add_node("route_after_video_analysis", _route_after_video_analysis)
    builder.add_node("route_after_video_clinical_extraction", _route_after_video_clinical_extraction)
    builder.add_node("clear_active_patient", build_clear_active_patient_node(control_model))
    builder.add_node("invalid_request", build_invalid_request_node(control_model))
    builder.add_node("processing_error", build_processing_error_node())
    builder.add_node("final_answer", build_finalize_response_node(control_model))

    builder.add_edge(START, "router")
    builder.add_edge("usage_instructions", "final_answer")
    builder.add_edge("patient_lookup", "route_after_lookup")
    builder.add_edge("symptom_analysis", "final_answer")
    builder.add_edge("video_analysis", "route_after_video_analysis")
    builder.add_edge("video_interpretation", "final_answer")
    builder.add_edge("video_clinical_extraction", "route_after_video_clinical_extraction")
    builder.add_edge("clear_active_patient", "final_answer")
    builder.add_edge("invalid_request", "final_answer")
    builder.add_edge("processing_error", "final_answer")
    builder.add_edge("final_answer", END)

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
        specialist_invoker=create_specialist_invoker(resolved_settings),
        repository=repository,
        video_analyst_model=create_video_analyst_model(resolved_settings),
        video_pipeline_settings=resolved_settings.video_pipeline,
        checkpointer=InMemorySaver() if resolved_settings.use_in_memory_checkpointer else None,
    )



def _route_after_lookup(
    state: AssistantState,
) -> Command[Literal["symptom_analysis", "final_answer"]]:
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
    goto: Literal["symptom_analysis", "final_answer"]
    if should_continue_to_analysis:
        goto = "symptom_analysis"
        detail = "Lookup succeeded in a combined request; continuing to symptom analysis."
    else:
        goto = "final_answer"
        detail = "Lookup flow will finalize with the final-answer node."
    event = create_audit_event(
        event_type="route_after_lookup",
        status="success",
        node_name="route_after_lookup",
        detail=detail,
    )
    emit_console_audit(event)
    return Command(update={"audit_events": [event]}, goto=goto)


def _route_after_video_analysis(
    state: AssistantState,
) -> Command[Literal["video_interpretation", "video_clinical_extraction", "final_answer"]]:
    """Route after the deterministic video pipeline finishes."""

    if state.get("video_analysis_status") != "completed":
        goto: Literal["video_interpretation", "video_clinical_extraction", "final_answer"] = (
            "final_answer"
        )
        detail = "Video pipeline did not complete; final-answer node will explain the failure."
    elif _should_extract_video_clinical_context(state):
        goto = "video_clinical_extraction"
        detail = "Video will be converted into structured clinical context before symptom analysis."
    else:
        goto = "video_interpretation"
        detail = "Video will receive narrative interpretation without structured clinical extraction."

    event = create_audit_event(
        event_type="route_after_video_analysis",
        status="success",
        node_name="route_after_video_analysis",
        detail=detail,
    )
    emit_console_audit(event)
    return Command(update={"audit_events": [event]}, goto=goto)


def _route_after_video_clinical_extraction(
    state: AssistantState,
) -> Command[Literal["symptom_analysis", "final_answer"]]:
    """Route after structured clinical video extraction."""

    has_clinical_context = bool(str(state.get("video_clinical_context_json") or "").strip())
    if has_clinical_context:
        goto: Literal["symptom_analysis", "final_answer"] = "symptom_analysis"
        detail = "Structured video clinical context is available; continuing to symptom analysis."
    else:
        goto = "final_answer"
        detail = "Structured video clinical context is unavailable; final-answer node will explain the failure."

    event = create_audit_event(
        event_type="route_after_video_clinical_extraction",
        status="success",
        node_name="route_after_video_clinical_extraction",
        detail=detail,
    )
    emit_console_audit(event)
    return Command(update={"audit_events": [event]}, goto=goto)


def _should_extract_video_clinical_context(state: AssistantState) -> bool:
    if state.get("router_intent") == "video_symptom_analysis":
        return True
    pending_request = state.get("pending_video_request")
    if not isinstance(pending_request, dict):
        return False
    return pending_request.get("intent") == "video_symptom_analysis"
