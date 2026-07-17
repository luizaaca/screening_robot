"""Router node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from langchain.messages import SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.graph.message_utils import get_last_human_message_text
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
    "video_analysis",
    "video_interpretation",
    "video_clinical_extraction",
    "clear_active_patient",
    "invalid_request",
    "final_answer",
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
        "video_analysis",
        "video_interpretation",
        "video_clinical_extraction",
        "clear_active_patient",
        "invalid_request",
        "final_answer",
        "processing_error",
    ]]:
        """Choose the next workflow node from the latest user turn.

        Args:
            state: Current graph state.

        Returns:
            A `Command` with updated router metadata and the target node.
        """

        pending_video_command = _route_pending_video_request(state)
        if pending_video_command is not None:
            return pending_video_command

        contextual_followup_command = _route_contextual_followup(state)
        if contextual_followup_command is not None:
            return contextual_followup_command

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
                status="warning",
                node_name="router",
                detail=(
                    "Routing classification failed after repeated structured-output attempts; "
                    "defaulting to final_answer with the available conversation context. "
                    f"Error: {error}"
                ),
            )
            emit_console_audit(event)
            return Command(
                update={
                    "router_intent": "final_answer",
                    "router_rationale": (
                        "The router could not classify the turn reliably, so the graph will "
                        "answer from the existing conversation and state context."
                    ),
                    "processing_error_detail": None,
                    "turn_outcome": None,
                    "audit_events": [event],
                },
                goto="final_answer",
            )
        goto = _map_intent_to_node(decision.intent)
        turn_outcome: dict[str, object] | None = None
        pending_video_request: dict[str, object] | None = None
        video_input_status = state.get("video_input_status") or "none"
        if _should_request_video_confirmation(decision.intent, state):
            pending_video_request = _build_pending_video_request(decision.intent, state)
            video_input_status = "awaiting_confirmation"
            turn_outcome = {
                "type": "video_upload_confirmation_requested",
                "requested_intent": pending_video_request["intent"],
            }
            goto = "final_answer"

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
                "pending_video_request": pending_video_request
                if pending_video_request is not None
                else state.get("pending_video_request"),
                "video_input_status": video_input_status,
                "video_input_event": None,
                "turn_outcome": turn_outcome,
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
        f"- Pending video request: {'yes' if state.get('pending_video_request') else 'no'}",
        f"- Video input status: {state.get('video_input_status') or 'none'}",
        f"- New video path provided this turn: {'yes' if state.get('incoming_video_path') else 'no'}",
        f"- Current video path available: {'yes' if state.get('video_path') else 'no'}",
        f"- Video analysis status: {state.get('video_analysis_status')}",
        f"- Video analysis result available: {'yes' if state.get('video_analysis_json') else 'no'}",
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
    "video_analysis",
    "video_interpretation",
    "video_clinical_extraction",
    "clear_active_patient",
    "invalid_request",
    "final_answer",
]:
    """Map an intent to the actual node name used in the root graph.

    Args:
        intent: Structured router intent.

    Returns:
        Concrete node name.
    """

    if intent in {"patient_lookup", "patient_lookup_then_analysis"}:
        return "patient_lookup"
    if intent == "video_qa":
        return "video_interpretation"
    if intent == "video_symptom_analysis":
        return "video_clinical_extraction"
    if intent == "video_upload_confirmation":
        return "invalid_request"
    if intent == "final_answer":
        return "final_answer"
    return cast(
        Literal[
            "usage_instructions",
            "patient_lookup",
            "symptom_analysis",
            "video_analysis",
            "video_interpretation",
            "video_clinical_extraction",
            "clear_active_patient",
            "invalid_request",
            "final_answer",
        ],
        intent,
    )


def _route_contextual_followup(
    state: AssistantState,
) -> Command[Literal["final_answer"]] | None:
    """Route short acknowledgements to the final answer when prior context exists."""

    latest_user_message = _latest_user_text(state)
    if not _is_contextual_followup(latest_user_message):
        return None
    if not _has_prior_response_context(state):
        return None

    rationale = (
        "The latest turn is a short contextual follow-up to the previous assistant response."
    )
    event = create_audit_event(
        event_type="routing",
        status="success",
        node_name="router",
        detail=f"Routed request to final_answer: {rationale}",
    )
    emit_console_audit(event)
    return Command(
        update={
            "router_intent": "final_answer",
            "router_rationale": rationale,
            "processing_error_detail": None,
            "turn_outcome": None,
            "audit_events": [event],
        },
        goto="final_answer",
    )


def _is_contextual_followup(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    if normalized in {
        "sim",
        "s",
        "sim faça isso",
        "sim faca isso",
        "faça isso",
        "faca isso",
        "pode",
        "pode fazer",
        "pode organizar",
        "claro",
        "ok",
        "okay",
        "yes",
        "y",
        "yes do that",
        "do that",
        "sure",
        "go ahead",
        "continue",
    }:
        return True
    if len(normalized.split()) > 6:
        return False
    starts_like_confirmation = normalized.startswith(
        ("sim ", "yes ", "ok ", "okay ", "sure ", "pode ", "claro "),
    )
    asks_to_continue = any(
        marker in normalized
        for marker in (
            "faça",
            "faca",
            "isso",
            "organize",
            "organizar",
            "continue",
            "that",
        )
    )
    return starts_like_confirmation and asks_to_continue


def _has_prior_response_context(state: AssistantState) -> bool:
    if _coerce_non_empty_string(state.get("last_response")) is not None:
        return True

    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return False
    for message in messages[:-1]:
        message_type = str(getattr(message, "type", message.__class__.__name__)).lower()
        if message_type not in {"ai", "assistant"}:
            continue
        if _coerce_non_empty_string(getattr(message, "content", None)) is not None:
            return True
    return False


def _route_pending_video_request(
    state: AssistantState,
) -> Command[Literal["video_analysis", "final_answer"]] | None:
    """Handle the two-turn upload confirmation state before LLM routing."""

    pending_request = state.get("pending_video_request")
    video_input_status = state.get("video_input_status") or "none"
    if not isinstance(pending_request, dict) or video_input_status == "none":
        return None

    latest_user_message = _latest_user_text(state)
    incoming_video_path = _coerce_non_empty_string(state.get("incoming_video_path"))
    if incoming_video_path is not None:
        intent = _coerce_pending_video_intent(pending_request)
        return _pending_video_command(
            goto="video_analysis",
            intent=intent,
            rationale="A pending video request received a video file or local path.",
            pending_video_request=pending_request,
            video_input_status="none",
            turn_outcome={
                "type": "video_upload_received",
                "requested_intent": intent,
            },
        )

    video_input_event = _coerce_non_empty_string(state.get("video_input_event"))
    if video_input_status == "awaiting_upload" and video_input_event in {
        "upload_timeout",
        "upload_cancelled",
    }:
        outcome_type = (
            "video_upload_timeout"
            if video_input_event == "upload_timeout"
            else "video_upload_cancelled"
        )
        return _pending_video_command(
            goto="final_answer",
            intent="video_upload_confirmation",
            rationale="The pending video upload did not provide a file.",
            pending_video_request=None,
            video_input_status="none",
            turn_outcome={"type": outcome_type},
        )

    if _is_negative_confirmation(latest_user_message):
        return _pending_video_command(
            goto="final_answer",
            intent="video_upload_confirmation",
            rationale="The user declined the pending video upload request.",
            pending_video_request=None,
            video_input_status="none",
            turn_outcome={"type": "video_upload_declined"},
        )

    if video_input_status == "awaiting_confirmation":
        if _is_affirmative_confirmation(latest_user_message):
            intent = _coerce_pending_video_intent(pending_request)
            return _pending_video_command(
                goto="final_answer",
                intent="video_upload_confirmation",
                rationale="The user confirmed that they want to provide a video.",
                pending_video_request=pending_request,
                video_input_status="awaiting_upload",
                turn_outcome={
                    "type": "video_upload_confirmed",
                    "requested_intent": intent,
                },
            )
        return _pending_video_command(
            goto="final_answer",
            intent="video_upload_confirmation",
            rationale="The pending video upload confirmation is still ambiguous.",
            pending_video_request=pending_request,
            video_input_status="awaiting_confirmation",
            turn_outcome={"type": "video_upload_confirmation_unclear"},
        )

    if video_input_status == "awaiting_upload":
        return _pending_video_command(
            goto="final_answer",
            intent="video_upload_confirmation",
            rationale="The graph is waiting for a video upload or local path.",
            pending_video_request=pending_request,
            video_input_status="awaiting_upload",
            turn_outcome={"type": "video_upload_still_needed"},
        )

    return None


def _pending_video_command(
    *,
    goto: Literal["video_analysis", "final_answer"],
    intent: RouteIntent,
    rationale: str,
    pending_video_request: dict[str, object] | None,
    video_input_status: Literal["none", "awaiting_confirmation", "awaiting_upload"],
    turn_outcome: dict[str, object],
) -> Command[Literal["video_analysis", "final_answer"]]:
    event = create_audit_event(
        event_type="routing",
        status="success",
        node_name="router",
        detail=f"Handled pending video request: {rationale}",
    )
    emit_console_audit(event)
    return Command(
        update={
            "router_intent": intent,
            "router_rationale": rationale,
            "pending_video_request": pending_video_request,
            "video_input_status": video_input_status,
            "video_input_event": None,
            "turn_outcome": turn_outcome,
            "audit_events": [event],
        },
        goto=goto,
    )


def _should_request_video_confirmation(intent: RouteIntent, state: AssistantState) -> bool:
    """Return whether this turn needs user confirmation before requesting a video."""

    if intent not in {
        "video_analysis",
        "video_interpretation",
        "video_qa",
        "video_symptom_analysis",
    }:
        return False
    if _coerce_non_empty_string(state.get("incoming_video_path")) is not None:
        return False
    return not _has_usable_video(state)


def _has_usable_video(state: AssistantState) -> bool:
    return (
        _coerce_non_empty_string(state.get("video_path")) is not None
        or _coerce_non_empty_string(state.get("video_analysis_json")) is not None
    )


def _build_pending_video_request(
    intent: RouteIntent,
    state: AssistantState,
) -> dict[str, object]:
    pending_intent: RouteIntent
    if intent == "video_qa":
        pending_intent = "video_interpretation"
    else:
        pending_intent = intent
    return {
        "intent": pending_intent,
        "request_text": _latest_user_text(state),
    }


def _coerce_pending_video_intent(pending_request: dict[str, object]) -> RouteIntent:
    raw_intent = pending_request.get("intent")
    if raw_intent == "video_symptom_analysis":
        return "video_symptom_analysis"
    if raw_intent in {"video_analysis", "video_interpretation", "video_qa"}:
        return "video_analysis"
    return "video_analysis"


def _latest_user_text(state: AssistantState) -> str:
    try:
        return get_last_human_message_text(state.get("messages", []))
    except ValueError:
        return ""


def _is_affirmative_confirmation(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "sim",
        "s",
        "pode",
        "quero",
        "enviarei",
        "vou enviar",
    }


def _is_negative_confirmation(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "no",
        "n",
        "nope",
        "cancel",
        "cancelar",
        "nao",
        "não",
        "negativo",
    }


def _coerce_non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
