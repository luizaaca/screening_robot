"""Router node for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Literal, cast

from langchain.messages import SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from screening_agent.audit import emit_console_audit
from screening_agent.graph.state import AssistantState, RouteIntent, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.model.structured_output import ResilientStructuredOutputInvoker
from screening_agent.prompts import ROUTER_SYSTEM_PROMPT


class RouteDecision(BaseModel):
    """Structured router output used by the root graph."""

    intent: RouteIntent = Field(description="Chosen workflow intent.")
    rationale: str = Field(description="Short explanation for the routing choice.")


_VALID_ROUTE_INTENTS: tuple[RouteIntent, ...] = (
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "patient_lookup_then_analysis",
    "clear_active_patient",
    "invalid_request",
)

_ROUTE_SYMPTOM_TERMS: tuple[str, ...] = (
    "pain",
    "dor",
    "fever",
    "febre",
    "cough",
    "tosse",
    "fatigue",
    "fadiga",
    "shortness of breath",
    "falta de ar",
    "palpitations",
    "palpitações",
    "palpitacoes",
    "headache",
    "cefaleia",
    "nausea",
    "náusea",
    "urination",
    "urinar",
    "thirst",
    "sede",
    "chest tightness",
    "chest pain",
    "aperto no peito",
)



def build_router_node(
    control_model: ControlModel,
) -> Callable[[AssistantState], Command[Literal[
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "clear_active_patient",
    "invalid_request",
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
        text_parser=_parse_route_decision_text,
    )

    def route_request(
        state: AssistantState,
    ) -> Command[Literal[
        "usage_instructions",
        "patient_lookup",
        "symptom_analysis",
        "clear_active_patient",
        "invalid_request",
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
        try:
            decision = structured_router.invoke(router_messages)
            event_status = "success"
            event_detail = None
        except Exception as error:
            decision = _heuristic_route_decision(state)
            event_status = "warning"
            event_detail = str(error)
        goto = _map_intent_to_node(decision.intent)
        event = create_audit_event(
            event_type="routing",
            status=event_status,
            node_name="router",
            detail=(
                f"Routed request to {goto}: {decision.rationale}"
                if event_detail is None
                else (
                    f"Routed request to {goto} with heuristic fallback after structured parsing failed: "
                    f"{decision.rationale}. Error: {event_detail}"
                )
            ),
        )
        emit_console_audit(event)
        return Command(
            update={
                "router_intent": decision.intent,
                "router_rationale": decision.rationale,
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


def _parse_route_decision_text(raw_text: str) -> RouteDecision | None:
    """Parse a plain-text route response into a `RouteDecision`.

    Args:
        raw_text: Plain-text model output.

    Returns:
        Parsed route decision when the text contains a recognizable intent,
        otherwise `None`.
    """

    normalized_text = raw_text.strip()
    if not normalized_text:
        return None

    intent_pattern = "|".join(re.escape(intent) for intent in _VALID_ROUTE_INTENTS)
    direct_match = re.search(
        rf"^(?P<intent>{intent_pattern})\s*[:\-–]?\s*(?P<rationale>.*)$",
        normalized_text,
        re.IGNORECASE | re.DOTALL,
    )
    if direct_match is not None:
        return RouteDecision(
            intent=cast(RouteIntent, direct_match.group("intent").lower()),
            rationale=_normalize_rationale(
                direct_match.group("rationale"),
                cast(RouteIntent, direct_match.group("intent").lower()),
            ),
        )

    labeled_match = re.search(
        rf"intent\s*[:=]\s*(?P<intent>{intent_pattern}).*?rationale\s*[:=]\s*(?P<rationale>.+)$",
        normalized_text,
        re.IGNORECASE | re.DOTALL,
    )
    if labeled_match is not None:
        return RouteDecision(
            intent=cast(RouteIntent, labeled_match.group("intent").lower()),
            rationale=_normalize_rationale(
                labeled_match.group("rationale"),
                cast(RouteIntent, labeled_match.group("intent").lower()),
            ),
        )

    for intent in _VALID_ROUTE_INTENTS:
        if re.search(rf"\b{re.escape(intent)}\b", normalized_text, re.IGNORECASE):
            return RouteDecision(intent=intent, rationale=_normalize_rationale(normalized_text, intent))
    return None


def _normalize_rationale(raw_rationale: str, intent: RouteIntent) -> str:
    """Normalize or synthesize a rationale string.

    Args:
        raw_rationale: Raw rationale text returned by the model.
        intent: Parsed routing intent.

    Returns:
        Cleaned rationale text.
    """

    normalized_rationale = raw_rationale.strip().strip("-: ")
    return normalized_rationale or _default_rationale(intent)


def _default_rationale(intent: RouteIntent) -> str:
    """Return the default rationale for a route intent.

    Args:
        intent: Routed intent.

    Returns:
        Default rationale text.
    """

    rationale_map: dict[RouteIntent, str] = {
        "usage_instructions": "The message asks for product guidance or capabilities.",
        "patient_lookup": "The message focuses on identifying a patient or choosing from candidates.",
        "symptom_analysis": "The message describes symptoms or requests a screening interpretation.",
        "patient_lookup_then_analysis": "The message combines patient identification with a clinical complaint.",
        "clear_active_patient": "The message asks to reset the active patient context.",
        "invalid_request": "The message is outside the assistant scope.",
    }
    return rationale_map[intent]


def _heuristic_route_decision(state: AssistantState) -> RouteDecision:
    """Fallback classifier used when model-based routing cannot be parsed.

    Args:
        state: Current graph state.

    Returns:
        Deterministic routing decision inferred from the latest user message and
        session context.
    """

    latest_user_text = _get_latest_user_text(state)
    pending_candidates = len(state.get("patient_lookup_candidates", []))
    lowered_text = latest_user_text.lower()

    if pending_candidates > 0 and re.fullmatch(r"\s*\d+\s*", latest_user_text):
        return RouteDecision(intent="patient_lookup", rationale=_default_rationale("patient_lookup"))
    if _contains_any(
        lowered_text,
        (
            "how to use",
            "how do i use",
            "use this assistant",
            "what can you do",
            "help",
            "como usar",
            "como eu uso",
            "ajuda",
            "o que você pode fazer",
            "o que voce pode fazer",
        ),
    ):
        return RouteDecision(intent="usage_instructions", rationale=_default_rationale("usage_instructions"))
    if _contains_any(
        lowered_text,
        (
            "clear active patient",
            "clear patient",
            "forget patient",
            "reset patient",
            "limpar paciente",
            "esquecer paciente",
            "remover paciente ativo",
        ),
    ):
        return RouteDecision(intent="clear_active_patient", rationale=_default_rationale("clear_active_patient"))
    if _contains_any(
        lowered_text,
        (
            "weather",
            "capital of",
            "tell me a joke",
            "write a poem",
            "piada",
            "previsão do tempo",
        ),
    ):
        return RouteDecision(intent="invalid_request", rationale=_default_rationale("invalid_request"))

    has_identifier = _has_patient_identifier(latest_user_text)
    has_symptom_request = _contains_any(lowered_text, _ROUTE_SYMPTOM_TERMS)
    if has_identifier and has_symptom_request:
        return RouteDecision(
            intent="patient_lookup_then_analysis",
            rationale=_default_rationale("patient_lookup_then_analysis"),
        )
    if has_identifier:
        return RouteDecision(intent="patient_lookup", rationale=_default_rationale("patient_lookup"))
    if has_symptom_request:
        return RouteDecision(intent="symptom_analysis", rationale=_default_rationale("symptom_analysis"))
    return RouteDecision(intent="invalid_request", rationale=_default_rationale("invalid_request"))


def _get_latest_user_text(state: AssistantState) -> str:
    """Return the text of the latest user-authored message.

    Args:
        state: Current graph state.

    Returns:
        Latest user message text, or an empty string when unavailable.
    """

    for message in reversed(state.get("messages", [])):
        message_type = getattr(message, "type", "")
        if message_type in {"human", "user"}:
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _contains_any(text: str, options: tuple[str, ...]) -> bool:
    """Return whether any candidate substring appears in the text.

    Args:
        text: Text to scan.
        options: Candidate substrings.

    Returns:
        `True` when at least one candidate appears.
    """

    return any(option in text for option in options)


def _has_patient_identifier(text: str) -> bool:
    """Return whether the text appears to include a patient identifier.

    Args:
        text: Latest user text.

    Returns:
        `True` when an 8-digit identifier or likely full name is present.
    """

    if re.search(r"\b\d{8}\b", text):
        return True
    return re.search(r"[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+", text) is not None
