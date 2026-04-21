"""Shared state and helper contracts for the screening graph."""

from __future__ import annotations

from datetime import datetime, timezone
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

RouteIntent = Literal[
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "patient_lookup_then_analysis",
    "clear_active_patient",
    "invalid_request",
]
ResponseKind = Literal["usage", "lookup", "analysis", "invalid", "system"]
AuditStatus = Literal["info", "success", "warning", "error"]
ClinicalStatus = Literal[
    "analysis_ready",
    "insufficient_information",
    "urgent_attention",
]
PatientLookupStatus = Literal["loaded", "selection_required", "not_found"]


class VitalSignsSnapshot(TypedDict, total=False):
    """Latest vitals associated with a patient record."""

    recorded_at: str
    blood_pressure: str
    heart_rate_bpm: int
    respiratory_rate_bpm: int
    temperature_c: float
    oxygen_saturation_pct: int


class ExamSummary(TypedDict):
    """Condensed exam summary kept in state for contextual analysis."""

    exam_name: str
    exam_date: str | None
    result_summary: str | None


class PatientCandidate(TypedDict):
    """Short patient representation used during name disambiguation."""

    security_number: str
    full_name: str
    age_years: int | None
    sex: str | None


class PatientRecord(TypedDict):
    """Full patient record stored as active session context."""

    security_number: str
    full_name: str
    birth_date: str | None
    age_years: int | None
    sex: str | None
    allergies: list[str]
    conditions: list[str]
    medications: list[str]
    last_vitals: VitalSignsSnapshot | None
    recent_exams: list[ExamSummary]


class ClinicalAnalysisResult(TypedDict):
    """Structured clinical output persisted for audit and testing."""

    status: ClinicalStatus
    primary_hypothesis: str | None
    differential_hypotheses: list[str]
    recommended_exams: list[str]
    reasoning_summary: str
    safety_notes: list[str]
    user_response: str
    backend_name: str


class AuditEvent(TypedDict):
    """Structured audit event recorded during graph execution."""

    timestamp_utc: str
    event_type: str
    status: AuditStatus
    node_name: str
    detail: str


class AssistantState(MessagesState):
    """LangGraph session state for the screening assistant."""

    active_patient: PatientRecord | None
    active_patient_header: str | None
    patient_lookup_status: PatientLookupStatus | None
    router_intent: RouteIntent | None
    router_rationale: str | None
    response_kind: ResponseKind | None
    response_body: str | None
    last_response: str | None
    response_requires_disclaimer: bool
    patient_lookup_candidates: list[PatientCandidate]
    analysis_result: ClinicalAnalysisResult | None
    audit_events: Annotated[list[AuditEvent], operator.add]



def build_active_patient_header(patient: PatientRecord) -> str:
    """Create the header displayed above final responses with active patient context.

    Args:
        patient: Active patient record from session state.

    Returns:
        A concise header with masked identifier and demographics.
    """

    demographics: list[str] = []
    if patient.get("age_years") is not None:
        demographics.append(f"{patient['age_years']} years")
    if patient.get("sex"):
        demographics.append(str(patient["sex"]))
    demographics_text = " • ".join(demographics) if demographics else "Demographics unavailable"
    return (
        f"Active patient: {patient['full_name']} • "
        f"ID {mask_security_number(patient['security_number'])} • "
        f"{demographics_text}"
    )



def create_audit_event(
    *,
    event_type: str,
    status: AuditStatus,
    node_name: str,
    detail: str,
) -> AuditEvent:
    """Create a timestamped audit event.

    Args:
        event_type: Category of the event.
        status: Severity or result classification.
        node_name: Graph node or tool responsible for the event.
        detail: Human-readable event explanation.

    Returns:
        A serializable audit event dictionary.
    """

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "status": status,
        "node_name": node_name,
        "detail": detail,
    }



def mask_security_number(security_number: str) -> str:
    """Mask a fictional security number for user-facing displays.

    Args:
        security_number: Raw patient security number.

    Returns:
        A masked representation that preserves only the last four digits.
    """

    normalized = security_number.strip()
    if len(normalized) <= 4:
        return "*" * len(normalized)
    return f"{'*' * (len(normalized) - 4)}{normalized[-4:]}"
