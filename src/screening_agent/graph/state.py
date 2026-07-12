"""Shared state and helper contracts for the screening graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from langgraph.graph import MessagesState

RouteIntent = Literal[
    "usage_instructions",
    "patient_lookup",
    "symptom_analysis",
    "patient_lookup_then_analysis",
    "video_analysis",
    "video_qa",
    "clear_active_patient",
    "invalid_request",
]
PatientLookupStatus = Literal["loaded", "selection_required", "not_found"]
VideoAnalysisStatus = Literal["completed", "missing_video", "failed"]
AuditStatus = Literal["success", "warning", "error", "info"]


class PatientCandidate(TypedDict):
    """Short patient representation used during name disambiguation."""

    security_number: str
    full_name: str


class PatientRecord(TypedDict):
    """Simplified patient record stored as active session context."""

    security_number: str
    full_name: str
    clinical_context: str


class AuditEvent(TypedDict):
    """Console-audit event emitted by nodes and tools."""

    timestamp_utc: str
    event_type: str
    status: AuditStatus
    node_name: str
    detail: str


class AssistantState(MessagesState):
    """LangGraph session state for the screening assistant."""

    active_patient: PatientRecord | None
    patient_lookup_status: PatientLookupStatus | None
    router_intent: RouteIntent | None
    router_rationale: str | None
    last_response: str | None
    patient_lookup_candidates: list[PatientCandidate]
    specialist_output_json: str | None
    video_path: str | None
    video_artifact_dir: str | None
    video_analysis_summary: str | None
    video_analysis_json: str | None
    video_analysis_status: VideoAnalysisStatus | None
    video_analysis_error: str | None



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


def build_active_patient_header(patient: PatientRecord) -> str:
    """Build the compact header shown when a patient is active.

    Args:
        patient: Active patient record.

    Returns:
        User-facing active-patient header.
    """

    return (
        f"Active patient: {patient['full_name']} "
        f"({mask_security_number(patient['security_number'])})"
    )


def create_audit_event(
    *,
    event_type: str,
    status: AuditStatus,
    node_name: str,
    detail: str,
) -> AuditEvent:
    """Create a normalized audit event payload.

    Args:
        event_type: Stable event category.
        status: Outcome severity for the event.
        node_name: Node or tool name that emitted the event.
        detail: Human-readable event detail.

    Returns:
        A JSON-compatible audit event dictionary.
    """

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "status": status,
        "node_name": node_name,
        "detail": detail,
    }
