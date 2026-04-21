"""Clinical backend contracts shared by multiple runtime implementations."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from screening_agent.graph.state import ClinicalStatus, PatientRecord


class ClinicalAnalysisPayload(BaseModel):
    """Structured clinical analysis returned by backend implementations."""

    status: ClinicalStatus = Field(description="Clinical status for audit and routing.")
    primary_hypothesis: str | None = Field(
        default=None,
        description="Primary screening hypothesis, if available.",
    )
    differential_hypotheses: list[str] = Field(
        default_factory=list,
        description="Alternative hypotheses relevant to the screening context.",
    )
    recommended_exams: list[str] = Field(
        default_factory=list,
        description="Confirmatory exams or next assessment steps.",
    )
    reasoning_summary: str = Field(
        description="Short clinically coherent reasoning summary.",
    )
    safety_notes: list[str] = Field(
        default_factory=list,
        description="Urgent warning signs or safety-oriented caveats.",
    )
    user_response: str = Field(
        description="User-facing response in the same language as the request.",
    )


class ClinicalBackend(Protocol):
    """Protocol implemented by clinical analysis runtimes."""

    backend_name: str

    def analyze(
        self,
        *,
        user_message: str,
        active_patient: PatientRecord | None,
    ) -> ClinicalAnalysisPayload:
        """Analyze a clinical complaint and return a structured result.

        Args:
            user_message: Latest user request to analyze.
            active_patient: Optional active patient context.

        Returns:
            Structured analysis payload.
        """



def build_clinical_input(user_message: str, active_patient: PatientRecord | None) -> str:
    """Compose the textual context sent to a clinical backend.

    Args:
        user_message: Latest user complaint or question.
        active_patient: Optional active patient record.

    Returns:
        A prompt-ready text block with complaint and patient context.
    """

    patient_context = _format_patient_context(active_patient)
    return (
        "Clinical request:\n"
        f"{user_message.strip()}\n\n"
        "Active patient context:\n"
        f"{patient_context}"
    )



def _format_patient_context(active_patient: PatientRecord | None) -> str:
    """Render patient context for model consumption.

    Args:
        active_patient: Optional active patient record.

    Returns:
        Plain-text patient context.
    """

    if active_patient is None:
        return "No active patient loaded. Analyze based only on the user's description."

    last_vitals = active_patient.get("last_vitals") or {}
    recent_exams = active_patient.get("recent_exams") or []
    exam_lines = [
        f"- {exam['exam_name']} ({exam['exam_date'] or 'date unavailable'}): {exam['result_summary'] or 'no summary'}"
        for exam in recent_exams
    ]
    return "\n".join(
        [
            f"Name: {active_patient['full_name']}",
            f"Security number: {active_patient['security_number']}",
            f"Age: {active_patient.get('age_years')}",
            f"Sex: {active_patient.get('sex')}",
            f"Allergies: {', '.join(active_patient.get('allergies', [])) or 'None reported'}",
            f"Conditions: {', '.join(active_patient.get('conditions', [])) or 'None reported'}",
            f"Medications: {', '.join(active_patient.get('medications', [])) or 'None reported'}",
            f"Latest vitals: {last_vitals or 'Unavailable'}",
            "Recent exams:",
            *(exam_lines or ["- None available"]),
        ]
    )
