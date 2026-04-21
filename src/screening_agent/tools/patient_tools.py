"""Patient lookup tools for the LangGraph workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from screening_agent.audit import emit_console_audit
from screening_agent.data import PatientRepository
from screening_agent.graph.state import (
    AssistantState,
    PatientCandidate,
    PatientRecord,
    build_active_patient_header,
    create_audit_event,
    mask_security_number,
)


class LookupBySecurityNumberInput(BaseModel):
    """Arguments for looking up a patient by fictional security number."""

    security_number: str = Field(
        description="Fictional 8-digit security number used to retrieve one patient.",
    )


class LookupByNameInput(BaseModel):
    """Arguments for looking up patient candidates by name."""

    full_name: str = Field(
        description="Full or partial patient name used for disambiguation.",
    )


class ActivatePatientSelectionInput(BaseModel):
    """Arguments for activating a patient candidate from an enumerated list."""

    selection_index: int = Field(
        ge=1,
        description="One-based index chosen by the user from the enumerated candidate list.",
    )


def execute_patient_lookup_tool_call(
    repository: PatientRepository,
    state: AssistantState,
    tool_call: Mapping[str, Any],
) -> dict[str, object]:
    """Execute one patient-lookup tool call against repository and graph state.

    Args:
        repository: Repository used to retrieve synthetic patient data.
        state: Current assistant state.
        tool_call: Tool-call payload emitted by the control model.

    Returns:
        State update produced by the tool execution.

    Raises:
        ValueError: If the tool name is not supported.
    """

    tool_name = str(tool_call.get("name", "")).strip()
    tool_call_id = str(tool_call.get("id", tool_name or "tool_call"))
    raw_args = tool_call.get("args", {})
    args = raw_args if isinstance(raw_args, Mapping) else {}

    if tool_name == "lookup_patient_by_security_number":
        return _lookup_by_security_number_update(
            repository=repository,
            security_number=str(args.get("security_number", "")).strip(),
            tool_call_id=tool_call_id,
        )
    if tool_name == "lookup_patient_by_name":
        return _lookup_by_name_update(
            repository=repository,
            full_name=str(args.get("full_name", "")).strip(),
            tool_call_id=tool_call_id,
        )
    if tool_name == "activate_patient_selection":
        selection_index = int(args.get("selection_index", 0))
        return _activate_patient_selection_update(
            repository=repository,
            state=state,
            selection_index=selection_index,
            tool_call_id=tool_call_id,
        )

    raise ValueError(f"Unsupported patient lookup tool call: {tool_name!r}")


def build_patient_lookup_tools(repository: PatientRepository) -> list[object]:
    """Build tool definitions used by the patient lookup subgraph.

    Args:
        repository: Repository used to retrieve synthetic patient data.

    Returns:
        A list of LangChain tools.
    """

    @tool(args_schema=LookupBySecurityNumberInput)
    def lookup_patient_by_security_number(
        security_number: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Load a patient directly from the SQLite repository by security number."""

        return Command(
            update=_lookup_by_security_number_update(
                repository=repository,
                security_number=security_number,
                tool_call_id=runtime.tool_call_id,
            ),
        )

    @tool(args_schema=LookupByNameInput)
    def lookup_patient_by_name(full_name: str, runtime: ToolRuntime) -> Command:
        """Search patient candidates by name and request user disambiguation when needed."""

        return Command(
            update=_lookup_by_name_update(
                repository=repository,
                full_name=full_name,
                tool_call_id=runtime.tool_call_id,
            ),
        )

    @tool(args_schema=ActivatePatientSelectionInput)
    def activate_patient_selection(selection_index: int, runtime: ToolRuntime) -> Command:
        """Activate one candidate from the current enumerated patient list."""

        state = cast(AssistantState, runtime.state)
        return Command(
            update=_activate_patient_selection_update(
                repository=repository,
                state=state,
                selection_index=selection_index,
                tool_call_id=runtime.tool_call_id,
            ),
        )

    return [
        lookup_patient_by_security_number,
        lookup_patient_by_name,
        activate_patient_selection,
    ]


def _lookup_by_security_number_update(
    *,
    repository: PatientRepository,
    security_number: str,
    tool_call_id: str,
) -> dict[str, object]:
    """Return the state update for a direct security-number lookup."""

    patient = repository.find_by_security_number(security_number)
    if patient is None:
        event = create_audit_event(
            event_type="patient_lookup",
            status="warning",
            node_name="lookup_patient_by_security_number",
            detail=f"No patient found for security number {security_number!r}.",
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "not_found",
            "patient_lookup_candidates": [],
            "messages": [
                ToolMessage(
                    content=(
                        "No patient was found for that security number. "
                        "Ask the user to verify the identifier or try another one."
                    ),
                    tool_call_id=tool_call_id,
                    name="lookup_patient_by_security_number",
                ),
            ],
            "audit_events": [event],
        }

    return _activate_patient_update(
        patient=patient,
        tool_call_id=tool_call_id,
        tool_name="lookup_patient_by_security_number",
        detail=(
            "Loaded patient by security number "
            f"{mask_security_number(patient['security_number'])}."
        ),
    )


def _lookup_by_name_update(
    *,
    repository: PatientRepository,
    full_name: str,
    tool_call_id: str,
) -> dict[str, object]:
    """Return the state update for a name-based patient lookup."""

    candidates = repository.search_by_name(full_name)
    if not candidates:
        event = create_audit_event(
            event_type="patient_lookup",
            status="warning",
            node_name="lookup_patient_by_name",
            detail=f"No patient matched the name query {full_name!r}.",
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "not_found",
            "patient_lookup_candidates": [],
            "messages": [
                ToolMessage(
                    content=(
                        "No patient matched that name. Ask the user to try another spelling or "
                        "provide the fictional security number."
                    ),
                    tool_call_id=tool_call_id,
                    name="lookup_patient_by_name",
                ),
            ],
            "audit_events": [event],
        }

    if len(candidates) == 1:
        patient = repository.find_by_security_number(candidates[0]["security_number"])
        if patient is None:
            event = create_audit_event(
                event_type="patient_lookup",
                status="error",
                node_name="lookup_patient_by_name",
                detail="Candidate existed but could not be hydrated into a patient record.",
            )
            emit_console_audit(event)
            return {
                "patient_lookup_status": "not_found",
                "messages": [
                    ToolMessage(
                        content=(
                            "A matching patient candidate was found, but the full record could "
                            "not be loaded. Ask the user to try again."
                        ),
                        tool_call_id=tool_call_id,
                        name="lookup_patient_by_name",
                    ),
                ],
                "audit_events": [event],
            }
        return _activate_patient_update(
            patient=patient,
            tool_call_id=tool_call_id,
            tool_name="lookup_patient_by_name",
            detail=(
                "Loaded patient by name query "
                f"{full_name!r}: {patient['full_name']}."
            ),
        )

    event = create_audit_event(
        event_type="patient_lookup_disambiguation",
        status="info",
        node_name="lookup_patient_by_name",
        detail=f"Multiple patients matched the name query {full_name!r}.",
    )
    emit_console_audit(event)
    return {
        "patient_lookup_status": "selection_required",
        "patient_lookup_candidates": candidates,
        "messages": [
            ToolMessage(
                content=_format_candidate_options(candidates),
                tool_call_id=tool_call_id,
                name="lookup_patient_by_name",
            ),
        ],
        "audit_events": [event],
    }


def _activate_patient_selection_update(
    *,
    repository: PatientRepository,
    state: AssistantState,
    selection_index: int,
    tool_call_id: str,
) -> dict[str, object]:
    """Return the state update for a candidate-selection tool call."""

    candidates = state.get("patient_lookup_candidates", [])
    if not candidates:
        event = create_audit_event(
            event_type="patient_lookup_selection",
            status="warning",
            node_name="activate_patient_selection",
            detail="User tried to select a patient without pending candidates.",
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "not_found",
            "messages": [
                ToolMessage(
                    content=(
                        "There is no pending patient selection. Ask the user to search for a "
                        "patient by name or security number first."
                    ),
                    tool_call_id=tool_call_id,
                    name="activate_patient_selection",
                ),
            ],
            "audit_events": [event],
        }

    zero_based_index = selection_index - 1
    if zero_based_index < 0 or zero_based_index >= len(candidates):
        event = create_audit_event(
            event_type="patient_lookup_selection",
            status="warning",
            node_name="activate_patient_selection",
            detail=(
                "User selected an invalid candidate index "
                f"{selection_index} for {len(candidates)} candidates."
            ),
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "selection_required",
            "messages": [
                ToolMessage(
                    content=(
                        "That selection is invalid. Ask the user to choose one of the numbered "
                        "options shown in the list."
                    ),
                    tool_call_id=tool_call_id,
                    name="activate_patient_selection",
                ),
            ],
            "audit_events": [event],
        }

    patient = repository.find_by_security_number(candidates[zero_based_index]["security_number"])
    if patient is None:
        event = create_audit_event(
            event_type="patient_lookup_selection",
            status="error",
            node_name="activate_patient_selection",
            detail="Selected candidate could not be hydrated into a patient record.",
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "not_found",
            "messages": [
                ToolMessage(
                    content=(
                        "The selected patient could not be loaded. Ask the user to repeat the "
                        "search."
                    ),
                    tool_call_id=tool_call_id,
                    name="activate_patient_selection",
                ),
            ],
            "audit_events": [event],
        }

    return _activate_patient_update(
        patient=patient,
        tool_call_id=tool_call_id,
        tool_name="activate_patient_selection",
        detail=(
            "Activated patient after disambiguation: "
            f"{patient['full_name']} ({mask_security_number(patient['security_number'])})."
        ),
    )


def _activate_patient_update(
    *,
    patient: PatientRecord,
    tool_call_id: str,
    tool_name: str,
    detail: str,
) -> dict[str, object]:
    """Create a state update that activates a patient in session state.

    Args:
        patient: Fully hydrated patient record.
        tool_call_id: Tool-call identifier associated with the execution.
        tool_name: Name of the tool emitting the update.
        detail: Human-readable audit detail.

    Returns:
        Dictionary of state updates that activates the patient and appends a `ToolMessage`.
    """

    event = create_audit_event(
        event_type="patient_activation",
        status="success",
        node_name=tool_name,
        detail=detail,
    )
    emit_console_audit(event)
    return {
        "active_patient": patient,
        "active_patient_header": build_active_patient_header(patient),
        "patient_lookup_status": "loaded",
        "patient_lookup_candidates": [],
        "analysis_result": None,
        "messages": [
            ToolMessage(
                content=_format_loaded_patient_summary(patient),
                tool_call_id=tool_call_id,
                name=tool_name,
            ),
        ],
        "audit_events": [event],
    }


def _format_candidate_options(candidates: list[PatientCandidate]) -> str:
    """Format an enumerated candidate list for the model and user.

    Args:
        candidates: Patient candidates found during name lookup.

    Returns:
        An enumerated list with concise demographics.
    """

    lines = [
        "Multiple patients matched the name. Ask the user to choose one numbered option:",
    ]
    for index, candidate in enumerate(candidates, start=1):
        demographics: list[str] = []
        if candidate.get("age_years") is not None:
            demographics.append(f"{candidate['age_years']} years")
        if candidate.get("sex"):
            demographics.append(str(candidate["sex"]))
        lines.append(
            f"{index}. {candidate['full_name']} • "
            f"ID {mask_security_number(candidate['security_number'])}"
            + (f" • {' • '.join(demographics)}" if demographics else "")
        )
    return "\n".join(lines)


def _format_loaded_patient_summary(patient: PatientRecord) -> str:
    """Summarize a loaded patient for the tool result message.

    Args:
        patient: Fully loaded patient record.

    Returns:
        A concise patient summary visible to the model.
    """

    recent_exam_names = [exam["exam_name"] for exam in patient.get("recent_exams", [])]
    return "\n".join(
        [
            "Patient context loaded successfully.",
            f"Name: {patient['full_name']}",
            f"Security number: {patient['security_number']}",
            f"Age: {patient.get('age_years')}",
            f"Sex: {patient.get('sex')}",
            f"Allergies: {', '.join(patient.get('allergies', [])) or 'None reported'}",
            f"Conditions: {', '.join(patient.get('conditions', [])) or 'None reported'}",
            f"Medications: {', '.join(patient.get('medications', [])) or 'None reported'}",
            f"Recent exams: {', '.join(recent_exam_names) or 'None available'}",
        ]
    )
