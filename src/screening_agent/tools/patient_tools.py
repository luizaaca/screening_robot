"""Patient lookup tools for the LangGraph workflow."""

from collections.abc import Mapping
from typing import cast

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from screening_agent.audit import create_debug_event, emit_console_audit
from screening_agent.data import PatientRepository
from screening_agent.graph.state import (
    AssistantState,
    PatientCandidate,
    PatientRecord,
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
        description=(
            "Full or partial patient name used for disambiguation. The search tolerates "
            "common name particles and minor spelling differences."
        ),
    )


class ListPatientsInput(BaseModel):
    """Arguments for listing all available patient candidates."""

    include_all: bool = Field(
        default=True,
        description="Set true when the user asks to list all available patients.",
    )


class ActivatePatientSelectionInput(BaseModel):
    """Arguments for activating a patient candidate from an enumerated list."""

    selection_index: int = Field(
        ge=1,
        description="One-based index chosen by the user from the enumerated candidate list.",
    )


def build_patient_lookup_tools(repository: PatientRepository) -> list[BaseTool]:
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

        _emit_tool_debug_event(
            runtime,
            event_name="tool_invocation",
            tool_name="lookup_patient_by_security_number",
            payload={"security_number": mask_security_number(security_number)},
        )
        update = _lookup_by_security_number_update(
            repository=repository,
            security_number=security_number,
            tool_call_id=runtime.tool_call_id,
        )
        _emit_tool_debug_event(
            runtime,
            event_name="tool_result",
            tool_name="lookup_patient_by_security_number",
            payload=_summarize_tool_update(update),
        )
        return Command(update=update)

    @tool(args_schema=LookupByNameInput)
    def lookup_patient_by_name(full_name: str, runtime: ToolRuntime) -> Command:
        """Search patient candidates by name and request user disambiguation when needed."""

        _emit_tool_debug_event(
            runtime,
            event_name="tool_invocation",
            tool_name="lookup_patient_by_name",
            payload={"full_name": full_name},
        )
        update = _lookup_by_name_update(
            repository=repository,
            full_name=full_name,
            tool_call_id=runtime.tool_call_id,
        )
        _emit_tool_debug_event(
            runtime,
            event_name="tool_result",
            tool_name="lookup_patient_by_name",
            payload=_summarize_tool_update(update),
        )
        return Command(update=update)

    @tool(args_schema=ListPatientsInput)
    def list_patients(include_all: bool, runtime: ToolRuntime) -> Command:
        """List all patient candidates available in the repository."""

        _emit_tool_debug_event(
            runtime,
            event_name="tool_invocation",
            tool_name="list_patients",
            payload={"include_all": include_all},
        )
        update = _list_patients_update(
            repository=repository,
            tool_call_id=runtime.tool_call_id,
        )
        _emit_tool_debug_event(
            runtime,
            event_name="tool_result",
            tool_name="list_patients",
            payload=_summarize_tool_update(update),
        )
        return Command(update=update)

    @tool(args_schema=ActivatePatientSelectionInput)
    def activate_patient_selection(selection_index: int, runtime: ToolRuntime) -> Command:
        """Activate one candidate from the current enumerated patient list."""

        state = cast(AssistantState, runtime.state)
        _emit_tool_debug_event(
            runtime,
            event_name="tool_invocation",
            tool_name="activate_patient_selection",
            payload={
                "selection_index": selection_index,
                "pending_candidate_count": len(state.get("patient_lookup_candidates", [])),
            },
        )
        update = _activate_patient_selection_update(
            repository=repository,
            state=state,
            selection_index=selection_index,
            tool_call_id=runtime.tool_call_id,
        )
        _emit_tool_debug_event(
            runtime,
            event_name="tool_result",
            tool_name="activate_patient_selection",
            payload=_summarize_tool_update(update),
        )
        return Command(update=update)

    return [
        cast(BaseTool, lookup_patient_by_security_number),
        cast(BaseTool, lookup_patient_by_name),
        cast(BaseTool, list_patients),
        cast(BaseTool, activate_patient_selection),
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
            "specialist_output_json": None,
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
                        "No patient matched that specific name query. Try an untried simplified "
                        "variant, another spelling, a partial given or family name, or ask the "
                        "user for the fictional security number if no reasonable query remains."
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
                "specialist_output_json": None,
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
        "specialist_output_json": None,
        "messages": [
            ToolMessage(
                content=_format_candidate_options(candidates),
                tool_call_id=tool_call_id,
                name="lookup_patient_by_name",
            ),
        ],
        "audit_events": [event],
    }


def _list_patients_update(
    *,
    repository: PatientRepository,
    tool_call_id: str,
) -> dict[str, object]:
    """Return the state update for listing available patient candidates."""

    candidates = repository.list_patients()
    if not candidates:
        event = create_audit_event(
            event_type="patient_lookup",
            status="warning",
            node_name="list_patients",
            detail="No patient records are available in the repository.",
        )
        emit_console_audit(event)
        return {
            "patient_lookup_status": "not_found",
            "patient_lookup_candidates": [],
            "specialist_output_json": None,
            "messages": [
                ToolMessage(
                    content="No patients are available in the repository.",
                    tool_call_id=tool_call_id,
                    name="list_patients",
                ),
            ],
            "audit_events": [event],
        }

    event = create_audit_event(
        event_type="patient_lookup_listing",
        status="info",
        node_name="list_patients",
        detail=f"Listed {len(candidates)} patient candidates.",
    )
    emit_console_audit(event)
    return {
        "patient_lookup_status": "selection_required",
        "patient_lookup_candidates": candidates,
        "specialist_output_json": None,
        "messages": [
            ToolMessage(
                content=_format_all_patient_options(candidates),
                tool_call_id=tool_call_id,
                name="list_patients",
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
            "specialist_output_json": None,
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
            "specialist_output_json": None,
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
            "specialist_output_json": None,
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
        "patient_lookup_status": "loaded",
        "patient_lookup_candidates": [],
        "specialist_output_json": None,
        "messages": [
            ToolMessage(
                content=_format_loaded_patient_summary(patient),
                tool_call_id=tool_call_id,
                name=tool_name,
            ),
        ],
        "audit_events": [event],
    }


def _format_candidate_options(
    candidates: list[PatientCandidate],
    *,
    intro: str = "Multiple patients matched the name. Ask the user to choose one numbered option:",
) -> str:
    """Format an enumerated candidate list for the model and user.

    Args:
        candidates: Patient candidates found during name lookup.

    Returns:
        An enumerated list with masked identifiers.
    """

    lines = [intro]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"{index}. {candidate['full_name']} - "
            f"ID {mask_security_number(candidate['security_number'])}"
        )
    return "\n".join(lines)


def _format_all_patient_options(candidates: list[PatientCandidate]) -> str:
    """Format all available patients as an enumerated selection list."""

    return _format_candidate_options(
        candidates,
        intro="Available patients. Ask the user to choose one numbered option to load a record:",
    )


def _format_loaded_patient_summary(patient: PatientRecord) -> str:
    """Summarize a loaded patient for the tool result message.

    Args:
        patient: Fully loaded patient record.

    Returns:
        A concise patient summary visible to the model.
    """

    return "\n".join(
        [
            "Patient context loaded successfully.",
            f"Name: {patient['full_name']}",
            f"Security number: {patient['security_number']}",
            f"Clinical context: {patient['clinical_context']}",
        ]
    )


def _emit_tool_debug_event(
    runtime: ToolRuntime,
    *,
    event_name: str,
    tool_name: str,
    payload: Mapping[str, object],
) -> None:
    """Emit a tool-level debug event through the runtime stream writer.

    Args:
        runtime: LangChain tool runtime injected by `ToolNode`.
        event_name: Stable debug event name.
        tool_name: Tool name associated with the event.
        payload: Event payload to emit.
    """

    writer = getattr(runtime, "stream_writer", None)
    if not callable(writer):
        return

    execution_info = getattr(runtime, "execution_info", None)
    writer(
        create_debug_event(
            event_name,
            node_name=tool_name,
            tool_call_id=runtime.tool_call_id,
            thread_id=getattr(execution_info, "thread_id", None),
            run_id=getattr(execution_info, "run_id", None),
            node_attempt=getattr(execution_info, "node_attempt", None),
            payload=payload,
        ),
    )


def _summarize_tool_update(update: Mapping[str, object]) -> dict[str, object]:
    """Build a compact, masked summary of a tool state update.

    Args:
        update: State update dictionary produced by a patient lookup tool.

    Returns:
        Masked summary of the tool result for console debug.
    """

    patient_lookup_candidates = update.get("patient_lookup_candidates", [])
    active_patient = update.get("active_patient")
    summary: dict[str, object] = {
        "patient_lookup_status": update.get("patient_lookup_status"),
        "candidate_count": len(patient_lookup_candidates)
        if isinstance(patient_lookup_candidates, list)
        else 0,
        "message_count": len(update.get("messages", []))
        if isinstance(update.get("messages", []), list)
        else 0,
    }
    if isinstance(active_patient, Mapping):
        security_number = active_patient.get("security_number")
        summary["active_patient"] = {
            "full_name": active_patient.get("full_name"),
            "security_number": (
                mask_security_number(str(security_number))
                if isinstance(security_number, str)
                else None
            ),
        }
    return summary
