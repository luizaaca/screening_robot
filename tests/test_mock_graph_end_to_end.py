"""End-to-end graph tests using deterministic mock runtimes."""

from __future__ import annotations

import json

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from screening_agent.data import PatientRepository
from screening_agent.graph.builder import build_screening_graph
from screening_agent.model.mock_runtime import MockControlModel
from screening_agent.tools import create_mock_specialist_invoker



def test_mock_graph_handles_usage_request(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure the graph can answer a usage question without external model access."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="How do I use this assistant?")]},
        config={"configurable": {"thread_id": "usage-thread"}},
    )

    assert "look up a patient" in result["last_response"].lower()
    assert result["router_intent"] == "usage_instructions"



def test_mock_graph_supports_name_disambiguation_across_turns(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure candidate selection works across turns in the same thread."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "lookup-thread"}}

    first_turn = graph.invoke(
        {"messages": [HumanMessage(content="Find patient Maria Silva")]},
        config=config,
    )
    second_turn = graph.invoke(
        {"messages": [HumanMessage(content="2")]},
        config=config,
    )

    assert "multiple patients" in first_turn["last_response"].lower()
    assert second_turn["active_patient"] is not None
    assert second_turn["active_patient"]["security_number"] == "87654321"
    assert "active patient: maria silva" in second_turn["last_response"].lower()
    assert "Hypertension" in second_turn["last_response"]
    assert "Losartan" in second_turn["last_response"]
    assert "87654321" not in second_turn["last_response"]



def test_mock_graph_runs_combined_lookup_and_analysis_flow(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure combined patient identification and analysis completes in one turn."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Patient 11112222 has fatigue and frequent urination")
            ]
        },
        config={"configurable": {"thread_id": "combined-thread"}},
    )

    assert result["router_intent"] == "patient_lookup_then_analysis"
    specialist_output = json.loads(str(result["specialist_output_json"]))
    assert specialist_output["support_status"] == "supported"
    assert specialist_output["candidate_diseases"] == [
        "Diabetes mellitus or poor glycemic control"
    ]
    assert "Serum glucose" in specialist_output["recommended_exams_tests"]
    assert "Active patient: João Souza" in result["last_response"]
    assert "support only" not in result["last_response"]


def test_mock_graph_summarizes_loaded_patient_record(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure lookup-only turns describe the loaded record in the final answer."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Lookup patient 11112222")]},
        config={"configurable": {"thread_id": "lookup-summary-thread"}},
    )

    assert result["router_intent"] == "patient_lookup"
    assert "Active patient:" in result["last_response"]
    assert "Record summary:" in result["last_response"]
    assert "Type 2 diabetes mellitus" in result["last_response"]
    assert "Metformin" in result["last_response"]
    assert "11112222" not in result["last_response"]


def test_mock_graph_patient_lookup_iterates_over_multiple_identifiers(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure lookup can try another query after a normal not-found result."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Lookup patient 00000000 or 11112222")
            ]
        },
        config={"configurable": {"thread_id": "lookup-multiple-identifiers-thread"}},
    )

    assert result["router_intent"] == "patient_lookup"
    assert result["patient_lookup_status"] == "loaded"
    assert result["active_patient"]["security_number"] == "11112222"
    assert "Type 2 diabetes mellitus" in result["last_response"]
    assert "00000000" not in result["last_response"]


def test_mock_graph_patient_lookup_handles_name_particles(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure lookup tolerates particles that are absent from stored names."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="encontre os dados da paciente maria da silva")
            ]
        },
        config={"configurable": {"thread_id": "lookup-name-particle-thread"}},
    )

    assert result["router_intent"] == "patient_lookup"
    assert result["patient_lookup_status"] == "selection_required"
    assert len(result["patient_lookup_candidates"]) == 2
    assert "Maria Silva" in result["last_response"]
    assert "No patient matched" not in result["last_response"]


def test_mock_graph_patient_lookup_can_list_all_patients(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure all-patient listing requests produce an enumerated candidate list."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="liste todos os pacientes")]},
        config={"configurable": {"thread_id": "lookup-list-all-thread"}},
    )

    assert result["router_intent"] == "patient_lookup"
    assert result["patient_lookup_status"] == "selection_required"
    assert len(result["patient_lookup_candidates"]) == 3
    assert "Available patients" in result["last_response"]
    assert "Maria Silva" in result["last_response"]



def test_mock_graph_clears_active_patient_context(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure clear requests remove the active patient from the thread state."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "clear-thread"}}

    graph.invoke(
        {"messages": [HumanMessage(content="Lookup patient 11112222")]},
        config=config,
    )
    cleared = graph.invoke(
        {"messages": [HumanMessage(content="Clear active patient")]},
        config=config,
    )

    assert cleared["active_patient"] is None
    assert "cleared" in cleared["last_response"].lower()
    assert "Active patient:" not in cleared["last_response"]
