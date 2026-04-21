"""End-to-end graph tests using deterministic mock runtimes."""

from __future__ import annotations

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from screening_agent.data import PatientRepository
from screening_agent.graph.builder import build_screening_graph
from screening_agent.model.mock_runtime import MockClinicalBackend, MockControlModel



def test_mock_graph_handles_usage_request(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure the graph can answer a usage question without external model access."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        clinical_backend=MockClinicalBackend(),
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
        clinical_backend=MockClinicalBackend(),
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



def test_mock_graph_runs_combined_lookup_and_analysis_flow(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure combined patient identification and analysis completes in one turn."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        clinical_backend=MockClinicalBackend(),
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
    assert result["analysis_result"] is not None
    assert result["analysis_result"]["primary_hypothesis"] == "Diabetes mellitus or poor glycemic control"
    assert "Serum glucose" in result["analysis_result"]["recommended_exams"]
    assert "Active patient: João Souza" in result["last_response"]
    assert "Clinical screening support only" in result["last_response"]



def test_mock_graph_clears_active_patient_context(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure clear requests remove the active patient from the thread state."""

    graph = build_screening_graph(
        control_model=MockControlModel(),
        clinical_backend=MockClinicalBackend(),
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
