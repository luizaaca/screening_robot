"""Tests for deterministic builder helpers."""

from __future__ import annotations

from screening_agent.graph.builder import _route_after_lookup



def test_route_after_lookup_continues_to_analysis_only_when_lookup_loaded() -> None:
    """Ensure combined requests continue to analysis only after a successful lookup."""

    command = _route_after_lookup(
        {
            "messages": [],
            "audit_events": [],
            "router_intent": "patient_lookup_then_analysis",
            "patient_lookup_status": "loaded",
        },
    )

    assert command.goto == "symptom_analysis"



def test_route_after_lookup_finalizes_when_lookup_did_not_load_patient() -> None:
    """Ensure the graph does not analyze symptoms with a stale or missing patient."""

    command = _route_after_lookup(
        {
            "messages": [],
            "audit_events": [],
            "router_intent": "patient_lookup_then_analysis",
            "patient_lookup_status": "not_found",
        },
    )

    assert command.goto == "finalize_response"
