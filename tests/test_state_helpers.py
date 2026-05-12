"""Tests for state helper functions."""

from __future__ import annotations

from screening_agent.graph.state import build_active_patient_header, mask_security_number



def test_mask_security_number_preserves_only_last_four_digits() -> None:
    """Ensure patient identifiers are masked before user-facing display."""

    assert mask_security_number("12345678") == "****5678"



def test_build_active_patient_header_includes_masked_identifier() -> None:
    """Ensure the active-patient header includes masked ID and patient name."""

    header = build_active_patient_header(
        {
            "security_number": "12345678",
            "full_name": "Maria Silva",
            "clinical_context": "37-year-old female. Conditions: Asthma.",
        },
    )

    assert "Maria Silva" in header
    assert "****5678" in header
    assert header.startswith("Active patient:")
