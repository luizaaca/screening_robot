"""Tests for state helper functions."""

from __future__ import annotations

from screening_agent.graph.state import build_active_patient_header, mask_security_number



def test_mask_security_number_preserves_only_last_four_digits() -> None:
    """Ensure patient identifiers are masked before user-facing display."""

    assert mask_security_number("12345678") == "****5678"



def test_build_active_patient_header_includes_masked_identifier() -> None:
    """Ensure the active-patient header includes masked ID and demographics."""

    header = build_active_patient_header(
        {
            "security_number": "12345678",
            "full_name": "Maria Silva",
            "birth_date": "1988-04-02",
            "age_years": 37,
            "sex": "F",
            "allergies": ["Dipyrone"],
            "conditions": ["Asthma"],
            "medications": ["Salbutamol"],
            "last_vitals": None,
            "recent_exams": [],
        },
    )

    assert "Maria Silva" in header
    assert "****5678" in header
    assert "37 years" in header
