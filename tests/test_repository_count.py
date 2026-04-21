"""Tests for lightweight repository metadata queries."""

from __future__ import annotations

from screening_agent.data import PatientRepository



def test_count_patients_returns_number_of_seeded_rows(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure repository metadata can report how many patients are available."""

    assert seeded_repository.count_patients() == 3
