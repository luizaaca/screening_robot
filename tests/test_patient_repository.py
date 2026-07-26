"""Tests for the SQLite patient repository."""

from __future__ import annotations

from screening_agent.data import PatientRepository



def test_find_by_security_number_returns_full_patient_record(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure lookup by fictional security number hydrates the full record."""

    patient = seeded_repository.find_by_security_number("12345678")

    assert patient is not None
    assert patient["full_name"] == "Maria Silva"
    assert patient["security_number"] == "12345678"
    assert "Asthma" in patient["clinical_context"]



def test_search_by_name_returns_ranked_candidates(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure name lookup returns candidates sorted by relevance and name."""

    candidates = seeded_repository.search_by_name("Maria")

    assert len(candidates) == 2
    assert candidates[0]["security_number"] == "12345678"
    assert candidates[1]["security_number"] == "87654321"


def test_search_by_name_ignores_common_name_particles(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure name lookup tolerates particles absent from stored names."""

    candidates = seeded_repository.search_by_name("Maria da Silva")

    assert len(candidates) == 2
    assert candidates[0]["security_number"] == "12345678"
    assert candidates[1]["security_number"] == "87654321"



def test_search_by_name_matches_reordered_name_tokens(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure name lookup can match first/last names in a different order."""

    candidates = seeded_repository.search_by_name("Silva Maria")

    assert len(candidates) == 2
    assert candidates[0]["security_number"] == "12345678"
    assert candidates[1]["security_number"] == "87654321"



def test_list_patients_returns_all_candidates(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure all patients can be listed for selection."""

    candidates = seeded_repository.list_patients()

    assert len(candidates) == 3
    assert {candidate["security_number"] for candidate in candidates} == {
        "12345678",
        "87654321",
        "11112222",
    }



def test_search_by_name_raises_for_empty_query(
    seeded_repository: PatientRepository,
) -> None:
    """Ensure empty name queries are rejected explicitly."""

    try:
        seeded_repository.search_by_name("   ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for empty name query.")
