"""Tests for deterministic demo-data seeding."""

from __future__ import annotations

from pathlib import Path

from screening_agent.data import PatientRepository, seed_demo_repository



def test_seed_demo_repository_populates_empty_database(tmp_path: Path) -> None:
    """Ensure the demo seeder inserts the expected number of base records."""

    repository = PatientRepository(tmp_path / "demo.sqlite3")

    report = seed_demo_repository(repository)

    assert report.skipped is False
    assert report.inserted_patients == 6
    assert repository.count_patients() == 6
    assert repository.find_by_security_number("12003456") is not None



def test_seed_demo_repository_skips_when_data_already_exists(tmp_path: Path) -> None:
    """Ensure seeding is idempotent unless replacement is explicitly requested."""

    repository = PatientRepository(tmp_path / "demo.sqlite3")
    first_report = seed_demo_repository(repository)
    second_report = seed_demo_repository(repository)

    assert first_report.skipped is False
    assert second_report.skipped is True
    assert repository.count_patients() == 6
