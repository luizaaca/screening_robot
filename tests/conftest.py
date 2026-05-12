"""Shared pytest fixtures for the screening assistant test suite."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from screening_agent.data import PatientRepository


@pytest.fixture()
def seeded_repository(tmp_path: Path) -> PatientRepository:
    """Create a temporary SQLite repository populated with sample patients.

    Args:
        tmp_path: Temporary directory managed by pytest.

    Returns:
        A seeded patient repository instance.
    """

    database_path = tmp_path / "patients.sqlite3"
    repository = PatientRepository(database_path)
    repository.initialize_database()
    _seed_database(repository.database_path)
    return repository



def _seed_database(database_path: Path) -> None:
    """Populate the temporary SQLite database with representative test data.

    Args:
        database_path: Path to the temporary SQLite database.
    """

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        cursor.executemany(
            """
            INSERT INTO patients (security_number, full_name, clinical_context)
            VALUES (?, ?, ?)
            """,
            [
                (
                    "12345678",
                    "Maria Silva",
                    "37-year-old female. Conditions: Asthma. Medications: Salbutamol inhaler.",
                ),
                (
                    "87654321",
                    "Maria Silva",
                    "34-year-old female. Conditions: Hypertension. Medications: Losartan.",
                ),
                (
                    "11112222",
                    "João Souza",
                    "47-year-old male. Conditions: Type 2 diabetes mellitus. Medications: Metformin.",
                ),
            ],
        )
        connection.commit()
