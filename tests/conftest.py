"""Shared pytest fixtures for the screening assistant test suite."""

from __future__ import annotations

import json
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
            INSERT INTO patients (
                security_number,
                full_name,
                birth_date,
                age_years,
                sex,
                allergies_json,
                conditions_json,
                medications_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "12345678",
                    "Maria Silva",
                    "1988-04-02",
                    37,
                    "F",
                    json.dumps(["Dipyrone"]),
                    json.dumps(["Asthma"]),
                    json.dumps(["Salbutamol"]),
                ),
                (
                    "87654321",
                    "Maria Silva",
                    "1991-09-10",
                    34,
                    "F",
                    json.dumps([]),
                    json.dumps(["Hypertension"]),
                    json.dumps(["Losartan"]),
                ),
                (
                    "11112222",
                    "João Souza",
                    "1979-01-15",
                    47,
                    "M",
                    json.dumps(["Penicillin"]),
                    json.dumps(["Diabetes"]),
                    json.dumps(["Metformin"]),
                ),
            ],
        )
        cursor.execute(
            """
            INSERT INTO patient_vitals (
                patient_id,
                recorded_at,
                blood_pressure,
                heart_rate_bpm,
                respiratory_rate_bpm,
                temperature_c,
                oxygen_saturation_pct
            )
            VALUES (1, '2026-04-20T10:00:00Z', '118/78', 82, 18, 36.7, 98)
            """,
        )
        cursor.executemany(
            """
            INSERT INTO patient_exams (patient_id, exam_name, exam_date, result_summary)
            VALUES (?, ?, ?, ?)
            """,
            [
                (1, "Chest X-Ray", "2026-03-18", "No acute infiltrate."),
                (1, "CBC", "2026-04-01", "Mild eosinophilia."),
                (3, "HbA1c", "2026-03-01", "7.3%."),
            ],
        )
        connection.commit()
