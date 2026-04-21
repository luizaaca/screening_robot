"""Deterministic demo-data seeding for the screening assistant."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final
import sqlite3

from screening_agent.data.patient_repository import PatientRepository


@dataclass(frozen=True)
class DemoVitalSeed:
    """Vital-sign payload associated with a demo patient."""

    recorded_at: str
    blood_pressure: str | None
    heart_rate_bpm: int | None
    respiratory_rate_bpm: int | None
    temperature_c: float | None
    oxygen_saturation_pct: int | None


@dataclass(frozen=True)
class DemoExamSeed:
    """Exam payload associated with a demo patient."""

    exam_name: str
    exam_date: str | None
    result_summary: str | None


@dataclass(frozen=True)
class DemoPatientSeed:
    """Synthetic patient payload used to populate the SQLite demo database."""

    security_number: str
    full_name: str
    birth_date: str
    age_years: int
    sex: str
    allergies: list[str]
    conditions: list[str]
    medications: list[str]
    vitals: list[DemoVitalSeed]
    exams: list[DemoExamSeed]


@dataclass(frozen=True)
class SeedReport:
    """Outcome summary for the demo-data seeding operation."""

    inserted_patients: int
    inserted_vitals: int
    inserted_exams: int
    skipped: bool


DEMO_PATIENTS: Final[tuple[DemoPatientSeed, ...]] = (
    DemoPatientSeed(
        security_number="12003456",
        full_name="Maria Silva",
        birth_date="1988-04-02",
        age_years=37,
        sex="F",
        allergies=["Dipyrone"],
        conditions=["Asthma"],
        medications=["Salbutamol inhaler"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-20T10:00:00Z",
                blood_pressure="118/78",
                heart_rate_bpm=82,
                respiratory_rate_bpm=18,
                temperature_c=36.7,
                oxygen_saturation_pct=98,
            ),
        ],
        exams=[
            DemoExamSeed("Chest X-Ray", "2026-03-18", "No acute infiltrate."),
            DemoExamSeed("CBC", "2026-04-01", "Mild eosinophilia."),
        ],
    ),
    DemoPatientSeed(
        security_number="99887766",
        full_name="Maria Silva",
        birth_date="1991-09-10",
        age_years=34,
        sex="F",
        allergies=[],
        conditions=["Hypertension"],
        medications=["Losartan"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-19T08:30:00Z",
                blood_pressure="144/92",
                heart_rate_bpm=76,
                respiratory_rate_bpm=16,
                temperature_c=36.5,
                oxygen_saturation_pct=99,
            ),
        ],
        exams=[
            DemoExamSeed("Basic metabolic panel", "2026-03-22", "Normal renal function."),
        ],
    ),
    DemoPatientSeed(
        security_number="55667788",
        full_name="João Souza",
        birth_date="1979-01-15",
        age_years=47,
        sex="M",
        allergies=["Penicillin"],
        conditions=["Type 2 diabetes mellitus"],
        medications=["Metformin"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-18T14:00:00Z",
                blood_pressure="126/84",
                heart_rate_bpm=88,
                respiratory_rate_bpm=17,
                temperature_c=36.9,
                oxygen_saturation_pct=97,
            ),
        ],
        exams=[
            DemoExamSeed("HbA1c", "2026-03-01", "7.3%."),
            DemoExamSeed("Urinalysis", "2026-02-11", "Trace glucose."),
        ],
    ),
    DemoPatientSeed(
        security_number="22334455",
        full_name="Ana Costa",
        birth_date="1996-11-23",
        age_years=29,
        sex="F",
        allergies=["Ibuprofen"],
        conditions=["Migraine"],
        medications=["Sumatriptan"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-17T11:15:00Z",
                blood_pressure="110/70",
                heart_rate_bpm=72,
                respiratory_rate_bpm=15,
                temperature_c=36.4,
                oxygen_saturation_pct=99,
            ),
        ],
        exams=[
            DemoExamSeed("Brain MRI", "2026-01-08", "No acute intracranial abnormality."),
        ],
    ),
    DemoPatientSeed(
        security_number="66778899",
        full_name="Carlos Mendes",
        birth_date="1964-06-30",
        age_years=61,
        sex="M",
        allergies=[],
        conditions=["COPD", "Former smoker"],
        medications=["Tiotropium", "Albuterol inhaler"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-20T09:20:00Z",
                blood_pressure="132/80",
                heart_rate_bpm=90,
                respiratory_rate_bpm=22,
                temperature_c=36.8,
                oxygen_saturation_pct=94,
            ),
        ],
        exams=[
            DemoExamSeed("Spirometry", "2026-02-20", "Moderate obstructive defect."),
            DemoExamSeed("Chest CT", "2025-12-12", "Hyperinflation without focal mass."),
        ],
    ),
    DemoPatientSeed(
        security_number="10293847",
        full_name="Beatriz Lima",
        birth_date="1973-08-14",
        age_years=52,
        sex="F",
        allergies=["Sulfa drugs"],
        conditions=["Hypothyroidism"],
        medications=["Levothyroxine"],
        vitals=[
            DemoVitalSeed(
                recorded_at="2026-04-16T07:50:00Z",
                blood_pressure="124/76",
                heart_rate_bpm=68,
                respiratory_rate_bpm=14,
                temperature_c=36.3,
                oxygen_saturation_pct=100,
            ),
        ],
        exams=[
            DemoExamSeed("TSH", "2026-03-28", "Mildly elevated at 5.8 mIU/L."),
        ],
    ),
)



def seed_demo_repository(
    repository: PatientRepository,
    *,
    replace_existing: bool = False,
) -> SeedReport:
    """Populate the SQLite repository with deterministic synthetic demo patients.

    Args:
        repository: Repository whose database should be populated.
        replace_existing: Whether to delete existing patient data before seeding.

    Returns:
        A summary of the seeding operation.
    """

    repository.initialize_database()
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        existing_patient_count = _count_patients(connection)
        if existing_patient_count > 0 and not replace_existing:
            return SeedReport(
                inserted_patients=0,
                inserted_vitals=0,
                inserted_exams=0,
                skipped=True,
            )
        if replace_existing:
            _clear_existing_data(connection)
        inserted_patients, inserted_vitals, inserted_exams = _insert_demo_patients(connection)
        connection.commit()
    return SeedReport(
        inserted_patients=inserted_patients,
        inserted_vitals=inserted_vitals,
        inserted_exams=inserted_exams,
        skipped=False,
    )



def _count_patients(connection: sqlite3.Connection) -> int:
    """Count patient records currently stored in the database.

    Args:
        connection: Active SQLite connection.

    Returns:
        Number of patient rows.
    """

    row = connection.execute("SELECT COUNT(*) AS total FROM patients").fetchone()
    return int(row[0]) if row is not None else 0



def _clear_existing_data(connection: sqlite3.Connection) -> None:
    """Delete existing demo tables in dependency-safe order.

    Args:
        connection: Active SQLite connection.
    """

    connection.execute("DELETE FROM patient_exams")
    connection.execute("DELETE FROM patient_vitals")
    connection.execute("DELETE FROM patients")



def _insert_demo_patients(connection: sqlite3.Connection) -> tuple[int, int, int]:
    """Insert the built-in demo patients and related records.

    Args:
        connection: Active SQLite connection.

    Returns:
        Counts of inserted patients, vitals, and exams.
    """

    inserted_patients = 0
    inserted_vitals = 0
    inserted_exams = 0
    for patient in DEMO_PATIENTS:
        cursor = connection.execute(
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
            (
                patient.security_number,
                patient.full_name,
                patient.birth_date,
                patient.age_years,
                patient.sex,
                json.dumps(patient.allergies),
                json.dumps(patient.conditions),
                json.dumps(patient.medications),
            ),
        )
        patient_id = int(cursor.lastrowid)
        inserted_patients += 1
        for vital in patient.vitals:
            connection.execute(
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
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    vital.recorded_at,
                    vital.blood_pressure,
                    vital.heart_rate_bpm,
                    vital.respiratory_rate_bpm,
                    vital.temperature_c,
                    vital.oxygen_saturation_pct,
                ),
            )
            inserted_vitals += 1
        for exam in patient.exams:
            connection.execute(
                """
                INSERT INTO patient_exams (
                    patient_id,
                    exam_name,
                    exam_date,
                    result_summary
                )
                VALUES (?, ?, ?, ?)
                """,
                (patient_id, exam.exam_name, exam.exam_date, exam.result_summary),
            )
            inserted_exams += 1
    return inserted_patients, inserted_vitals, inserted_exams
