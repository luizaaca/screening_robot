"""Deterministic demo-data seeding for the screening assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
import sqlite3

from screening_agent.data.patient_repository import PatientRepository


@dataclass(frozen=True)
class DemoPatientSeed:
    """Synthetic patient payload used to populate the SQLite demo database."""

    security_number: str
    full_name: str
    clinical_context: str


@dataclass(frozen=True)
class SeedReport:
    """Outcome summary for the demo-data seeding operation."""

    inserted_patients: int
    skipped: bool


DEMO_PATIENTS: Final[tuple[DemoPatientSeed, ...]] = (
    DemoPatientSeed(
        security_number="12003456",
        full_name="Maria Silva",
        clinical_context=(
            "37-year-old female. Allergies: Dipyrone. Conditions: Asthma. "
            "Medications: Salbutamol inhaler. "
            "Recent exams: Chest X-Ray (2026-03-18, no acute infiltrate), "
            "CBC (2026-04-01, mild eosinophilia). "
            "Vitals (2026-04-20): BP 118/78, HR 82, RR 18, Temp 36.7C, O2 98%."
        ),
    ),
    DemoPatientSeed(
        security_number="99887766",
        full_name="Maria Silva",
        clinical_context=(
            "34-year-old female. Allergies: none. Conditions: Hypertension. "
            "Medications: Losartan. "
            "Recent exams: Basic metabolic panel (2026-03-22, normal renal function). "
            "Vitals (2026-04-19): BP 144/92, HR 76, RR 16, Temp 36.5C, O2 99%."
        ),
    ),
    DemoPatientSeed(
        security_number="55667788",
        full_name="João Souza",
        clinical_context=(
            "47-year-old male. Allergies: Penicillin. Conditions: Type 2 diabetes mellitus. "
            "Medications: Metformin. "
            "Recent exams: HbA1c (2026-03-01, 7.3%), Urinalysis (2026-02-11, trace glucose). "
            "Vitals (2026-04-18): BP 126/84, HR 88, RR 17, Temp 36.9C, O2 97%."
        ),
    ),
    DemoPatientSeed(
        security_number="22334455",
        full_name="Ana Costa",
        clinical_context=(
            "29-year-old female. Allergies: Ibuprofen. Conditions: Migraine. "
            "Medications: Sumatriptan. "
            "Recent exams: Brain MRI (2026-01-08, no acute intracranial abnormality). "
            "Vitals (2026-04-17): BP 110/70, HR 72, RR 15, Temp 36.4C, O2 99%."
        ),
    ),
    DemoPatientSeed(
        security_number="66778899",
        full_name="Carlos Mendes",
        clinical_context=(
            "61-year-old male. Allergies: none. Conditions: COPD, Former smoker. "
            "Medications: Tiotropium, Albuterol inhaler. "
            "Recent exams: Spirometry (2026-02-20, moderate obstructive defect), "
            "Chest CT (2025-12-12, hyperinflation without focal mass). "
            "Vitals (2026-04-20): BP 132/80, HR 90, RR 22, Temp 36.8C, O2 94%."
        ),
    ),
    DemoPatientSeed(
        security_number="10293847",
        full_name="Beatriz Lima",
        clinical_context=(
            "52-year-old female. Allergies: Sulfa drugs. Conditions: Hypothyroidism. "
            "Medications: Levothyroxine. "
            "Recent exams: TSH (2026-03-28, mildly elevated at 5.8 mIU/L). "
            "Vitals (2026-04-16): BP 124/76, HR 68, RR 14, Temp 36.3C, O2 100%."
        ),
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
            return SeedReport(inserted_patients=0, skipped=True)
        if replace_existing:
            connection.execute("DELETE FROM patients")
        inserted_patients = _insert_demo_patients(connection)
        connection.commit()
    return SeedReport(inserted_patients=inserted_patients, skipped=False)


def _count_patients(connection: sqlite3.Connection) -> int:
    """Count patient records currently stored in the database.

    Args:
        connection: Active SQLite connection.

    Returns:
        Number of patient rows.
    """

    row = connection.execute("SELECT COUNT(*) AS total FROM patients").fetchone()
    return int(row[0]) if row is not None else 0


def _insert_demo_patients(connection: sqlite3.Connection) -> int:
    """Insert the built-in demo patients.

    Args:
        connection: Active SQLite connection.

    Returns:
        Count of inserted patients.
    """

    inserted_patients = 0
    for patient in DEMO_PATIENTS:
        connection.execute(
            """
            INSERT INTO patients (security_number, full_name, clinical_context)
            VALUES (?, ?, ?)
            """,
            (patient.security_number, patient.full_name, patient.clinical_context),
        )
        inserted_patients += 1
    return inserted_patients
