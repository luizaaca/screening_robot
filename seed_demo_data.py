"""CLI entry point for populating the SQLite demo patient database."""

from __future__ import annotations

import argparse

from screening_agent.config import AppSettings
from screening_agent.data import PatientRepository, seed_demo_repository



def main() -> None:
    """Seed the configured SQLite database with deterministic demo data."""

    parser = argparse.ArgumentParser(
        description="Populate the screening assistant SQLite database with demo patients.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing patient data before inserting the demo dataset.",
    )
    args = parser.parse_args()

    settings = AppSettings.from_env()
    repository = PatientRepository(settings.patient_database_path)
    report = seed_demo_repository(repository, replace_existing=args.replace)

    if report.skipped:
        print(
            "Seed skipped because patient data already exists. "
            "Run again with --replace to overwrite the demo dataset.",
        )
        return

    print(
        "Seed completed successfully: "
        f"{report.inserted_patients} patients, "
        f"{report.inserted_vitals} vitals, "
        f"{report.inserted_exams} exams.",
    )


if __name__ == "__main__":
    main()
