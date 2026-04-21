"""SQLite patient repository for the screening assistant."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from screening_agent.graph.state import ExamSummary, PatientCandidate, PatientRecord, VitalSignsSnapshot


class PatientRepository:
    """Provide typed access to synthetic patient data stored in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the repository.

        Args:
            database_path: Path to the SQLite database file.
        """

        self._database_path = Path(database_path)

    @property
    def database_path(self) -> Path:
        """Return the path of the SQLite database file.

        Returns:
            The configured database path.
        """

        return self._database_path

    def initialize_database(self, schema_path: str | Path | None = None) -> None:
        """Create the SQLite schema when it does not exist.

        Args:
            schema_path: Optional custom schema file path.
        """

        resolved_schema_path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        script = resolved_schema_path.read_text(encoding="utf-8")
        with closing(self._connect()) as connection:
            connection.executescript(script)
            connection.commit()

    def count_patients(self) -> int:
        """Return how many patient rows exist in the SQLite database.

        Returns:
            Number of patient records currently stored.
        """

        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM patients").fetchone()
        return int(row["total"]) if row is not None else 0

    def find_by_security_number(self, security_number: str) -> PatientRecord | None:
        """Look up a full patient record by fictional security number.

        Args:
            security_number: Fictional patient security number.

        Returns:
            The matching patient record, if found.
        """

        normalized_security_number = security_number.strip()
        if not normalized_security_number:
            raise ValueError("security_number must not be empty.")

        query = """
            SELECT id, security_number, full_name, birth_date, age_years, sex,
                   allergies_json, conditions_json, medications_json
            FROM patients
            WHERE security_number = ?
        """
        with closing(self._connect()) as connection:
            row = connection.execute(query, (normalized_security_number,)).fetchone()
            if row is None:
                return None
            return self._build_patient_record(connection, row)

    def search_by_name(self, name_query: str, limit: int = 5) -> list[PatientCandidate]:
        """Search patient candidates by partial or exact name.

        Args:
            name_query: Partial or exact patient name.
            limit: Maximum number of candidates to return.

        Returns:
            Candidate patients for disambiguation.

        Raises:
            ValueError: If the query is empty or the limit is invalid.
        """

        normalized_name_query = name_query.strip()
        if not normalized_name_query:
            raise ValueError("name_query must not be empty.")
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        search_pattern = f"%{normalized_name_query}%"
        prefix_pattern = f"{normalized_name_query}%"
        query = """
            SELECT security_number, full_name, age_years, sex
            FROM patients
            WHERE full_name LIKE ? COLLATE NOCASE
            ORDER BY
                CASE
                    WHEN lower(full_name) = lower(?) THEN 0
                    WHEN lower(full_name) LIKE lower(?) THEN 1
                    ELSE 2
                END,
                full_name ASC,
                security_number ASC
            LIMIT ?
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                query,
                (search_pattern, normalized_name_query, prefix_pattern, limit),
            ).fetchall()
        return [self._build_patient_candidate(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured with row access by name.

        Returns:
            A SQLite connection instance.
        """

        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _build_patient_candidate(self, row: sqlite3.Row) -> PatientCandidate:
        """Convert a patient row into a disambiguation candidate.

        Args:
            row: SQLite row with candidate columns.

        Returns:
            A short patient candidate dictionary.
        """

        return {
            "security_number": str(row["security_number"]),
            "full_name": str(row["full_name"]),
            "age_years": int(row["age_years"]) if row["age_years"] is not None else None,
            "sex": str(row["sex"]) if row["sex"] is not None else None,
        }

    def _build_patient_record(self, connection: sqlite3.Connection, row: sqlite3.Row) -> PatientRecord:
        """Hydrate the full patient record including vitals and exams.

        Args:
            connection: Active SQLite connection.
            row: Base patient row.

        Returns:
            A full patient record.
        """

        patient_id = int(row["id"])
        return {
            "security_number": str(row["security_number"]),
            "full_name": str(row["full_name"]),
            "birth_date": str(row["birth_date"]) if row["birth_date"] is not None else None,
            "age_years": int(row["age_years"]) if row["age_years"] is not None else None,
            "sex": str(row["sex"]) if row["sex"] is not None else None,
            "allergies": _parse_json_string_list(row["allergies_json"]),
            "conditions": _parse_json_string_list(row["conditions_json"]),
            "medications": _parse_json_string_list(row["medications_json"]),
            "last_vitals": self._load_latest_vitals(connection, patient_id),
            "recent_exams": self._load_recent_exams(connection, patient_id),
        }

    def _load_latest_vitals(
        self,
        connection: sqlite3.Connection,
        patient_id: int,
    ) -> VitalSignsSnapshot | None:
        """Load the latest vital signs snapshot for a patient.

        Args:
            connection: Active SQLite connection.
            patient_id: Internal patient primary key.

        Returns:
            The latest vital signs snapshot or `None`.
        """

        query = """
            SELECT recorded_at, blood_pressure, heart_rate_bpm, respiratory_rate_bpm,
                   temperature_c, oxygen_saturation_pct
            FROM patient_vitals
            WHERE patient_id = ?
            ORDER BY recorded_at DESC, id DESC
            LIMIT 1
        """
        row = connection.execute(query, (patient_id,)).fetchone()
        if row is None:
            return None
        snapshot: VitalSignsSnapshot = {"recorded_at": str(row["recorded_at"])}
        if row["blood_pressure"] is not None:
            snapshot["blood_pressure"] = str(row["blood_pressure"])
        if row["heart_rate_bpm"] is not None:
            snapshot["heart_rate_bpm"] = int(row["heart_rate_bpm"])
        if row["respiratory_rate_bpm"] is not None:
            snapshot["respiratory_rate_bpm"] = int(row["respiratory_rate_bpm"])
        if row["temperature_c"] is not None:
            snapshot["temperature_c"] = float(row["temperature_c"])
        if row["oxygen_saturation_pct"] is not None:
            snapshot["oxygen_saturation_pct"] = int(row["oxygen_saturation_pct"])
        return snapshot

    def _load_recent_exams(
        self,
        connection: sqlite3.Connection,
        patient_id: int,
        limit: int = 5,
    ) -> list[ExamSummary]:
        """Load the most recent exams for a patient.

        Args:
            connection: Active SQLite connection.
            patient_id: Internal patient primary key.
            limit: Maximum number of exams to return.

        Returns:
            A list of recent exam summaries.
        """

        query = """
            SELECT exam_name, exam_date, result_summary
            FROM patient_exams
            WHERE patient_id = ?
            ORDER BY exam_date DESC, id DESC
            LIMIT ?
        """
        rows = connection.execute(query, (patient_id, limit)).fetchall()
        return [
            {
                "exam_name": str(row["exam_name"]),
                "exam_date": str(row["exam_date"]) if row["exam_date"] is not None else None,
                "result_summary": str(row["result_summary"]) if row["result_summary"] is not None else None,
            }
            for row in rows
        ]



def _parse_json_string_list(raw_value: str | None) -> list[str]:
    """Parse a JSON string list stored in SQLite.

    Args:
        raw_value: Raw JSON string.

    Returns:
        A normalized list of strings.
    """

    if raw_value is None or not raw_value.strip():
        return []
    parsed = json.loads(raw_value)
    return [str(item) for item in parsed]
