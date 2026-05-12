"""SQLite patient repository for the screening assistant."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

from screening_agent.graph.state import PatientCandidate, PatientRecord


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
            SELECT security_number, full_name, clinical_context
            FROM patients
            WHERE security_number = ?
        """
        with closing(self._connect()) as connection:
            row = connection.execute(query, (normalized_security_number,)).fetchone()
            if row is None:
                return None
            return {
                "security_number": str(row["security_number"]),
                "full_name": str(row["full_name"]),
                "clinical_context": str(row["clinical_context"]),
            }

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
            SELECT security_number, full_name
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
        return [
            {
                "security_number": str(row["security_number"]),
                "full_name": str(row["full_name"]),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured with row access by name.

        Returns:
            A SQLite connection instance.
        """

        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
