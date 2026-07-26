"""SQLite patient repository for the screening assistant."""

from __future__ import annotations

from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path
import re
import sqlite3
import unicodedata

from screening_agent.graph.state import PatientCandidate, PatientRecord

_NAME_PARTICLES = frozenset(
    {
        "da",
        "de",
        "di",
        "do",
        "das",
        "des",
        "dos",
        "del",
        "della",
        "e",
    }
)


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

    def list_patients(self) -> list[PatientCandidate]:
        """Return all available patient candidates ordered for display.

        Returns:
            Candidate patients with names and security numbers.
        """

        query = """
            SELECT security_number, full_name
            FROM patients
            ORDER BY full_name COLLATE NOCASE ASC, security_number ASC
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query).fetchall()
        return [
            {
                "security_number": str(row["security_number"]),
                "full_name": str(row["full_name"]),
            }
            for row in rows
        ]

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

        query = """
            SELECT security_number, full_name
            FROM patients
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query).fetchall()

        query_key = _normalize_name_for_matching(normalized_name_query)
        query_tokens = _name_match_tokens(normalized_name_query)
        scored_candidates: list[tuple[int, str, str, PatientCandidate]] = []
        for row in rows:
            security_number = str(row["security_number"])
            full_name = str(row["full_name"])
            score = _score_name_match(
                query_key=query_key,
                query_tokens=query_tokens,
                candidate_name=full_name,
            )
            if score is None:
                continue
            scored_candidates.append(
                (
                    score,
                    _normalize_name_for_matching(full_name),
                    security_number,
                    {
                        "security_number": security_number,
                        "full_name": full_name,
                    },
                )
            )

        scored_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [candidate for *_unused, candidate in scored_candidates[:limit]]

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured with row access by name.

        Returns:
            A SQLite connection instance.
        """

        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _score_name_match(
    *,
    query_key: str,
    query_tokens: list[str],
    candidate_name: str,
) -> int | None:
    """Return a ranking score for a candidate name or `None` when it does not match."""

    candidate_key = _normalize_name_for_matching(candidate_name)
    candidate_tokens = _name_match_tokens(candidate_name)
    if candidate_key == query_key:
        return 0
    if candidate_key.startswith(query_key):
        return 1
    if query_key in candidate_key:
        return 2
    if query_tokens and _all_tokens_match(query_tokens, candidate_tokens):
        return 3
    if len(query_tokens) > 1 and _all_tokens_fuzzy_match(query_tokens, candidate_tokens):
        return 4
    return None


def _all_tokens_match(query_tokens: list[str], candidate_tokens: list[str]) -> bool:
    """Return whether every query token has an exact or prefix match."""

    return all(
        any(
            candidate_token == query_token
            or candidate_token.startswith(query_token)
            or query_token.startswith(candidate_token)
            for candidate_token in candidate_tokens
        )
        for query_token in query_tokens
    )


def _all_tokens_fuzzy_match(query_tokens: list[str], candidate_tokens: list[str]) -> bool:
    """Return whether every query token has a close candidate-token match."""

    return all(
        any(_tokens_are_similar(query_token, candidate_token) for candidate_token in candidate_tokens)
        for query_token in query_tokens
    )


def _tokens_are_similar(query_token: str, candidate_token: str) -> bool:
    """Return whether two name tokens are close enough for typo-tolerant lookup."""

    if len(query_token) < 4 or len(candidate_token) < 4:
        return False
    return SequenceMatcher(None, query_token, candidate_token).ratio() >= 0.84


def _name_match_tokens(text: str) -> list[str]:
    """Normalize a name into significant tokens for matching."""

    tokens = re.findall(r"[a-z0-9]+", _normalize_name_for_matching(text))
    significant_tokens = [token for token in tokens if token not in _NAME_PARTICLES]
    return significant_tokens or tokens


def _normalize_name_for_matching(text: str) -> str:
    """Fold case, accents, punctuation, and spacing for name comparisons."""

    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_text.casefold())
    return re.sub(r"\s+", " ", normalized).strip()
