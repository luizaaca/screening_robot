"""Configuration models for the screening assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal

from dotenv import load_dotenv

ControlBackendKind = Literal["openai_compatible", "mock"]
ClinicalBackendKind = Literal["openai_compatible", "gguf", "mock"]


@dataclass(frozen=True)
class ControlModelSettings:
    """Configuration for the control model used by router and tools.

    Attributes:
        backend: Selected control-model runtime backend.
        model: Model identifier understood by the OpenAI-compatible endpoint.
        base_url: Base URL for the OpenAI-compatible API.
        api_key: API key used to authenticate with the endpoint.
        temperature: Sampling temperature for deterministic control flows.
    """

    backend: ControlBackendKind
    model: str
    base_url: str | None
    api_key: str | None
    temperature: float = 0.0


@dataclass(frozen=True)
class ClinicalBackendSettings:
    """Configuration for the clinical analysis backend.

    Attributes:
        backend: Selected runtime backend.
        model: Remote model identifier for OpenAI-compatible backends.
        base_url: Base URL for the remote clinical backend.
        api_key: API key for the remote clinical backend.
        gguf_model_path: Local file path for the GGUF artifact.
        temperature: Sampling temperature for the clinical analysis model.
    """

    backend: ClinicalBackendKind
    model: str | None
    base_url: str | None
    api_key: str | None
    gguf_model_path: Path | None
    temperature: float = 0.0


@dataclass(frozen=True)
class AppSettings:
    """Top-level configuration for the screening assistant.

    Attributes:
        patient_database_path: Location of the SQLite database with synthetic patient data.
        control_model: Configuration for the router and tool-calling model.
        clinical_backend: Configuration for the complaint analysis runtime.
        use_in_memory_checkpointer: Whether to default to an in-memory LangGraph checkpointer.
    """

    patient_database_path: Path
    control_model: ControlModelSettings
    clinical_backend: ClinicalBackendSettings
    use_in_memory_checkpointer: bool = True

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "AppSettings":
        """Build settings from environment variables.

        Args:
            root_dir: Optional repository root used to resolve relative paths.

        Returns:
            A validated application settings instance.
        """

        resolved_root_dir = root_dir or Path(__file__).resolve().parents[2]
        load_dotenv(resolved_root_dir / ".env", override=False)
        patient_database_path = _resolve_path(
            resolved_root_dir,
            os.getenv("SCREENING_AGENT_PATIENT_DB_PATH", "data/patients.sqlite3"),
        )
        gguf_model_path = os.getenv("SCREENING_AGENT_GGUF_MODEL_PATH")
        return cls(
            patient_database_path=patient_database_path,
            control_model=ControlModelSettings(
                backend=_read_control_backend_kind(
                    os.getenv("SCREENING_AGENT_CONTROL_BACKEND", "mock"),
                ),
                model=os.getenv("SCREENING_AGENT_CONTROL_MODEL", "gpt-4.1-mini"),
                base_url=os.getenv("SCREENING_AGENT_CONTROL_BASE_URL"),
                api_key=os.getenv("SCREENING_AGENT_CONTROL_API_KEY"),
                temperature=_read_float_env("SCREENING_AGENT_CONTROL_TEMPERATURE", 0.0),
            ),
            clinical_backend=ClinicalBackendSettings(
                backend=_read_backend_kind(
                    os.getenv("SCREENING_AGENT_CLINICAL_BACKEND", "mock"),
                ),
                model=os.getenv("SCREENING_AGENT_CLINICAL_MODEL"),
                base_url=os.getenv("SCREENING_AGENT_CLINICAL_BASE_URL"),
                api_key=os.getenv("SCREENING_AGENT_CLINICAL_API_KEY"),
                gguf_model_path=_resolve_path(resolved_root_dir, gguf_model_path)
                if gguf_model_path
                else None,
                temperature=_read_float_env("SCREENING_AGENT_CLINICAL_TEMPERATURE", 0.0),
            ),
            use_in_memory_checkpointer=_read_bool_env(
                "SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER",
                default=True,
            ),
        )


def _read_control_backend_kind(value: str) -> ControlBackendKind:
    """Validate the configured control-model backend kind.

    Args:
        value: Raw environment variable value.

    Returns:
        A valid control backend literal.

    Raises:
        ValueError: If the control backend is not supported.
    """

    normalized_value = value.strip().lower()
    if normalized_value not in {"openai_compatible", "mock"}:
        raise ValueError(
            "SCREENING_AGENT_CONTROL_BACKEND must be 'openai_compatible' or 'mock'.",
        )
    return normalized_value  # type: ignore[return-value]


def _read_backend_kind(value: str) -> ClinicalBackendKind:
    """Validate the configured clinical backend kind.

    Args:
        value: Raw environment variable value.

    Returns:
        A valid backend literal.

    Raises:
        ValueError: If the backend is not supported.
    """

    normalized_value = value.strip().lower()
    if normalized_value not in {"openai_compatible", "gguf", "mock"}:
        raise ValueError(
            "SCREENING_AGENT_CLINICAL_BACKEND must be 'openai_compatible', 'gguf', or 'mock'.",
        )
    return normalized_value  # type: ignore[return-value]



def _read_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is not set.

    Returns:
        The parsed boolean value.
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}



def _read_float_env(name: str, default: float) -> float:
    """Parse a floating-point environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is not set.

    Returns:
        The parsed floating-point value.

    Raises:
        ValueError: If the variable cannot be parsed as a float.
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)



def _resolve_path(root_dir: Path, raw_path: str | None) -> Path:
    """Resolve a path relative to the repository root when necessary.

    Args:
        root_dir: Repository root directory.
        raw_path: Raw path string.

    Returns:
        An absolute or root-relative path.

    Raises:
        ValueError: If the raw path is empty.
    """

    if raw_path is None or not raw_path.strip():
        raise ValueError("A non-empty path value is required.")
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else root_dir / candidate
