from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from video_pipeline.contracts import PipelineConfig


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"
DEFAULT_HOLISTIC_LANDMARKER_PATH = (
    PACKAGE_ROOT / "processors" / "holistic_landmarker.task"
)


def resolve_project_path(
    path: str | PathLike[str],
    *,
    field_name: str = "path",
) -> Path:
    """Resolve relative paths against the repository root."""
    raw_value = str(path).strip()
    if not raw_value:
        raise ValueError(f"{field_name} must not be empty")

    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate

    return (PROJECT_ROOT / candidate).resolve()


def normalize_pipeline_config_paths(config: PipelineConfig) -> PipelineConfig:
    """Return a copy with project-relative filesystem fields normalized."""
    normalized = config.model_copy(deep=True)

    if normalized.output_dir is not None:
        normalized.output_dir = str(
            resolve_project_path(normalized.output_dir, field_name="output_dir")
        )

    if normalized.pose.model_asset_path is not None:
        normalized.pose.model_asset_path = str(
            resolve_project_path(
                normalized.pose.model_asset_path,
                field_name="pose.model_asset_path",
            )
        )

    return normalized
