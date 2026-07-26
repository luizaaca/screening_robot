from typing import Optional

import yaml

from video_pipeline.contracts import PipelineConfig
from video_pipeline.paths import DEFAULT_CONFIG_PATH, resolve_project_path


def load_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load a YAML config file and return a validated PipelineConfig.

    When config_path is None, loads video_pipeline/configs/default.yaml.
    Relative config paths are resolved from the repository root.
    """
    path_to_load = (
        DEFAULT_CONFIG_PATH
        if config_path is None
        else resolve_project_path(config_path, field_name="config_path")
    )

    if not path_to_load.exists():
        if config_path:
            raise FileNotFoundError(f"Configuration file not found: {path_to_load}")

        return PipelineConfig()

    with open(path_to_load, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return PipelineConfig()

    return PipelineConfig(**data)
