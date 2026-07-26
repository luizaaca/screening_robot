"""Tests for video-related application settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from screening_agent.config import AppSettings


_VIDEO_ENV_NAMES = [
    "SCREENING_AGENT_VIDEO_ANALYST_BACKEND",
    "SCREENING_AGENT_VIDEO_ANALYST_MODEL",
    "SCREENING_AGENT_VIDEO_ANALYST_BASE_URL",
    "SCREENING_AGENT_VIDEO_ANALYST_API_KEY",
    "SCREENING_AGENT_VIDEO_ANALYST_TEMPERATURE",
    "SCREENING_AGENT_VIDEO_PIPELINE_OUTPUT_DIR",
    "SCREENING_AGENT_VIDEO_PIPELINE_DEBUG",
    "SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB",
    "SCREENING_AGENT_VIDEO_PIPELINE_WINDOW_S",
    "SCREENING_AGENT_VIDEO_PIPELINE_STRIDE_S",
]


def test_app_settings_reads_video_defaults(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure video settings have safe defaults."""

    for name in _VIDEO_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings.from_env(root_dir=tmp_path)

    assert settings.video_analyst.backend == "mock"
    assert settings.video_analyst.model == "gpt-4.1-mini"
    assert settings.video_pipeline.output_dir == tmp_path / "outputs" / "videos"
    assert settings.video_pipeline.debug is False
    assert settings.video_pipeline.upload_max_mb == 100
    assert settings.video_pipeline.window_s == 8.0
    assert settings.video_pipeline.stride_s == 5.0


def test_app_settings_reads_video_env_overrides(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure video settings are read from SCREENING_AGENT_VIDEO_* variables."""

    monkeypatch.setenv("SCREENING_AGENT_VIDEO_ANALYST_BACKEND", "openai_compatible")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_ANALYST_MODEL", "video-model")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_ANALYST_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_ANALYST_API_KEY", "video-key")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_ANALYST_TEMPERATURE", "0.2")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_PIPELINE_OUTPUT_DIR", "custom/video-output")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_PIPELINE_DEBUG", "true")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB", "42")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_PIPELINE_WINDOW_S", "9.5")
    monkeypatch.setenv("SCREENING_AGENT_VIDEO_PIPELINE_STRIDE_S", "4.5")

    settings = AppSettings.from_env(root_dir=tmp_path)

    assert settings.video_analyst.backend == "openai_compatible"
    assert settings.video_analyst.model == "video-model"
    assert settings.video_analyst.base_url == "http://localhost:9999/v1"
    assert settings.video_analyst.api_key == "video-key"
    assert settings.video_analyst.temperature == 0.2
    assert settings.video_pipeline.output_dir == tmp_path / "custom" / "video-output"
    assert settings.video_pipeline.debug is True
    assert settings.video_pipeline.upload_max_mb == 42
    assert settings.video_pipeline.window_s == 9.5
    assert settings.video_pipeline.stride_s == 4.5

