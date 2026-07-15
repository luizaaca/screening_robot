"""Tests for Chainlit video input helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import app_chainlit


@dataclass
class _FakeUpload:
    path: str
    name: str
    mime: str
    size: int


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Analyze video_path=clips/session01.mp4", "clips/session01.mp4"),
        ("Analyze video_path='clips/session 01.mp4'", "clips/session 01.mp4"),
        ("path: C:/videos/session01.webm", "C:/videos/session01.webm"),
        ("path: C:/videos/session01.mp4 please analyze posture", "C:/videos/session01.mp4"),
    ],
)
def test_extract_explicit_video_path(text: str, expected: str) -> None:
    """Ensure explicit video paths can be parsed from user text."""

    assert app_chainlit._extract_explicit_video_path(text) == expected


def test_extract_uploaded_video_path_accepts_chainlit_like_file() -> None:
    """Ensure uploaded video elements are converted to local paths."""

    upload = _FakeUpload(
        path="C:/tmp/uploaded.mp4",
        name="uploaded.mp4",
        mime="video/mp4",
        size=5 * 1024 * 1024,
    )

    assert app_chainlit._extract_uploaded_video_path([upload], max_mb=100) == "C:/tmp/uploaded.mp4"


def test_extract_uploaded_video_path_rejects_large_video() -> None:
    """Ensure configured upload limits are enforced before graph execution."""

    upload = _FakeUpload(
        path="C:/tmp/large.mp4",
        name="large.mp4",
        mime="video/mp4",
        size=101 * 1024 * 1024,
    )

    with pytest.raises(ValueError, match="SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB=100"):
        app_chainlit._extract_uploaded_video_path([upload], max_mb=100)


def test_resolve_message_video_path_does_not_prompt_on_video_mention(monkeypatch) -> None:
    """Ensure Chainlit does not open AskFileMessage from keyword heuristics."""

    class _FakeMessage:
        content = "What does the video show?"
        elements: list[object] = []

    def fail_ask_file(*args, **kwargs):
        raise AssertionError("AskFileMessage should not be opened by keyword mention.")

    monkeypatch.setattr(app_chainlit.cl, "AskFileMessage", fail_ask_file)
    settings = SimpleNamespace(video_pipeline=SimpleNamespace(upload_max_mb=100))

    resolved_path = asyncio.run(
        app_chainlit._resolve_message_video_path(_FakeMessage(), settings),
    )

    assert resolved_path is None
