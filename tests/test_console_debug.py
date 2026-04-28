"""Regression tests for console debug streaming helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from screening_agent.audit import emit_console_stream_part, emit_custom_debug_event
from screening_agent.config import AppSettings
import app_chainlit


def test_app_settings_reads_console_debug_flag(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ensure the console debug flag is parsed from the environment."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG", "true")

    settings = AppSettings.from_env(root_dir=tmp_path)

    assert settings.console_debug is True


def test_emit_console_stream_part_masks_identifiers_and_truncates_text(capsys: Any) -> None:
    """Ensure terminal stream output is JSON, masked, and truncated when needed."""

    emit_console_stream_part(
        {
            "type": "custom",
            "ns": ("patient_lookup",),
            "data": {
                "message": "Patient 12345678 " + ("x" * 5_000),
            },
        },
        thread_id="thread-123",
    )

    payload = json.loads(capsys.readouterr().out.strip())

    assert payload["event_class"] == "langgraph_stream"
    assert payload["thread_id"] == "thread-123"
    assert payload["type"] == "custom"
    assert payload["ns"] == ["patient_lookup"]
    assert "****5678" in payload["data"]["message"]
    assert "[truncated" in payload["data"]["message"]


def test_emit_custom_debug_event_uses_stream_writer(monkeypatch: Any) -> None:
    """Ensure custom debug events are emitted through the LangGraph stream writer."""

    emitted_events: list[dict[str, object]] = []

    monkeypatch.setattr("screening_agent.audit.get_stream_writer", lambda: emitted_events.append)

    emit_custom_debug_event(
        "router_prompt",
        node_name="router",
        payload={"message": "Lookup patient 12345678"},
    )

    assert len(emitted_events) == 1
    assert emitted_events[0]["event_name"] == "router_prompt"
    assert emitted_events[0]["node_name"] == "router"
    assert emitted_events[0]["payload"] == {"message": "Lookup patient ****5678"}


def test_invoke_graph_with_console_debug_returns_last_values_and_emits_non_values_parts(
    monkeypatch: Any,
) -> None:
    """Ensure console debug streaming preserves the final state and prints other parts."""

    emitted_parts: list[tuple[dict[str, object], str | None]] = []

    class _FakeGraph:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def astream(self, inputs: dict[str, object], **kwargs: object):
            self.calls.append({"inputs": inputs, **kwargs})
            yield {
                "type": "custom",
                "ns": (),
                "data": {"event_name": "router_prompt"},
            }
            yield {
                "type": "values",
                "ns": (),
                "data": {"last_response": "Done."},
            }

    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        app_chainlit,
        "emit_console_stream_part",
        lambda part, *, thread_id=None: emitted_parts.append((part, thread_id)),
    )

    result = asyncio.run(
        app_chainlit._invoke_graph_with_console_debug(
            fake_graph,
            user_message="hello",
            thread_id="thread-abc",
        ),
    )

    assert result == {"last_response": "Done."}
    assert emitted_parts == [
        ({"type": "custom", "ns": (), "data": {"event_name": "router_prompt"}}, "thread-abc"),
    ]
    assert fake_graph.calls[0]["subgraphs"] is True
    assert fake_graph.calls[0]["version"] == "v2"
    assert fake_graph.calls[0]["stream_mode"] == list(app_chainlit._DEBUG_STREAM_MODES)