"""Regression tests for console debug streaming helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from screening_agent.audit import (
    emit_console_audit,
    emit_console_stream_part,
    emit_custom_debug_event,
    emit_verbose_console_stream_part,
)
from screening_agent.config import AppSettings
import app_chainlit


class _FakeChunk:
    """Minimal streamed chunk stub used by UI-streaming tests."""

    def __init__(self, content: object) -> None:
        """Store the chunk content.

        Args:
            content: Streamed content payload.
        """

        self.content = content


class _FakeStateSnapshot:
    """Minimal state snapshot stub returned by fake graphs."""

    def __init__(self, values: dict[str, object]) -> None:
        """Store the final graph values.

        Args:
            values: Root graph state.
        """

        self.values = values


class _FakeResponseMessage:
    """Chainlit-like response message used for streaming tests."""

    def __init__(self) -> None:
        """Initialize an empty streamed response."""

        self.content = ""
        self.tokens: list[str] = []

    async def stream_token(self, token: str) -> None:
        """Capture a streamed token.

        Args:
            token: Streamed token text.
        """

        self.tokens.append(token)


@pytest.mark.parametrize("mode", ["none", "info", "debug"])
def test_app_settings_reads_console_debug_mode(
    tmp_path: Path,
    monkeypatch: Any,
    mode: str,
) -> None:
    """Ensure the console debug mode is parsed from the environment."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG_MODE", mode)

    settings = AppSettings.from_env(root_dir=tmp_path)

    assert settings.console_debug_mode == mode


def test_app_settings_rejects_invalid_console_debug_mode(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ensure invalid console debug modes fail with a clear error."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG_MODE", "verbose")

    with pytest.raises(ValueError, match="SCREENING_AGENT_CONSOLE_DEBUG_MODE"):
        AppSettings.from_env(root_dir=tmp_path)


@pytest.mark.parametrize(
    ("mode", "should_emit_json"),
    [("none", False), ("info", False), ("debug", True)],
)
def test_emit_console_audit_obeys_console_debug_mode(
    monkeypatch: Any,
    capsys: Any,
    mode: str,
    should_emit_json: bool,
) -> None:
    """Ensure audit JSON is emitted only in debug mode."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG_MODE", mode)

    emit_console_audit(
        {
            "timestamp_utc": "2026-05-12T00:00:00+00:00",
            "event_type": "routing",
            "status": "success",
            "node_name": "router",
            "detail": "Processed patient 12345678 successfully.",
        },
    )

    console_output = capsys.readouterr().out.strip()
    if not should_emit_json:
        assert console_output == ""
        return

    payload = json.loads(console_output)
    assert payload["event_class"] == "audit"
    assert payload["thread_id"] == "n/a"
    assert payload["execution"]["event_type"] == "routing"
    assert payload["execution"]["detail"] == "Processed patient ****5678 successfully."


def test_emit_console_stream_part_masks_identifiers_and_truncates_text(capsys: Any) -> None:
    """Ensure debug task events stay compact and omit heavy payload fields."""

    emitted = emit_console_stream_part(
        {
            "type": "debug",
            "ns": ("patient_lookup",),
            "data": {
                "step": 2,
                "timestamp": "2026-05-12T00:00:01+00:00",
                "type": "task",
                "payload": {
                    "id": "task-123",
                    "name": "patient_lookup_agent",
                    "input": {
                        "message": "Patient 12345678 " + ("x" * 5_000),
                    },
                    "triggers": ["branch:to:patient_lookup_agent"],
                },
            },
        },
        thread_id="thread-123",
    )

    raw_output = capsys.readouterr().out.strip()
    payload = json.loads(raw_output)

    assert emitted is True
    assert payload["event_class"] == "langgraph_stream"
    assert payload["thread_id"] == "thread-123"
    assert payload["execution"]["stream_type"] == "debug"
    assert payload["execution"]["namespace"] == "patient_lookup"
    assert payload["execution"]["internal_type"] == "task"
    assert payload["execution"]["node_name"] == "patient_lookup_agent"
    assert payload["execution"]["id"] == "task-123"
    assert payload["execution"]["triggers"] == ["branch:to:patient_lookup_agent"]
    assert payload["execution"]["step"] == 2
    assert "input" not in payload["execution"]
    assert "message" not in payload["execution"]
    assert "12345678" not in raw_output
    assert "xxxxx" not in raw_output


def test_emit_console_stream_part_skips_non_flow_events_by_default(capsys: Any) -> None:
    """Ensure standard console mode suppresses non-flow stream events."""

    emitted = emit_console_stream_part(
        {
            "type": "debug",
            "ns": (),
            "data": {
                "step": 2,
                "timestamp": "2026-05-12T00:00:01+00:00",
                "type": "checkpoint",
                "payload": {
                    "next": ["final_answer"],
                    "values": {"last_response": "hello"},
                },
            },
        },
        thread_id="thread-123",
    )

    assert emitted is False
    assert capsys.readouterr().out == ""


def test_emit_verbose_console_stream_part_emits_message_parts(capsys: Any) -> None:
    """Ensure verbose console mode includes raw token-level stream events."""

    emitted = emit_verbose_console_stream_part(
        {
            "type": "messages",
            "ns": (),
            "data": [
                {"content": "token"},
                {"langgraph_node": "final_answer"},
            ],
        },
        thread_id="thread-123",
    )

    payload = json.loads(capsys.readouterr().out.strip())

    assert emitted is True
    assert payload["type"] == "messages"
    assert payload["thread_id"] == "thread-123"


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


@pytest.mark.parametrize(
    ("mode", "expected_stream_mode", "should_emit_json", "should_pretty_print"),
    [
        ("none", ["messages"], False, False),
        ("info", ["messages"], False, True),
        ("debug", ["messages", "debug"], True, True),
    ],
)
def test_stream_graph_turn_respects_console_debug_mode(
    monkeypatch: Any,
    capsys: Any,
    mode: str,
    expected_stream_mode: list[str],
    should_emit_json: bool,
    should_pretty_print: bool,
) -> None:
    """Ensure UI streaming honors the configured console debug mode."""

    class _FakeGraph:
        """Minimal graph stub implementing streaming and checkpoint retrieval."""

        def __init__(self) -> None:
            """Initialize the fake graph call trackers."""

            self.astream_calls: list[dict[str, object]] = []
            self.state_calls: list[dict[str, dict[str, str]]] = []

        async def astream(self, inputs: dict[str, object], **kwargs: object):
            """Yield a mix of debug and streamed-token events.

            Args:
                inputs: Graph input payload.
                **kwargs: LangGraph stream arguments.
            """

            self.astream_calls.append({"inputs": inputs, **kwargs})
            yield {
                "type": "debug",
                "ns": (),
                "data": {
                    "step": 1,
                    "timestamp": "2026-05-12T00:00:00+00:00",
                    "type": "task",
                    "payload": {
                        "id": "task-router-1",
                        "name": "router",
                        "input": {"messages": ["hello"]},
                        "triggers": ["start:router"],
                    },
                },
            }
            yield {
                "type": "custom",
                "ns": (),
                "data": {"event_name": "router_prompt"},
            }
            yield {
                "type": "messages",
                "ns": (),
                "data": [_FakeChunk("internal-token"), {"langgraph_node": "router"}],
            }
            yield {
                "type": "messages",
                "ns": (),
                "data": [_FakeChunk("Hello "), {"langgraph_node": "final_answer"}],
            }
            yield {
                "type": "messages",
                "ns": (),
                "data": [_FakeChunk("world"), {"langgraph_node": "final_answer"}],
            }

        async def aget_state(
            self,
            config: dict[str, dict[str, str]],
        ) -> _FakeStateSnapshot:
            """Return the authoritative final state for the current thread.

            Args:
                config: LangGraph thread config.

            Returns:
                Fake state snapshot.
            """

            self.state_calls.append(config)
            return _FakeStateSnapshot({"last_response": "Hello world"})

    fake_graph = _FakeGraph()
    fake_response_message = _FakeResponseMessage()
    pretty_print_calls: list[dict[str, object]] = []

    monkeypatch.setattr(app_chainlit, "_get_console_debug_mode", lambda: mode)
    monkeypatch.setattr(
        app_chainlit,
        "_pretty_print_history",
        lambda final_state: pretty_print_calls.append(dict(final_state)),
    )

    result = asyncio.run(
        app_chainlit._stream_graph_turn(
            fake_graph,
            user_message="hello",
            thread_id="thread-abc",
            response_message=fake_response_message,
        ),
    )

    output_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert result == {"last_response": "Hello world"}
    assert fake_response_message.tokens == ["Hello ", "world"]
    assert fake_response_message.content == "Hello world"
    assert fake_graph.astream_calls[0]["subgraphs"] is True
    assert fake_graph.astream_calls[0]["version"] == "v2"
    assert fake_graph.astream_calls[0]["stream_mode"] == expected_stream_mode
    assert fake_graph.state_calls == [{"configurable": {"thread_id": "thread-abc"}}]

    if should_emit_json:
        assert len(output_lines) == 1
        payload = json.loads(output_lines[0])
        assert payload["event_class"] == "langgraph_stream"
        assert payload["thread_id"] == "thread-abc"
        assert payload["execution"]["stream_type"] == "debug"
        assert payload["execution"]["internal_type"] == "task"
        assert payload["execution"]["node_name"] == "router"
        assert payload["execution"]["id"] == "task-router-1"
        assert "data" not in payload
    else:
        assert output_lines == []

    if should_pretty_print:
        assert pretty_print_calls == [{"last_response": "Hello world"}]
    else:
        assert pretty_print_calls == []
