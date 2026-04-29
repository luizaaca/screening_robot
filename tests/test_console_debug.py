"""Regression tests for console debug streaming helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from screening_agent.audit import (
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


def test_app_settings_reads_console_debug_flags(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ensure the console debug flags are parsed from the environment."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG", "true")
    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG_VERBOSE", "true")

    settings = AppSettings.from_env(root_dir=tmp_path)

    assert settings.console_debug is True
    assert settings.console_debug_verbose is True


def test_emit_console_stream_part_masks_identifiers_and_truncates_text(capsys: Any) -> None:
    """Ensure standard terminal stream output is JSON, masked, and truncated."""

    emitted = emit_console_stream_part(
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

    assert emitted is True
    assert payload["event_class"] == "langgraph_stream"
    assert payload["thread_id"] == "thread-123"
    assert payload["type"] == "custom"
    assert payload["ns"] == ["patient_lookup"]
    assert "****5678" in payload["data"]["message"]
    assert "[truncated" in payload["data"]["message"]


def test_emit_console_stream_part_skips_message_parts_by_default(capsys: Any) -> None:
    """Ensure standard console mode suppresses raw token-level stream events."""

    emitted = emit_console_stream_part(
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


def test_stream_graph_turn_uses_checkpoint_state_and_streams_only_final_answer(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Ensure UI streaming uses only final-answer tokens and final checkpoint state."""

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

    monkeypatch.setattr(app_chainlit, "_is_console_debug_enabled", lambda: True)
    monkeypatch.setattr(app_chainlit, "_is_console_debug_verbose_enabled", lambda: False)

    result = asyncio.run(
        app_chainlit._stream_graph_turn(
            fake_graph,
            user_message="hello",
            thread_id="thread-abc",
            response_message=fake_response_message,
        ),
    )

    output_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    payload = json.loads(output_lines[0])

    assert result == {"last_response": "Hello world"}
    assert fake_response_message.tokens == ["Hello ", "world"]
    assert fake_response_message.content == "Hello world"
    assert len(output_lines) == 1
    assert payload["type"] == "custom"
    assert payload["data"]["event_name"] == "router_prompt"
    assert fake_graph.astream_calls[0]["subgraphs"] is True
    assert fake_graph.astream_calls[0]["version"] == "v2"
    assert fake_graph.astream_calls[0]["stream_mode"] == ["messages", "debug", "custom"]
    assert fake_graph.state_calls == [{"configurable": {"thread_id": "thread-abc"}}]
