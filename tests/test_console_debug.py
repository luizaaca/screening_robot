"""Regression tests for console debug streaming helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from screening_agent.audit import (
    emit_console_audit,
    emit_console_stream_part,
    emit_custom_debug_event,
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


class _FakeMessage:
    """Chainlit-like message used by final-response ordering tests."""

    event_log: list[str] = []
    created_messages: list["_FakeMessage"] = []

    def __init__(self, content: str = "") -> None:
        """Initialize a message with Chainlit-compatible content."""

        self.content = content
        self.tokens: list[str] = []
        self.sent = False
        self.updated = False
        self.parent_id: str | None = "active-step-parent"
        _FakeMessage.created_messages.append(self)

    async def send(self) -> "_FakeMessage":
        """Capture message creation."""

        self.sent = True
        _FakeMessage.event_log.append(f"message.send:{self.content}")
        return self

    async def stream_token(self, token: str) -> None:
        """Fail if the final response path streams tokens again.

        Args:
            token: Streamed token text.
        """

        raise AssertionError(f"Unexpected streamed final-response token: {token}")

    async def update(self) -> bool:
        """Capture message update."""

        self.updated = True
        return True


class _FakeStep:
    """Chainlit Step stub used outside a Chainlit runtime context."""

    created_steps: list["_FakeStep"] = []
    event_log: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.name = str(kwargs.get("name") or "")
        self.output = ""
        self.input = ""
        self.is_error = False
        self.start = None
        self.end = None
        self.sent = False
        self.updated = False
        _FakeStep.created_steps.append(self)

    async def send(self) -> "_FakeStep":
        self.sent = True
        _FakeStep.event_log.append(f"step.send:{self.name}")
        _FakeMessage.event_log.append(f"step.send:{self.name}")
        return self

    async def update(self) -> bool:
        self.updated = True
        _FakeStep.event_log.append(f"step.update:{self.name}")
        _FakeMessage.event_log.append(f"step.update:{self.name}")
        return True

    async def __aenter__(self) -> "_FakeStep":
        self.start = "started"
        await self.send()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        self.end = "ended"
        if exc_type:
            self.is_error = True
            self.output = str(exc_val)
        await self.update()


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


def test_emit_console_audit_escapes_unicode_for_windows_stdout(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """JSON terminal logs should remain ASCII-safe and parseable."""

    monkeypatch.setenv("SCREENING_AGENT_CONSOLE_DEBUG_MODE", "debug")

    emit_console_audit(
        {
            "timestamp_utc": "2026-05-12T00:00:00+00:00",
            "event_type": "routing",
            "status": "success",
            "node_name": "router",
            "detail": "Resposta gerada ✅ para paciente 12345678.",
        },
    )

    console_output = capsys.readouterr().out.strip()
    payload = json.loads(console_output)

    assert "\\u2705" in console_output
    assert payload["execution"]["detail"] == "Resposta gerada ✅ para paciente ****5678."


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


def test_safe_console_print_escapes_unencodable_characters(monkeypatch: Any) -> None:
    """Console diagnostics should not fail on Windows code-page limitations."""

    class _AsciiStdout:
        encoding = "ascii"

        def __init__(self) -> None:
            self.chunks: list[str] = []

        def write(self, text: str) -> int:
            text.encode(self.encoding)
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    fake_stdout = _AsciiStdout()
    monkeypatch.setattr(app_chainlit.sys, "stdout", fake_stdout)

    app_chainlit._safe_console_print("Resposta gerada ✅")

    assert "Resposta gerada \\u2705" in "".join(fake_stdout.chunks)


def test_console_history_debug_errors_do_not_escape(monkeypatch: Any, capsys: Any) -> None:
    """A terminal debug failure should not break the Chainlit user response."""

    def fail_history(final_state: dict[str, object]) -> None:
        raise UnicodeEncodeError("charmap", "✅", 0, 1, "character maps to <undefined>")

    monkeypatch.setattr(app_chainlit, "_pretty_print_history", fail_history)

    app_chainlit._try_pretty_print_history({"messages": []})

    output = capsys.readouterr().out
    assert "Console history debug output failed: UnicodeEncodeError" in output


@pytest.mark.parametrize(
    ("mode", "expected_stream_mode", "should_emit_json", "should_pretty_print"),
    [
        ("none", ["messages", "debug"], False, False),
        ("info", ["messages", "debug"], False, True),
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
            self.stream_completed = False

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
            self.stream_completed = True
            _FakeMessage.event_log.append("graph.stream_done")

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
            _FakeMessage.event_log.append("graph.state_read")
            return _FakeStateSnapshot({"last_response": "Hello world"})

    fake_graph = _FakeGraph()
    pretty_print_calls: list[dict[str, object]] = []
    _FakeStep.created_steps = []
    _FakeStep.event_log = []
    _FakeMessage.event_log = []
    _FakeMessage.created_messages = []

    monkeypatch.setattr(app_chainlit, "_get_console_debug_mode", lambda: mode)
    monkeypatch.setattr(app_chainlit.cl, "Step", _FakeStep)
    monkeypatch.setattr(app_chainlit.cl, "Message", _FakeMessage)
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
        ),
    )
    final_message = asyncio.run(app_chainlit._send_final_response_message(result))

    output_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert result == {"last_response": "Hello world"}
    assert final_message.content == "Hello world"
    assert final_message.parent_id is None
    assert final_message.tokens == []
    assert final_message.sent is True
    assert fake_graph.stream_completed is True
    assert [step.name for step in _FakeStep.created_steps] == ["Classificando solicitação"]
    assert _FakeStep.created_steps[0].updated is True
    assert _FakeStep.created_steps[0].end == "ended"
    assert _FakeStep.created_steps[0].output == "Solicitação classificada."
    assert _FakeMessage.event_log == [
        "step.send:Classificando solicitação",
        "graph.stream_done",
        "step.update:Classificando solicitação",
        "graph.state_read",
        "message.send:Hello world",
    ]
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


def test_on_message_sends_upload_followup_response_after_second_turn(
    monkeypatch: Any,
) -> None:
    """Ensure upload follow-up also sends the final message after graph streaming."""

    class _FakeUserSession:
        def __init__(self) -> None:
            self.values: dict[str, str] = {"thread_id": "thread-upload"}

        def get(self, key: str) -> str | None:
            return self.values.get(key)

        def set(self, key: str, value: str) -> None:
            self.values[key] = value

    graph = object()
    events: list[str] = []
    stream_calls: list[dict[str, object]] = []
    states = [
        {
            "last_response": "Please upload the video.",
            "video_input_status": "awaiting_upload",
            "pending_video_request": {"request_text": "Analyze the gait video"},
        },
        {"last_response": "Video interpretation complete."},
    ]
    _FakeMessage.created_messages = []
    _FakeMessage.event_log = []
    original_send_final_response_message = app_chainlit._send_final_response_message

    async def fake_resolve_message_video_path(
        message: object,
        settings: object,
    ) -> None:
        return None

    async def fake_stream_graph_turn(
        received_graph: object,
        **kwargs: object,
    ) -> dict[str, object]:
        assert received_graph is graph
        stream_calls.append(kwargs)
        events.append(f"graph.turn:{len(stream_calls)}")
        return states[len(stream_calls) - 1]

    async def fake_request_video_upload(settings: object, **kwargs: object) -> str:
        events.append("upload.request")
        return "uploaded-session.mp4"

    async def fake_send_final_response_message(
        final_state: dict[str, object],
        **kwargs: object,
    ) -> object:
        events.append(f"message.send:{final_state['last_response']}")
        return await original_send_final_response_message(final_state, **kwargs)

    monkeypatch.setattr(app_chainlit.cl, "Message", _FakeMessage)
    monkeypatch.setattr(app_chainlit.cl, "user_session", _FakeUserSession())
    monkeypatch.setattr(app_chainlit, "_get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_chainlit, "_get_graph", lambda: graph)
    monkeypatch.setattr(app_chainlit, "_resolve_message_video_path", fake_resolve_message_video_path)
    monkeypatch.setattr(app_chainlit, "_request_video_upload", fake_request_video_upload)
    monkeypatch.setattr(app_chainlit, "_stream_graph_turn", fake_stream_graph_turn)
    monkeypatch.setattr(
        app_chainlit,
        "_send_final_response_message",
        fake_send_final_response_message,
    )

    asyncio.run(app_chainlit.on_message(SimpleNamespace(content="Analyze video")))

    assert events == [
        "graph.turn:1",
        "message.send:Please upload the video.",
        "upload.request",
        "graph.turn:2",
        "message.send:Video interpretation complete.",
    ]
    assert stream_calls == [
        {
            "user_message": "Analyze video",
            "thread_id": "thread-upload",
            "locale": "pt-BR",
            "video_path": None,
        },
        {
            "user_message": "Analyze the gait video",
            "thread_id": "thread-upload",
            "locale": "pt-BR",
            "video_path": "uploaded-session.mp4",
        },
    ]
    assert [message.content for message in _FakeMessage.created_messages] == [
        "Please upload the video.",
        "Video interpretation complete.",
    ]
    assert all(message.sent for message in _FakeMessage.created_messages)
    assert all(message.parent_id is None for message in _FakeMessage.created_messages)
    assert not any(message.updated for message in _FakeMessage.created_messages)


def test_progress_steps_filter_internal_nodes_and_mark_errors(monkeypatch: Any) -> None:
    """Ensure progress steps stay limited to main nodes and surface failures safely."""

    _FakeStep.created_steps = []
    _FakeStep.event_log = []
    _FakeMessage.event_log = []
    monkeypatch.setattr(app_chainlit.cl, "Step", _FakeStep)
    active_steps: dict[str, tuple[str, Any]] = {}

    async def run_events() -> None:
        await app_chainlit._handle_progress_step_event(
            {
                "type": "debug",
                "data": {
                    "type": "task",
                    "payload": {"id": "internal-1", "name": "symptom_analysis_agent"},
                },
            },
            active_steps,
        )
        await app_chainlit._handle_progress_step_event(
            {
                "type": "debug",
                "data": {
                    "type": "task",
                    "payload": {"id": "video-1", "name": "video_analysis"},
                },
            },
            active_steps,
        )
        await app_chainlit._handle_progress_step_event(
            {
                "type": "debug",
                "data": {
                    "type": "task_result",
                    "payload": {
                        "id": "video-1",
                        "name": "video_analysis",
                        "error": "RuntimeError: patient 12345678 details " + ("x" * 400),
                    },
                },
            },
            active_steps,
        )

    asyncio.run(run_events())

    assert [step.name for step in _FakeStep.created_steps] == ["Processando vídeo"]
    assert _FakeStep.created_steps[0].sent is True
    assert _FakeStep.created_steps[0].updated is True
    assert _FakeStep.created_steps[0].is_error is True
    assert "****5678" in _FakeStep.created_steps[0].output
    assert len(_FakeStep.created_steps[0].output) < 280
