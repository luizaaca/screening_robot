"""Chainlit user interface for the screening assistant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any
from uuid import uuid4

import chainlit as cl
from langchain.messages import HumanMessage

from screening_agent.audit import emit_console_stream_part, emit_verbose_console_stream_part
from screening_agent.config import AppSettings
from screening_agent.data import PatientRepository
from screening_agent.graph import build_default_graph
from screening_agent.graph.message_utils import coerce_message_text

_BASE_STREAM_MODES: tuple[str, ...] = ("messages",)
_CONSOLE_DEBUG_STREAM_MODES: tuple[str, ...] = ("debug", "custom")


@lru_cache(maxsize=1)
def _get_settings() -> AppSettings:
    """Load and cache application settings for the current process.

    Returns:
        Frozen application settings loaded from the environment.
    """

    return AppSettings.from_env()


@lru_cache(maxsize=1)
def _get_graph() -> Any:
    """Build and cache the compiled LangGraph application.

    Returns:
        A compiled graph instance ready for invocation.

    Raises:
        ValueError: If checkpoint-backed state is disabled for the Chainlit app.
    """

    settings = _get_settings()
    if not settings.use_in_memory_checkpointer:
        raise ValueError(
            "Chainlit streaming requires SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true.",
        )
    return build_default_graph(settings)


def _get_patient_count() -> int:
    """Read the number of patients currently available in the SQLite database.

    Returns:
        Number of patient records in the configured database.
    """

    settings = _get_settings()
    repository = PatientRepository(settings.patient_database_path)
    repository.initialize_database()
    return repository.count_patients()


def _build_welcome_message(patient_count: int) -> str:
    """Create the welcome message displayed when a chat session starts.

    Args:
        patient_count: Number of patient records currently available.

    Returns:
        User-facing welcome text.
    """

    lines = [
        "# Clinical Screening Assistant",
        "",
        "You can ask me to:",
        "- explain how to use the assistant;",
        "- look up a patient by fictional security number or by name;",
        "- clear the active patient context;",
        "- analyze symptoms and suggest likely conditions or relevant exams.",
        "",
        f"Current patient records available: {patient_count}.",
    ]
    if patient_count == 0:
        lines.extend(
            [
                "",
                "The database is currently empty.",
                "Run `python seed_demo_data.py` to load the demo patients before testing lookup flows.",
            ],
        )
    else:
        lines.extend(
            [
                "",
                "Example prompts:",
                "- `Find patient Maria Silva`",
                "- `Lookup patient 12003456`",
                "- `Patient 55667788 has fatigue and frequent urination`",
                "- `Clear active patient`",
            ],
        )
    return "\n".join(lines)


def _build_graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Create the LangGraph invocation configuration for a chat session.

    Args:
        thread_id: Stable thread identifier for the current Chainlit session.

    Returns:
        LangGraph config dictionary.
    """

    return {"configurable": {"thread_id": thread_id}}


def _extract_response_text(result: Mapping[str, object]) -> str:
    """Extract the final assistant response from a graph invocation result.

    Args:
        result: Graph output state.

    Returns:
        Final user-facing response.
    """

    last_response = result.get("last_response")
    if isinstance(last_response, str) and last_response.strip():
        return last_response
    return "I could not produce a response for this turn."


def _is_console_debug_enabled() -> bool:
    """Return whether console debug streaming is enabled.

    Returns:
        `True` when the terminal debug stream should be emitted.
    """

    return _get_settings().console_debug


def _is_console_debug_verbose_enabled() -> bool:
    """Return whether console debug should include raw token-level events.

    Returns:
        `True` when raw `messages` stream events should be printed to the terminal.
    """

    return _get_settings().console_debug_verbose


def _build_stream_modes() -> list[str]:
    """Build the LangGraph stream modes required by the Chainlit UI.

    Returns:
        Ordered, de-duplicated list of stream modes.
    """

    modes: list[str] = list(_BASE_STREAM_MODES)
    if _is_console_debug_enabled():
        modes.extend(_CONSOLE_DEBUG_STREAM_MODES)
    return _deduplicate_stream_modes(modes)


def _deduplicate_stream_modes(modes: Sequence[str]) -> list[str]:
    """Preserve stream-mode order while removing duplicates.

    Args:
        modes: Candidate stream modes.

    Returns:
        De-duplicated stream-mode list.
    """

    unique_modes: list[str] = []
    for mode in modes:
        if mode not in unique_modes:
            unique_modes.append(mode)
    return unique_modes


async def _stream_graph_turn(
    graph: Any,
    *,
    user_message: str,
    thread_id: str,
    response_message: Any | None = None,
) -> dict[str, object]:
    """Invoke the graph through LangGraph streaming and retain the final state.

    Args:
        graph: Compiled LangGraph application.
        user_message: Latest user message text.
        thread_id: Stable chat thread identifier.
        response_message: Optional Chainlit-like message used for UI token streaming.

    Returns:
        Final graph state extracted from the authoritative checkpoint snapshot.
    """

    async for part in graph.astream(
        {"messages": [HumanMessage(content=user_message)]},
        config=_build_graph_config(thread_id),
        stream_mode=_build_stream_modes(),
        subgraphs=True,
        version="v2",
    ):
        if _is_console_debug_enabled():
            _emit_console_stream_part(part, thread_id=thread_id)
        token_text = _extract_final_answer_token(part)
        if response_message is not None and token_text:
            await response_message.stream_token(token_text)

    final_state = await _get_authoritative_graph_state(graph, thread_id=thread_id)
    if response_message is not None:
        response_message.content = _extract_response_text(final_state)
    return final_state


async def _invoke_graph_with_console_debug(
    graph: Any,
    *,
    user_message: str,
    thread_id: str,
) -> dict[str, object]:
    """Backward-compatible wrapper for tests exercising the streaming path.

    Args:
        graph: Compiled LangGraph application.
        user_message: Latest user message text.
        thread_id: Stable chat thread identifier.

    Returns:
        Final graph state extracted from the authoritative checkpoint snapshot.
    """

    return await _stream_graph_turn(
        graph,
        user_message=user_message,
        thread_id=thread_id,
    )


def _emit_console_stream_part(part: Mapping[str, object], *, thread_id: str) -> None:
    """Emit a console stream part using the configured verbosity policy.

    Args:
        part: LangGraph stream part in `version="v2"` format.
        thread_id: Stable chat thread identifier.
    """

    if _is_console_debug_verbose_enabled():
        emit_verbose_console_stream_part(part, thread_id=thread_id)
        return
    emit_console_stream_part(part, thread_id=thread_id)


def _coerce_graph_state(value: object) -> dict[str, object]:
    """Coerce a graph state snapshot into a dictionary.

    Args:
        value: Graph state payload.

    Returns:
        Dictionary-like graph state.

    Raises:
        TypeError: If the state cannot be represented as a dictionary.
    """

    if isinstance(value, Mapping):
        return {str(key): nested_value for key, nested_value in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped_value = model_dump()
        if isinstance(dumped_value, Mapping):
            return {str(key): nested_value for key, nested_value in dumped_value.items()}
    raise TypeError(f"Unsupported streamed state type: {type(value).__name__}")


async def _get_authoritative_graph_state(
    graph: Any,
    *,
    thread_id: str,
) -> dict[str, object]:
    """Read the latest root checkpoint state for the current thread.

    Args:
        graph: Compiled LangGraph application.
        thread_id: Stable chat thread identifier.

    Returns:
        Final root graph state for the thread.

    Raises:
        RuntimeError: If the graph does not expose a readable state snapshot.
    """

    config = _build_graph_config(thread_id)
    aget_state = getattr(graph, "aget_state", None)
    if callable(aget_state):
        snapshot = await aget_state(config)
    else:
        get_state = getattr(graph, "get_state", None)
        if not callable(get_state):
            raise RuntimeError("The compiled graph does not expose get_state/aget_state.")
        snapshot = await cl.make_async(get_state)(config)

    values = getattr(snapshot, "values", None)
    return _coerce_graph_state(values)


def _extract_final_answer_token(part: Mapping[str, object]) -> str | None:
    """Extract token text for the `final_answer` node from a stream part.

    Args:
        part: LangGraph stream part in `version="v2"` format.

    Returns:
        Token text for the final-answer node, or `None` when the part should not
        be shown in the user-facing Chainlit stream.
    """

    if part.get("type") != "messages":
        return None

    chunk, metadata = _unpack_message_stream_data(part.get("data"))
    if metadata.get("langgraph_node") != "final_answer":
        return None

    token_text = _coerce_stream_chunk_text(chunk)
    return token_text or None


def _unpack_message_stream_data(data: object) -> tuple[object, dict[str, object]]:
    """Normalize a streamed `messages` payload into chunk and metadata objects.

    Args:
        data: Raw `messages` payload emitted by LangGraph.

    Returns:
        Tuple containing the message chunk object and its metadata dictionary.
    """

    if isinstance(data, tuple) and len(data) == 2:
        return data[0], _coerce_metadata_dict(data[1])
    if isinstance(data, list) and len(data) == 2:
        return data[0], _coerce_metadata_dict(data[1])
    if isinstance(data, Mapping):
        chunk = data.get("chunk")
        if chunk is None:
            chunk = data.get("message")
        if chunk is None:
            chunk = data.get("data")
        return chunk, _coerce_metadata_dict(data.get("metadata"))
    return data, {}


def _coerce_metadata_dict(value: object) -> dict[str, object]:
    """Coerce streamed metadata into a plain dictionary.

    Args:
        value: Raw metadata object.

    Returns:
        String-keyed metadata dictionary.
    """

    if isinstance(value, Mapping):
        return {str(key): nested_value for key, nested_value in value.items()}
    return {}


def _coerce_stream_chunk_text(chunk: object) -> str:
    """Normalize a streamed message chunk into token text.

    Args:
        chunk: Raw streamed message chunk.

    Returns:
        Extracted token text.
    """

    if hasattr(chunk, "content"):
        return coerce_message_text(getattr(chunk, "content"))
    if isinstance(chunk, Mapping) and "content" in chunk:
        return coerce_message_text(chunk["content"])
    return coerce_message_text(chunk)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize a new Chainlit chat session."""

    thread_id = str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    patient_count = _get_patient_count()
    await cl.Message(content=_build_welcome_message(patient_count)).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle a user message by invoking the LangGraph workflow.

    Args:
        message: Incoming Chainlit user message.
    """

    thread_id = cl.user_session.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        thread_id = str(uuid4())
        cl.user_session.set("thread_id", thread_id)

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        graph = _get_graph()
        await _stream_graph_turn(
            graph,
            user_message=message.content,
            thread_id=thread_id,
            response_message=response_message,
        )
    except Exception as exc:  # pragma: no cover - UI safety fallback
        response_message.content = (
            "I could not process the request with the current configuration. "
            f"Details: {type(exc).__name__}: {exc}"
        )

    await response_message.update()
