"""Chainlit user interface for the screening assistant."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from uuid import uuid4

import chainlit as cl
from langchain.messages import HumanMessage

from screening_agent.audit import emit_console_stream_part
from screening_agent.config import AppSettings
from screening_agent.data import PatientRepository
from screening_agent.graph import build_default_graph

_DEBUG_STREAM_MODES: tuple[str, ...] = ("debug", "messages", "custom", "values")


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
    """

    return build_default_graph(_get_settings())



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


async def _invoke_graph_with_console_debug(
    graph: Any,
    *,
    user_message: str,
    thread_id: str,
) -> dict[str, object]:
    """Invoke the graph through LangGraph streaming and retain the final state.

    Args:
        graph: Compiled LangGraph application.
        user_message: Latest user message text.
        thread_id: Stable chat thread identifier.

    Returns:
        Final graph state extracted from the latest `values` stream part.

    Raises:
        RuntimeError: If the debug stream completes without a `values` event.
        TypeError: If a `values` event cannot be coerced into a state dictionary.
    """

    latest_values: dict[str, object] | None = None
    async for part in graph.astream(
        {"messages": [HumanMessage(content=user_message)]},
        config=_build_graph_config(thread_id),
        stream_mode=list(_DEBUG_STREAM_MODES),
        subgraphs=True,
        version="v2",
    ):
        if part.get("type") == "values":
            latest_values = _coerce_graph_state(part.get("data"))
            continue
        emit_console_stream_part(part, thread_id=thread_id)

    if latest_values is None:
        raise RuntimeError("LangGraph debug streaming completed without a final values event.")
    return latest_values


def _coerce_graph_state(value: object) -> dict[str, object]:
    """Coerce a streamed state snapshot into a dictionary.

    Args:
        value: Streamed state payload from a `values` stream part.

    Returns:
        Dictionary-like graph state.

    Raises:
        TypeError: If the streamed value cannot be represented as a dictionary.
    """

    if isinstance(value, dict):
        return {str(key): nested_value for key, nested_value in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped_value = model_dump()
        if isinstance(dumped_value, dict):
            return {str(key): nested_value for key, nested_value in dumped_value.items()}
    raise TypeError(f"Unsupported streamed state type: {type(value).__name__}")


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

    response_message = cl.Message(content="Processing your request...")
    await response_message.send()

    try:
        graph = _get_graph()
        if _is_console_debug_enabled():
            result = await _invoke_graph_with_console_debug(
                graph,
                user_message=message.content,
                thread_id=thread_id,
            )
        else:
            result = await cl.make_async(graph.invoke)(
                {"messages": [HumanMessage(content=message.content)]},
                config=_build_graph_config(thread_id),
            )
        response_message.content = _extract_response_text(result)
    except Exception as exc:  # pragma: no cover - UI safety fallback
        response_message.content = (
            "I could not process the request with the current configuration. "
            f"Details: {type(exc).__name__}: {exc}"
        )

    await response_message.update()
