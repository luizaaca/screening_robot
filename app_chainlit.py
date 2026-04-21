"""Chainlit user interface for the screening assistant."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import uuid4

import chainlit as cl
from langchain.messages import HumanMessage

from screening_agent.config import AppSettings
from screening_agent.data import PatientRepository
from screening_agent.graph import build_default_graph


@lru_cache(maxsize=1)
def _get_graph() -> Any:
    """Build and cache the compiled LangGraph application.

    Returns:
        A compiled graph instance ready for invocation.
    """

    return build_default_graph()



def _get_patient_count() -> int:
    """Read the number of patients currently available in the SQLite database.

    Returns:
        Number of patient records in the configured database.
    """

    settings = AppSettings.from_env()
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



def _extract_response_text(result: dict[str, object]) -> str:
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
