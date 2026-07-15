"""Chainlit user interface for the screening assistant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import chainlit as cl
from langchain.messages import AIMessage, HumanMessage

from screening_agent.audit import emit_console_stream_part
from screening_agent.config import AppSettings, ConsoleDebugMode
from screening_agent.data import PatientRepository
from screening_agent.graph import build_default_graph
from screening_agent.graph.message_utils import get_message_text

_BASE_STREAM_MODES: tuple[str, ...] = ("messages",)
_PROGRESS_STREAM_MODES: tuple[str, ...] = ("debug",)
_CONSOLE_DEBUG_STREAM_MODES: tuple[str, ...] = ("debug",)
_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
)
_VIDEO_ACCEPT_TYPES: list[str] = [
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
]
_EXPLICIT_VIDEO_PATH_PATTERN = re.compile(
    r"(?:video_path\s*=\s*|path\s*:\s*)(?P<path>.+)",
    re.IGNORECASE,
)
_VIDEO_FILE_PATTERN = re.compile(
    r"(?P<path>.+?\.(?:mp4|mov|avi|mkv|webm|m4v))(?:\s|$)",
    re.IGNORECASE,
)
_PROGRESS_NODE_LABELS: dict[str, str] = {
    "router": "Classificando solicitacao",
    "patient_lookup": "Buscando paciente",
    "video_analysis": "Processando video",
    "video_interpretation": "Interpretando video",
    "video_qa": "Interpretando video",
    "video_clinical_extraction": "Extraindo contexto clinico do video",
    "symptom_analysis": "Analisando sintomas",
    "final_answer": "Gerando resposta",
    "processing_error": "Tratando erro",
}
_PROGRESS_RUNNING_OUTPUTS: dict[str, str] = {
    "router": "Classificando a solicitacao.",
    "patient_lookup": "Buscando o paciente solicitado.",
    "video_analysis": "Executando o pipeline de video.",
    "video_interpretation": "Gerando interpretacao narrativa do video.",
    "video_qa": "Gerando interpretacao narrativa do video.",
    "video_clinical_extraction": "Extraindo contexto clinico estruturado do video.",
    "symptom_analysis": "Executando analise de sintomas.",
    "final_answer": "Compondo a resposta final.",
    "processing_error": "Tratando uma falha de processamento.",
}
_PROGRESS_COMPLETED_OUTPUTS: dict[str, str] = {
    "router": "Solicitacao classificada.",
    "patient_lookup": "Busca de paciente concluida.",
    "video_analysis": "Pipeline de video concluido.",
    "video_interpretation": "Interpretacao de video concluida.",
    "video_qa": "Interpretacao de video concluida.",
    "video_clinical_extraction": "Contexto clinico de video extraido.",
    "symptom_analysis": "Analise de sintomas concluida.",
    "final_answer": "Resposta final gerada.",
    "processing_error": "Erro tratado pelo fluxo seguro.",
}
_MAX_STEP_OUTPUT_CHARS = 240


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
        "- upload or reference a video for expression, posture, and transcription analysis.",
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
                "- `Analyze this video with video_path=concepts_video/sample.mp4`",
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
    messages = result.get("messages", [])
    if isinstance(messages, Sequence):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                response_text = get_message_text(message).strip()
                if response_text:
                    return response_text
    return "I could not produce a response for this turn."


def _get_console_debug_mode() -> ConsoleDebugMode:
    """Return the configured terminal debug mode.

    Returns:
        Normalized console debug mode.
    """

    return _get_settings().console_debug_mode


def _is_console_debug_info_enabled() -> bool:
    """Return whether first-level terminal debug output is enabled.

    Returns:
        `True` when final message history should be printed.
    """

    return _get_console_debug_mode() in {"info", "debug"}


def _is_console_debug_json_enabled() -> bool:
    """Return whether second-level JSON terminal debug output is enabled.

    Returns:
        `True` when JSON stream events should be printed.
    """

    return _get_console_debug_mode() == "debug"


def _build_stream_modes() -> list[str]:
    """Build the LangGraph stream modes required by the Chainlit UI.

    Returns:
        Ordered, de-duplicated list of stream modes.
    """

    modes: list[str] = list(_BASE_STREAM_MODES)
    modes.extend(_PROGRESS_STREAM_MODES)
    if _is_console_debug_json_enabled():
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
    video_path: str | None = None,
    video_input_event: str | None = None,
    response_message: Any | None = None,
) -> dict[str, object]:
    """Invoke the graph through LangGraph streaming and retain the final state.

    Args:
        graph: Compiled LangGraph application.
        user_message: Latest user message text.
        thread_id: Stable chat thread identifier.
        video_path: Optional uploaded or explicit video path for this turn.
        video_input_event: Optional upload timeout/cancel event emitted by Chainlit.
        response_message: Optional Chainlit-like message used for UI token streaming.

    Returns:
        Final graph state extracted from the authoritative checkpoint snapshot.
    """

    graph_input: dict[str, object] = {"messages": [HumanMessage(content=user_message)]}
    if video_path:
        graph_input["video_path"] = video_path
        graph_input["incoming_video_path"] = video_path
    if video_input_event:
        graph_input["video_input_event"] = video_input_event

    active_steps: dict[str, tuple[str, Any]] = {}
    async for part in graph.astream(
        graph_input,
        config=_build_graph_config(thread_id),
        stream_mode=_build_stream_modes(),
        subgraphs=True,
        version="v2",
    ):
        if _is_console_debug_json_enabled():
            _emit_console_stream_part(part, thread_id=thread_id)
        await _handle_progress_step_event(part, active_steps)
        token_text = _extract_final_answer_token(part)
        if response_message is not None and token_text:
            await response_message.stream_token(token_text)

    final_state = await _get_authoritative_graph_state(graph, thread_id=thread_id)
    if _is_console_debug_info_enabled():
        _pretty_print_history(final_state)
    if response_message is not None:
        response_message.content = _extract_response_text(final_state)
    return final_state


async def _handle_progress_step_event(
    part: Mapping[str, object],
    active_steps: dict[str, tuple[str, Any]],
) -> None:
    """Create or complete Chainlit steps from sanitized LangGraph debug events."""

    progress_event = _extract_progress_event(part)
    if progress_event is None:
        return

    task_id = progress_event["task_id"]
    node_name = progress_event["node_name"]
    event_type = progress_event["event_type"]

    if event_type == "task":
        step = cl.Step(
            name=_PROGRESS_NODE_LABELS[node_name],
            type="run",
            show_input=False,
            default_open=False,
            auto_collapse=True,
        )
        step.output = _PROGRESS_RUNNING_OUTPUTS[node_name]
        step.start = _utc_now_iso()
        await step.send()
        active_steps[task_id] = (node_name, step)
        return

    active_entry = active_steps.pop(task_id, None)
    if active_entry is None:
        step = cl.Step(
            name=_PROGRESS_NODE_LABELS[node_name],
            type="run",
            show_input=False,
            default_open=False,
            auto_collapse=True,
        )
        await step.send()
    else:
        _, step = active_entry

    error_text = progress_event.get("error")
    is_error = bool(error_text) or node_name == "processing_error"
    step.is_error = is_error
    step.output = (
        f"Erro tecnico: {_sanitize_step_output(str(error_text))}"
        if error_text
        else _PROGRESS_COMPLETED_OUTPUTS[node_name]
    )
    step.end = _utc_now_iso()
    await step.update()


def _extract_progress_event(part: Mapping[str, object]) -> dict[str, str] | None:
    """Extract a main-node task transition from a LangGraph debug stream part."""

    if part.get("type") != "debug":
        return None
    data = part.get("data")
    if not isinstance(data, Mapping):
        return None
    event_type = data.get("type")
    if event_type not in {"task", "task_result"}:
        return None
    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        return None
    node_name = payload.get("name")
    if not isinstance(node_name, str) or node_name not in _PROGRESS_NODE_LABELS:
        return None
    task_id = payload.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = f"{node_name}:{data.get('step', '')}"
    error_text = payload.get("error")
    return {
        "event_type": str(event_type),
        "node_name": node_name,
        "task_id": task_id,
        "error": str(error_text) if error_text else "",
    }


def _sanitize_step_output(text: str) -> str:
    """Keep step output compact and free of obvious patient identifiers."""

    collapsed = re.sub(r"\s+", " ", text).strip()
    masked = re.sub(r"\b(\d{4})(\d{4})\b", r"****\2", collapsed)
    if len(masked) <= _MAX_STEP_OUTPUT_CHARS:
        return masked
    return f"{masked[:_MAX_STEP_OUTPUT_CHARS].rstrip()}..."


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_explicit_video_path(text: str) -> str | None:
    """Extract `video_path=...` or `path: ...` from user text.

    Args:
        text: Raw user message text.

    Returns:
        Parsed path string, if present.
    """

    for line in text.splitlines():
        match = _EXPLICIT_VIDEO_PATH_PATTERN.search(line)
        if not match:
            continue
        raw_path = match.group("path").strip()
        if not raw_path:
            continue
        return _clean_explicit_video_path(raw_path)
    return None


def _clean_explicit_video_path(raw_path: str) -> str:
    """Normalize a path value extracted from text."""

    candidate = raw_path.strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        return candidate[1:-1].strip()

    file_match = _VIDEO_FILE_PATTERN.search(candidate)
    if file_match:
        return file_match.group("path").strip().strip("'\"")
    return candidate.strip().strip("'\"")


def _extract_uploaded_video_path(
    elements: Sequence[object] | None,
    *,
    max_mb: int,
) -> str | None:
    """Extract the first uploaded video path from Chainlit elements.

    Args:
        elements: Chainlit message elements or AskFile responses.
        max_mb: Maximum accepted file size in megabytes.

    Returns:
        Local uploaded file path, if a video file is present.

    Raises:
        ValueError: If the uploaded video exceeds the configured size limit.
    """

    for element in elements or []:
        path = _extract_element_path(element)
        if path is None or not _looks_like_video_element(element, path):
            continue
        size_bytes = _extract_element_size_bytes(element, path)
        if size_bytes is not None and size_bytes > max_mb * 1024 * 1024:
            raise ValueError(
                f"Uploaded video exceeds SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB={max_mb}.",
            )
        return path
    return None


def _extract_element_path(element: object) -> str | None:
    """Read a Chainlit-like uploaded file path from a loose object."""

    for attribute in ("path", "file_path"):
        value = getattr(element, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(element, Mapping):
        for key in ("path", "file_path"):
            value = element.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _looks_like_video_element(element: object, path: str) -> bool:
    """Return whether an uploaded element is a supported video file."""

    mime = getattr(element, "mime", None) or getattr(element, "type", None)
    if isinstance(element, Mapping):
        mime = mime or element.get("mime") or element.get("type")
    if isinstance(mime, str) and mime.lower().startswith("video/"):
        return True

    name = getattr(element, "name", None)
    if isinstance(element, Mapping):
        name = name or element.get("name")
    suffix_source = str(name or path)
    return Path(suffix_source).suffix.lower() in _VIDEO_EXTENSIONS


def _extract_element_size_bytes(element: object, path: str) -> int | None:
    """Read upload size from metadata or filesystem when available."""

    size_value = getattr(element, "size", None)
    if isinstance(element, Mapping):
        size_value = size_value or element.get("size")
    if isinstance(size_value, int):
        return size_value
    if isinstance(size_value, float):
        return int(size_value)

    try:
        return Path(path).stat().st_size
    except OSError:
        return None


async def _resolve_message_video_path(message: object, settings: AppSettings) -> str | None:
    """Resolve only an explicit/uploaded video path for a Chainlit message."""

    content = str(getattr(message, "content", "") or "")
    explicit_path = _extract_explicit_video_path(content)
    if explicit_path:
        return explicit_path

    max_mb = settings.video_pipeline.upload_max_mb
    uploaded_path = _extract_uploaded_video_path(
        getattr(message, "elements", None),
        max_mb=max_mb,
    )
    if uploaded_path:
        return uploaded_path

    return None


async def _request_video_upload(settings: AppSettings) -> str | None:
    """Ask Chainlit for one video file after the graph requested upload."""

    requested_files = await cl.AskFileMessage(
        content="Upload one video file to continue, or send a local path with `video_path=...`.",
        accept=_VIDEO_ACCEPT_TYPES,
        max_size_mb=settings.video_pipeline.upload_max_mb,
        max_files=1,
        timeout=180,
    ).send()
    return _extract_uploaded_video_path(
        requested_files,
        max_mb=settings.video_pipeline.upload_max_mb,
    )


def _emit_console_stream_part(part: Mapping[str, object], *, thread_id: str) -> None:
    """Emit a console stream part in JSON debug mode.

    Args:
        part: LangGraph stream part in `version="v2"` format.
        thread_id: Stable chat thread identifier.
    """

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

    text_attr = getattr(chunk, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if text_attr is not None and not callable(text_attr):
        return str(text_attr)
    if hasattr(chunk, "content"):
        return str(getattr(chunk, "content"))
    if isinstance(chunk, Mapping) and "content" in chunk:
        return str(chunk["content"])
    return str(chunk)


def _pretty_print_history(final_state: Mapping[str, object]) -> None:
    """Pretty-print the authoritative message history when debug is enabled.

    Args:
        final_state: Final graph state for the current turn.
    """

    messages = final_state.get("messages", [])
    if not isinstance(messages, Sequence) or not messages:
        return

    print("=== Message history ===")
    for message in messages:
        pretty_print = getattr(message, "pretty_print", None)
        if callable(pretty_print):
            pretty_print()
        else:
            print(get_message_text(message))
    print("=== End message history ===")


def _should_prompt_for_video_upload(final_state: Mapping[str, object]) -> bool:
    """Return whether the graph has moved a pending video request to upload."""

    return (
        final_state.get("video_input_status") == "awaiting_upload"
        and isinstance(final_state.get("pending_video_request"), Mapping)
    )


def _pending_video_request_text(final_state: Mapping[str, object], fallback: str) -> str:
    """Recover the original request text for the graph turn after upload."""

    pending_request = final_state.get("pending_video_request")
    if isinstance(pending_request, Mapping):
        request_text = pending_request.get("request_text")
        if isinstance(request_text, str) and request_text.strip():
            return request_text.strip()
    return fallback


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
    sent_messages: list[Any] = []

    try:
        settings = _get_settings()
        video_path = await _resolve_message_video_path(message, settings)
        await response_message.send()
        sent_messages.append(response_message)
        graph = _get_graph()
        final_state = await _stream_graph_turn(
            graph,
            user_message=message.content,
            thread_id=thread_id,
            video_path=video_path,
            response_message=response_message,
        )
        await response_message.update()

        if _should_prompt_for_video_upload(final_state):
            uploaded_video_path = await _request_video_upload(settings)
            followup_message = cl.Message(content="")
            await followup_message.send()
            sent_messages.append(followup_message)
            if uploaded_video_path:
                await _stream_graph_turn(
                    graph,
                    user_message=_pending_video_request_text(final_state, message.content),
                    thread_id=thread_id,
                    video_path=uploaded_video_path,
                    response_message=followup_message,
                )
            else:
                await _stream_graph_turn(
                    graph,
                    user_message="Video upload timed out before a file was provided.",
                    thread_id=thread_id,
                    video_input_event="upload_timeout",
                    response_message=followup_message,
                )
            await followup_message.update()
    except Exception as exc:  # pragma: no cover - UI safety fallback
        fallback_message = sent_messages[-1] if sent_messages else response_message
        if not sent_messages:
            await fallback_message.send()
            sent_messages.append(fallback_message)
        fallback_message.content = (
            "I could not process the request with the current configuration. "
            f"Details: {type(exc).__name__}: {exc}"
        )
        await fallback_message.update()
