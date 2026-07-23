"""Chainlit user interface for the screening assistant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
import re
import sys
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
DEFAULT_LOCALE = "pt-BR"
_SUPPORTED_LOCALES: frozenset[str] = frozenset({DEFAULT_LOCALE, "en-US"})
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
_UI_STRINGS: dict[str, dict[str, str]] = {
    DEFAULT_LOCALE: {
        "empty_response": "Não consegui produzir uma resposta para este turno.",
        "request_failure": (
            "Não consegui processar a solicitação com a configuração atual. "
            "Detalhes: {error_type}: {error}"
        ),
        "upload_prompt": (
            "Envie um arquivo de vídeo para continuar ou informe um caminho local "
            "com `video_path=...`."
        ),
        "upload_timeout": "O envio do vídeo expirou antes que um arquivo fosse fornecido.",
    },
    "en-US": {
        "empty_response": "I could not produce a response for this turn.",
        "request_failure": (
            "I could not process the request with the current configuration. "
            "Details: {error_type}: {error}"
        ),
        "upload_prompt": (
            "Upload one video file to continue, or send a local path with `video_path=...`."
        ),
        "upload_timeout": "Video upload timed out before a file was provided.",
    },
}
_WELCOME_TEXT: dict[str, dict[str, object]] = {
    DEFAULT_LOCALE: {
        "title": "# Assistente de Triagem Clínica",
        "intro": "Você pode me pedir para:",
        "capabilities": [
            "- explicar como usar o assistente;",
            "- buscar um paciente por número de segurança fictício ou por nome;",
            "- limpar o contexto do paciente ativo;",
            "- analisar sintomas e sugerir condições prováveis ou exames relevantes.",
            "- enviar ou referenciar um vídeo para análise de expressão, postura e transcrição.",
        ],
        "patient_count": "Registros de pacientes disponíveis no momento: {patient_count}.",
        "empty_database": "O banco de dados está vazio no momento.",
        "seed_database": (
            "Execute `python seed_demo_data.py` para carregar os pacientes de demonstração "
            "antes de testar os fluxos de busca."
        ),
        "examples_title": "Exemplos de prompts:",
        "examples": [
            "- `Encontrar paciente Maria Silva`",
            "- `Consultar paciente 12003456`",
            "- `Paciente 55667788 tem fadiga e micção frequente`",
            "- `Analise este vídeo com video_path=concepts_video/sample.mp4`",
            "- `Limpar paciente ativo`",
        ],
    },
    "en-US": {
        "title": "# Clinical Screening Assistant",
        "intro": "You can ask me to:",
        "capabilities": [
            "- explain how to use the assistant;",
            "- look up a patient by fictional security number or by name;",
            "- clear the active patient context;",
            "- analyze symptoms and suggest likely conditions or relevant exams.",
            "- upload or reference a video for expression, posture, and transcription analysis.",
        ],
        "patient_count": "Current patient records available: {patient_count}.",
        "empty_database": "The database is currently empty.",
        "seed_database": (
            "Run `python seed_demo_data.py` to load the demo patients before testing lookup flows."
        ),
        "examples_title": "Example prompts:",
        "examples": [
            "- `Find patient Maria Silva`",
            "- `Lookup patient 12003456`",
            "- `Patient 55667788 has fatigue and frequent urination`",
            "- `Analyze this video with video_path=concepts_video/sample.mp4`",
            "- `Clear active patient`",
        ],
    },
}
_PROGRESS_NODE_LABELS: dict[str, dict[str, str]] = {
    DEFAULT_LOCALE: {
        "router": "Classificando solicitação",
        "patient_lookup": "Buscando paciente",
        "video_analysis": "Processando vídeo",
        "video_interpretation": "Interpretando vídeo",
        "video_qa": "Interpretando vídeo",
        "video_clinical_extraction": "Extraindo contexto clínico do vídeo",
        "symptom_analysis": "Analisando sintomas",
        "final_answer": "Gerando resposta",
    },
    "en-US": {
        "router": "Classifying request",
        "patient_lookup": "Looking up patient",
        "video_analysis": "Processing video",
        "video_interpretation": "Interpreting video",
        "video_qa": "Interpreting video",
        "video_clinical_extraction": "Extracting clinical video context",
        "symptom_analysis": "Analyzing symptoms",
        "final_answer": "Generating response",
    },
}
_PROGRESS_RUNNING_OUTPUTS: dict[str, dict[str, str]] = {
    DEFAULT_LOCALE: {
        "router": "Classificando a solicitação.",
        "patient_lookup": "Buscando o paciente solicitado.",
        "video_analysis": "Executando o pipeline de vídeo.",
        "video_interpretation": "Gerando interpretação narrativa do vídeo.",
        "video_qa": "Gerando interpretação narrativa do vídeo.",
        "video_clinical_extraction": "Extraindo contexto clínico estruturado do vídeo.",
        "symptom_analysis": "Executando análise de sintomas.",
        "final_answer": "Compondo a resposta final.",
    },
    "en-US": {
        "router": "Classifying the request.",
        "patient_lookup": "Looking up the requested patient.",
        "video_analysis": "Running the video pipeline.",
        "video_interpretation": "Generating narrative video interpretation.",
        "video_qa": "Generating narrative video interpretation.",
        "video_clinical_extraction": "Extracting structured clinical video context.",
        "symptom_analysis": "Running symptom analysis.",
        "final_answer": "Composing the final response.",
    },
}
_PROGRESS_COMPLETED_OUTPUTS: dict[str, dict[str, str]] = {
    DEFAULT_LOCALE: {
        "router": "Solicitação classificada.",
        "patient_lookup": "Busca de paciente concluída.",
        "video_analysis": "Pipeline de vídeo concluído.",
        "video_interpretation": "Interpretação de vídeo concluída.",
        "video_qa": "Interpretação de vídeo concluída.",
        "video_clinical_extraction": "Contexto clínico de vídeo extraído.",
        "symptom_analysis": "Análise de sintomas concluída.",
        "final_answer": "Resposta final gerada.",
    },
    "en-US": {
        "router": "Request classified.",
        "patient_lookup": "Patient lookup complete.",
        "video_analysis": "Video pipeline complete.",
        "video_interpretation": "Video interpretation complete.",
        "video_qa": "Video interpretation complete.",
        "video_clinical_extraction": "Clinical video context extracted.",
        "symptom_analysis": "Symptom analysis complete.",
        "final_answer": "Final response generated.",
    },
}
_PROGRESS_ERROR_PREFIXES: dict[str, str] = {
    DEFAULT_LOCALE: "Erro técnico",
    "en-US": "Technical error",
}
_PROGRESS_NODE_NAMES: frozenset[str] = frozenset(_PROGRESS_NODE_LABELS[DEFAULT_LOCALE])
_MAX_STEP_OUTPUT_CHARS = 240


def _normalize_locale(language: str | None) -> str:
    """Normalize a browser/Chainlit language code to a supported UI locale."""

    raw_language = str(language or "").strip().replace("_", "-")
    primary_language = re.split(r"[,;]", raw_language, maxsplit=1)[0].strip()
    normalized_language = primary_language.lower()
    if normalized_language == "pt" or normalized_language.startswith("pt-"):
        return DEFAULT_LOCALE
    if normalized_language == "en" or normalized_language.startswith("en-"):
        return "en-US"
    return DEFAULT_LOCALE


def _current_locale() -> str:
    """Infer the current Chainlit session locale, falling back to pt-BR."""

    try:
        session_language = getattr(cl.context.session, "language", None)
    except Exception:
        return DEFAULT_LOCALE
    return _normalize_locale(session_language)


def _ui_string(locale: str | None, key: str, **format_values: object) -> str:
    """Return a localized UI string for the app-controlled Chainlit text."""

    template = _UI_STRINGS[_normalize_locale(locale)][key]
    if format_values:
        return template.format(**format_values)
    return template


def _progress_node_label(node_name: str, locale: str | None) -> str:
    """Return a localized Chainlit progress step label."""

    return _PROGRESS_NODE_LABELS[_normalize_locale(locale)][node_name]


def _progress_running_output(node_name: str, locale: str | None) -> str:
    """Return the localized in-progress text for a graph node."""

    return _PROGRESS_RUNNING_OUTPUTS[_normalize_locale(locale)][node_name]


def _progress_completed_output(node_name: str, locale: str | None) -> str:
    """Return the localized completed text for a graph node."""

    return _PROGRESS_COMPLETED_OUTPUTS[_normalize_locale(locale)][node_name]


def _progress_error_output(error_text: str, locale: str | None) -> str:
    """Return a localized, sanitized technical error output for a progress step."""

    error_prefix = _PROGRESS_ERROR_PREFIXES[_normalize_locale(locale)]
    return f"{error_prefix}: {_sanitize_step_output(error_text)}"


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


def _build_welcome_message(patient_count: int, *, locale: str | None = None) -> str:
    """Create the welcome message displayed when a chat session starts.

    Args:
        patient_count: Number of patient records currently available.
        locale: Optional UI locale for app-controlled text.

    Returns:
        User-facing welcome text.
    """

    welcome_text = _WELCOME_TEXT[_normalize_locale(locale)]
    capabilities = welcome_text["capabilities"]
    examples = welcome_text["examples"]
    if not isinstance(capabilities, list) or not isinstance(examples, list):
        raise TypeError("Welcome text lists are misconfigured.")

    lines = [
        str(welcome_text["title"]),
        "",
        str(welcome_text["intro"]),
        *[str(item) for item in capabilities],
        "",
        str(welcome_text["patient_count"]).format(patient_count=patient_count),
    ]
    if patient_count == 0:
        lines.extend(
            [
                "",
                str(welcome_text["empty_database"]),
                str(welcome_text["seed_database"]),
            ],
        )
    else:
        lines.extend(
            [
                "",
                str(welcome_text["examples_title"]),
                *[str(item) for item in examples],
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


def _extract_response_text(
    result: Mapping[str, object],
    *,
    locale: str | None = None,
) -> str:
    """Extract the final assistant response from a graph invocation result.

    Args:
        result: Graph output state.
        locale: Optional UI locale for app-controlled fallback text.

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
    return _ui_string(locale, "empty_response")


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
    locale: str | None = None,
    video_path: str | None = None,
    video_input_event: str | None = None,
) -> dict[str, object]:
    """Invoke the graph through LangGraph streaming and retain the final state.

    Args:
        graph: Compiled LangGraph application.
        user_message: Latest user message text.
        thread_id: Stable chat thread identifier.
        locale: Optional UI locale for app-controlled progress text.
        video_path: Optional uploaded or explicit video path for this turn.
        video_input_event: Optional upload timeout/cancel event emitted by Chainlit.

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
    progress_locale = _normalize_locale(locale)
    try:
        async for part in graph.astream(
            graph_input,
            config=_build_graph_config(thread_id),
            stream_mode=_build_stream_modes(),
            subgraphs=True,
            version="v2",
        ):
            if _is_console_debug_json_enabled():
                _emit_console_stream_part(part, thread_id=thread_id)
            await _handle_progress_step_event(part, active_steps, locale=progress_locale)
    except Exception as exc:
        await _close_remaining_progress_steps(
            active_steps,
            locale=progress_locale,
            error=exc,
        )
        raise

    await _close_remaining_progress_steps(active_steps, locale=progress_locale)

    final_state = await _get_authoritative_graph_state(graph, thread_id=thread_id)
    if _is_console_debug_info_enabled():
        _try_pretty_print_history(final_state)
    return final_state


async def _send_final_response_message(
    final_state: Mapping[str, object],
    *,
    locale: str | None = None,
) -> Any:
    """Send the final answer after all streamed graph steps are complete."""

    return await _send_top_level_message(_extract_response_text(final_state, locale=locale))


async def _send_top_level_message(content: str) -> Any:
    """Send a Chainlit message without inheriting the active step parent."""

    message = cl.Message(content=content)
    message.parent_id = None
    await message.send()
    return message


async def _handle_progress_step_event(
    part: Mapping[str, object],
    active_steps: dict[str, tuple[str, Any]],
    *,
    locale: str | None = None,
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
            name=_progress_node_label(node_name, locale),
            type="run",
            show_input=False,
            default_open=False,
            auto_collapse=True,
        )
        step.output = _progress_running_output(node_name, locale)
        await step.__aenter__()
        active_steps[task_id] = (node_name, step)
        return

    active_entry = active_steps.pop(task_id, None)
    if active_entry is None:
        step = cl.Step(
            name=_progress_node_label(node_name, locale),
            type="run",
            show_input=False,
            default_open=False,
            auto_collapse=True,
        )
        await step.__aenter__()
    else:
        _, step = active_entry

    error_text = progress_event.get("error")
    is_error = bool(error_text)
    step.is_error = is_error
    step.output = (
        _progress_error_output(str(error_text), locale)
        if error_text
        else _progress_completed_output(node_name, locale)
    )
    await step.__aexit__(None, None, None)


async def _close_remaining_progress_steps(
    active_steps: dict[str, tuple[str, Any]],
    *,
    locale: str | None = None,
    error: Exception | None = None,
) -> None:
    """Close any progress steps that did not receive a terminal debug event."""

    while active_steps:
        _, (node_name, step) = active_steps.popitem()
        if error is not None:
            step.is_error = True
            step.output = _progress_error_output(str(error), locale)
        elif not getattr(step, "output", "") or step.output == _progress_running_output(
            node_name,
            locale,
        ):
            step.output = _progress_completed_output(node_name, locale)
        await step.__aexit__(None, None, None)


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
    if not isinstance(node_name, str) or node_name not in _PROGRESS_NODE_NAMES:
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


async def _request_video_upload(
    settings: AppSettings,
    *,
    locale: str | None = None,
) -> str | None:
    """Ask Chainlit for one video file after the graph requested upload."""

    requested_files = await cl.AskFileMessage(
        content=_ui_string(locale, "upload_prompt"),
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


def _safe_console_print(text: object) -> None:
    """Print terminal diagnostics without raising on Windows code pages."""

    output_text = str(text)
    try:
        print(output_text)
    except UnicodeEncodeError:
        output_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = output_text.encode(output_encoding, errors="backslashreplace").decode(
            output_encoding,
            errors="replace",
        )
        print(safe_text)


def _try_pretty_print_history(final_state: Mapping[str, object]) -> None:
    """Emit message history diagnostics without letting console errors affect UI."""

    try:
        _pretty_print_history(final_state)
    except Exception as exc:
        _safe_console_print(
            "Console history debug output failed: "
            f"{type(exc).__name__}: {_sanitize_step_output(str(exc))}"
        )


def _pretty_print_history(final_state: Mapping[str, object]) -> None:
    """Pretty-print the authoritative message history when debug is enabled.

    Args:
        final_state: Final graph state for the current turn.
    """

    messages = final_state.get("messages", [])
    if not isinstance(messages, Sequence) or not messages:
        return

    _safe_console_print("=== Message history ===")
    for message in messages:
        message_type = type(message).__name__
        _safe_console_print(f"{message_type}: {get_message_text(message)}")
    _safe_console_print("=== End message history ===")


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

    locale = _current_locale()
    thread_id = str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    patient_count = _get_patient_count()
    await cl.Message(content=_build_welcome_message(patient_count, locale=locale)).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle a user message by invoking the LangGraph workflow.

    Args:
        message: Incoming Chainlit user message.
    """

    locale = _current_locale()
    thread_id = cl.user_session.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        thread_id = str(uuid4())
        cl.user_session.set("thread_id", thread_id)

    try:
        settings = _get_settings()
        video_path = await _resolve_message_video_path(message, settings)
        graph = _get_graph()
        final_state = await _stream_graph_turn(
            graph,
            user_message=message.content,
            thread_id=thread_id,
            locale=locale,
            video_path=video_path,
        )
        await _send_final_response_message(final_state, locale=locale)

        if _should_prompt_for_video_upload(final_state):
            uploaded_video_path = await _request_video_upload(settings, locale=locale)
            if uploaded_video_path:
                followup_state = await _stream_graph_turn(
                    graph,
                    user_message=_pending_video_request_text(final_state, message.content),
                    thread_id=thread_id,
                    locale=locale,
                    video_path=uploaded_video_path,
                )
            else:
                followup_state = await _stream_graph_turn(
                    graph,
                    user_message=_ui_string(locale, "upload_timeout"),
                    thread_id=thread_id,
                    locale=locale,
                    video_input_event="upload_timeout",
                )
            await _send_final_response_message(followup_state, locale=locale)
    except Exception as exc:  # pragma: no cover - UI safety fallback
        await _send_top_level_message(
            _ui_string(
                locale,
                "request_failure",
                error_type=type(exc).__name__,
                error=exc,
            ),
        )
