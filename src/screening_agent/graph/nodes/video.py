"""Video analysis and video QA nodes for the screening assistant graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import re
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from screening_agent.audit import emit_console_audit, emit_custom_debug_event
from screening_agent.config import VideoPipelineSettings
from screening_agent.graph.message_utils import get_last_human_message_text, get_message_text
from screening_agent.graph.state import AssistantState, create_audit_event
from screening_agent.model.control_models import ControlModel
from screening_agent.prompts import VIDEO_QA_SYSTEM_PROMPT
from video_pipeline.contracts import PipelineConfig, VideoAnalysisResult

VideoProcessor = Callable[[str, PipelineConfig], VideoAnalysisResult]

_MISSING_VIDEO_RESPONSE = (
    "Please upload a video file or provide a local path using `video_path=...` "
    "so I can process the video before answering."
)
_VIDEO_ANALYSIS_FAILURE_RESPONSE = (
    "I could not process the video safely with the current pipeline configuration. "
    "Please review the video file, optional video dependencies, and pipeline settings."
)
_VIDEO_QA_FAILURE_RESPONSE = (
    "I could not generate a safe answer from the video analysis with the current video analyst configuration."
)
_MAX_SUMMARY_WINDOWS = 4
_MAX_TRANSCRIPT_CHARS = 700


def build_video_analysis_node(
    settings: VideoPipelineSettings,
    video_processor: VideoProcessor | None = None,
) -> Callable[[AssistantState, RunnableConfig], dict[str, object]]:
    """Build the deterministic video-processing node.

    Args:
        settings: Video pipeline settings loaded from configuration.
        video_processor: Optional processing function injected by tests.

    Returns:
        A LangGraph node callable.
    """

    processor = video_processor or _default_video_processor

    def video_analysis(
        state: AssistantState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return _run_video_analysis(
            state,
            config=config,
            settings=settings,
            video_processor=processor,
        )

    return video_analysis


def build_video_qa_node(
    video_analyst_model: ControlModel,
    settings: VideoPipelineSettings,
    video_processor: VideoProcessor | None = None,
) -> Callable[[AssistantState, RunnableConfig], dict[str, object]]:
    """Build the node that answers questions using video-analysis context.

    Args:
        video_analyst_model: Chat model used for video QA.
        settings: Video pipeline settings loaded from configuration.
        video_processor: Optional processing function injected by tests.

    Returns:
        A LangGraph node callable.
    """

    processor = video_processor or _default_video_processor

    def video_qa(
        state: AssistantState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        analysis_json = _coerce_non_empty_string(state.get("video_analysis_json"))
        current_video_path = _coerce_non_empty_string(state.get("video_path"))
        analysis_payload = _load_analysis_payload(analysis_json)

        if _should_process_before_qa(
            current_video_path=current_video_path,
            analysis_payload=analysis_payload,
        ):
            processing_update = _run_video_analysis(
                state,
                config=config,
                settings=settings,
                video_processor=processor,
            )
            if processing_update.get("video_analysis_status") != "completed":
                return processing_update
            analysis_json = _coerce_non_empty_string(processing_update.get("video_analysis_json"))
            analysis_payload = _load_analysis_payload(analysis_json)
            prior_events = list(processing_update.get("audit_events", []))
        else:
            processing_update = {}
            prior_events = []

        if not isinstance(analysis_payload, Mapping):
            event = create_audit_event(
                event_type="video_qa",
                status="error",
                node_name="video_qa",
                detail="Video QA could not find a valid video analysis payload.",
            )
            emit_console_audit(event)
            return {
                **processing_update,
                "last_response": _MISSING_VIDEO_RESPONSE,
                "video_analysis_status": "missing_video",
                "audit_events": [*prior_events, event],
            }

        latest_user_message = _resolve_latest_user_message(state)
        prompt_payload = {
            "latest_user_message": latest_user_message,
            "active_patient": state.get("active_patient"),
            "video_analysis_summary": _coerce_non_empty_string(
                processing_update.get("video_analysis_summary"),
            )
            or state.get("video_analysis_summary"),
            "video_analysis": analysis_payload,
        }
        messages = [
            SystemMessage(content=VIDEO_QA_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=False, indent=2)),
        ]
        emit_custom_debug_event(
            "video_qa_prompt",
            node_name="video_qa",
            payload={"messages": messages},
        )

        try:
            response = video_analyst_model.invoke(messages)
            response_text = get_message_text(response).strip()
            if not response_text:
                raise ValueError("Video analyst returned empty content.")
        except Exception as exc:  # pragma: no cover - defensive provider fallback
            event = create_audit_event(
                event_type="video_qa",
                status="error",
                node_name="video_qa",
                detail=f"Video QA failed: {type(exc).__name__}: {exc}",
            )
            emit_console_audit(event)
            return {
                **processing_update,
                "last_response": _VIDEO_QA_FAILURE_RESPONSE,
                "specialist_output_json": None,
                "audit_events": [*prior_events, event],
            }

        event = create_audit_event(
            event_type="video_qa",
            status="success",
            node_name="video_qa",
            detail="Generated a video QA response from stored video analysis.",
        )
        emit_console_audit(event)
        emit_custom_debug_event(
            "video_qa_response",
            node_name="video_qa",
            payload={"response": response_text},
        )
        return {
            **processing_update,
            "last_response": response_text,
            "specialist_output_json": None,
            "audit_events": [*prior_events, event],
        }

    return video_qa


def _run_video_analysis(
    state: AssistantState,
    *,
    config: RunnableConfig | None,
    settings: VideoPipelineSettings,
    video_processor: VideoProcessor,
) -> dict[str, object]:
    """Run the video pipeline and build the state update."""

    video_path = _coerce_non_empty_string(state.get("video_path"))
    if video_path is None:
        event = create_audit_event(
            event_type="video_analysis",
            status="warning",
            node_name="video_analysis",
            detail="Video analysis requested without an uploaded file or explicit path.",
        )
        emit_console_audit(event)
        return {
            "video_analysis_status": "missing_video",
            "video_analysis_error": "No video path was provided.",
            "last_response": _MISSING_VIDEO_RESPONSE,
            "specialist_output_json": None,
            "audit_events": [event],
        }

    thread_id = _resolve_thread_id(config)
    artifact_dir = settings.output_dir / _safe_path_component(thread_id)
    pipeline_config = PipelineConfig(
        window_s=settings.window_s,
        stride_s=settings.stride_s,
        debug=settings.debug,
        output_dir=str(artifact_dir),
    )

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = video_processor(video_path, pipeline_config)
        result_payload = _analysis_result_to_payload(result)
        result_payload["_screening_agent"] = {
            "source_path": video_path,
            "artifact_dir": str(artifact_dir),
        }
        analysis_json = json.dumps(result_payload, ensure_ascii=False)
        summary = _build_video_summary(result_payload, artifact_dir=artifact_dir)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        event = create_audit_event(
            event_type="video_analysis",
            status="error",
            node_name="video_analysis",
            detail=f"Video processing failed: {error_text}",
        )
        emit_console_audit(event)
        emit_custom_debug_event(
            "video_analysis_failure",
            node_name="video_analysis",
            payload={"video_path": video_path, "error": error_text},
        )
        return {
            "video_path": video_path,
            "video_artifact_dir": str(artifact_dir),
            "video_analysis_summary": None,
            "video_analysis_json": None,
            "video_analysis_status": "failed",
            "video_analysis_error": error_text,
            "specialist_output_json": None,
            "last_response": _VIDEO_ANALYSIS_FAILURE_RESPONSE,
            "audit_events": [event],
        }

    event = create_audit_event(
        event_type="video_analysis",
        status="success",
        node_name="video_analysis",
        detail=f"Processed video and stored artifacts under {artifact_dir}.",
    )
    emit_console_audit(event)
    emit_custom_debug_event(
        "video_analysis_result",
        node_name="video_analysis",
        payload={
            "video_path": video_path,
            "artifact_dir": str(artifact_dir),
            "summary": summary,
        },
    )
    return {
        "video_path": video_path,
        "video_artifact_dir": str(artifact_dir),
        "video_analysis_summary": summary,
        "video_analysis_json": analysis_json,
        "video_analysis_status": "completed",
        "video_analysis_error": None,
        "specialist_output_json": None,
        "last_response": summary,
        "audit_events": [event],
    }


def _default_video_processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
    """Call the real video pipeline lazily so optional imports stay isolated."""

    from video_pipeline.orchestrator import process_video

    return process_video(video_path, config=config)


def _analysis_result_to_payload(result: VideoAnalysisResult | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a video analysis result into a JSON-compatible dictionary."""

    if isinstance(result, Mapping):
        return {str(key): value for key, value in result.items()}
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", by_alias=True, exclude_none=True))
    raise TypeError(f"Unsupported video processor result type: {type(result).__name__}")


def _build_video_summary(payload: Mapping[str, Any], *, artifact_dir: Path) -> str:
    """Build a compact text summary for downstream final-answer composition."""

    lines = [
        (
            "Video analysis completed"
            f" for {payload.get('video_id', 'unknown video')}"
            f" ({_format_seconds(payload.get('duration_s'))})."
        ),
        (
            "Pipeline window/stride: "
            f"{payload.get('window_s', 'n/a')}s / {payload.get('stride_s', 'n/a')}s."
        ),
    ]
    lines.extend(_summarize_detection_module(payload.get("expression"), label="Expression"))
    lines.extend(_summarize_detection_module(payload.get("pose"), label="Posture"))
    lines.extend(_summarize_transcription(payload.get("transcription")))
    lines.append(f"Artifacts directory: {artifact_dir}")
    return "\n".join(lines)


def _summarize_detection_module(value: object, *, label: str) -> list[str]:
    """Summarize dominant detections from a visual module."""

    if not isinstance(value, Mapping):
        return [f"{label}: no result available."]
    windows = value.get("windows")
    if not isinstance(windows, list) or not windows:
        return [f"{label}: no detection windows captured."]

    lines = [f"{label} timeline:"]
    emitted = 0
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        dominant = window.get("dominant")
        if not isinstance(dominant, Mapping):
            detections = window.get("detections")
            dominant = detections[0] if isinstance(detections, list) and detections else None
        if not isinstance(dominant, Mapping):
            continue
        lines.append(
            "  - "
            f"{_format_time_range(window)}: "
            f"{dominant.get('label', 'unknown')} "
            f"(score {_format_score(dominant.get('score'))})."
        )
        emitted += 1
        if emitted >= _MAX_SUMMARY_WINDOWS:
            break
    if emitted == 0:
        return [f"{label}: no dominant detections captured."]
    return lines


def _summarize_transcription(value: object) -> list[str]:
    """Summarize transcription text and audio availability."""

    if not isinstance(value, Mapping):
        return ["Transcription: no result available."]
    if value.get("has_audio") is False:
        return ["Transcription: the video has no audio track."]

    text = str(value.get("text") or "").strip()
    if text:
        return [f"Transcription: {_truncate_text(text, _MAX_TRANSCRIPT_CHARS)}"]

    segments = value.get("segments")
    if isinstance(segments, list) and segments:
        segment_texts = [
            str(segment.get("text") or "").strip()
            for segment in segments
            if isinstance(segment, Mapping)
        ]
        joined = " ".join(text for text in segment_texts if text).strip()
        if joined:
            return [f"Transcription: {_truncate_text(joined, _MAX_TRANSCRIPT_CHARS)}"]
    return ["Transcription: no speech text detected."]


def _should_process_before_qa(
    *,
    current_video_path: str | None,
    analysis_payload: Mapping[str, Any] | None,
) -> bool:
    """Return whether QA must run video processing before answering."""

    if analysis_payload is None:
        return True
    if current_video_path is None:
        return False

    stored_source_path = _stored_source_path(analysis_payload)
    if stored_source_path is None:
        return False
    return not _same_path(current_video_path, stored_source_path)


def _stored_source_path(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("_screening_agent")
    if not isinstance(metadata, Mapping):
        return None
    return _coerce_non_empty_string(metadata.get("source_path"))


def _load_analysis_payload(raw_json: str | None) -> dict[str, Any] | None:
    if raw_json is None:
        return None
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_latest_user_message(state: AssistantState) -> str:
    try:
        return get_last_human_message_text(state.get("messages", []))
    except ValueError:
        return ""


def _resolve_thread_id(config: RunnableConfig | None) -> str:
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
    return "default"


def _safe_path_component(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe_value.strip("._") or "default"


def _coerce_non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))


def _format_seconds(value: object) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "unknown duration"


def _format_time_range(window: Mapping[str, Any]) -> str:
    return f"{_format_timestamp(window.get('start_s'))}-{_format_timestamp(window.get('end_s'))}"


def _format_timestamp(value: object) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "n/a"


def _format_score(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."
