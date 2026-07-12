"""Tests for video analysis and QA graph flows."""

from __future__ import annotations

import json
from pathlib import Path

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from screening_agent.config import VideoPipelineSettings
from screening_agent.data import PatientRepository
from screening_agent.graph.builder import build_screening_graph
from screening_agent.model.mock_runtime import MockControlModel
from screening_agent.tools import create_mock_specialist_invoker
from video_pipeline.contracts import (
    Detection,
    DetectionWindow,
    DominantDetection,
    ExpressionResult,
    PipelineConfig,
    PoseResult,
    TranscriptSegment,
    TranscriptionResult,
    VideoAnalysisResult,
)


def _fake_result(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
    """Create a deterministic video pipeline result for graph tests."""

    return VideoAnalysisResult(
        video_id=Path(video_path).stem,
        duration_s=12.0,
        window_s=config.window_s,
        stride_s=config.stride_s,
        expression=ExpressionResult(
            windows=[
                DetectionWindow(
                    start_s=0.0,
                    end_s=8.0,
                    detections=[Detection(label="sad_expression", score=0.82, support=0.75)],
                    dominant=DominantDetection(label="sad_expression", score=0.82),
                ),
            ],
        ),
        pose=PoseResult(
            windows=[
                DetectionWindow(
                    start_s=0.0,
                    end_s=8.0,
                    detections=[Detection(label="head_down", score=0.71, support=0.6)],
                    dominant=DominantDetection(label="head_down", score=0.71),
                ),
            ],
        ),
        transcription=TranscriptionResult(
            language="en",
            text="I have felt tired and withdrawn during the session.",
            segments=[
                TranscriptSegment(
                    start_s=1.0,
                    end_s=4.0,
                    text="I have felt tired and withdrawn during the session.",
                ),
            ],
            has_audio=True,
        ),
    )


def _build_graph(
    seeded_repository: PatientRepository,
    tmp_path: Path,
    *,
    processor,
):
    """Build a video-enabled graph for tests."""

    return build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=create_mock_specialist_invoker(),
        repository=seeded_repository,
        video_analyst_model=MockControlModel(),
        video_pipeline_settings=VideoPipelineSettings(
            output_dir=tmp_path / "video-artifacts",
            debug=False,
            upload_max_mb=100,
            window_s=8.0,
            stride_s=5.0,
        ),
        video_processor=processor,
        checkpointer=InMemorySaver(),
    )


def test_graph_processes_video_with_injected_processor(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure video analysis stores summary, JSON, and artifact metadata."""

    calls: list[tuple[str, PipelineConfig]] = []

    def processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
        calls.append((video_path, config))
        return _fake_result(video_path, config)

    graph = _build_graph(seeded_repository, tmp_path, processor=processor)
    video_path = str(tmp_path / "session01.mp4")

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze this video")],
            "video_path": video_path,
        },
        config={"configurable": {"thread_id": "video-analysis-thread"}},
    )

    assert result["router_intent"] == "video_analysis"
    assert result["video_analysis_status"] == "completed"
    assert len(calls) == 1
    assert calls[0][0] == video_path
    assert calls[0][1].window_s == 8.0
    assert calls[0][1].stride_s == 5.0
    assert calls[0][1].debug is False
    payload = json.loads(result["video_analysis_json"])
    assert payload["video_id"] == "session01"
    assert payload["_screening_agent"]["source_path"] == video_path
    assert "Video analysis completed" in result["video_analysis_summary"]
    assert str(tmp_path / "video-artifacts" / "video-analysis-thread") in result["video_artifact_dir"]


def test_graph_answers_video_qa_with_existing_result_and_active_patient(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure video QA reuses prior analysis and includes optional patient context."""

    calls: list[str] = []

    def processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
        calls.append(video_path)
        return _fake_result(video_path, config)

    graph = _build_graph(seeded_repository, tmp_path, processor=processor)
    config = {"configurable": {"thread_id": "video-qa-thread"}}
    video_path = str(tmp_path / "session02.mp4")

    graph.invoke({"messages": [HumanMessage(content="Lookup patient 11112222")]}, config=config)
    graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze this video")],
            "video_path": video_path,
        },
        config=config,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What does the video suggest about posture?")]},
        config=config,
    )

    assert result["router_intent"] == "video_qa"
    assert calls == [video_path]
    assert "Based on the processed video" in result["last_response"]
    assert "Active patient:" in result["last_response"]


def test_graph_video_qa_without_video_asks_for_upload(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure video QA does not fabricate analysis when no video exists."""

    graph = _build_graph(seeded_repository, tmp_path, processor=_fake_result)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What does the video show?")]},
        config={"configurable": {"thread_id": "missing-video-thread"}},
    )

    assert result["router_intent"] == "video_qa"
    assert result["video_analysis_status"] == "missing_video"
    assert result.get("video_analysis_json") is None
    assert "upload a video" in result["last_response"].lower()


def test_graph_video_failure_clears_stale_result_for_new_video(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure failed processing does not reuse a previous video's JSON."""

    calls: list[str] = []

    def processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
        calls.append(video_path)
        if video_path.endswith("bad.mp4"):
            raise RuntimeError("optional video dependency missing")
        return _fake_result(video_path, config)

    graph = _build_graph(seeded_repository, tmp_path, processor=processor)
    config = {"configurable": {"thread_id": "video-failure-thread"}}
    good_path = str(tmp_path / "good.mp4")
    bad_path = str(tmp_path / "bad.mp4")

    first = graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze this video")],
            "video_path": good_path,
        },
        config=config,
    )
    second = graph.invoke(
        {
            "messages": [HumanMessage(content="What does this video show?")],
            "video_path": bad_path,
        },
        config=config,
    )

    assert first["video_analysis_status"] == "completed"
    assert second["video_analysis_status"] == "failed"
    assert second["video_analysis_json"] is None
    assert second["video_analysis_error"] == "RuntimeError: optional video dependency missing"
    assert calls == [good_path, bad_path]

