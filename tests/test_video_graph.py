"""Tests for video analysis and QA graph flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from screening_agent.config import VideoPipelineSettings
from screening_agent.data import PatientRepository
from screening_agent.graph.builder import build_screening_graph
from screening_agent.model.mock_runtime import MockControlModel
from screening_agent.tools import create_mock_specialist_invoker
from screening_agent.tools.specialist_tool import ClinicalScreeningOutput
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
    video_analyst_model: Any | None = None,
    specialist_invoker: Any | None = None,
):
    """Build a video-enabled graph for tests."""

    return build_screening_graph(
        control_model=MockControlModel(),
        specialist_invoker=specialist_invoker or create_mock_specialist_invoker(),
        repository=seeded_repository,
        video_analyst_model=video_analyst_model or MockControlModel(),
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


class _RecordingVideoAnalyst(MockControlModel):
    """Video analyst stub that records interpretation and extraction prompts."""

    def __init__(self) -> None:
        self.interpretation_payloads: list[dict[str, object]] = []
        self.extraction_payloads: list[dict[str, object]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        system_text = next(
            (str(message.content) for message in messages if isinstance(message, SystemMessage)),
            "",
        )
        payload_text = next(
            (str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)),
            "{}",
        )
        payload = json.loads(payload_text)
        if "structured clinical-context extractor" in system_text:
            self.extraction_payloads.append(payload)
            return AIMessage(
                content=json.dumps(
                    {
                        "reported_or_inferred_symptoms": [
                            "fatigue mentioned in video transcription"
                        ],
                        "observable_signs": ["sad facial expression", "head-down posture"],
                        "evidence": [
                            {
                                "observation": "fatigue mentioned in video transcription",
                                "source": "transcription",
                                "time_range_s": [1.0, 4.0],
                            }
                        ],
                        "limitations": ["Video evidence is supportive only."],
                        "uncertainties": ["Requires clinical correlation."],
                        "clinical_attention_points": ["Compare video signs with patient history."],
                    },
                ),
            )

        self.interpretation_payloads.append(payload)
        return AIMessage(content="Narrative video interpretation from the recording analyst.")


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
    assert result["video_interpretation"]
    assert result.get("video_clinical_context_json") is None
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

    assert result["router_intent"] == "video_interpretation"
    assert calls == [video_path]
    assert "Based on the processed video" in result["last_response"]
    assert "Active patient:" in result["last_response"]


def test_general_video_upload_uses_interpretation_without_clinical_extraction(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure general upload processes video and skips structured clinical extraction."""

    video_analyst = _RecordingVideoAnalyst()
    calls: list[str] = []

    def processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
        calls.append(video_path)
        return _fake_result(video_path, config)

    graph = _build_graph(
        seeded_repository,
        tmp_path,
        processor=processor,
        video_analyst_model=video_analyst,
    )
    video_path = str(tmp_path / "general.mp4")

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze this video")],
            "video_path": video_path,
            "incoming_video_path": video_path,
        },
        config={"configurable": {"thread_id": "general-upload-thread"}},
    )

    assert calls == [video_path]
    assert result["router_intent"] == "video_analysis"
    assert result["video_analysis_status"] == "completed"
    assert result["video_interpretation"] == "Narrative video interpretation from the recording analyst."
    assert result.get("video_clinical_context_json") is None
    assert len(video_analyst.interpretation_payloads) == 1
    assert video_analyst.extraction_payloads == []


def test_video_interpretation_receives_optional_active_patient_context(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure patient context is passed to narrative video interpretation when loaded."""

    video_analyst = _RecordingVideoAnalyst()
    graph = _build_graph(
        seeded_repository,
        tmp_path,
        processor=_fake_result,
        video_analyst_model=video_analyst,
    )
    config = {"configurable": {"thread_id": "video-patient-context-thread"}}
    video_path = str(tmp_path / "patient-context.mp4")

    graph.invoke({"messages": [HumanMessage(content="Lookup patient 11112222")]}, config=config)
    graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze this video")],
            "video_path": video_path,
            "incoming_video_path": video_path,
        },
        config=config,
    )

    assert video_analyst.interpretation_payloads[0]["active_patient"]["full_name"] == "João Souza"
    assert video_analyst.extraction_payloads == []


def test_video_symptom_flow_extracts_clinical_context_before_symptom_analysis(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure symptom analysis based on video uses structured video context."""

    video_analyst = _RecordingVideoAnalyst()
    specialist_contexts: list[str] = []

    def specialist_invoker(clinical_request: str, clinical_context: str) -> ClinicalScreeningOutput:
        specialist_contexts.append(clinical_context)
        return ClinicalScreeningOutput(
            support_status="supported",
            candidate_diseases=["Further video-informed screening"],
            recommended_exams_tests=["Focused clinical examination"],
        )

    graph = _build_graph(
        seeded_repository,
        tmp_path,
        processor=_fake_result,
        video_analyst_model=video_analyst,
        specialist_invoker=specialist_invoker,
    )
    video_path = str(tmp_path / "symptom-video.mp4")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Could symptoms be worsening based on this video?")
            ],
            "video_path": video_path,
            "incoming_video_path": video_path,
        },
        config={"configurable": {"thread_id": "video-symptom-thread"}},
    )

    assert result["router_intent"] == "video_symptom_analysis"
    assert result["video_analysis_status"] == "completed"
    assert json.loads(result["video_clinical_context_json"])["observable_signs"] == [
        "sad facial expression",
        "head-down posture",
    ]
    assert video_analyst.interpretation_payloads == []
    assert len(video_analyst.extraction_payloads) == 1
    assert "Structured clinical context extracted from video" in specialist_contexts[0]
    assert "fatigue mentioned in video transcription" in specialist_contexts[0]


def test_video_symptom_flow_reuses_pipeline_and_extraction_for_same_request(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure repeated video symptom requests reuse pipeline and clinical extraction."""

    video_analyst = _RecordingVideoAnalyst()
    calls: list[str] = []

    def processor(video_path: str, config: PipelineConfig) -> VideoAnalysisResult:
        calls.append(video_path)
        return _fake_result(video_path, config)

    graph = _build_graph(
        seeded_repository,
        tmp_path,
        processor=processor,
        video_analyst_model=video_analyst,
    )
    config = {"configurable": {"thread_id": "video-symptom-reuse-thread"}}
    video_path = str(tmp_path / "symptom-reuse.mp4")
    request = "Could symptoms be worsening based on this video?"

    first = graph.invoke(
        {
            "messages": [HumanMessage(content=request)],
            "video_path": video_path,
            "incoming_video_path": video_path,
        },
        config=config,
    )
    second = graph.invoke({"messages": [HumanMessage(content=request)]}, config=config)

    assert first["video_analysis_status"] == "completed"
    assert second["video_analysis_status"] == "completed"
    assert calls == [video_path]
    assert len(video_analyst.extraction_payloads) == 1
    assert second["video_clinical_context_json"] == first["video_clinical_context_json"]


def test_video_upload_confirmation_acceptance_decline_and_timeout(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure graph-owned two-turn confirmation states work without Chainlit heuristics."""

    graph = _build_graph(seeded_repository, tmp_path, processor=_fake_result)
    config = {"configurable": {"thread_id": "video-confirmation-thread"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="What does the video show?")]},
        config=config,
    )
    confirmed = graph.invoke({"messages": [HumanMessage(content="yes")]}, config=config)
    timeout = graph.invoke(
        {
            "messages": [HumanMessage(content="Video upload timed out before a file was provided.")],
            "video_input_event": "upload_timeout",
        },
        config=config,
    )

    assert first["video_input_status"] == "awaiting_confirmation"
    assert confirmed["video_input_status"] == "awaiting_upload"
    assert timeout["video_input_status"] == "none"
    assert timeout["pending_video_request"] is None
    assert "timeout" in timeout["last_response"].lower()

    decline_config = {"configurable": {"thread_id": "video-decline-thread"}}
    graph.invoke(
        {"messages": [HumanMessage(content="What does the video show?")]},
        config=decline_config,
    )
    declined = graph.invoke({"messages": [HumanMessage(content="no")]}, config=decline_config)

    assert declined["video_input_status"] == "none"
    assert declined["pending_video_request"] is None
    assert "declined" in declined["last_response"].lower()


def test_graph_video_question_without_video_requests_confirmation(
    seeded_repository: PatientRepository,
    tmp_path: Path,
) -> None:
    """Ensure video QA does not fabricate analysis when no video exists."""

    graph = _build_graph(seeded_repository, tmp_path, processor=_fake_result)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What does the video show?")]},
        config={"configurable": {"thread_id": "missing-video-thread"}},
    )

    assert result["router_intent"] == "video_interpretation"
    assert result["video_input_status"] == "awaiting_confirmation"
    assert result["pending_video_request"]["intent"] == "video_interpretation"
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
