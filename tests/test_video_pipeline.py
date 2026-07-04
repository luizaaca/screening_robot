import os
import shutil
import tempfile

import pytest

import video_pipeline.orchestrator as orchestrator
import video_pipeline.processors.expression_deepface as expression_module
from video_pipeline import PipelineConfig, VideoAnalysisResult, process_video
from video_pipeline.audio_extractor import extract_audio
from video_pipeline.config import load_pipeline_config
from video_pipeline.contracts import (
    FrameAnalysisRecord,
    FrameDetection,
    FramePacket,
    VideoMeta,
)
from video_pipeline.processors.expression_deepface import (
    ExpressionDeepFaceProcessor,
    _normalize_score,
    _select_largest_face,
)
from video_pipeline.processors.pose_mediapipe import PoseMediaPipeProcessor
from video_pipeline.processors.transcription_whisper import TranscriptionWhisperProcessor
from video_pipeline.video_reader import read_frames


class FakeDeepFace:
    analyze_results = []
    analyze_calls = []
    build_model_calls = []

    @classmethod
    def reset(cls):
        cls.analyze_results = []
        cls.analyze_calls = []
        cls.build_model_calls = []

    @classmethod
    def build_model(cls, model_name, task=None):
        cls.build_model_calls.append({"model_name": model_name, "task": task})
        return {"model_name": model_name, "task": task}

    @classmethod
    def analyze(cls, **kwargs):
        cls.analyze_calls.append(kwargs)
        if cls.analyze_results:
            result = cls.analyze_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return {
            "dominant_emotion": "happy",
            "emotion": {"happy": 92.0},
            "region": {"x": 1, "y": 2, "w": 20, "h": 20},
        }


@pytest.fixture(autouse=True)
def fake_deepface(monkeypatch):
    FakeDeepFace.reset()
    monkeypatch.setattr(expression_module, "_import_deepface", lambda: FakeDeepFace)
    return FakeDeepFace


def fake_frame_stream(video_meta):
    for idx in range(int(video_meta.duration_s)):
        yield FramePacket(timestamp_s=float(idx), frame_index=idx), object()


def make_meta(duration_s=10.0, fps=30.0):
    return VideoMeta(
        video_id="test",
        source_path="",
        duration_s=duration_s,
        fps=fps,
        width=640,
        height=480,
        has_audio=True,
    )


def test_process_video_in_memory(monkeypatch):
    monkeypatch.setattr(orchestrator, "read_frames", fake_frame_stream)
    config = PipelineConfig(window_s=8.0, stride_s=2.0, debug=False, output_dir=None)

    result = process_video("non_existent_mock_video.mp4", config=config)

    assert isinstance(result, VideoAnalysisResult)
    assert result.video_id == "non_existent_mock_video"
    assert result.duration_s == 30.0
    assert result.window_s == 8.0
    assert result.stride_s == 2.0

    assert result.expression is not None
    assert len(result.expression.windows) > 0
    assert result.expression.windows[0].start_s == 0.0
    assert result.expression.windows[0].detections[0].label == "joy_expression"

    assert result.pose is not None
    assert len(result.pose.windows) > 0

    assert result.transcription is not None
    assert len(result.transcription.windows) > 0
    assert result.transcription.has_audio is True


def test_process_video_saving_files(monkeypatch):
    monkeypatch.setattr(orchestrator, "read_frames", fake_frame_stream)
    temp_dir = tempfile.mkdtemp()
    try:
        config = PipelineConfig(
            window_s=8.0,
            stride_s=2.0,
            debug=True,
            output_dir=temp_dir,
        )

        process_video("test_video.mp4", config=config)

        expr_file = os.path.join(temp_dir, "test_video.expression.json")
        pose_file = os.path.join(temp_dir, "test_video.pose.json")
        trans_file = os.path.join(temp_dir, "test_video.transcription.json")
        assert os.path.exists(expr_file)
        assert os.path.exists(pose_file)
        assert os.path.exists(trans_file)

        expr_debug_file = os.path.join(temp_dir, "test_video.expression.debug.json")
        pose_debug_file = os.path.join(temp_dir, "test_video.pose.debug.json")
        trans_debug_file = os.path.join(temp_dir, "test_video.transcription.debug.json")
        assert os.path.exists(expr_debug_file)
        assert os.path.exists(pose_debug_file)
        assert os.path.exists(trans_debug_file)
    finally:
        shutil.rmtree(temp_dir)


def test_process_video_no_audio(monkeypatch):
    monkeypatch.setattr(orchestrator, "read_frames", fake_frame_stream)
    config = PipelineConfig(window_s=8.0, stride_s=2.0, debug=False, output_dir=None)

    result = process_video("some_silence_video.mp4", config=config)

    assert result.transcription is not None
    assert result.transcription.has_audio is False
    assert len(result.transcription.windows) > 0
    for win in result.transcription.windows:
        assert win.text == ""
        assert len(win.segments) == 0
        assert win.coverage_s == 0.0


def test_config_loading():
    config = load_pipeline_config()
    assert isinstance(config, PipelineConfig)
    assert config.window_s == 8.0
    assert config.audio.sample_rate == 16000
    assert config.expression.min_confidence == 0.30
    assert config.pose.target_sample_fps == 5.0
    assert config.transcription.model_name == "base"


def test_processor_setup_and_aggregate_signature(fake_deepface):
    meta = make_meta()
    config = PipelineConfig()

    expr = ExpressionDeepFaceProcessor()
    expr.setup(meta, config)
    assert expr.pipeline_config == config
    assert fake_deepface.build_model_calls == [
        {"model_name": "Emotion", "task": "facial_attribute"}
    ]
    assert isinstance(expr.aggregate([]), list)

    pose = PoseMediaPipeProcessor()
    pose.setup(meta, config)
    assert pose.pipeline_config == config
    assert isinstance(pose.aggregate([]), list)

    trans = TranscriptionWhisperProcessor()
    trans.setup(meta, config)
    assert trans.pipeline_config == config
    assert isinstance(trans.aggregate([]), list)


def test_audio_extractor_uses_config():
    meta = make_meta()
    meta.source_path = "non_existent.mp4"
    config = PipelineConfig()
    config.audio.sample_rate = 22050
    config.audio.channels = 2
    config.audio.codec = "libmp3lame"
    config.audio.format = "mp3"

    packet = extract_audio(meta, config.audio)

    assert packet.sample_rate == 22050
    assert packet.channels == 2
    assert packet.codec == "libmp3lame"
    assert packet.format == "mp3"


def test_video_reader_sequential_timestamps():
    meta = VideoMeta(
        video_id="test",
        source_path="non_existent.mp4",
        duration_s=2.0,
        fps=10.0,
        width=640,
        height=480,
        has_audio=False,
    )

    frames = list(read_frames(meta))

    assert len(frames) == 20
    for idx, (packet, frame) in enumerate(frames):
        assert frame is None
        assert packet.frame_index == idx
        assert packet.timestamp_s == round(idx / 10.0, 3)


def test_normalize_score():
    assert _normalize_score(None) is None
    assert _normalize_score(82.0) == 0.82
    assert _normalize_score(0.42) == 0.42
    assert _normalize_score(180.0) == 1.0
    assert _normalize_score(-0.2) == 0.0


def test_select_largest_face_from_dict_and_list():
    face, count = _select_largest_face(
        {"dominant_emotion": "sad", "region": {"w": 10, "h": 5}}
    )
    assert face is not None
    assert face["dominant_emotion"] == "sad"
    assert count == 1

    face, count = _select_largest_face(
        [
            {"dominant_emotion": "happy", "region": {"w": 3, "h": 3}},
            {"dominant_emotion": "sad", "region": {"w": 8, "h": 4}},
            {"dominant_emotion": "fear", "region": {"w": 0, "h": 10}},
            "not-a-face",
        ]
    )
    assert face is not None
    assert face["dominant_emotion"] == "sad"
    assert count == 2


def test_expression_process_frame_scorable(fake_deepface):
    fake_deepface.analyze_results = [
        {
            "dominant_emotion": "sad",
            "emotion": {"sad": 87.0},
            "region": {"x": 1, "y": 2, "w": 10, "h": 10},
        }
    ]
    config = PipelineConfig()
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), config)

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), object())

    assert record.status == "scorable"
    assert len(record.detections) == 1
    assert record.detections[0].label == "sad_expression"
    assert record.detections[0].score == 0.87
    assert fake_deepface.analyze_calls[0]["actions"] == ["emotion"]
    assert fake_deepface.analyze_calls[0]["detector_backend"] == "opencv"


def test_expression_process_frame_uncertain_low_score(fake_deepface):
    fake_deepface.analyze_results = [
        {
            "dominant_emotion": "happy",
            "emotion": {"happy": 0.1},
            "region": {"w": 10, "h": 10},
        }
    ]
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), PipelineConfig())

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), object())

    assert record.status == "uncertain"
    assert record.detections == []
    assert record.debug["reason"] == "low_confidence"


def test_expression_process_frame_uncertain_unknown_label(fake_deepface):
    fake_deepface.analyze_results = [
        {
            "dominant_emotion": "confused",
            "emotion": {"confused": 99.0},
            "region": {"w": 10, "h": 10},
        }
    ]
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), PipelineConfig())

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), object())

    assert record.status == "uncertain"
    assert record.detections == []
    assert record.debug["reason"] == "unknown_label"


def test_expression_process_frame_no_detection(fake_deepface):
    fake_deepface.analyze_results = [
        [
            {"dominant_emotion": "happy", "region": {"w": 0, "h": 10}},
            {"dominant_emotion": "sad", "region": {"w": 5, "h": 0}},
        ]
    ]
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), PipelineConfig())

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), object())

    assert record.status == "no_detection"
    assert record.detections == []
    assert record.debug["reason"] == "no_valid_face"


def test_expression_process_frame_insufficient_data(fake_deepface):
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), PipelineConfig())

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), None)

    assert record.status == "insufficient_data"
    assert record.detections == []
    assert record.debug["reason"] == "missing_frame"
    assert fake_deepface.analyze_calls == []


def test_expression_process_frame_analyze_failure(fake_deepface):
    fake_deepface.analyze_results = [RuntimeError("backend failed")]
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(), PipelineConfig())

    record = expr.process_frame(FramePacket(timestamp_s=0.0, frame_index=0), object())

    assert record.status == "insufficient_data"
    assert record.detections == []
    assert record.debug["reason"] == "deepface_analyze_failed"
    assert record.debug["error"] == "backend failed"


def test_synthetic_aggregation_expression_and_pose():
    meta = make_meta(duration_s=10.0)
    config = PipelineConfig()
    config.window_s = 5.0
    config.stride_s = 2.0

    expr = ExpressionDeepFaceProcessor()
    expr.setup(meta, config)

    records = [
        FrameAnalysisRecord(
            timestamp_s=1.0,
            frame_index=30,
            status="scorable",
            detections=[FrameDetection(timestamp_s=1.0, label="happy", score=0.9)],
        ),
        FrameAnalysisRecord(
            timestamp_s=3.0,
            frame_index=90,
            status="scorable",
            detections=[
                FrameDetection(timestamp_s=3.0, label="happy", score=0.7),
                FrameDetection(timestamp_s=3.0, label="sad", score=0.2),
            ],
        ),
        FrameAnalysisRecord(
            timestamp_s=4.0,
            frame_index=120,
            status="uncertain",
            detections=[FrameDetection(timestamp_s=4.0, label="sad", score=0.9)],
        ),
    ]

    windows = expr.aggregate(records)
    assert len(windows) > 0
    w1 = windows[0]
    assert w1.start_s == 0.0
    assert w1.end_s == 5.0
    happy_det = [d for d in w1.detections if d.label == "happy"][0]
    sad_det = [d for d in w1.detections if d.label == "sad"][0]
    assert happy_det.score == 0.8
    assert happy_det.support == 1.0
    assert sad_det.score == 0.2
    assert sad_det.support == 0.5
    assert w1.dominant is not None
    assert w1.dominant.label == "happy"

    empty_windows = expr.aggregate([])
    assert empty_windows[0].detections == []
    assert empty_windows[0].dominant is None

    pose = PoseMediaPipeProcessor()
    pose.setup(meta, config)
    pose_records = [
        FrameAnalysisRecord(
            timestamp_s=1.0,
            frame_index=30,
            status="scorable",
            detections=[FrameDetection(timestamp_s=1.0, label="hand_on_head", score=0.6)],
        ),
        FrameAnalysisRecord(
            timestamp_s=3.0,
            frame_index=90,
            status="scorable",
            detections=[FrameDetection(timestamp_s=3.0, label="hand_on_head", score=0.8)],
        ),
    ]

    pose_windows = pose.aggregate(pose_records)
    hand_det = [d for d in pose_windows[0].detections if d.label == "hand_on_head"][0]
    assert hand_det.score == 0.8
    assert hand_det.support == 1.0


def test_expression_debug_payload_counters_segments_and_percentages(fake_deepface):
    fake_deepface.analyze_results = [
        {
            "dominant_emotion": "happy",
            "emotion": {"happy": 90.0},
            "region": {"w": 10, "h": 10},
        },
        {
            "dominant_emotion": "happy",
            "emotion": {"happy": 80.0},
            "region": {"w": 10, "h": 10},
        },
        {
            "dominant_emotion": "sad",
            "emotion": {"sad": 90.0},
            "region": {"w": 10, "h": 10},
        },
        {"dominant_emotion": "happy", "region": {"w": 0, "h": 0}},
    ]
    config = PipelineConfig()
    config.expression.min_segment_duration_s = 1.5
    expr = ExpressionDeepFaceProcessor()
    expr.setup(make_meta(duration_s=4.0, fps=1.0), config)

    for idx in range(4):
        expr.process_frame(FramePacket(timestamp_s=float(idx), frame_index=idx), object())

    payload = expr.debug_payload()

    assert payload["backend"] == "opencv"
    assert payload["counters"]["total_frames"] == 4
    assert payload["counters"]["scorable_frames"] == 3
    assert payload["counters"]["no_detection_frames"] == 1
    assert payload["segments"] == [
        {"label": "joy_expression", "start_s": 0.0, "end_s": 2.0}
    ]
    assert payload["percentages"] == {"joy_expression": 50.0}
    assert len(payload["frames"]) == 4


def test_video_reader_packets_can_feed_expression_processor(fake_deepface):
    meta = VideoMeta(
        video_id="test",
        source_path="non_existent.mp4",
        duration_s=1.0,
        fps=2.0,
        width=640,
        height=480,
        has_audio=False,
    )
    config = PipelineConfig(window_s=1.0, stride_s=1.0)
    expr = ExpressionDeepFaceProcessor()
    expr.setup(meta, config)

    records = [
        expr.process_frame(packet, object())
        for packet, _frame in read_frames(meta)
    ]
    windows = expr.aggregate(records)

    assert len(records) == 2
    assert len(fake_deepface.analyze_calls) == 2
    assert windows[0].detections[0].label == "joy_expression"


def test_json_default_does_not_apply_global_thresholds():
    meta = make_meta()
    config = PipelineConfig()
    config.pose.display_score_threshold = 0.9

    pose = PoseMediaPipeProcessor()
    pose.setup(meta, config)
    pose_records = [
        FrameAnalysisRecord(
            timestamp_s=1.0,
            frame_index=30,
            status="scorable",
            detections=[FrameDetection(timestamp_s=1.0, label="hand_on_head", score=0.8)],
        )
    ]

    windows = pose.aggregate(pose_records)

    assert len(windows[0].detections) == 0


def test_pose_aggregate_uses_detections_not_debug():
    meta = make_meta()
    config = PipelineConfig()
    pose = PoseMediaPipeProcessor()
    pose.setup(meta, config)

    pose_records = [
        FrameAnalysisRecord(
            timestamp_s=1.0,
            frame_index=30,
            status="scorable",
            detections=[],
            debug={"rules": {"hand_on_head": 0.8}},
        )
    ]
    windows = pose.aggregate(pose_records)

    assert len(windows[0].detections) == 0
