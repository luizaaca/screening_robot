import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest

import video_pipeline.orchestrator as orchestrator
import video_pipeline.paths as path_module
import video_pipeline.processors.expression_deepface as expression_module
import video_pipeline.processors.transcription_whisper as transcription_module
from video_pipeline import PipelineConfig, VideoAnalysisResult, process_video
from video_pipeline.audio_extractor import extract_audio
from video_pipeline.config import load_pipeline_config
from video_pipeline.contracts import (
    AudioPacket,
    FrameAnalysisRecord,
    FrameDetection,
    FramePacket,
    TranscriptSegment,
    TranscriptionResult,
    VideoMeta,
)
from video_pipeline.paths import (
    PROJECT_ROOT,
    normalize_pipeline_config_paths,
    resolve_project_path,
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


def install_fake_orchestrator_processors(monkeypatch):
    instances = {}

    class FakeProcessor:
        key = "processor"

        def __init__(self):
            instances[self.key] = self

        def setup(self, video_meta, config):
            self.video_meta = video_meta
            self.config = config

        def wants_frame(self, frame_packet):
            return False

        def process_frame(self, frame_packet, frame_bgr):
            raise AssertionError("fake processor should not receive frames")

        def process_audio(self, audio_packet):
            self.audio_packet = audio_packet
            if audio_packet.available:
                assert Path(audio_packet.audio_path).exists()
            return []

        def aggregate(self, records):
            self.aggregate_input = records
            return []

        def debug_payload(self):
            return {
                "output_dir": self.config.output_dir,
                "model_asset_path": self.config.pose.model_asset_path,
            }

    class FakeExpressionProcessor(FakeProcessor):
        key = "expression"

    class FakePoseProcessor(FakeProcessor):
        key = "pose"

    class FakeTranscriptionProcessor(FakeProcessor):
        key = "transcription"

        def process_audio(self, audio_packet):
            self.audio_packet = audio_packet
            if audio_packet.available:
                assert Path(audio_packet.audio_path).exists()
            return TranscriptionResult(
                language=self.config.transcription.language,
                text="",
                segments=[],
                has_audio=self.video_meta.has_audio,
            )

    monkeypatch.setattr(orchestrator, "ExpressionDeepFaceProcessor", FakeExpressionProcessor)
    monkeypatch.setattr(orchestrator, "PoseMediaPipeProcessor", FakePoseProcessor)
    monkeypatch.setattr(orchestrator, "TranscriptionWhisperProcessor", FakeTranscriptionProcessor)
    monkeypatch.setattr(orchestrator, "read_frames", lambda video_meta: iter(()))

    return instances


def fake_probe_video(path: str) -> VideoMeta:
    return VideoMeta(
        video_id=Path(path).stem,
        source_path=path,
        duration_s=1.0,
        fps=1.0,
        width=640,
        height=480,
        has_audio=True,
    )


def test_resolve_project_path_keeps_absolute_path(tmp_path):
    absolute_path = tmp_path / "asset.txt"

    assert resolve_project_path(str(absolute_path)) == absolute_path


def test_resolve_project_path_resolves_relative_path_from_project_root():
    resolved = resolve_project_path("relative/out")

    assert resolved == (PROJECT_ROOT / "relative" / "out").resolve()


@pytest.mark.parametrize("blank_path", ["", "   "])
def test_resolve_project_path_rejects_blank_required_path(blank_path):
    with pytest.raises(ValueError, match="video_path"):
        resolve_project_path(blank_path, field_name="video_path")


def test_load_pipeline_config_relative_path_uses_project_root(monkeypatch, tmp_path):
    config_path = tmp_path / "relative.yaml"
    config_path.write_text("window_s: 3.0\naudio:\n  sample_rate: 22050\n", encoding="utf-8")
    monkeypatch.setattr(path_module, "PROJECT_ROOT", tmp_path)

    config = load_pipeline_config("relative.yaml")

    assert config.window_s == 3.0
    assert config.audio.sample_rate == 22050


def test_normalize_pipeline_config_paths_resolves_copy():
    config = PipelineConfig(output_dir="relative_out")
    config.pose.model_asset_path = "models/holistic.task"

    normalized = normalize_pipeline_config_paths(config)

    assert normalized is not config
    assert normalized.output_dir == str((PROJECT_ROOT / "relative_out").resolve())
    assert normalized.pose.model_asset_path == str(
        (PROJECT_ROOT / "models" / "holistic.task").resolve()
    )
    assert config.output_dir == "relative_out"
    assert config.pose.model_asset_path == "models/holistic.task"


def test_process_video_relative_paths_and_debug_audio_preserved(monkeypatch, tmp_path):
    instances = install_fake_orchestrator_processors(monkeypatch)
    monkeypatch.setattr(path_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "probe_video", fake_probe_video)
    extract_calls = []

    def fake_extract_audio(video_meta, audio_config, output_dir=None, cleanup_dir=None):
        extract_calls.append({"output_dir": output_dir, "cleanup_dir": cleanup_dir})
        audio_path = Path(output_dir) / f"{video_meta.video_id}.{audio_config.format}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")
        return AudioPacket(
            audio_path=str(audio_path),
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            codec=audio_config.codec,
            format=audio_config.format,
            duration_s=video_meta.duration_s,
            available=True,
            extraction_status="extracted",
            cleanup_dir=cleanup_dir,
        )

    monkeypatch.setattr(orchestrator, "extract_audio", fake_extract_audio)

    config = PipelineConfig(debug=True, output_dir="relative_out")
    config.pose.model_asset_path = "models/holistic.task"

    process_video("videos/test_video.mp4", config=config)

    output_dir = (tmp_path / "relative_out").resolve()
    assert (output_dir / "test_video.expression.json").exists()
    assert (output_dir / "test_video.pose.json").exists()
    assert (output_dir / "test_video.transcription.json").exists()
    assert (output_dir / "test_video.wav").exists()

    expr_payload = json.loads((output_dir / "test_video.expression.json").read_text(encoding="utf-8"))
    assert expr_payload == {"module": "expression", "windows": []}

    pose_payload = json.loads((output_dir / "test_video.pose.json").read_text(encoding="utf-8"))
    assert pose_payload == {"module": "pose", "windows": []}

    trans_payload = json.loads((output_dir / "test_video.transcription.json").read_text(encoding="utf-8"))
    assert trans_payload == {
        "module": "transcription",
        "language": "pt",
        "text": "",
        "segments": [],
        "has_audio": True,
    }

    expr_debug = json.loads((output_dir / "test_video.expression.debug.json").read_text(encoding="utf-8"))
    pose_debug = json.loads((output_dir / "test_video.pose.debug.json").read_text(encoding="utf-8"))
    trans_debug = json.loads((output_dir / "test_video.transcription.debug.json").read_text(encoding="utf-8"))

    assert set(expr_debug["config"]) == {"window_s", "stride_s", "debug", "output_dir", "expression"}
    assert "pose" not in expr_debug["config"]
    assert set(pose_debug["config"]) == {"window_s", "stride_s", "debug", "output_dir", "pose"}
    assert "expression" not in pose_debug["config"]
    assert set(trans_debug["config"]) == {
        "window_s",
        "stride_s",
        "debug",
        "output_dir",
        "transcription",
    }
    assert "expression" not in trans_debug["config"]
    assert "pose" not in trans_debug["config"]
    assert "windows_summary" not in trans_debug
    assert trans_debug["result_summary"] == trans_payload

    assert extract_calls == [{"output_dir": str(output_dir), "cleanup_dir": None}]
    assert instances["pose"].config.pose.model_asset_path == str(
        (tmp_path / "models" / "holistic.task").resolve()
    )
    assert config.output_dir == "relative_out"
    assert config.pose.model_asset_path == "models/holistic.task"


def test_process_video_removes_non_debug_audio_cleanup_dir(monkeypatch):
    install_fake_orchestrator_processors(monkeypatch)
    monkeypatch.setattr(orchestrator, "probe_video", fake_probe_video)
    extract_calls = []

    def fake_extract_audio(video_meta, audio_config, output_dir=None, cleanup_dir=None):
        extract_calls.append({"output_dir": output_dir, "cleanup_dir": cleanup_dir})
        audio_path = Path(output_dir) / f"{video_meta.video_id}.{audio_config.format}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")
        return AudioPacket(
            audio_path=str(audio_path),
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            codec=audio_config.codec,
            format=audio_config.format,
            duration_s=video_meta.duration_s,
            available=True,
            extraction_status="extracted",
            cleanup_dir=cleanup_dir,
        )

    monkeypatch.setattr(orchestrator, "extract_audio", fake_extract_audio)

    process_video("test_video.mp4", config=PipelineConfig(debug=False, output_dir=None))

    cleanup_dir = Path(extract_calls[0]["cleanup_dir"])
    assert extract_calls[0]["output_dir"] == extract_calls[0]["cleanup_dir"]
    assert ".tmp_audio" not in str(cleanup_dir)
    assert not cleanup_dir.exists()


def test_transcription_debug_artifacts_use_normalized_output_dir(monkeypatch, tmp_path):
    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            return {
                "language": "pt",
                "text": "ola",
                "segments": [{"start": 0.0, "end": 1.0, "text": " ola "}],
            }

    class FakeWhisper:
        @staticmethod
        def load_model(model_name):
            return FakeModel()

    monkeypatch.setattr(path_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(transcription_module, "_import_whisper", lambda: FakeWhisper)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    config = PipelineConfig(debug=True, output_dir="debug_out")
    config.transcription.fp16 = False

    proc = TranscriptionWhisperProcessor()
    proc.setup(make_meta(duration_s=1.0), config)
    result = proc.process_audio(
        AudioPacket(
            audio_path=str(audio_path),
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            format="wav",
            duration_s=1.0,
            available=True,
            extraction_status="extracted",
        )
    )

    output_dir = tmp_path / "debug_out"
    assert (output_dir / "test.transcription.segments.csv").exists()
    assert (output_dir / "test.transcription.srt").exists()


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
    assert result.transcription.has_audio is True
    assert result.transcription.text == ""
    assert result.transcription.segments == []

    payload = result.model_dump()
    assert "video_id" not in payload["expression"]
    assert "duration_s" not in payload["expression"]
    assert "window_s" not in payload["expression"]
    assert "stride_s" not in payload["expression"]
    assert "video_id" not in payload["pose"]
    assert "video_id" not in payload["transcription"]
    assert "windows" not in payload["transcription"]


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
    assert result.transcription.text == ""
    assert result.transcription.segments == []


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
    assert trans.debug_payload()["status"] == "ready"


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
    assert packet.available is False
    assert packet.extraction_status == "failed"
    assert packet.audio_path == ""
    assert packet.error is not None


def test_transcription_skips_unavailable_audio_without_loading_whisper(monkeypatch):
    def fail_import():
        raise AssertionError("whisper should not be imported for unavailable audio")

    monkeypatch.setattr(transcription_module, "_import_whisper", fail_import)

    proc = TranscriptionWhisperProcessor()
    proc.setup(make_meta(), PipelineConfig())
    result = proc.process_audio(
        AudioPacket(
            audio_path="",
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            format="wav",
            duration_s=0.0,
            available=False,
            extraction_status="failed",
            error="audio extraction failed",
        )
    )

    assert result.text == ""
    assert result.segments == []
    assert result.has_audio is True
    payload = proc.debug_payload()
    assert payload["status"] == "failed"
    assert payload["segment_count"] == 0
    assert payload["simulated"] is False


def test_transcription_whisper_uses_config_and_writes_debug_artifacts(
    monkeypatch,
    tmp_path,
):
    calls = {}

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            calls["audio_path"] = audio_path
            calls["kwargs"] = kwargs
            return {
                "language": "pt",
                "text": "ola mundo",
                "segments": [
                    {"start": 0.25, "end": 1.75, "text": " ola "},
                    {"start": 2.0, "end": 3.5, "text": " mundo "},
                ],
            }

    class FakeWhisper:
        @staticmethod
        def load_model(model_name):
            calls["model_name"] = model_name
            return FakeModel()

    monkeypatch.setattr(transcription_module, "_import_whisper", lambda: FakeWhisper)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    config = PipelineConfig(debug=True, output_dir=str(tmp_path))
    config.transcription.model_name = "tiny"
    config.transcription.language = "pt"
    config.transcription.fp16 = False

    proc = TranscriptionWhisperProcessor()
    proc.setup(make_meta(), config)
    result = proc.process_audio(
        AudioPacket(
            audio_path=str(audio_path),
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            format="wav",
            duration_s=4.0,
            available=True,
            extraction_status="extracted",
        )
    )
    segments = result.segments

    assert calls["model_name"] == "tiny"
    assert calls["audio_path"] == str(audio_path)
    assert calls["kwargs"] == {
        "task": "transcribe",
        "fp16": False,
        "verbose": False,
        "language": "pt",
    }
    assert result.language == "pt"
    assert result.text == "ola mundo"
    assert result.has_audio is True
    assert [seg.text for seg in segments] == ["ola", "mundo"]
    assert segments[0].start_s == 0.25
    assert segments[0].end_s == 1.75
    assert result.model_dump()["segments"] == [
        {"start": 0.25, "end": 1.75, "text": "ola"},
        {"start": 2.0, "end": 3.5, "text": "mundo"},
    ]

    csv_path = tmp_path / "test.transcription.segments.csv"
    srt_path = tmp_path / "test.transcription.srt"
    assert csv_path.exists()
    assert srt_path.exists()

    payload = proc.debug_payload()
    assert payload["status"] == "transcribed"
    assert payload["segment_count"] == 2
    assert payload["detected_language"] == "pt"
    assert payload["fp16_effective"] is False
    assert payload["artifacts"] == {
        "csv": str(csv_path),
        "srt": str(srt_path),
    }


def test_transcription_whisper_auto_language_omits_language(monkeypatch, tmp_path):
    calls = {}

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            calls["kwargs"] = kwargs
            return {"language": "en", "text": "hello", "segments": []}

    class FakeWhisper:
        @staticmethod
        def load_model(model_name):
            return FakeModel()

    monkeypatch.setattr(transcription_module, "_import_whisper", lambda: FakeWhisper)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    config = PipelineConfig(debug=False)
    config.transcription.language = "auto"
    config.transcription.fp16 = True

    proc = TranscriptionWhisperProcessor()
    proc.setup(make_meta(), config)
    result = proc.process_audio(
        AudioPacket(
            audio_path=str(audio_path),
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            format="wav",
            duration_s=1.0,
            available=True,
            extraction_status="extracted",
        )
    )

    assert result.language == "en"
    assert result.text == "hello"
    assert result.segments == []
    assert calls["kwargs"] == {
        "task": "transcribe",
        "fp16": True,
        "verbose": False,
    }
    assert not (tmp_path / "test.transcription.segments.csv").exists()
    assert not (tmp_path / "test.transcription.srt").exists()


def test_transcription_result_uses_whisper_style_segments_without_windows():
    result = TranscriptionResult(
        language="pt",
        text="primeira fala segunda fala",
        segments=[
            TranscriptSegment(start=0.0, end=5.88, text="primeira fala"),
            TranscriptSegment(start_s=5.88, end_s=10.0, text="segunda fala"),
        ],
        has_audio=True,
    )

    payload = result.model_dump()

    assert "windows" not in payload
    assert payload["segments"] == [
        {"start": 0.0, "end": 5.88, "text": "primeira fala"},
        {"start": 5.88, "end": 10.0, "text": "segunda fala"},
    ]


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
                FrameDetection(timestamp_s=3.0, label="surprise", score=0.99),
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
    surprise_det = [d for d in w1.detections if d.label == "surprise"][0]
    assert happy_det.score == 0.8
    assert happy_det.support == 1.0
    assert sad_det.score == 0.2
    assert sad_det.support == 0.5
    assert surprise_det.score == 0.99
    assert surprise_det.support == 0.5
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
            detections=[
                FrameDetection(timestamp_s=3.0, label="hand_on_head", score=0.8),
                FrameDetection(timestamp_s=3.0, label="hand_on_neck", score=1.0),
            ],
        ),
    ]

    pose_windows = pose.aggregate(pose_records)
    hand_det = [d for d in pose_windows[0].detections if d.label == "hand_on_head"][0]
    neck_det = [d for d in pose_windows[0].detections if d.label == "hand_on_neck"][0]
    assert hand_det.score == 0.8
    assert hand_det.support == 1.0
    assert neck_det.score == 1.0
    assert neck_det.support == 0.5
    assert pose_windows[0].dominant is not None
    assert pose_windows[0].dominant.label == "hand_on_head"


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

    assert len(windows[0].detections) == 1
    assert windows[0].detections[0].label == "hand_on_head"
    assert windows[0].detections[0].score == 0.8


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


def test_read_frames_resizing(monkeypatch, tmp_path):
    import numpy as np
    import cv2

    # Criamos um frame fake grande (1920x1080)
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class FakeVideoCapture:
        def __init__(self, path):
            self.opened = True
            self.frames_read = 0

        def isOpened(self):
            return self.opened

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            return 0

        def read(self):
            if self.frames_read < 2:
                self.frames_read += 1
                return True, fake_frame.copy()
            return False, None

        def release(self):
            self.opened = False

    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_text("fake video file content")

    meta = VideoMeta(
        video_id="dummy",
        source_path=str(dummy_file),
        duration_s=2.0,
        fps=10.0,
        width=1920,
        height=1080,
        has_audio=False,
    )

    frames = list(read_frames(meta))
    assert len(frames) == 2
    for packet, frame in frames:
        assert frame is not None
        h, w = frame.shape[:2]
        assert max(h, w) == 480
        assert w == 480
        assert h == 270
