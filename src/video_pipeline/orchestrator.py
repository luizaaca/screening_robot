import shutil
import tempfile

from video_pipeline.audio_extractor import extract_audio
from video_pipeline.config import load_pipeline_config
from video_pipeline.contracts import (
    ExpressionResult,
    PipelineConfig,
    PoseResult,
    TranscriptionResult,
    VideoAnalysisResult,
    VideoMeta,
)
from video_pipeline.media_probe import probe_video
from video_pipeline.paths import normalize_pipeline_config_paths, resolve_project_path
from video_pipeline.processors import (
    ExpressionDeepFaceProcessor,
    PoseMediaPipeProcessor,
    TranscriptionWhisperProcessor,
)
from video_pipeline.video_reader import read_frames
from video_pipeline.writers import write_results


def process_video(
    video_path: str,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> VideoAnalysisResult:
    """Process one video through expression, posture, and transcription modules."""

    if config is not None and config_path is not None:
        raise ValueError("config and config_path are mutually exclusive.")

    if config is None:
        config = load_pipeline_config(config_path)

    config = normalize_pipeline_config_paths(config)
    resolved_video_path = resolve_project_path(video_path, field_name="video_path")

    video_meta: VideoMeta = probe_video(str(resolved_video_path))
    video_id = video_meta.video_id
    duration = video_meta.duration_s

    expression_proc = ExpressionDeepFaceProcessor()
    pose_proc = PoseMediaPipeProcessor()
    trans_proc = TranscriptionWhisperProcessor()

    try:
        expression_proc.setup(video_meta, config)
        pose_proc.setup(video_meta, config)
        trans_proc.setup(video_meta, config)

        raw_frame_records = {
            "expression": [],
            "pose": [],
        }

        if duration > 0:
            for frame_packet, frame_bgr in read_frames(video_meta):
                if expression_proc.wants_frame(frame_packet):
                    rec = expression_proc.process_frame(frame_packet, frame_bgr)
                    raw_frame_records["expression"].append(rec)

                if pose_proc.wants_frame(frame_packet):
                    rec = pose_proc.process_frame(frame_packet, frame_bgr)
                    raw_frame_records["pose"].append(rec)

        expression_windows = expression_proc.aggregate(raw_frame_records["expression"])
        pose_windows = pose_proc.aggregate(raw_frame_records["pose"])

        expression_res = ExpressionResult(
            windows=expression_windows,
        )

        pose_res = PoseResult(
            windows=pose_windows,
        )

        trans_res = TranscriptionResult(
            language=config.transcription.language,
            text="",
            segments=[],
            has_audio=video_meta.has_audio,
        )
        audio_packet = None

        preserve_audio = bool(config.debug and config.output_dir)
        temp_audio_dir = None if preserve_audio else tempfile.mkdtemp(prefix="video_pipeline_audio_")

        try:
            audio_output_dir = config.output_dir if preserve_audio else temp_audio_dir
            audio_packet = extract_audio(
                video_meta,
                config.audio,
                audio_output_dir,
                cleanup_dir=None if preserve_audio else temp_audio_dir,
            )
            trans_res = trans_proc.process_audio(audio_packet)
        finally:
            cleanup_dir = (
                audio_packet.cleanup_dir
                if audio_packet is not None and audio_packet.cleanup_dir
                else temp_audio_dir
            )
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        analysis_result = VideoAnalysisResult(
            video_id=video_id,
            duration_s=duration,
            window_s=config.window_s,
            stride_s=config.stride_s,
            expression=expression_res,
            pose=pose_res,
            transcription=trans_res,
        )

        module_debug_payloads = {
            "expression": expression_proc.debug_payload(),
            "pose": pose_proc.debug_payload(),
            "transcription": trans_proc.debug_payload(),
        }

        write_results(
            result=analysis_result,
            video_meta=video_meta,
            config=config,
            module_debug_payloads=module_debug_payloads,
            raw_transcript_segments=trans_res.segments,
        )

        return analysis_result
    finally:
        for processor in (expression_proc, pose_proc, trans_proc):
            teardown = getattr(processor, "teardown", None)
            if callable(teardown):
                teardown()
