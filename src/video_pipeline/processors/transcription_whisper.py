from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from video_pipeline.contracts import (
    AudioPacket,
    PipelineConfig,
    TranscriptSegment,
    TranscriptionResult,
    VideoMeta,
)
from video_pipeline.paths import resolve_project_path


def _import_whisper():
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is required for transcription. "
            "Install the video extras or add openai-whisper to the environment."
        ) from exc
    return whisper


def _accelerator_supports_fp16() -> bool:
    try:
        import torch
    except ImportError:
        return False

    cuda_available = bool(torch.cuda.is_available())
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    return cuda_available or xpu_available


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


class TranscriptionWhisperProcessor:
    """Whisper ASR processor for audio transcription."""

    name: str = "transcription_whisper"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._debug: dict[str, object] = {
            "backend": "whisper",
            "simulated": False,
            "status": "not_setup",
        }

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        self.config = config.transcription
        self.pipeline_config = config
        self.video_meta = video_meta
        self._model = None
        self._debug = {
            "backend": "whisper",
            "simulated": False,
            "status": "ready",
            "model_name": self.config.model_name,
            "requested_language": self.config.language,
            "fp16_config": self.config.fp16,
        }

    def process_audio(self, audio_packet: AudioPacket) -> TranscriptionResult:
        """Transcribe an extracted audio file with Whisper."""
        self._ensure_setup()
        self._debug["audio_packet"] = audio_packet.model_dump()

        if not audio_packet.available or not audio_packet.audio_path:
            self._debug.update(
                {
                    "status": audio_packet.extraction_status,
                    "error": audio_packet.error,
                    "segment_count": 0,
                }
            )
            return self._empty_result()

        audio_path = Path(audio_packet.audio_path)
        if not audio_path.exists():
            error = f"AudioPacket points to a missing file: {audio_path}"
            self._debug.update(
                {
                    "status": "failed",
                    "error": error,
                    "segment_count": 0,
                }
            )
            raise FileNotFoundError(error)

        fp16 = self._effective_fp16()
        transcribe_kwargs: dict[str, object] = {
            "task": "transcribe",
            "fp16": fp16,
            "verbose": False,
        }
        if self.config.language and self.config.language != "auto":
            transcribe_kwargs["language"] = self.config.language

        try:
            model = self._load_model()
            result = model.transcribe(str(audio_path), **transcribe_kwargs)
        except Exception as exc:
            self._debug.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "segment_count": 0,
                    "fp16_effective": fp16,
                    "transcribe_kwargs": transcribe_kwargs,
                }
            )
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

        segments = self._normalize_segments(result)
        text = str(result.get("text", "")).strip()
        if not text:
            text = " ".join(segment.text for segment in segments)

        language = str(result.get("language") or self.config.language)
        artifact_paths = self._write_debug_artifacts(segments)

        self._debug.update(
            {
                "status": "transcribed",
                "segment_count": len(segments),
                "detected_language": result.get("language"),
                "fp16_effective": fp16,
                "transcribe_kwargs": transcribe_kwargs,
                "text": text,
                "raw_asr": self._summarize_asr_result(result),
                "artifacts": artifact_paths,
            }
        )
        return TranscriptionResult(
            language=language,
            text=text,
            segments=segments,
            has_audio=self.video_meta.has_audio,
        )

    def debug_payload(self) -> dict[str, object]:
        return dict(self._debug)

    def _ensure_setup(self) -> None:
        if not hasattr(self, "config") or not hasattr(self, "pipeline_config"):
            raise RuntimeError("TranscriptionWhisperProcessor.setup() must be called first")

    def _load_model(self):
        if self._model is None:
            whisper = _import_whisper()
            self._model = whisper.load_model(self.config.model_name)
        return self._model

    def _effective_fp16(self) -> bool:
        if self.config.fp16 is not None:
            return bool(self.config.fp16)
        return _accelerator_supports_fp16()

    def _normalize_segments(self, result: dict[str, Any]) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for raw_segment in result.get("segments", []):
            start_s = float(raw_segment.get("start", 0.0))
            end_s = float(raw_segment.get("end", start_s))
            if end_s < start_s:
                end_s = start_s

            text = str(raw_segment.get("text", "")).strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start_s=round(start_s, 3),
                    end_s=round(end_s, 3),
                    text=text,
                )
            )
        return segments

    def _empty_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            language=self.config.language,
            text="",
            segments=[],
            has_audio=self.video_meta.has_audio,
        )

    def _summarize_asr_result(self, result: dict[str, Any]) -> dict[str, object]:
        return {
            "language": result.get("language"),
            "text": str(result.get("text", "")).strip(),
            "segment_count": len(result.get("segments", [])),
            "keys": sorted(str(key) for key in result.keys()),
        }

    def _write_debug_artifacts(self, segments: list[TranscriptSegment]) -> dict[str, str]:
        if not self.pipeline_config.debug or not self.pipeline_config.output_dir:
            return {}

        out_dir = resolve_project_path(
            self.pipeline_config.output_dir,
            field_name="output_dir",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        if self.config.generate_debug_csv:
            csv_path = out_dir / f"{self.video_meta.video_id}.transcription.segments.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["start_s", "end_s", "duration_s", "text"],
                )
                writer.writeheader()
                for segment in segments:
                    writer.writerow(
                        {
                            "start_s": segment.start_s,
                            "end_s": segment.end_s,
                            "duration_s": round(
                                max(0.0, segment.end_s - segment.start_s),
                                3,
                            ),
                            "text": segment.text,
                        }
                    )
            artifacts["csv"] = str(csv_path)

        if self.config.generate_debug_srt:
            srt_path = out_dir / f"{self.video_meta.video_id}.transcription.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, segment in enumerate(segments, start=1):
                    f.write(f"{idx}\n")
                    f.write(
                        f"{_format_srt_timestamp(segment.start_s)} --> "
                        f"{_format_srt_timestamp(segment.end_s)}\n"
                    )
                    f.write(f"{segment.text}\n\n")
            artifacts["srt"] = str(srt_path)

        return artifacts
