import tempfile
from pathlib import Path

from video_pipeline.contracts import AudioExtractionConfig, AudioPacket, VideoMeta
from video_pipeline.paths import resolve_project_path


def _import_video_file_clip():
    try:
        from moviepy import VideoFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError as exc:
            raise ImportError("moviepy is required to extract audio from videos") from exc
    return VideoFileClip


def _unavailable_packet(
    audio_config: AudioExtractionConfig,
    *,
    status: str,
    error: str | None = None,
    cleanup_dir: str | None = None,
) -> AudioPacket:
    return AudioPacket(
        audio_path="",
        sample_rate=audio_config.sample_rate,
        channels=audio_config.channels,
        codec=audio_config.codec,
        format=audio_config.format,
        duration_s=0.0,
        available=False,
        extraction_status=status,
        error=error,
        cleanup_dir=cleanup_dir,
    )


def extract_audio(
    video_meta: VideoMeta,
    audio_config: AudioExtractionConfig,
    output_dir: str | None = None,
    cleanup_dir: str | None = None,
) -> AudioPacket:
    """Extract a video's audio track to a file for ASR processing."""
    effective_cleanup_dir = (
        str(resolve_project_path(cleanup_dir, field_name="cleanup_dir"))
        if cleanup_dir is not None
        else None
    )

    if not video_meta.has_audio:
        return _unavailable_packet(
            audio_config,
            status="no_audio",
            cleanup_dir=effective_cleanup_dir,
        )

    path = Path(video_meta.source_path)
    if not path.exists():
        return _unavailable_packet(
            audio_config,
            status="failed",
            error=f"video file not found: {path}",
            cleanup_dir=effective_cleanup_dir,
        )

    if output_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="video_pipeline_audio_"))
        effective_cleanup_dir = str(temp_dir)
    else:
        temp_dir = resolve_project_path(output_dir, field_name="output_dir")

    audio_path = temp_dir / f"{video_meta.video_id}.{audio_config.format}"

    clip = None
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        VideoFileClip = _import_video_file_clip()
        clip = VideoFileClip(str(path), audio=True)

        if clip.audio is None:
            return _unavailable_packet(
                audio_config,
                status="no_audio",
                cleanup_dir=effective_cleanup_dir,
            )

        clip.audio.write_audiofile(
            str(audio_path),
            fps=audio_config.sample_rate,
            nbytes=2,
            codec=audio_config.codec,
            ffmpeg_params=["-ac", str(audio_config.channels)],
            logger=None,
        )

        if not audio_path.exists():
            return _unavailable_packet(
                audio_config,
                status="failed",
                error=f"audio extraction produced no file: {audio_path}",
                cleanup_dir=effective_cleanup_dir,
            )

        return AudioPacket(
            audio_path=str(audio_path.resolve()),
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            codec=audio_config.codec,
            format=audio_config.format,
            duration_s=video_meta.duration_s,
            available=True,
            extraction_status="extracted",
            cleanup_dir=effective_cleanup_dir,
        )
    except Exception as exc:
        return _unavailable_packet(
            audio_config,
            status="failed",
            error=str(exc),
            cleanup_dir=effective_cleanup_dir,
        )
    finally:
        if clip is not None:
            clip.close()
