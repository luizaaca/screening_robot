from pathlib import Path
from video_pipeline.contracts import VideoMeta, AudioPacket, AudioExtractionConfig


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
    )

def extract_audio(
    video_meta: VideoMeta, 
    audio_config: AudioExtractionConfig,
    output_dir: str | None = None
) -> AudioPacket:
    """Extrai a faixa de áudio do vídeo para processamento ASR."""
    if not video_meta.has_audio:
        return _unavailable_packet(audio_config, status="no_audio")

    path = Path(video_meta.source_path)
    if not path.exists():
        return _unavailable_packet(
            audio_config,
            status="failed",
            error=f"video file not found: {path}",
        )

    temp_dir = Path(output_dir) if output_dir else Path("./.tmp_audio")
    audio_path = temp_dir / f"{video_meta.video_id}.{audio_config.format}"

    clip = None
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        VideoFileClip = _import_video_file_clip()
        clip = VideoFileClip(str(path), audio=True)

        if clip.audio is None:
            return _unavailable_packet(audio_config, status="no_audio")

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
        )
    except Exception as exc:
        return _unavailable_packet(
            audio_config,
            status="failed",
            error=str(exc),
        )
    finally:
        if clip is not None:
            clip.close()
