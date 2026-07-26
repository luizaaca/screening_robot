import logging
from pathlib import Path

from video_pipeline.contracts import VideoMeta

logger = logging.getLogger(__name__)


def probe_video(video_path: str) -> VideoMeta:
    """Extract metadata from a readable video file."""

    path = Path(video_path)
    if not path.exists():
        logger.error("Video file not found while probing metadata: %s", path)
        raise FileNotFoundError(f"video file not found: {path}")

    try:
        import cv2
    except ImportError as exc:
        logger.error("OpenCV is required to probe video metadata: %s", path)
        raise ImportError(
            "opencv-python is required to probe video metadata. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            logger.error("OpenCV could not open video file while probing metadata: %s", path)
            raise ValueError(f"could not open video file: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        logger.error(
            "Video metadata is invalid for %s: fps=%s frame_count=%s width=%s height=%s",
            path,
            fps,
            frame_count,
            width,
            height,
        )
        raise ValueError(f"could not read valid video metadata from: {path}")

    has_audio = True
    try:
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip

        clip = VideoFileClip(str(path))
        try:
            has_audio = clip.audio is not None
        finally:
            clip.close()
    except ImportError:
        logger.warning(
            "MoviePy is unavailable; audio presence could not be probed for %s.",
            path,
        )
    except Exception as exc:
        logger.warning("Audio probing failed for %s: %s", path, exc)

    return VideoMeta(
        video_id=path.stem,
        source_path=str(path.resolve()),
        duration_s=round(frame_count / fps, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
        has_audio=has_audio,
    )
