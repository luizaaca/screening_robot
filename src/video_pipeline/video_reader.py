import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from video_pipeline.contracts import FramePacket, VideoMeta

logger = logging.getLogger(__name__)


def read_frames(video_meta: VideoMeta) -> Iterator[Tuple[FramePacket, Optional[Any]]]:
    """Read real video frames in sequential order."""

    path = Path(video_meta.source_path)
    if not path.exists():
        logger.error("Video file not found while reading frames: %s", path)
        raise FileNotFoundError(f"video file not found: {path}")

    try:
        import cv2
    except ImportError as exc:
        logger.error("OpenCV is required to read video frames: %s", path)
        raise ImportError(
            "opencv-python is required to read video frames. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            logger.error("OpenCV could not open video file while reading frames: %s", path)
            raise ValueError(f"could not open video file: {path}")

        fps = video_meta.fps if video_meta.fps > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error("Video FPS is invalid while reading frames: %s fps=%s", path, fps)
            raise ValueError(f"could not read a valid FPS from video file: {path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            max_dim = 480
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            current_time = frame_idx / fps
            packet = FramePacket(timestamp_s=round(current_time, 3), frame_index=frame_idx)
            yield packet, frame
            frame_idx += 1

        if frame_idx == 0:
            logger.error("No frames could be read from video file: %s", path)
            raise ValueError(f"no frames could be read from video file: {path}")
    finally:
        cap.release()
