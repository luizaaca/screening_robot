from __future__ import annotations

from collections import Counter
import logging
import urllib.request
from typing import Any

import cv2
import numpy as np

from video_pipeline.contracts import (
    FramePacket,
    FrameDetection,
    VideoMeta,
    PipelineConfig,
    DetectionWindow,
    Detection,
    DominantDetection,
    FrameAnalysisRecord,
)
from video_pipeline.paths import DEFAULT_HOLISTIC_LANDMARKER_PATH, resolve_project_path

from video_pipeline.processors.posture_geometry import (
    PostureAnalysisConfig,
    extract_landmark_sets,
    build_empty_landmark_sets,
    build_smoothers,
    smooth_landmark_sets,
    compute_reference_geometry,
    evaluate_rules,
)

logger = logging.getLogger(__name__)


def _harmonic_mean(score: float, support: float) -> float:
    if score <= 0.0 or support <= 0.0:
        return 0.0
    return (2.0 * score * support) / (score + support)


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        return mp, python, vision
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is required to run the pose mediapipe processor. "
            "Install project dependencies with `pip install -e .`."
        ) from exc


class PoseMediaPipeProcessor:
    """Processador de sinais posturais usando MediaPipe Holistic."""

    name: str = "pose_mediapipe"
    sample_fps: float = 2.0

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        self.config = config.pose
        self.pipeline_config = config
        self.video_meta = video_meta

        self.sample_fps = self.config.target_sample_fps
        self._sample_interval = max(1, round(video_meta.fps / self.sample_fps))

        # Build PostureAnalysisConfig from self.config (which is PoseMediaPipeConfig)
        self._analysis_config = PostureAnalysisConfig(
            visibility_threshold=self.config.visibility_threshold,
            ema_alpha=self.config.ema_alpha,
            ema_max_gap_frames=self.config.ema_max_gap_frames,
            min_shoulder_width_norm=self.config.min_shoulder_width_norm,
            display_score_threshold=self.config.display_score_threshold,
            rule_score_threshold=self.config.rule_score_threshold,
            frame_strong_score_threshold=self.config.frame_strong_score_threshold,
            head_drop_neutral_ratio=self.config.head_drop_neutral_ratio,
            head_drop_strong_ratio=self.config.head_drop_strong_ratio,
            hand_source_quality=self.config.hand_source_quality,
            rule_order=self.config.rule_order,
        )

        self._smoothers = build_smoothers(self._analysis_config)
        self._warnings: list[str] = []
        self._sampled_count = 0

        # Lazy import of mediapipe
        self._mp, self._python_tasks, self._vision_tasks = _import_mediapipe()

        # Resolve model path through the shared project/package path helper.
        model_path_str = self.config.model_asset_path
        if model_path_str is not None:
            model_path = resolve_project_path(
                model_path_str,
                field_name="pose.model_asset_path",
            )
        else:
            model_path = DEFAULT_HOLISTIC_LANDMARKER_PATH

        # Download model if not present
        if not model_path.exists():
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading model from {self.config.model_asset_url} to {model_path}...")
                urllib.request.urlretrieve(self.config.model_asset_url, str(model_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Holistic model asset from {self.config.model_asset_url} to {model_path}: {e}"
                )

        self._model_path = model_path

        # Create landmarker in VIDEO running mode
        base_options = self._python_tasks.BaseOptions(model_asset_path=str(model_path))
        options = self._vision_tasks.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=self._vision_tasks.RunningMode.VIDEO,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_suppression_threshold=self.config.min_face_suppression_threshold,
            min_face_landmarks_confidence=self.config.min_face_landmarks_confidence,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_suppression_threshold=self.config.min_pose_suppression_threshold,
            min_pose_landmarks_confidence=self.config.min_pose_landmarks_confidence,
            min_hand_landmarks_confidence=self.config.min_hand_landmarks_confidence,
            output_face_blendshapes=self.config.output_face_blendshapes,
            output_segmentation_mask=self.config.output_segmentation_mask,
        )
        try:
            self._landmarker = self._vision_tasks.HolisticLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create HolisticLandmarker from {model_path}: {exc}"
            ) from exc

    def wants_frame(self, frame_packet: FramePacket) -> bool:
        return frame_packet.frame_index % self._sample_interval == 0

    def process_frame(
        self,
        frame_packet: FramePacket,
        frame_bgr: object | None = None,
    ) -> FrameAnalysisRecord:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return FrameAnalysisRecord(
                timestamp_s=frame_packet.timestamp_s,
                frame_index=frame_packet.frame_index,
                status="skipped",
                detections=[],
                debug={"error": "missing_or_invalid_frame"},
            )

        timestamp_ms = int(round(frame_packet.timestamp_s * 1000.0))
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            holistic_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            raw_sets = extract_landmark_sets(holistic_result)
        except Exception as exc:
            self._warnings.append(f"Holistic inference failed at frame {frame_packet.frame_index}: {exc}")
            raw_sets = build_empty_landmark_sets()

        smoothed_sets = smooth_landmark_sets(frame_packet.frame_index, raw_sets, self._smoothers)
        geometry = compute_reference_geometry(smoothed_sets)
        rules = evaluate_rules(geometry, self._analysis_config)

        detections = []
        for name, rule in rules.items():
            if rule.get("passed_threshold"):
                detections.append(
                    FrameDetection(
                        timestamp_s=frame_packet.timestamp_s,
                        label=name,
                        score=rule.get("score", 0.0),
                    )
                )

        status = "scorable" if geometry.get("shoulder_reference_ok") else "insufficient_data"
        self._sampled_count += 1

        # Build detailed debug dictionary
        debug_metrics = {
            "shoulder_width": geometry.get("shoulder_width"),
            "head_height_ratio": geometry.get("head_height_ratio"),
            "head_tilt_score": geometry.get("head_tilt_score"),
            "neck_fraction": geometry.get("neck_fraction"),
            "coordinate_mode": geometry.get("coordinate_mode"),
            "rules": rules,
        }

        return FrameAnalysisRecord(
            timestamp_s=frame_packet.timestamp_s,
            frame_index=frame_packet.frame_index,
            status=status,
            detections=detections,
            debug=debug_metrics,
        )

    def aggregate(self, frame_records: list[FrameAnalysisRecord]) -> list[DetectionWindow]:
        duration = self.video_meta.duration_s
        window_s = self.pipeline_config.window_s
        stride_s = self.pipeline_config.stride_s

        windows: list[DetectionWindow] = []
        start = 0.0
        while start < duration:
            end = min(start + window_s, duration)
            win_records = [r for r in frame_records if start <= r.timestamp_s <= end]
            scorable_records = [r for r in win_records if r.status == "scorable"]
            scorable_count = len(scorable_records)

            scores_by_label: dict[str, list[float]] = {}
            frames_by_label: Counter[str] = Counter()
            for record in scorable_records:
                labels_in_frame = set()
                for detection in record.detections:
                    scores_by_label.setdefault(detection.label, []).append(detection.score)
                    labels_in_frame.add(detection.label)
                for label in labels_in_frame:
                    frames_by_label[label] += 1

            ranked_detections = []
            for label, scores in scores_by_label.items():
                if not scores:
                    continue
                # Score: peak (max) score in window
                max_score = max(scores)
                # Support: detection ratio in scorable frames
                support = frames_by_label[label] / scorable_count if scorable_count else 0.0

                detection = Detection(
                    label=label,
                    score=round(max_score, 3),
                    support=round(support, 3),
                )
                ranked_detections.append((detection, _harmonic_mean(max_score, support)))

            ranked_detections.sort(
                key=lambda item: (item[1], item[0].score, item[0].support),
                reverse=True,
            )
            detections_agg = [detection for detection, _rank_score in ranked_detections]

            dominant = None
            if detections_agg:
                dominant = DominantDetection(
                    label=detections_agg[0].label,
                    score=detections_agg[0].score,
                )

            windows.append(
                DetectionWindow(
                    start_s=start,
                    end_s=end,
                    detections=detections_agg,
                    dominant=dominant,
                )
            )

            if start + window_s >= duration:
                break
            start += stride_s

        return windows

    def debug_payload(self) -> dict[str, object]:
        return {
            "backend": "mediapipe",
            "simulated": False,
            "model_path": str(self._model_path) if hasattr(self, "_model_path") else None,
            "sample_fps": self.sample_fps,
            "warnings": self._warnings,
            "sampled_count": self._sampled_count,
        }

    def teardown(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
