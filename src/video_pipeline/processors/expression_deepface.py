from collections import Counter
from typing import Any

from video_pipeline.contracts import (
    Detection,
    DetectionWindow,
    DominantDetection,
    FrameAnalysisRecord,
    FrameDetection,
    FramePacket,
    PipelineConfig,
    VideoMeta,
)


def _import_deepface() -> Any:
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise ImportError(
            "DeepFace is required to run the expression processor. "
            "Install the optional video dependencies with "
            "`pip install screening-robot-agent[video]`."
        ) from exc
    return DeepFace


def _normalize_score(raw_score: object | None) -> float | None:
    if raw_score is None:
        return None

    score = float(raw_score)
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _region_area(face: dict[str, Any]) -> int:
    region = face.get("region") or {}
    width = int(region.get("w") or 0)
    height = int(region.get("h") or 0)
    return width * height


def _select_largest_face(analysis_result: object) -> tuple[dict[str, Any] | None, int]:
    if isinstance(analysis_result, dict):
        faces = [analysis_result]
    elif isinstance(analysis_result, list):
        faces = [item for item in analysis_result if isinstance(item, dict)]
    else:
        faces = []

    valid_faces = [face for face in faces if _region_area(face) > 0]
    if not valid_faces:
        return None, 0

    return max(valid_faces, key=_region_area), len(valid_faces)


def _harmonic_mean(score: float, support: float) -> float:
    if score <= 0.0 or support <= 0.0:
        return 0.0
    return (2.0 * score * support) / (score + support)


class ExpressionDeepFaceProcessor:
    """DeepFace-based facial expression processor."""

    name: str = "expression_deepface"

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        self.config = config.expression
        self.pipeline_config = config
        self.video_meta = video_meta

        self._deepface = _import_deepface()
        self._model = None
        self._warnings: list[str] = []
        self._frames: list[dict[str, object]] = []
        self._counters: Counter[str] = Counter(
            {
                "total_frames": 0,
                "scorable_frames": 0,
                "uncertain_frames": 0,
                "no_detection_frames": 0,
                "insufficient_data_frames": 0,
            }
        )

        try:
            self._model = self._deepface.build_model(
                self.config.build_model_name,
                task=self.config.build_model_task,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the DeepFace emotion model. "
                "Verify the optional video dependencies and model download."
            ) from exc

    def wants_frame(self, frame_packet: FramePacket) -> bool:
        return True

    def process_frame(
        self,
        frame_packet: FramePacket,
        frame_bgr: object | None = None,
    ) -> FrameAnalysisRecord:
        if frame_bgr is None:
            return self._record(
                frame_packet=frame_packet,
                status="insufficient_data",
                reason="missing_frame",
            )

        try:
            analysis_result = self._deepface.analyze(
                img_path=frame_bgr,
                actions=list(self.config.actions),
                detector_backend=self.config.detector_backend,
                enforce_detection=self.config.enforce_detection,
                align=self.config.align,
                silent=self.config.silent,
            )
        except Exception as exc:
            return self._record(
                frame_packet=frame_packet,
                status="insufficient_data",
                reason="deepface_analyze_failed",
                extra_debug={"error": str(exc)},
            )

        face, face_count = _select_largest_face(analysis_result)
        if face is None:
            return self._record(
                frame_packet=frame_packet,
                status="no_detection",
                reason="no_valid_face",
                extra_debug={"face_count": 0},
            )

        raw_label = str(face.get("dominant_emotion") or "").lower().strip()
        mapped_label = self.config.label_map.get(raw_label)
        emotion_scores = face.get("emotion") or {}
        confidence = _normalize_score(emotion_scores.get(raw_label))
        region = self._region_payload(face)

        debug = {
            "face_count": face_count,
            "raw_label": raw_label,
            "mapped_label": mapped_label,
            "confidence": confidence,
            "region": region,
        }

        if mapped_label is None:
            return self._record(
                frame_packet=frame_packet,
                status="uncertain",
                reason="unknown_label",
                extra_debug=debug,
            )

        if confidence is None:
            return self._record(
                frame_packet=frame_packet,
                status="uncertain",
                reason="missing_score",
                extra_debug=debug,
            )

        if confidence < self.config.min_confidence:
            return self._record(
                frame_packet=frame_packet,
                status="uncertain",
                reason="low_confidence",
                extra_debug=debug,
            )

        detection = FrameDetection(
            timestamp_s=frame_packet.timestamp_s,
            label=mapped_label,
            score=confidence,
        )
        return self._record(
            frame_packet=frame_packet,
            status="scorable",
            detection=detection,
            reason="accepted",
            extra_debug=debug,
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
                score = sum(scores) / len(scores)
                support = frames_by_label[label] / scorable_count if scorable_count else 0.0
                detection = Detection(
                    label=label,
                    score=round(score, 3),
                    support=round(support, 3),
                )
                ranked_detections.append((detection, _harmonic_mean(score, support)))

            ranked_detections.sort(
                key=lambda item: (item[1], item[0].score, item[0].support),
                reverse=True,
            )
            detections = [detection for detection, _rank_score in ranked_detections]
            dominant = None
            if detections:
                dominant = DominantDetection(
                    label=detections[0].label,
                    score=detections[0].score,
                )

            windows.append(
                DetectionWindow(
                    start_s=start,
                    end_s=end,
                    detections=detections,
                    dominant=dominant,
                )
            )

            if start + window_s >= duration:
                break
            start += stride_s

        return windows

    def debug_payload(self) -> dict[str, object]:
        segments, percentages = self._segments_and_percentages()
        return {
            "backend": self.config.detector_backend,
            "parameters": {
                "actions": list(self.config.actions),
                "min_confidence": self.config.min_confidence,
                "min_segment_duration_s": self.config.min_segment_duration_s,
                "detector_backend": self.config.detector_backend,
                "enforce_detection": self.config.enforce_detection,
                "align": self.config.align,
                "silent": self.config.silent,
                "build_model_name": self.config.build_model_name,
                "build_model_task": self.config.build_model_task,
                "label_map": dict(self.config.label_map),
            },
            "counters": dict(self._counters),
            "warnings": list(self._warnings),
            "segments": segments,
            "percentages": percentages,
            "frames": list(self._frames),
        }

    def _record(
        self,
        frame_packet: FramePacket,
        status: str,
        detection: FrameDetection | None = None,
        reason: str | None = None,
        extra_debug: dict[str, object] | None = None,
    ) -> FrameAnalysisRecord:
        self._counters["total_frames"] += 1
        self._counters[f"{status}_frames"] += 1

        detections = [detection] if detection is not None else []
        debug = {"reason": reason} if reason else {}
        if extra_debug:
            debug.update(extra_debug)

        compact = {
            "frame_index": frame_packet.frame_index,
            "timestamp_s": frame_packet.timestamp_s,
            "status": status,
            "label": detection.label if detection is not None else None,
            "score": detection.score if detection is not None else None,
            "reason": reason,
        }
        for key in ("raw_label", "mapped_label", "confidence", "face_count"):
            if key in debug:
                compact[key] = debug[key]
        self._frames.append(compact)

        return FrameAnalysisRecord(
            timestamp_s=frame_packet.timestamp_s,
            frame_index=frame_packet.frame_index,
            status=status,
            detections=detections,
            debug=debug,
        )

    @staticmethod
    def _region_payload(face: dict[str, Any]) -> dict[str, int]:
        region = face.get("region") or {}
        return {
            "x": int(region.get("x") or 0),
            "y": int(region.get("y") or 0),
            "w": int(region.get("w") or 0),
            "h": int(region.get("h") or 0),
        }

    def _estimated_frame_duration_s(self) -> float:
        fps = self.video_meta.fps
        if fps > 0:
            return 1.0 / fps
        if self._frames and self.video_meta.duration_s > 0:
            return self.video_meta.duration_s / len(self._frames)
        return 0.0

    def _segments_and_percentages(self) -> tuple[list[dict[str, object]], dict[str, float]]:
        if not self._frames or self.video_meta.duration_s <= 0:
            return [], {}

        frame_duration_s = self._estimated_frame_duration_s()
        segments: list[dict[str, object]] = []
        current_label: str | None = None
        current_start_s = 0.0

        def close_segment(end_s: float) -> None:
            if current_label is None:
                return
            start_s = current_start_s
            duration_s = max(0.0, end_s - start_s)
            if duration_s >= self.config.min_segment_duration_s:
                segments.append(
                    {
                        "label": current_label,
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                    }
                )

        for frame in self._frames:
            timestamp_s = float(frame["timestamp_s"])
            label = frame["label"] if isinstance(frame.get("label"), str) else None
            if label == current_label:
                continue

            close_segment(timestamp_s)
            current_label = label
            current_start_s = timestamp_s

        last_timestamp_s = float(self._frames[-1]["timestamp_s"])
        final_end_s = min(self.video_meta.duration_s, last_timestamp_s + frame_duration_s)
        close_segment(final_end_s)

        duration_by_label: Counter[str] = Counter()
        for segment in segments:
            label = str(segment["label"])
            duration_by_label[label] += float(segment["end_s"]) - float(segment["start_s"])

        percentages = {
            label: round((duration / self.video_meta.duration_s) * 100.0, 1)
            for label, duration in sorted(
                duration_by_label.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        return segments, percentages
