"""Posture geometry analysis and rule evaluation. Ported from mediapipe_video_posture_timeline_v3.ipynb."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostureAnalysisConfig:
    """Central configuration for all posture geometry and rule evaluation."""

    visibility_threshold: float = 0.50
    ema_alpha: float = 0.35
    ema_max_gap_frames: int = 3
    min_shoulder_width_norm: float = 0.02
    display_score_threshold: float = 0.35
    rule_score_threshold: float = 0.35
    frame_strong_score_threshold: float = 0.70
    head_drop_neutral_ratio: float = 0.42
    head_drop_strong_ratio: float = 0.26

    hand_source_quality: dict[str, float] = field(default_factory=lambda: {
        'holistic_hand': 1.00,
        'pose_wrist_fallback': 0.75,
        'arm_proxy': 0.45,
        'unknown': 0.00,
    })

    rule_order: tuple[str, ...] = (
        'hand_on_head',
        'hand_on_neck',
        'hand_on_chest',
        'head_down',
        'forward_head',
        'rounded_shoulders_or_asymmetry',
    )


# ---------------------------------------------------------------------------
# Coordinate mode constants
# ---------------------------------------------------------------------------

COORDINATE_MODE_WORLD = 'mixed_world_pose_face_hands'
COORDINATE_MODE_2D = 'image_2d_only'


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointData:
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    space: str = 'image'


# ---------------------------------------------------------------------------
# Landmark index dictionaries (truly constant)
# ---------------------------------------------------------------------------

POSE_LANDMARK_INDEX: dict[str, int] = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
}

HAND_LANDMARK_INDEX: dict[str, int] = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}

FACE_LANDMARK_INDEX: dict[str, int] = {
    'face_forehead': 10,
    'face_nose_tip': 1,
    'face_nose_bridge': 6,
    'face_chin': 152,
    'face_left_eye_outer': 33,
    'face_left_eye_inner': 133,
    'face_right_eye_inner': 362,
    'face_right_eye_outer': 263,
    'face_mouth_left': 61,
    'face_mouth_right': 291,
    'face_mouth_upper': 13,
    'face_mouth_lower': 14,
    'face_lower_lip': 17,
    'face_jaw_left': 234,
    'face_jaw_right': 454,
    'face_cheek_left': 205,
    'face_cheek_right': 425,
}

FACE_REGION_NAMES: list[str] = [
    'face_nose_tip',
    'face_nose_bridge',
    'face_chin',
    'face_mouth_left',
    'face_mouth_right',
    'face_mouth_upper',
    'face_mouth_lower',
    'face_lower_lip',
    'face_jaw_left',
    'face_jaw_right',
    'face_cheek_left',
    'face_cheek_right',
    'face_left_eye_outer',
    'face_right_eye_outer',
]

FACE_PITCH_PNP_NAMES: list[str] = [
    'face_nose_tip',
    'face_chin',
    'face_left_eye_outer',
    'face_right_eye_outer',
    'face_mouth_left',
    'face_mouth_right',
]


# ---------------------------------------------------------------------------
# Raw landmark helpers
# ---------------------------------------------------------------------------

def make_point(landmark: Any, *, space: str, visibility: float | None = None) -> PointData:
    """Convert a raw MediaPipe landmark into a stable, serialisable structure."""
    landmark_visibility = getattr(landmark, 'visibility', visibility)
    return PointData(
        x=float(landmark.x),
        y=float(landmark.y),
        z=float(getattr(landmark, 'z', 0.0)),
        visibility=float(landmark_visibility) if landmark_visibility is not None else None,
        space=space,
    )


def normalize_landmark_container(raw_landmarks: Any) -> list[Any]:
    """Normalise different MediaPipe Tasks API containers into a plain landmark list."""
    if raw_landmarks is None:
        return []
    if hasattr(raw_landmarks, 'landmark'):
        raw_landmarks = getattr(raw_landmarks, 'landmark')
    if not raw_landmarks:
        return []

    first_item = raw_landmarks[0]
    if hasattr(first_item, 'x'):
        return list(raw_landmarks)
    if hasattr(first_item, 'landmark'):
        return list(first_item.landmark)
    if isinstance(first_item, (list, tuple)):
        return list(first_item)
    return list(raw_landmarks)


def landmark_list_to_points(
    landmark_list: Any,
    name_to_index: Mapping[str, int],
    *,
    space: str,
    require_visibility: bool = False,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, PointData]:
    """Extract named landmarks and apply a visibility filter when available.

    Args:
        landmark_list: Raw landmark container from MediaPipe.
        name_to_index: Mapping of landmark names to indices.
        space: Coordinate space label (``'image'`` or ``'world'``).
        require_visibility: Whether to discard landmarks below the threshold.
        config: Analysis configuration (uses ``config.visibility_threshold``).

    Returns:
        A dictionary of named :class:`PointData` instances.
    """
    if config is None:
        config = PostureAnalysisConfig()
    visibility_threshold = config.visibility_threshold

    landmarks = normalize_landmark_container(landmark_list)
    points: dict[str, PointData] = {}
    for name, index in name_to_index.items():
        if index >= len(landmarks):
            continue
        landmark = landmarks[index]
        visibility = getattr(landmark, 'visibility', None)
        if require_visibility and visibility is not None and visibility < visibility_threshold:
            continue
        points[name] = make_point(landmark, space=space, visibility=visibility)
    return points


def extract_landmark_sets(
    holistic_result: Any,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, dict[str, PointData]]:
    """Read a HolisticLandmarker result and organise pose, face and hands by group.

    Args:
        holistic_result: The raw holistic result object.
        config: Analysis configuration.

    Returns:
        A dictionary of named landmark groups.
    """
    if holistic_result is None:
        return build_empty_landmark_sets()
    if config is None:
        config = PostureAnalysisConfig()

    pose_landmarks = getattr(holistic_result, 'pose_landmarks', None)
    pose_world_landmarks = getattr(holistic_result, 'pose_world_landmarks', None)
    face_landmarks = getattr(holistic_result, 'face_landmarks', None)
    left_hand_landmarks = getattr(holistic_result, 'left_hand_landmarks', None)
    right_hand_landmarks = getattr(holistic_result, 'right_hand_landmarks', None)
    left_hand_world_landmarks = getattr(holistic_result, 'left_hand_world_landmarks', None)
    right_hand_world_landmarks = getattr(holistic_result, 'right_hand_world_landmarks', None)

    return {
        'pose_2d': landmark_list_to_points(
            pose_landmarks,
            POSE_LANDMARK_INDEX,
            space='image',
            require_visibility=True,
            config=config,
        ),
        'pose_world': landmark_list_to_points(
            pose_world_landmarks,
            POSE_LANDMARK_INDEX,
            space='world',
            require_visibility=False,
            config=config,
        ),
        'face_2d': landmark_list_to_points(
            face_landmarks,
            FACE_LANDMARK_INDEX,
            space='image',
            require_visibility=False,
            config=config,
        ),
        'left_hand_2d': landmark_list_to_points(
            left_hand_landmarks,
            HAND_LANDMARK_INDEX,
            space='image',
            require_visibility=False,
            config=config,
        ),
        'right_hand_2d': landmark_list_to_points(
            right_hand_landmarks,
            HAND_LANDMARK_INDEX,
            space='image',
            require_visibility=False,
            config=config,
        ),
        'left_hand_world': landmark_list_to_points(
            left_hand_world_landmarks,
            HAND_LANDMARK_INDEX,
            space='world',
            require_visibility=False,
            config=config,
        ),
        'right_hand_world': landmark_list_to_points(
            right_hand_world_landmarks,
            HAND_LANDMARK_INDEX,
            space='world',
            require_visibility=False,
            config=config,
        ),
    }


def build_empty_landmark_sets() -> dict[str, dict[str, PointData]]:
    """Return the same structure as the extractor but with no points."""
    return {
        'pose_2d': {},
        'pose_world': {},
        'face_2d': {},
        'left_hand_2d': {},
        'right_hand_2d': {},
        'left_hand_world': {},
        'right_hand_world': {},
    }


# ---------------------------------------------------------------------------
# EMA temporal smoothing
# ---------------------------------------------------------------------------

def blend_optional(
    current_value: float | None,
    previous_value: float | None,
    alpha: float,
) -> float | None:
    """Blend two optional floats using exponential moving average weights."""
    if current_value is None and previous_value is None:
        return None
    if current_value is None:
        return previous_value
    if previous_value is None:
        return current_value
    return (alpha * current_value) + ((1.0 - alpha) * previous_value)


class EMASmoother:
    """Per-landmark EMA smoother that resets after a configurable gap."""

    def __init__(
        self,
        alpha: float | None = None,
        max_gap_frames: int | None = None,
        config: PostureAnalysisConfig | None = None,
    ) -> None:
        if config is None:
            config = PostureAnalysisConfig()
        self.alpha = alpha if alpha is not None else config.ema_alpha
        self.max_gap_frames = max_gap_frames if max_gap_frames is not None else config.ema_max_gap_frames
        self.state: dict[str, dict[str, Any]] = {}

    def update(self, frame_idx: int, points: Mapping[str, PointData]) -> dict[str, PointData]:
        """Update a set of points and return the smoothed version for this frame."""
        smoothed: dict[str, PointData] = {}
        for name, point in points.items():
            previous = self.state.get(name)
            if previous is None or (frame_idx - previous['frame_idx']) > self.max_gap_frames:
                smoothed_point = point
            else:
                previous_point = previous['point']
                smoothed_point = PointData(
                    x=(self.alpha * point.x) + ((1.0 - self.alpha) * previous_point.x),
                    y=(self.alpha * point.y) + ((1.0 - self.alpha) * previous_point.y),
                    z=blend_optional(point.z, previous_point.z, self.alpha),
                    visibility=point.visibility,
                    space=point.space,
                )

            self.state[name] = {'frame_idx': frame_idx, 'point': smoothed_point}
            smoothed[name] = smoothed_point
        return smoothed


def build_smoothers(config: PostureAnalysisConfig | None = None) -> dict[str, EMASmoother]:
    """Create one EMA smoother per landmark group.

    Args:
        config: Analysis configuration.

    Returns:
        A dictionary of named :class:`EMASmoother` instances.
    """
    if config is None:
        config = PostureAnalysisConfig()
    return {
        'pose_2d': EMASmoother(config=config),
        'pose_world': EMASmoother(config=config),
        'face_2d': EMASmoother(config=config),
        'left_hand_2d': EMASmoother(config=config),
        'right_hand_2d': EMASmoother(config=config),
        'left_hand_world': EMASmoother(config=config),
        'right_hand_world': EMASmoother(config=config),
    }


def smooth_landmark_sets(
    frame_idx: int,
    raw_sets: Mapping[str, dict[str, PointData]],
    smoothers: Mapping[str, EMASmoother],
) -> dict[str, dict[str, PointData]]:
    """Apply all expected smoothers, keeping empty keys for missing groups."""
    return {
        name: smoothers[name].update(frame_idx, raw_sets.get(name, {}))
        for name in [
            'pose_2d',
            'pose_world',
            'face_2d',
            'left_hand_2d',
            'right_hand_2d',
            'left_hand_world',
            'right_hand_world',
        ]
    }


# ---------------------------------------------------------------------------
# Pure geometric helpers
# ---------------------------------------------------------------------------

def optional_round(value: float | None, digits: int = 4) -> float | None:
    """Round an optional float value for JSON/debug output.

    Args:
        value: Value to round when present.
        digits: Number of decimal digits to keep.

    Returns:
        The rounded float or ``None`` when the input is missing.
    """
    if value is None:
        return None
    return round(float(value), digits)


def clamp_value(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval.

    Args:
        value: Value to clamp.
        low: Lower inclusive bound.
        high: Upper inclusive bound.

    Returns:
        The clamped value.
    """
    return max(low, min(high, float(value)))


def point_distance(point_a: PointData | None, point_b: PointData | None) -> float | None:
    """Compute the Euclidean distance between two points."""
    if point_a is None or point_b is None:
        return None
    if point_a.space == point_b.space == 'world' and point_a.z is not None and point_b.z is not None:
        return math.dist((point_a.x, point_a.y, point_a.z), (point_b.x, point_b.y, point_b.z))
    return math.dist((point_a.x, point_a.y), (point_b.x, point_b.y))


def midpoint(point_a: PointData | None, point_b: PointData | None) -> PointData | None:
    """Return the midpoint between two points."""
    if point_a is None or point_b is None:
        return None
    z_value = None
    if point_a.z is not None and point_b.z is not None:
        z_value = (point_a.z + point_b.z) / 2.0
    visibility = None
    if point_a.visibility is not None and point_b.visibility is not None:
        visibility = min(point_a.visibility, point_b.visibility)
    return PointData(
        x=(point_a.x + point_b.x) / 2.0,
        y=(point_a.y + point_b.y) / 2.0,
        z=z_value,
        visibility=visibility,
        space=point_a.space,
    )


def offset_point(point: PointData | None, dx: float = 0.0, dy: float = 0.0) -> PointData | None:
    """Offset a point by a 2D delta."""
    if point is None:
        return None
    return PointData(
        x=point.x + dx,
        y=point.y + dy,
        z=point.z,
        visibility=point.visibility,
        space=point.space,
    )


def centroid(points: Iterable[PointData | None], *, fallback_space: str = 'image') -> PointData | None:
    """Compute the centroid of valid points."""
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    z_values = [point.z for point in valid if point.z is not None]
    visibility_values = [point.visibility for point in valid if point.visibility is not None]
    return PointData(
        x=float(np.mean([point.x for point in valid])),
        y=float(np.mean([point.y for point in valid])),
        z=float(np.mean(z_values)) if z_values else None,
        visibility=float(min(visibility_values)) if visibility_values else None,
        space=valid[0].space if valid else fallback_space,
    )


def valid_points(points: Iterable[PointData | None]) -> list[PointData]:
    """Filter out missing points from an iterable."""
    return [point for point in points if point is not None]


def score_ratio_below(ratio: float | None, strong_ratio: float, weak_ratio: float) -> float:
    """Score a ratio where smaller values indicate stronger evidence."""
    if ratio is None:
        return 0.0
    if ratio <= strong_ratio:
        return 1.0
    if ratio >= weak_ratio:
        return 0.0
    return 1.0 - ((ratio - strong_ratio) / max(weak_ratio - strong_ratio, 1e-6))


def score_ratio_above(ratio: float | None, weak_ratio: float, strong_ratio: float) -> float:
    """Score a ratio where larger values indicate stronger evidence."""
    if ratio is None:
        return 0.0
    if ratio <= weak_ratio:
        return 0.0
    if ratio >= strong_ratio:
        return 1.0
    return (ratio - weak_ratio) / max(strong_ratio - weak_ratio, 1e-6)


def weighted_mean_score(components: Iterable[tuple[float | None, float]]) -> float | None:
    """Compute a weighted mean over available score components."""
    available = [(score, weight) for score, weight in components if score is not None and weight > 0]
    if not available:
        return None
    total_weight = sum(weight for _, weight in available)
    return sum(float(score) * weight for score, weight in available) / total_weight


# ---------------------------------------------------------------------------
# Face pitch estimation
# ---------------------------------------------------------------------------

def estimate_face_pitch_degrees(face_2d: Mapping[str, PointData]) -> float | None:
    """Estimate face pitch in degrees from a small set of face landmarks."""
    if any(name not in face_2d for name in FACE_PITCH_PNP_NAMES):
        return None

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)
    image_points = np.array(
        [[face_2d[name].x, face_2d[name].y] for name in FACE_PITCH_PNP_NAMES],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        decomposition = cv2.RQDecomp3x3(rotation_matrix)
        angles = decomposition[0]
        pitch_degrees = float(angles[0])
        if not math.isfinite(pitch_degrees):
            return None
        return pitch_degrees
    except Exception:
        return None


def classify_face_pitch_direction(face_pitch_degrees: float | None) -> str:
    """Classify the face pitch direction for head-down reasoning."""
    if face_pitch_degrees is None or not math.isfinite(face_pitch_degrees):
        return 'unknown'
    if face_pitch_degrees <= -12.0:
        return 'down'
    if face_pitch_degrees >= 12.0:
        return 'up'
    return 'neutral'


def compute_directional_face_pitch_score(
    face_pitch_degrees: float | None,
) -> tuple[float | None, bool, str]:
    """Convert raw face pitch into a directional head-down score."""
    if face_pitch_degrees is None or not math.isfinite(face_pitch_degrees):
        return None, False, 'unknown'
    if abs(face_pitch_degrees) > 60.0:
        return None, False, 'implausible'
    direction = classify_face_pitch_direction(face_pitch_degrees)
    if direction != 'down':
        return 0.0, True, direction
    score = clamp_value((abs(face_pitch_degrees) - 12.0) / 18.0)
    return score, True, direction


# ---------------------------------------------------------------------------
# Neck height and head-width estimation
# ---------------------------------------------------------------------------

def compute_neck_height_from_shoulders(
    shoulder_width: float | None,
    head_tilt_score: float | None,
    head_drop_score: float | None,
) -> float | None:
    """Estimate a compact neck height anchored on shoulder width.

    Args:
        shoulder_width: Normalized shoulder width.
        head_tilt_score: Current head-down score estimate.
        head_drop_score: Static nose-to-shoulder drop support.

    Returns:
        A compact neck height estimate or ``None`` when shoulders are unavailable.
    """
    if shoulder_width is None or shoulder_width <= 0:
        return None
    stable_base = shoulder_width * 0.18
    tilt_bonus = shoulder_width * 0.025 * float(head_tilt_score or 0.0)
    drop_bonus = shoulder_width * 0.015 * float(head_drop_score or 0.0)
    return clamp_value(stable_base + tilt_bonus + drop_bonus, shoulder_width * 0.14, shoulder_width * 0.24)


def estimate_head_width(
    face_2d: Mapping[str, PointData],
    pose_2d: Mapping[str, PointData],
    shoulder_width: float | None,
) -> float | None:
    """Estimate head width from stable bilateral facial anchors.

    Args:
        face_2d: Named 2D face landmarks.
        pose_2d: Named 2D pose landmarks.
        shoulder_width: Normalized shoulder width.

    Returns:
        A bounded head-width estimate or ``None`` when normalization is unavailable.
    """
    if shoulder_width is None or shoulder_width <= 0:
        return None

    candidate_specs: list[tuple[PointData | None, PointData | None, float]] = [
        (pose_2d.get('left_ear'), pose_2d.get('right_ear'), 1.00),
        (face_2d.get('face_cheek_left'), face_2d.get('face_cheek_right'), 1.05),
        (face_2d.get('face_jaw_left'), face_2d.get('face_jaw_right'), 1.10),
        (face_2d.get('face_left_eye_outer'), face_2d.get('face_right_eye_outer'), 1.55),
    ]
    candidates: list[float] = []
    for left_point, right_point, scale in candidate_specs:
        distance = point_distance(left_point, right_point)
        if distance is not None:
            candidates.append(distance * scale)

    if not candidates:
        return clamp_value(shoulder_width * 0.36, shoulder_width * 0.26, shoulder_width * 0.52)

    estimate = float(np.median(candidates))
    return clamp_value(estimate, shoulder_width * 0.26, shoulder_width * 0.52)


# ---------------------------------------------------------------------------
# Neck pose mode inference
# ---------------------------------------------------------------------------

def infer_neck_pose_mode(
    face_2d: Mapping[str, PointData],
    pose_2d: Mapping[str, PointData],
    head_tilt_score: float | None,
    shoulder_width: float | None,
) -> dict[str, float | str | None]:
    """Infer whether the current frame needs a compact or posterior neck profile.

    Args:
        face_2d: Named 2D face landmarks.
        pose_2d: Named 2D pose landmarks.
        head_tilt_score: Current head-down support score.
        shoulder_width: Normalized shoulder width.

    Returns:
        A dictionary with pose mode and coarse profile-support metrics.
    """
    left_points = valid_points([
        pose_2d.get('left_ear'),
        pose_2d.get('left_eye_outer'),
        face_2d.get('face_left_eye_outer'),
        face_2d.get('face_cheek_left'),
        face_2d.get('face_jaw_left'),
    ])
    right_points = valid_points([
        pose_2d.get('right_ear'),
        pose_2d.get('right_eye_outer'),
        face_2d.get('face_right_eye_outer'),
        face_2d.get('face_cheek_right'),
        face_2d.get('face_jaw_right'),
    ])
    visible_pairs = len(left_points) + len(right_points)
    lateral_face_balance_ratio = 1.0
    if visible_pairs:
        lateral_face_balance_ratio = 1.0 - (abs(len(left_points) - len(right_points)) / visible_pairs)

    profile_support_score = 0.0
    left_ear = pose_2d.get('left_ear')
    right_ear = pose_2d.get('right_ear')
    if (left_ear is None) ^ (right_ear is None):
        profile_support_score += 0.45
    if abs(len(left_points) - len(right_points)) >= 2:
        profile_support_score += 0.20
    if head_tilt_score is not None and head_tilt_score >= 0.45:
        profile_support_score += 0.20
    if shoulder_width is not None and shoulder_width > 0:
        ear_span = point_distance(left_ear, right_ear)
        if ear_span is not None and (ear_span / shoulder_width) <= 0.22:
            profile_support_score += 0.15

    profile_support_score = clamp_value(profile_support_score)
    neck_pose_mode = 'posterior_nape_extended' if profile_support_score >= 0.55 else 'frontal_compact'
    return {
        'neck_pose_mode': neck_pose_mode,
        'profile_support_score': profile_support_score,
        'lateral_face_balance_ratio': lateral_face_balance_ratio,
    }


# ---------------------------------------------------------------------------
# Face vertical limits
# ---------------------------------------------------------------------------

def compute_face_vertical_limits(
    face_2d: Mapping[str, PointData],
    pose_2d: Mapping[str, PointData],
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
) -> dict[str, float | None]:
    """Compute stable vertical anchors for face, head, neck, and nape separation.

    Args:
        face_2d: Named 2D face landmarks.
        pose_2d: Named 2D pose landmarks.
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.

    Returns:
        Vertical anchor lines used by head, face, and neck rules.
    """
    if shoulder_mid is None or shoulder_width is None or shoulder_width <= 0:
        return {
            'nose_line_y': None,
            'eye_line_y': None,
            'mouth_line_y': None,
            'chin_line_y': None,
            'forehead_line_y': None,
            'ear_base_y': None,
        }

    nose_candidates = valid_points([
        pose_2d.get('nose'),
        face_2d.get('face_nose_tip'),
    ])
    eye_candidates = valid_points([
        face_2d.get('face_left_eye_outer'),
        face_2d.get('face_right_eye_outer'),
        pose_2d.get('left_eye_outer'),
        pose_2d.get('right_eye_outer'),
    ])
    mouth_candidates = valid_points([
        face_2d.get('face_mouth_upper'),
        face_2d.get('face_mouth_lower'),
        face_2d.get('face_mouth_left'),
        face_2d.get('face_mouth_right'),
        pose_2d.get('mouth_left'),
        pose_2d.get('mouth_right'),
    ])
    chin_candidates = valid_points([
        face_2d.get('face_chin'),
        face_2d.get('face_lower_lip'),
    ])
    forehead_candidates = valid_points([
        face_2d.get('face_forehead'),
    ])
    ear_candidates = valid_points([
        pose_2d.get('left_ear'),
        pose_2d.get('right_ear'),
    ])

    nose_line_y = float(np.mean([point.y for point in nose_candidates])) if nose_candidates else None
    eye_line_y = float(np.mean([point.y for point in eye_candidates])) if eye_candidates else None
    mouth_line_y = float(np.mean([point.y for point in mouth_candidates])) if mouth_candidates else None
    chin_line_y = float(np.mean([point.y for point in chin_candidates])) if chin_candidates else None
    forehead_line_y = float(np.mean([point.y for point in forehead_candidates])) if forehead_candidates else None
    ear_base_y = float(np.mean([point.y for point in ear_candidates])) if ear_candidates else None

    if eye_line_y is None:
        eye_line_y = shoulder_mid.y - (0.42 * shoulder_width)
    if nose_line_y is None:
        nose_line_y = eye_line_y + (0.10 * shoulder_width)
    if mouth_line_y is None:
        mouth_line_y = shoulder_mid.y - (0.24 * shoulder_width)
    if chin_line_y is None:
        chin_line_y = shoulder_mid.y - (0.14 * shoulder_width)
    if forehead_line_y is None:
        forehead_line_y = eye_line_y - (0.12 * shoulder_width)
    if ear_base_y is None:
        ear_base_y = min(chin_line_y - (0.03 * shoulder_width), shoulder_mid.y - (0.17 * shoulder_width))

    return {
        'nose_line_y': nose_line_y,
        'eye_line_y': eye_line_y,
        'mouth_line_y': mouth_line_y,
        'chin_line_y': chin_line_y,
        'forehead_line_y': forehead_line_y,
        'ear_base_y': ear_base_y,
    }


# ---------------------------------------------------------------------------
# Head tilt components
# ---------------------------------------------------------------------------

def compute_head_tilt_components(
    pose_2d: Mapping[str, PointData],
    face_2d: Mapping[str, PointData],
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
    head_height_ratio: float | None,
    face_limits: Mapping[str, float | None],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Combine static anatomical cues and directional face pitch into a head-down score.

    Args:
        pose_2d: Named 2D pose landmarks.
        face_2d: Named 2D face landmarks.
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.
        head_height_ratio: Ratio between shoulder-to-head height and shoulder width.
        face_limits: Stable vertical facial anchors.
        config: Analysis configuration.

    Returns:
        A dictionary with static, directional, and agreement head-down cues.
    """
    if config is None:
        config = PostureAnalysisConfig()

    face_pitch_degrees = estimate_face_pitch_degrees(face_2d)
    face_pitch_score, face_pitch_valid, face_pitch_direction = compute_directional_face_pitch_score(face_pitch_degrees)

    head_drop_score = None
    if head_height_ratio is not None:
        head_drop_score = clamp_value(
            (config.head_drop_neutral_ratio - head_height_ratio)
            / max(config.head_drop_neutral_ratio - config.head_drop_strong_ratio, 1e-6)
        )

    nose_drop_ratio = None
    mouth_drop_ratio = None
    chin_drop_ratio = None
    ear_drop_ratio = None
    nose_drop_score = None
    mouth_drop_score = None
    chin_drop_score = None
    ear_drop_score = None
    shoulder_ear_compression_score = None

    if shoulder_mid is not None and shoulder_width is not None and shoulder_width > 0:
        nose_point = pose_2d.get('nose') or face_2d.get('face_nose_tip')
        if nose_point is not None:
            nose_drop_ratio = (nose_point.y - (shoulder_mid.y - (0.42 * shoulder_width))) / shoulder_width
            nose_drop_score = score_ratio_above(nose_drop_ratio, 0.01, 0.16)

        mouth_line_y = face_limits.get('mouth_line_y')
        if mouth_line_y is not None:
            mouth_drop_ratio = (mouth_line_y - (shoulder_mid.y - (0.24 * shoulder_width))) / shoulder_width
            mouth_drop_score = score_ratio_above(mouth_drop_ratio, 0.01, 0.16)

        chin_line_y = face_limits.get('chin_line_y')
        if chin_line_y is not None:
            chin_drop_ratio = (chin_line_y - (shoulder_mid.y - (0.14 * shoulder_width))) / shoulder_width
            chin_drop_score = score_ratio_above(chin_drop_ratio, 0.02, 0.18)

        left_ear = pose_2d.get('left_ear')
        right_ear = pose_2d.get('right_ear')
        ear_points = [point for point in [left_ear, right_ear] if point is not None]
        if ear_points:
            mean_ear_y = float(np.mean([point.y for point in ear_points]))
            ear_drop_ratio = (mean_ear_y - (shoulder_mid.y - (0.30 * shoulder_width))) / shoulder_width
            ear_drop_score = score_ratio_above(ear_drop_ratio, 0.00, 0.12)

        left_ratio = point_distance(pose_2d.get('left_shoulder'), left_ear)
        right_ratio = point_distance(pose_2d.get('right_shoulder'), right_ear)
        ratios = [
            ratio / shoulder_width
            for ratio in [left_ratio, right_ratio]
            if ratio is not None
        ]
        if ratios:
            shoulder_ear_compression_score = score_ratio_below(float(np.mean(ratios)), 0.30, 0.45)

    static_pose_score = weighted_mean_score([
        (head_drop_score, 0.30),
        (chin_drop_score, 0.24),
        (mouth_drop_score, 0.18),
        (nose_drop_score, 0.16),
        (ear_drop_score, 0.07),
        (shoulder_ear_compression_score, 0.05),
    ])

    static_support_scores = [
        score
        for score in [head_drop_score, chin_drop_score, mouth_drop_score, nose_drop_score, ear_drop_score]
        if score is not None and score >= 0.20
    ]
    static_support_count = len(static_support_scores)
    agreement_score = float(np.mean(static_support_scores[:2])) if static_support_scores else None
    head_down_gate_passed = bool(
        static_support_count >= 2
        or (static_pose_score is not None and static_pose_score >= 0.42)
        or (
            face_pitch_score is not None
            and face_pitch_score >= 0.20
            and static_pose_score is not None
            and static_pose_score >= 0.24
        )
    )

    head_tilt_score = weighted_mean_score([
        (static_pose_score, 0.78),
        (face_pitch_score, 0.10),
        (agreement_score, 0.12),
    ])
    if head_tilt_score is not None and not head_down_gate_passed:
        head_tilt_score = min(head_tilt_score, 0.32)
    if head_tilt_score is not None and face_pitch_direction == 'up':
        head_tilt_score = min(head_tilt_score, 0.22)

    return {
        'face_pitch_degrees': face_pitch_degrees,
        'face_pitch_score': face_pitch_score,
        'face_pitch_valid': face_pitch_valid,
        'face_pitch_direction': face_pitch_direction,
        'head_drop_score': head_drop_score,
        'nose_drop_ratio': nose_drop_ratio,
        'nose_drop_score': nose_drop_score,
        'mouth_drop_ratio': mouth_drop_ratio,
        'mouth_drop_score': mouth_drop_score,
        'chin_drop_ratio': chin_drop_ratio,
        'chin_drop_score': chin_drop_score,
        'ear_drop_ratio': ear_drop_ratio,
        'ear_drop_score': ear_drop_score,
        'shoulder_ear_compression_score': shoulder_ear_compression_score,
        'static_pose_score': static_pose_score,
        'agreement_score': agreement_score,
        'static_support_count': static_support_count,
        'head_down_gate_passed': head_down_gate_passed,
        'head_tilt_score': head_tilt_score,
        'head_tilt_basis': 'static_pose_with_directional_face_pitch' if face_pitch_score is not None else 'static_pose_guarded',
    }


# ---------------------------------------------------------------------------
# Arm proxy estimation
# ---------------------------------------------------------------------------

def estimate_arm_proxy_point(
    shoulder: PointData | None,
    elbow: PointData | None,
) -> PointData | None:
    """Estimate a proxy wrist point from shoulder and elbow landmarks."""
    if shoulder is None or elbow is None:
        return None
    dx = elbow.x - shoulder.x
    dy = elbow.y - shoulder.y
    dz = None
    if shoulder.z is not None and elbow.z is not None:
        dz = elbow.z + ((elbow.z - shoulder.z) * 0.70)
    return PointData(
        x=clamp_value(elbow.x + (dx * 0.70), -0.25, 1.25),
        y=clamp_value(elbow.y + (dy * 0.70), -0.25, 1.25),
        z=dz,
        visibility=elbow.visibility,
        space=elbow.space,
    )


# ---------------------------------------------------------------------------
# Side contact evidence
# ---------------------------------------------------------------------------

def build_side_contact_evidence(
    side: str,
    pose_2d: Mapping[str, PointData],
    hand_2d: Mapping[str, PointData],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build contact evidence for one body side.

    Args:
        side: ``'left'`` or ``'right'``.
        pose_2d: Named 2D pose landmarks.
        hand_2d: Named 2D hand landmarks.
        config: Analysis configuration.

    Returns:
        A contact evidence dictionary.
    """
    if config is None:
        config = PostureAnalysisConfig()
    hand_source_quality = config.hand_source_quality

    if hand_2d:
        return {
            'side': side,
            'source': 'holistic_hand',
            'quality': hand_source_quality['holistic_hand'],
            'named_points': [(f'{side}_hand_2d.{name}', point) for name, point in hand_2d.items()],
            'landmarks_used': [f'{side}_hand_2d.{name}' for name in hand_2d],
            'notes': [],
        }

    wrist = pose_2d.get(f'{side}_wrist')
    if wrist is not None:
        return {
            'side': side,
            'source': 'pose_wrist_fallback',
            'quality': hand_source_quality['pose_wrist_fallback'],
            'named_points': [(f'pose_2d.{side}_wrist', wrist)],
            'landmarks_used': [f'pose_2d.{side}_wrist'],
            'notes': ['Hand landmarks missing; using pose wrist fallback.'],
        }

    shoulder = pose_2d.get(f'{side}_shoulder')
    elbow = pose_2d.get(f'{side}_elbow')
    proxy_point = estimate_arm_proxy_point(shoulder, elbow)
    if elbow is not None and proxy_point is not None:
        return {
            'side': side,
            'source': 'arm_proxy',
            'quality': hand_source_quality['arm_proxy'],
            'named_points': [
                (f'pose_2d.{side}_elbow', elbow),
                (f'pose_2d.{side}_estimated_wrist', proxy_point),
            ],
            'landmarks_used': [f'pose_2d.{side}_elbow', f'pose_2d.{side}_estimated_wrist'],
            'notes': ['Hand and wrist missing; using low-confidence arm proxy.'],
        }

    return {
        'side': side,
        'source': 'unknown',
        'quality': hand_source_quality['unknown'],
        'named_points': [],
        'landmarks_used': [],
        'notes': ['No usable hand, wrist or elbow evidence for this side.'],
    }


# ---------------------------------------------------------------------------
# Neck lateral anchor summary
# ---------------------------------------------------------------------------

def build_neck_lateral_anchor_summary(
    face_2d: Mapping[str, PointData],
    pose_2d: Mapping[str, PointData],
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
    head_width: float | None,
) -> dict[str, Any]:
    """Infer lateral head anchors used to widen neck and nape regions.

    Args:
        face_2d: Named 2D face landmarks.
        pose_2d: Named 2D pose landmarks.
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.
        head_width: Estimated head width.

    Returns:
        A dictionary with visible lateral boundaries, center, and width reference.
    """
    if shoulder_mid is None or shoulder_width is None or shoulder_width <= 0:
        return {
            'neck_lateral_center_x': None,
            'neck_lateral_left_boundary_x': None,
            'neck_lateral_right_boundary_x': None,
            'neck_lateral_reference_width': None,
            'neck_lateral_reference_source': 'unavailable',
            'ear_span': None,
            'visible_head_span': None,
        }

    left_candidates = valid_points([
        pose_2d.get('left_ear'),
        face_2d.get('face_cheek_left'),
        face_2d.get('face_jaw_left'),
        face_2d.get('face_left_eye_outer'),
    ])
    right_candidates = valid_points([
        pose_2d.get('right_ear'),
        face_2d.get('face_cheek_right'),
        face_2d.get('face_jaw_right'),
        face_2d.get('face_right_eye_outer'),
    ])

    left_boundary_x = min((point.x for point in left_candidates), default=None)
    right_boundary_x = max((point.x for point in right_candidates), default=None)

    visible_head_span = None
    if (
        left_boundary_x is not None
        and right_boundary_x is not None
        and right_boundary_x > left_boundary_x
    ):
        visible_head_span = float(right_boundary_x - left_boundary_x)

    left_ear = pose_2d.get('left_ear')
    right_ear = pose_2d.get('right_ear')
    ear_span = point_distance(left_ear, right_ear)

    head_width_value = head_width if head_width is not None and head_width > 0 else clamp_value(
        shoulder_width * 0.38,
        shoulder_width * 0.26,
        shoulder_width * 0.56,
    )

    width_candidates = [
        value
        for value in [
            ear_span,
            visible_head_span,
            head_width_value * 0.92 if head_width_value is not None else None,
        ]
        if value is not None and value > 0
    ]
    lateral_reference_width = (
        float(max(width_candidates)) if width_candidates else float(head_width_value)
    )

    if visible_head_span is not None:
        lateral_reference_source = 'visible_head_span'
    elif ear_span is not None:
        lateral_reference_source = 'ear_span'
    else:
        lateral_reference_source = 'head_width_fallback'

    lateral_center_x = shoulder_mid.x
    if left_boundary_x is not None and right_boundary_x is not None:
        visible_center_x = (float(left_boundary_x) + float(right_boundary_x)) / 2.0
        lateral_center_x = (0.55 * shoulder_mid.x) + (0.45 * visible_center_x)
    elif left_ear is not None and right_ear is not None:
        ear_center_x = (left_ear.x + right_ear.x) / 2.0
        lateral_center_x = (0.60 * shoulder_mid.x) + (0.40 * ear_center_x)

    return {
        'neck_lateral_center_x': lateral_center_x,
        'neck_lateral_left_boundary_x': left_boundary_x,
        'neck_lateral_right_boundary_x': right_boundary_x,
        'neck_lateral_reference_width': lateral_reference_width,
        'neck_lateral_reference_source': lateral_reference_source,
        'ear_span': ear_span,
        'visible_head_span': visible_head_span,
    }


# ---------------------------------------------------------------------------
# Neck regions
# ---------------------------------------------------------------------------

def build_neck_regions(
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
    head_width: float | None,
    head_tilt_score: float | None,
    head_drop_score: float | None,
    face_limits: Mapping[str, float | None],
    neck_pose_mode: str,
    lateral_anchor_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build pose-aware anterior-neck and posterior-nape regions.

    Args:
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.
        head_width: Estimated head width.
        head_tilt_score: Current head-down support score.
        head_drop_score: Static drop support.
        face_limits: Stable vertical facial anchors.
        neck_pose_mode: Compact frontal or posterior-extended profile mode.
        lateral_anchor_summary: Optional ear/head-based lateral anchor summary.

    Returns:
        A dictionary with vertical and horizontal neck/nape geometry.
    """
    _empty_neck: dict[str, Any] = {
        'neck_fraction': None,
        'neck_top_y': None,
        'neck_bottom_y': None,
        'neck_center': None,
        'stable_neck_height': None,
        'neck_anchor_mode': 'unavailable',
        'neck_pose_mode': 'unavailable',
        'anterior_neck_top_y': None,
        'anterior_neck_bottom_y': None,
        'anterior_neck_center': None,
        'anterior_neck_half_width': None,
        'anterior_neck_left_x': None,
        'anterior_neck_right_x': None,
        'posterior_nape_top_y': None,
        'posterior_nape_bottom_y': None,
        'posterior_nape_center': None,
        'posterior_nape_half_width': None,
        'posterior_nape_left_x': None,
        'posterior_nape_right_x': None,
        'nape_extension_ratio': None,
    }

    if shoulder_mid is None or shoulder_width is None or shoulder_width <= 0:
        return dict(_empty_neck)

    stable_neck_height = compute_neck_height_from_shoulders(shoulder_width, head_tilt_score, head_drop_score)
    if stable_neck_height is None:
        return dict(_empty_neck)

    mouth_line_y = face_limits.get('mouth_line_y')
    chin_line_y = face_limits.get('chin_line_y')
    nose_line_y = face_limits.get('nose_line_y')
    ear_base_y = face_limits.get('ear_base_y')

    anterior_top_candidate = shoulder_mid.y - stable_neck_height
    anterior_lower_bound = min([
        shoulder_mid.y - (0.20 * shoulder_width),
        *(value for value in [
            (mouth_line_y + (0.06 * shoulder_width)) if mouth_line_y is not None else None,
            (chin_line_y + (0.01 * shoulder_width)) if chin_line_y is not None else None,
        ] if value is not None),
    ])
    anterior_upper_bound = max([
        shoulder_mid.y - (0.08 * shoulder_width),
        *(value for value in [
            (chin_line_y + (0.09 * shoulder_width)) if chin_line_y is not None else None,
            (mouth_line_y + (0.18 * shoulder_width)) if mouth_line_y is not None else None,
        ] if value is not None),
    ])
    if anterior_lower_bound > anterior_upper_bound:
        anterior_lower_bound, anterior_upper_bound = anterior_upper_bound, anterior_lower_bound
    anterior_neck_top_y = clamp_value(anterior_top_candidate, anterior_lower_bound, anterior_upper_bound)

    anterior_neck_bottom_y = clamp_value(
        shoulder_mid.y + (0.035 * shoulder_width),
        shoulder_mid.y + (0.015 * shoulder_width),
        shoulder_mid.y + (0.060 * shoulder_width),
    )

    compact_mode = neck_pose_mode != 'posterior_nape_extended'
    nape_extension_ratio = clamp_value(
        0.025 + ((0.060 if not compact_mode else 0.020) * float(head_tilt_score or 0.0)),
        0.02,
        0.10,
    )
    posterior_top_candidate = anterior_neck_top_y - ((0.020 if compact_mode else nape_extension_ratio) * shoulder_width)

    posterior_lower_candidates = [shoulder_mid.y - ((0.22 if compact_mode else 0.30) * shoulder_width)]
    if compact_mode and chin_line_y is not None:
        posterior_lower_candidates.append(chin_line_y + (0.01 * shoulder_width))
    if ear_base_y is not None:
        posterior_lower_candidates.append(ear_base_y + ((0.05 if compact_mode else 0.00) * shoulder_width))
    if not compact_mode and nose_line_y is not None:
        posterior_lower_candidates.append(nose_line_y + (0.05 * shoulder_width))
    posterior_lower_bound = min(posterior_lower_candidates)

    posterior_upper_candidates = [anterior_neck_top_y - (0.01 * shoulder_width)]
    if compact_mode and chin_line_y is not None:
        posterior_upper_candidates.append(chin_line_y + (0.08 * shoulder_width))
    posterior_upper_bound = max(posterior_upper_candidates)
    if posterior_lower_bound > posterior_upper_bound:
        posterior_lower_bound, posterior_upper_bound = posterior_upper_bound, posterior_lower_bound
    posterior_nape_top_y = clamp_value(posterior_top_candidate, posterior_lower_bound, posterior_upper_bound)

    posterior_nape_bottom_y = clamp_value(
        shoulder_mid.y + ((0.015 if compact_mode else 0.025) * shoulder_width),
        shoulder_mid.y - (0.005 * shoulder_width),
        shoulder_mid.y + (0.050 * shoulder_width),
    )

    lateral_anchor_summary = dict(lateral_anchor_summary or {})
    lateral_reference_width_value = lateral_anchor_summary.get('neck_lateral_reference_width')
    lateral_reference_width = (
        float(lateral_reference_width_value)
        if isinstance(lateral_reference_width_value, (int, float)) and float(lateral_reference_width_value) > 0
        else float(head_width if head_width is not None and head_width > 0 else clamp_value(shoulder_width * 0.38, shoulder_width * 0.26, shoulder_width * 0.56))
    )
    lateral_center_value = lateral_anchor_summary.get('neck_lateral_center_x')
    lateral_center_x = (
        float(lateral_center_value)
        if isinstance(lateral_center_value, (int, float))
        else shoulder_mid.x
    )
    left_boundary_value = lateral_anchor_summary.get('neck_lateral_left_boundary_x')
    right_boundary_value = lateral_anchor_summary.get('neck_lateral_right_boundary_x')
    left_boundary_x = float(left_boundary_value) if isinstance(left_boundary_value, (int, float)) else None
    right_boundary_x = float(right_boundary_value) if isinstance(right_boundary_value, (int, float)) else None

    base_half_width = clamp_value(
        (0.50 * lateral_reference_width) + max(lateral_reference_width * 0.05, shoulder_width * 0.015),
        shoulder_width * 0.12,
        shoulder_width * 0.32,
    )
    anterior_default_half_width = clamp_value(
        base_half_width * (0.94 if compact_mode else 0.98),
        shoulder_width * 0.12,
        shoulder_width * 0.30,
    )
    posterior_default_half_width = clamp_value(
        base_half_width * (1.00 if compact_mode else 1.08),
        shoulder_width * 0.13,
        shoulder_width * 0.34,
    )

    boundary_margin = max(shoulder_width * 0.012, lateral_reference_width * 0.035)
    max_left_extent = shoulder_mid.x - (0.40 * shoulder_width)
    max_right_extent = shoulder_mid.x + (0.40 * shoulder_width)

    if left_boundary_x is not None and right_boundary_x is not None and right_boundary_x > left_boundary_x:
        anterior_neck_left_x = max(left_boundary_x - boundary_margin, max_left_extent)
        anterior_neck_right_x = min(right_boundary_x + boundary_margin, max_right_extent)
        posterior_margin = boundary_margin * (1.00 if compact_mode else 1.12)
        posterior_nape_left_x = max(left_boundary_x - posterior_margin, max_left_extent)
        posterior_nape_right_x = min(right_boundary_x + posterior_margin, max_right_extent)
    else:
        anterior_neck_left_x = lateral_center_x - anterior_default_half_width
        anterior_neck_right_x = lateral_center_x + anterior_default_half_width
        posterior_nape_left_x = lateral_center_x - posterior_default_half_width
        posterior_nape_right_x = lateral_center_x + posterior_default_half_width

    anterior_neck_half_width = max((anterior_neck_right_x - anterior_neck_left_x) / 2.0, shoulder_width * 0.12)
    posterior_nape_half_width = max((posterior_nape_right_x - posterior_nape_left_x) / 2.0, shoulder_width * 0.13)
    anterior_center_x = (anterior_neck_left_x + anterior_neck_right_x) / 2.0
    posterior_center_x = (posterior_nape_left_x + posterior_nape_right_x) / 2.0

    anterior_neck_center = PointData(
        x=anterior_center_x,
        y=(anterior_neck_top_y + anterior_neck_bottom_y) / 2.0,
        z=shoulder_mid.z,
        visibility=shoulder_mid.visibility,
        space='image',
    )
    posterior_nape_center = PointData(
        x=posterior_center_x,
        y=(posterior_nape_top_y + posterior_nape_bottom_y) / 2.0,
        z=shoulder_mid.z,
        visibility=shoulder_mid.visibility,
        space='image',
    )

    combined_top_y = min(anterior_neck_top_y, posterior_nape_top_y)
    combined_bottom_y = max(anterior_neck_bottom_y, posterior_nape_bottom_y)
    combined_center = PointData(
        x=(anterior_center_x + posterior_center_x) / 2.0,
        y=(combined_top_y + combined_bottom_y) / 2.0,
        z=shoulder_mid.z,
        visibility=shoulder_mid.visibility,
        space='image',
    )

    return {
        'neck_fraction': stable_neck_height / shoulder_width,
        'neck_top_y': combined_top_y,
        'neck_bottom_y': combined_bottom_y,
        'neck_center': combined_center,
        'stable_neck_height': stable_neck_height,
        'neck_anchor_mode': 'pose_aware_split_neck_nape_ear_aligned',
        'neck_pose_mode': neck_pose_mode,
        'anterior_neck_top_y': anterior_neck_top_y,
        'anterior_neck_bottom_y': anterior_neck_bottom_y,
        'anterior_neck_center': anterior_neck_center,
        'anterior_neck_half_width': anterior_neck_half_width,
        'anterior_neck_left_x': anterior_neck_left_x,
        'anterior_neck_right_x': anterior_neck_right_x,
        'posterior_nape_top_y': posterior_nape_top_y,
        'posterior_nape_bottom_y': posterior_nape_bottom_y,
        'posterior_nape_center': posterior_nape_center,
        'posterior_nape_half_width': posterior_nape_half_width,
        'posterior_nape_left_x': posterior_nape_left_x,
        'posterior_nape_right_x': posterior_nape_right_x,
        'nape_extension_ratio': nape_extension_ratio,
    }


# ---------------------------------------------------------------------------
# Head reference points
# ---------------------------------------------------------------------------

def build_head_reference_points(
    face_2d: Mapping[str, PointData],
    pose_2d: Mapping[str, PointData],
    face_limits: Mapping[str, float | None],
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
    head_width: float | None,
    neck_regions: Mapping[str, Any],
    fallback_center: PointData | None,
) -> dict[str, Any]:
    """Build synthetic head anchors for head-contact reasoning.

    Args:
        face_2d: Named 2D face landmarks.
        pose_2d: Named 2D pose landmarks.
        face_limits: Stable vertical facial anchors.
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.
        head_width: Estimated head width.
        neck_regions: Neck and nape geometry.
        fallback_center: Precomputed face centroid fallback.

    Returns:
        A dictionary with head anchors and vertical limits for head contact.
    """
    if shoulder_mid is None or shoulder_width is None or shoulder_width <= 0:
        return {
            'head_width': None,
            'head_center': None,
            'head_top_y': None,
            'head_contact_top_y': None,
            'head_contact_bottom_y': None,
            'head_crown': None,
            'head_left_temple': None,
            'head_right_temple': None,
            'head_left_side': None,
            'head_right_side': None,
            'head_nape_anchor': None,
        }

    head_width_value = head_width if head_width is not None and head_width > 0 else clamp_value(
        shoulder_width * 0.36,
        shoulder_width * 0.26,
        shoulder_width * 0.52,
    )
    head_center_candidates = valid_points([
        face_2d.get('face_forehead'),
        pose_2d.get('nose'),
        face_2d.get('face_nose_tip'),
        pose_2d.get('left_ear'),
        pose_2d.get('right_ear'),
        face_2d.get('face_cheek_left'),
        face_2d.get('face_cheek_right'),
        fallback_center,
    ])
    head_center = centroid(head_center_candidates) or PointData(
        x=shoulder_mid.x,
        y=shoulder_mid.y - (0.30 * shoulder_width),
        z=shoulder_mid.z,
        visibility=shoulder_mid.visibility,
        space='image',
    )

    forehead_line_y = face_limits.get('forehead_line_y')
    eye_line_y = face_limits.get('eye_line_y')
    chin_line_y = face_limits.get('chin_line_y')
    anterior_neck_top_y = neck_regions.get('anterior_neck_top_y')
    posterior_nape_top_y = neck_regions.get('posterior_nape_top_y')

    head_top_y = (
        forehead_line_y - (0.08 * head_width_value)
        if forehead_line_y is not None
        else shoulder_mid.y - (0.50 * shoulder_width)
    )
    if eye_line_y is not None:
        temple_y = float(np.mean([head_top_y + (0.10 * shoulder_width), eye_line_y - (0.01 * shoulder_width)]))
    else:
        temple_y = head_center.y - (0.08 * shoulder_width)

    head_contact_bottom_y = (
        chin_line_y + (0.04 * shoulder_width)
        if chin_line_y is not None
        else head_center.y + (0.16 * shoulder_width)
    )
    if anterior_neck_top_y is not None:
        head_contact_bottom_y = min(head_contact_bottom_y, anterior_neck_top_y + (0.02 * shoulder_width))

    head_crown = PointData(
        x=head_center.x,
        y=head_top_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )
    head_left_temple = PointData(
        x=head_center.x - (0.34 * head_width_value),
        y=temple_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )
    head_right_temple = PointData(
        x=head_center.x + (0.34 * head_width_value),
        y=temple_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )
    head_side_y = float(np.mean([temple_y, head_contact_bottom_y]))
    head_left_side = PointData(
        x=head_center.x - (0.42 * head_width_value),
        y=head_side_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )
    head_right_side = PointData(
        x=head_center.x + (0.42 * head_width_value),
        y=head_side_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )
    nape_anchor_y = posterior_nape_top_y if posterior_nape_top_y is not None else head_contact_bottom_y
    nape_anchor_y = max(nape_anchor_y, head_top_y + (0.16 * shoulder_width))
    head_nape_anchor = PointData(
        x=head_center.x,
        y=nape_anchor_y,
        z=head_center.z,
        visibility=head_center.visibility,
        space='image',
    )

    return {
        'head_width': head_width_value,
        'head_center': head_center,
        'head_top_y': head_top_y,
        'head_contact_top_y': head_top_y,
        'head_contact_bottom_y': head_contact_bottom_y,
        'head_crown': head_crown,
        'head_left_temple': head_left_temple,
        'head_right_temple': head_right_temple,
        'head_left_side': head_left_side,
        'head_right_side': head_right_side,
        'head_nape_anchor': head_nape_anchor,
    }


# ---------------------------------------------------------------------------
# Forward-head 2D components
# ---------------------------------------------------------------------------

def compute_forward_head_2d_components(
    pose_2d: Mapping[str, PointData],
    face_limits: Mapping[str, float | None],
    shoulder_mid: PointData | None,
    shoulder_width: float | None,
    head_height_ratio: float | None,
) -> dict[str, float | None]:
    """Estimate forward-head evidence from 2D face and shoulder geometry.

    Args:
        pose_2d: Named 2D pose landmarks.
        face_limits: Face vertical anchors.
        shoulder_mid: Midpoint between shoulders.
        shoulder_width: Normalized shoulder width.
        head_height_ratio: Ratio between shoulder-to-head height and shoulder width.

    Returns:
        A dictionary with 2D forward-head cues and combined score.
    """
    if shoulder_mid is None or shoulder_width is None or shoulder_width <= 0:
        return {
            'eye_drop_ratio': None,
            'chin_drop_ratio': None,
            'ear_compression_ratio': None,
            'eye_drop_score': None,
            'chin_drop_score': None,
            'ear_compression_score': None,
            'forward_head_2d_score': None,
        }

    eye_line_y = face_limits.get('eye_line_y')
    chin_line_y = face_limits.get('chin_line_y')
    left_ear = pose_2d.get('left_ear')
    right_ear = pose_2d.get('right_ear')

    eye_drop_ratio = None
    if eye_line_y is not None:
        eye_drop_ratio = (eye_line_y - (shoulder_mid.y - (0.42 * shoulder_width))) / shoulder_width

    chin_drop_ratio = None
    if chin_line_y is not None:
        chin_drop_ratio = (chin_line_y - (shoulder_mid.y - (0.14 * shoulder_width))) / shoulder_width

    ear_compression_ratio = None
    ear_distances = [
        point_distance(pose_2d.get('left_shoulder'), left_ear),
        point_distance(pose_2d.get('right_shoulder'), right_ear),
    ]
    valid_ear_distances = [distance for distance in ear_distances if distance is not None]
    if valid_ear_distances:
        ear_compression_ratio = float(np.mean(valid_ear_distances)) / shoulder_width

    eye_drop_score = score_ratio_above(eye_drop_ratio, 0.02, 0.16) if eye_drop_ratio is not None else None
    chin_drop_score = score_ratio_above(chin_drop_ratio, 0.02, 0.18) if chin_drop_ratio is not None else None
    ear_compression_score = score_ratio_below(ear_compression_ratio, 0.28, 0.42) if ear_compression_ratio is not None else None
    head_drop_support_score = score_ratio_below(head_height_ratio, 0.26, 0.38) if head_height_ratio is not None else None

    forward_head_2d_score = weighted_mean_score([
        (eye_drop_score, 0.35),
        (chin_drop_score, 0.30),
        (ear_compression_score, 0.20),
        (head_drop_support_score, 0.15),
    ])

    return {
        'eye_drop_ratio': eye_drop_ratio,
        'chin_drop_ratio': chin_drop_ratio,
        'ear_compression_ratio': ear_compression_ratio,
        'eye_drop_score': eye_drop_score,
        'chin_drop_score': chin_drop_score,
        'ear_compression_score': ear_compression_score,
        'head_drop_support_score': head_drop_support_score,
        'forward_head_2d_score': forward_head_2d_score,
    }


# ---------------------------------------------------------------------------
# Reference geometry (main builder)
# ---------------------------------------------------------------------------

def compute_reference_geometry(
    landmark_sets: Mapping[str, dict[str, PointData]],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build stable geometric references used by all posture rules.

    Args:
        landmark_sets: Smoothed 2D and world-space landmark groups.
        config: Analysis configuration.

    Returns:
        A geometry dictionary shared by all rule evaluators.
    """
    if config is None:
        config = PostureAnalysisConfig()

    pose_2d = landmark_sets.get('pose_2d', {})
    pose_world = landmark_sets.get('pose_world', {})
    face_2d = landmark_sets.get('face_2d', {})
    left_hand_2d = landmark_sets.get('left_hand_2d', {})
    right_hand_2d = landmark_sets.get('right_hand_2d', {})

    left_shoulder = pose_2d.get('left_shoulder')
    right_shoulder = pose_2d.get('right_shoulder')
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    shoulder_width = point_distance(left_shoulder, right_shoulder)
    shoulder_reference_ok = shoulder_width is not None and shoulder_width >= config.min_shoulder_width_norm

    nose_2d = pose_2d.get('nose') or face_2d.get('face_nose_tip')
    shoulder_to_nose = None
    head_height_ratio = None
    if shoulder_reference_ok and shoulder_mid is not None and nose_2d is not None and shoulder_width:
        shoulder_to_nose = max(shoulder_mid.y - nose_2d.y, shoulder_width * 0.10)
        head_height_ratio = shoulder_to_nose / shoulder_width

    face_region_points_named = [
        (name, face_2d[name])
        for name in FACE_REGION_NAMES
        if name in face_2d
    ]
    if not face_region_points_named:
        fallback_face_names = ['nose', 'mouth_left', 'mouth_right', 'left_ear', 'right_ear']
        face_region_points_named = [
            (f'pose_2d.{name}', pose_2d[name])
            for name in fallback_face_names
            if name in pose_2d
        ]
    face_points = [point for _, point in face_region_points_named]
    face_center = centroid(face_points)

    face_limits = compute_face_vertical_limits(face_2d, pose_2d, shoulder_mid, shoulder_width)
    head_width = estimate_head_width(face_2d, pose_2d, shoulder_width)
    head_tilt_components = compute_head_tilt_components(
        pose_2d,
        face_2d,
        shoulder_mid,
        shoulder_width,
        head_height_ratio,
        face_limits,
        config=config,
    )
    head_tilt_score_value = head_tilt_components.get('head_tilt_score')
    head_drop_score_value = head_tilt_components.get('head_drop_score')
    head_tilt_score = float(head_tilt_score_value) if isinstance(head_tilt_score_value, (int, float)) else None
    head_drop_score = float(head_drop_score_value) if isinstance(head_drop_score_value, (int, float)) else None

    pose_profile = infer_neck_pose_mode(face_2d, pose_2d, head_tilt_score, shoulder_width)
    lateral_anchor_summary = build_neck_lateral_anchor_summary(
        face_2d,
        pose_2d,
        shoulder_mid,
        shoulder_width,
        head_width,
    )
    neck_regions = build_neck_regions(
        shoulder_mid,
        shoulder_width,
        head_width,
        head_tilt_score,
        head_drop_score,
        face_limits,
        str(pose_profile.get('neck_pose_mode') or 'frontal_compact'),
        lateral_anchor_summary=lateral_anchor_summary,
    )
    head_references = build_head_reference_points(
        face_2d,
        pose_2d,
        face_limits,
        shoulder_mid,
        shoulder_width,
        head_width,
        neck_regions,
        face_center,
    )

    hip_mid = midpoint(pose_2d.get('left_hip'), pose_2d.get('right_hip'))
    upper_chest_center = None
    if shoulder_mid is not None and hip_mid is not None:
        upper_chest_center = PointData(
            x=(shoulder_mid.x * 0.70) + (hip_mid.x * 0.30),
            y=(shoulder_mid.y * 0.70) + (hip_mid.y * 0.30),
            z=None,
            visibility=shoulder_mid.visibility,
            space='image',
        )
    elif shoulder_reference_ok and shoulder_mid is not None and shoulder_width:
        upper_chest_center = offset_point(shoulder_mid, dy=(0.22 * shoulder_width))

    left_evidence = build_side_contact_evidence('left', pose_2d, left_hand_2d, config=config)
    right_evidence = build_side_contact_evidence('right', pose_2d, right_hand_2d, config=config)
    contact_evidence = [
        evidence for evidence in [left_evidence, right_evidence]
        if evidence.get('source') != 'unknown' and evidence.get('named_points')
    ]

    world_left_shoulder = pose_world.get('left_shoulder')
    world_right_shoulder = pose_world.get('right_shoulder')
    shoulder_mid_world = midpoint(world_left_shoulder, world_right_shoulder)
    shoulder_width_world = point_distance(world_left_shoulder, world_right_shoulder)
    nose_world = pose_world.get('nose')
    forward_head_world_ratio = None
    if (
        shoulder_mid_world is not None
        and shoulder_width_world is not None
        and shoulder_width_world > 0
        and nose_world is not None
        and shoulder_mid_world.z is not None
        and nose_world.z is not None
    ):
        forward_head_world_ratio = (shoulder_mid_world.z - nose_world.z) / shoulder_width_world

    forward_head_2d_components = compute_forward_head_2d_components(
        pose_2d,
        face_limits,
        shoulder_mid,
        shoulder_width,
        head_height_ratio,
    )

    left_shoulder_ear_ratio = None
    right_shoulder_ear_ratio = None
    if shoulder_reference_ok and shoulder_width:
        left_shoulder_ear = point_distance(left_shoulder, pose_2d.get('left_ear'))
        right_shoulder_ear = point_distance(right_shoulder, pose_2d.get('right_ear'))
        if left_shoulder_ear is not None:
            left_shoulder_ear_ratio = left_shoulder_ear / shoulder_width
        if right_shoulder_ear is not None:
            right_shoulder_ear_ratio = right_shoulder_ear / shoulder_width

    shoulder_asymmetry_ratio = None
    if shoulder_reference_ok and left_shoulder is not None and right_shoulder is not None and shoulder_width:
        shoulder_asymmetry_ratio = abs(left_shoulder.y - right_shoulder.y) / shoulder_width

    coordinate_mode = COORDINATE_MODE_WORLD if forward_head_world_ratio is not None else COORDINATE_MODE_2D

    return {
        'pose_2d': pose_2d,
        'pose_world': pose_world,
        'face_2d': face_2d,
        'left_hand_2d': left_hand_2d,
        'right_hand_2d': right_hand_2d,
        'shoulder_mid': shoulder_mid,
        'shoulder_width': shoulder_width,
        'shoulder_reference_ok': shoulder_reference_ok,
        'shoulder_to_nose': shoulder_to_nose,
        'head_height_ratio': head_height_ratio,
        'head_tilt_components': head_tilt_components,
        'head_tilt_score': head_tilt_score,
        'neck_fraction': neck_regions['neck_fraction'],
        'neck_top_y': neck_regions['neck_top_y'],
        'neck_bottom_y': neck_regions['neck_bottom_y'],
        'neck_center': neck_regions['neck_center'],
        'stable_neck_height': neck_regions['stable_neck_height'],
        'neck_anchor_mode': neck_regions['neck_anchor_mode'],
        'neck_pose_mode': neck_regions['neck_pose_mode'],
        'anterior_neck_top_y': neck_regions['anterior_neck_top_y'],
        'anterior_neck_bottom_y': neck_regions['anterior_neck_bottom_y'],
        'anterior_neck_center': neck_regions['anterior_neck_center'],
        'anterior_neck_half_width': neck_regions['anterior_neck_half_width'],
        'anterior_neck_left_x': neck_regions['anterior_neck_left_x'],
        'anterior_neck_right_x': neck_regions['anterior_neck_right_x'],
        'posterior_nape_top_y': neck_regions['posterior_nape_top_y'],
        'posterior_nape_bottom_y': neck_regions['posterior_nape_bottom_y'],
        'posterior_nape_center': neck_regions['posterior_nape_center'],
        'posterior_nape_half_width': neck_regions['posterior_nape_half_width'],
        'posterior_nape_left_x': neck_regions['posterior_nape_left_x'],
        'posterior_nape_right_x': neck_regions['posterior_nape_right_x'],
        'nape_extension_ratio': neck_regions['nape_extension_ratio'],
        'nose_line_y': face_limits['nose_line_y'],
        'eye_line_y': face_limits['eye_line_y'],
        'mouth_line_y': face_limits['mouth_line_y'],
        'chin_line_y': face_limits['chin_line_y'],
        'forehead_line_y': face_limits['forehead_line_y'],
        'ear_base_y': face_limits['ear_base_y'],
        'head_width': head_references['head_width'],
        'head_center': head_references['head_center'],
        'head_top_y': head_references['head_top_y'],
        'head_contact_top_y': head_references['head_contact_top_y'],
        'head_contact_bottom_y': head_references['head_contact_bottom_y'],
        'head_crown': head_references['head_crown'],
        'head_left_temple': head_references['head_left_temple'],
        'head_right_temple': head_references['head_right_temple'],
        'head_left_side': head_references['head_left_side'],
        'head_right_side': head_references['head_right_side'],
        'head_nape_anchor': head_references['head_nape_anchor'],
        'profile_support_score': pose_profile['profile_support_score'],
        'lateral_face_balance_ratio': pose_profile['lateral_face_balance_ratio'],
        **lateral_anchor_summary,
        'forward_head_2d_components': forward_head_2d_components,
        'upper_chest_center': upper_chest_center,
        'face_points': face_points,
        'face_region_points_named': face_region_points_named,
        'face_center': face_center,
        'face_anchor_count': len(face_points),
        'contact_evidence': contact_evidence,
        'left_evidence': left_evidence,
        'right_evidence': right_evidence,
        'hands_visible': bool(left_hand_2d or right_hand_2d),
        'left_hand_visible': bool(left_hand_2d),
        'right_hand_visible': bool(right_hand_2d),
        'left_wrist_visible': pose_2d.get('left_wrist') is not None,
        'right_wrist_visible': pose_2d.get('right_wrist') is not None,
        'left_elbow_visible': pose_2d.get('left_elbow') is not None,
        'right_elbow_visible': pose_2d.get('right_elbow') is not None,
        'left_shoulder_ear_ratio': left_shoulder_ear_ratio,
        'right_shoulder_ear_ratio': right_shoulder_ear_ratio,
        'shoulder_asymmetry_ratio': shoulder_asymmetry_ratio,
        'forward_head_world_ratio': forward_head_world_ratio,
        'coordinate_mode': coordinate_mode,
    }


# ---------------------------------------------------------------------------
# Score helpers and contact targets (cell 8)
# ---------------------------------------------------------------------------

def serialize_metric_value(value: Any) -> Any:
    """Serialize nested metric values into JSON-friendly primitives."""
    if isinstance(value, float):
        return optional_round(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serialize_metric_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_metric_value(item) for item in value]
    return value


def build_scored_rule_result(
    *,
    score: float | None,
    evaluable: bool,
    raw_distance_ratio: float | None = None,
    zone_score: float | None = None,
    evidence_quality: float | None = None,
    source: str | None = None,
    closest_side: str | None = None,
    landmarks_used: list[str] | None = None,
    score_components: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build a normalized rule result payload.

    Args:
        score: Rule score (0–1).
        evaluable: Whether the rule could be evaluated.
        raw_distance_ratio: Raw proximity ratio.
        zone_score: Zone occupancy score.
        evidence_quality: Source quality factor.
        source: Evidence source label.
        closest_side: Body side with strongest evidence.
        landmarks_used: Landmarks contributing to the score.
        score_components: Detailed score components dict.
        notes: Human-readable evaluation notes.
        config: Analysis configuration.

    Returns:
        A normalised rule result dictionary.
    """
    if config is None:
        config = PostureAnalysisConfig()

    safe_score = clamp_value(score or 0.0)
    passed_threshold = bool(evaluable and safe_score >= config.rule_score_threshold)
    if not evaluable:
        state = 'unknown'
        strength = 'none'
    elif passed_threshold:
        state = 'true'
        strength = 'strong' if safe_score >= config.frame_strong_score_threshold else 'weak'
    else:
        state = 'false'
        strength = 'none'

    return {
        'state': state,
        'strength': strength,
        'score': optional_round(safe_score),
        'passed_threshold': passed_threshold,
        'threshold': optional_round(config.rule_score_threshold),
        'raw_distance_ratio': optional_round(raw_distance_ratio),
        'zone_score': optional_round(zone_score),
        'evidence_quality': optional_round(evidence_quality),
        'source': source,
        'closest_side': closest_side,
        'landmarks_used': landmarks_used or [],
        'score_components': serialize_metric_value(dict(score_components or {})),
        'notes': notes or [],
    }


def face_vertical_score(point: PointData, geometry: Mapping[str, Any]) -> float:
    """Score whether a point remains vertically compatible with the face region."""
    chin_line_y = geometry.get('chin_line_y')
    eye_line_y = geometry.get('eye_line_y')
    shoulder_width = geometry.get('shoulder_width') or 0.0
    if shoulder_width <= 0:
        return 1.0
    if eye_line_y is None or chin_line_y is None:
        return 1.0
    upper_limit = eye_line_y - (0.08 * shoulder_width)
    lower_limit = chin_line_y + (0.10 * shoulder_width)
    if upper_limit <= point.y <= lower_limit:
        return 1.0
    if point.y < upper_limit:
        return clamp_value(1.0 - ((upper_limit - point.y) / max(shoulder_width * 0.18, 1e-6)))
    return clamp_value(1.0 - ((point.y - lower_limit) / max(shoulder_width * 0.22, 1e-6)), 0.0, 1.0)


def head_vertical_score(point: PointData, geometry: Mapping[str, Any]) -> float:
    """Score whether a point stays vertically compatible with the head region.

    Args:
        point: Candidate hand, wrist, or proxy point.
        geometry: Shared frame geometry dictionary.

    Returns:
        A 0-1 vertical compatibility score for head contact.
    """
    shoulder_width = geometry.get('shoulder_width') or 0.0
    head_top_y = geometry.get('head_contact_top_y')
    head_bottom_y = geometry.get('head_contact_bottom_y')
    if shoulder_width <= 0 or head_top_y is None or head_bottom_y is None:
        return 1.0
    if head_top_y <= point.y <= head_bottom_y:
        return 1.0
    if point.y < head_top_y:
        return clamp_value(1.0 - ((head_top_y - point.y) / max(shoulder_width * 0.20, 1e-6)))
    return clamp_value(1.0 - ((point.y - head_bottom_y) / max(shoulder_width * 0.16, 1e-6)))


def neck_zone_score(
    point: PointData,
    geometry: Mapping[str, Any],
    zone_name: str | None = None,
) -> float:
    """Score whether a point lies inside the explicit neck or nape polygons.

    Args:
        point: Candidate hand, wrist, or proxy point.
        geometry: Shared frame geometry dictionary.
        zone_name: Optional specific zone to evaluate.

    Returns:
        A 0-1 zone occupancy score.
    """
    shoulder_width = geometry.get('shoulder_width') or 0.0
    if shoulder_width <= 0:
        return 0.0

    zone_specs: list[tuple[str, float | None, float | None, float | None, float | None]] = [
        (
            'anterior_neck',
            geometry.get('anterior_neck_top_y'),
            geometry.get('anterior_neck_bottom_y'),
            geometry.get('anterior_neck_left_x'),
            geometry.get('anterior_neck_right_x'),
        ),
        (
            'posterior_nape',
            geometry.get('posterior_nape_top_y'),
            geometry.get('posterior_nape_bottom_y'),
            geometry.get('posterior_nape_left_x'),
            geometry.get('posterior_nape_right_x'),
        ),
    ]

    scores: list[float] = []
    for current_zone_name, top_y, bottom_y, left_x, right_x in zone_specs:
        if zone_name is not None and current_zone_name != zone_name:
            continue
        if None in {top_y, bottom_y, left_x, right_x}:
            continue

        top_y_value = float(top_y)  # type: ignore[arg-type]
        bottom_y_value = float(bottom_y)  # type: ignore[arg-type]
        left_x_value = float(left_x)  # type: ignore[arg-type]
        right_x_value = float(right_x)  # type: ignore[arg-type]
        vertical_margin = max((bottom_y_value - top_y_value) * 0.40, shoulder_width * 0.06)
        horizontal_margin = max((right_x_value - left_x_value) * 0.24, shoulder_width * 0.04)

        if top_y_value <= point.y <= bottom_y_value:
            vertical_score = 1.0
        elif point.y < top_y_value:
            vertical_score = clamp_value(
                1.0 - ((top_y_value - point.y) / max(vertical_margin, 1e-6))
            )
        else:
            vertical_score = clamp_value(
                1.0 - ((point.y - bottom_y_value) / max(vertical_margin, 1e-6))
            )

        if left_x_value <= point.x <= right_x_value:
            horizontal_score = 1.0
        elif point.x < left_x_value:
            horizontal_score = clamp_value(
                1.0 - ((left_x_value - point.x) / max(horizontal_margin, 1e-6))
            )
        else:
            horizontal_score = clamp_value(
                1.0 - ((point.x - right_x_value) / max(horizontal_margin, 1e-6))
            )

        scores.append(vertical_score * horizontal_score)

    return max(scores, default=0.0)


def chest_vertical_score(point: PointData, geometry: Mapping[str, Any]) -> float:
    """Score whether a point remains vertically compatible with the chest region."""
    neck_bottom_y = geometry.get('anterior_neck_bottom_y') or geometry.get('neck_bottom_y')
    shoulder_width = geometry.get('shoulder_width') or 0.0
    if neck_bottom_y is None or shoulder_width <= 0:
        return 1.0
    if point.y >= neck_bottom_y:
        return 1.0
    return clamp_value(1.0 - ((neck_bottom_y - point.y) / max(shoulder_width * 0.35, 1e-6)))


def build_face_contact_targets(geometry: Mapping[str, Any]) -> list[tuple[str, PointData]]:
    """Build the subset of face anchors used for face-contact scoring."""
    preferred_names = [
        'face_cheek_left',
        'face_cheek_right',
        'face_jaw_left',
        'face_jaw_right',
        'face_mouth_left',
        'face_mouth_right',
        'face_chin',
    ]
    face_2d = geometry.get('face_2d', {})
    targets = [(name, face_2d[name]) for name in preferred_names if name in face_2d]
    if targets:
        return targets
    return list(geometry.get('face_region_points_named', []))


def build_head_contact_targets(geometry: Mapping[str, Any]) -> list[tuple[str, PointData]]:
    """Build synthetic and landmark-based head anchors for head-contact scoring.

    Args:
        geometry: Shared frame geometry dictionary.

    Returns:
        Named head targets suitable for proximity scoring.
    """
    pose_2d = geometry.get('pose_2d', {})
    face_2d = geometry.get('face_2d', {})
    target_specs: list[tuple[str, PointData | None]] = [
        ('synthetic.head_crown', geometry.get('head_crown')),
        ('synthetic.head_left_temple', geometry.get('head_left_temple')),
        ('synthetic.head_right_temple', geometry.get('head_right_temple')),
        ('synthetic.head_left_side', geometry.get('head_left_side')),
        ('synthetic.head_right_side', geometry.get('head_right_side')),
        ('synthetic.head_nape_anchor', geometry.get('head_nape_anchor')),
        ('face_forehead', face_2d.get('face_forehead')),
        ('pose_2d.left_ear', pose_2d.get('left_ear')),
        ('pose_2d.right_ear', pose_2d.get('right_ear')),
    ]
    return [(name, point) for name, point in target_specs if point is not None]


def build_head_contact_summary(
    evidence: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build a head-contact summary from hand evidence and head geometry."""
    named_points = evidence.get('named_points', [])
    shoulder_width = geometry.get('shoulder_width')
    if not named_points or shoulder_width is None or shoulder_width <= 0:
        return None

    hand_centroid = centroid([point for _, point in named_points])
    head_targets = build_head_contact_targets(geometry)
    head_centroid = centroid([point for _, point in head_targets])

    if hand_centroid is None or head_centroid is None:
        return None

    centroid_distance = point_distance(hand_centroid, head_centroid)
    if centroid_distance is None:
        return None

    centroid_ratio = centroid_distance / shoulder_width

    # Adjust here to define head contact score sensitivity
    centroid_score = score_ratio_below(centroid_ratio, 0.30, 0.65)

    return {
        'centroid_ratio': centroid_ratio,
        'centroid_score': centroid_score,
        'hand_centroid': hand_centroid,
        'head_centroid': head_centroid,
    }


def score_head_contact_summary(
    summary: Mapping[str, Any],
    evidence: Any,
    source: str,
) -> tuple[float, dict[str, Any], list[str]]:
    """Score a head-contact summary into a final score and components."""
    raw_score = float(summary.get('centroid_score') or 0.0)

    final_score = clamp_value(raw_score)
    return final_score, {
        'centroid_score': summary.get('centroid_score'),
        'centroid_ratio': summary.get('centroid_ratio'),
        'raw_score': raw_score,
    }, []


def point_target_overlap_score(
    point: PointData,
    targets: Iterable[tuple[str, PointData]],
    shoulder_width: float,
    *,
    strong_ratio: float,
    weak_ratio: float,
) -> tuple[float, str | None]:
    """Return the strongest proximity overlap between a point and named targets.

    Args:
        point: Candidate contact point.
        targets: Named target points.
        shoulder_width: Normalized shoulder width.
        strong_ratio: Ratio threshold for full overlap.
        weak_ratio: Ratio threshold for zero overlap.

    Returns:
        The strongest overlap score and the best-matching target name.
    """
    best_score = 0.0
    best_target = None
    for target_name, target_point in targets:
        distance = point_distance(point, target_point)
        if distance is None:
            continue
        ratio = distance / shoulder_width
        score = score_ratio_below(ratio, strong_ratio, weak_ratio)
        if score > best_score:
            best_score = score
            best_target = target_name
    return best_score, best_target


def score_neck_contact_point(
    point: PointData,
    geometry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Score a single point against a contact rule target region.

    Args:
        point: Candidate hand, wrist, or proxy point.
        geometry: Shared frame geometry dictionary.

    Returns:
        A score bundle or ``None`` when the rule cannot be evaluated.
    """
    shoulder_width = geometry.get('shoulder_width')
    if not geometry.get('shoulder_reference_ok') or not shoulder_width:
        return None

    anterior_center = geometry.get('anterior_neck_center')
    posterior_center = geometry.get('posterior_nape_center')
    center_candidates = [
        ('anterior_neck_zone', 'anterior_neck', anterior_center),
        ('posterior_nape_zone', 'posterior_nape', posterior_center),
    ]
    face_targets = build_face_contact_targets(geometry)
    head_targets = build_head_contact_targets(geometry)
    scored_candidates: list[dict[str, Any]] = []
    for target_name, zone_name, center_point in center_candidates:
        center_distance = point_distance(point, center_point)
        if center_distance is None:
            continue
        raw_ratio = center_distance / shoulder_width
        distance_score = score_ratio_below(raw_ratio, 0.08, 0.24)
        zone_sc = neck_zone_score(point, geometry, zone_name=zone_name)
        face_overlap_score, face_overlap_target = point_target_overlap_score(
            point,
            face_targets,
            shoulder_width,
            strong_ratio=0.16,
            weak_ratio=0.30,
        )
        face_overlap_score *= face_vertical_score(point, geometry)
        head_overlap_score, head_overlap_target = point_target_overlap_score(
            point,
            head_targets,
            shoulder_width,
            strong_ratio=0.18,
            weak_ratio=0.34,
        )
        head_overlap_score *= head_vertical_score(point, geometry)
        overlap_dominance = max(head_overlap_score, face_overlap_score) - zone_sc
        overlap_penalty = 1.0 - (0.55 * clamp_value(overlap_dominance, 0.0, 1.0))
        base_score = (0.30 * distance_score) + (0.70 * zone_sc)
        sc = clamp_value(base_score * overlap_penalty)
        scored_candidates.append({
            'score': sc,
            'raw_distance_ratio': raw_ratio,
            'zone_score': zone_sc,
            'target_name': target_name,
            'components': {
                'distance_score': distance_score,
                'zone_score': zone_sc,
                'zone_name': zone_name,
                'neck_fraction': geometry.get('neck_fraction'),
                'head_tilt_score': geometry.get('head_tilt_score'),
                'neck_pose_mode': geometry.get('neck_pose_mode'),
                'face_overlap_score': face_overlap_score,
                'face_overlap_target': face_overlap_target,
                'head_overlap_score': head_overlap_score,
                'head_overlap_target': head_overlap_target,
                'overlap_penalty': overlap_penalty,
                'anterior_neck_half_width': geometry.get('anterior_neck_half_width'),
                'posterior_nape_half_width': geometry.get('posterior_nape_half_width'),
            },
        })

    if not scored_candidates:
        return None
    return max(scored_candidates, key=lambda item: item['score'])


def score_chest_contact_point(
    point: PointData,
    geometry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Score one point against the upper-chest target zone."""
    shoulder_width = geometry.get('shoulder_width')
    if not geometry.get('shoulder_reference_ok') or not shoulder_width:
        return None

    chest_center = geometry.get('upper_chest_center')
    center_distance = point_distance(point, chest_center)
    if center_distance is None:
        return None
    raw_ratio = center_distance / shoulder_width
    distance_score = score_ratio_below(raw_ratio, 0.12, 0.36)
    vertical_sc = chest_vertical_score(point, geometry)
    sc = distance_score * vertical_sc
    return {
        'score': sc,
        'raw_distance_ratio': raw_ratio,
        'zone_score': vertical_sc,
        'target_name': 'upper_chest_center',
        'components': {'distance_score': distance_score, 'vertical_score': vertical_sc},
    }


# ---------------------------------------------------------------------------
# Rule evaluators (cell 9)
# ---------------------------------------------------------------------------

def contact_rule_precheck(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any] | None]:
    """Return usable contact evidence or the standard unevaluable contact-rule result."""
    if config is None:
        config = PostureAnalysisConfig()
    if not geometry.get('shoulder_reference_ok'):
        return None, build_scored_rule_result(
            score=None, evaluable=False,
            notes=['Shoulder reference unavailable; contact distance cannot be normalized.'],
            config=config,
        )
    evidence_candidates = list(geometry.get('contact_evidence', []))
    if not evidence_candidates:
        return None, build_scored_rule_result(
            score=None, evaluable=False,
            notes=['No usable hand, wrist or arm proxy evidence.'],
            config=config,
        )
    return evidence_candidates, None


def build_contact_rule_result(
    best: Mapping[str, Any] | None,
    missing_note: str,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Normalize the best contact candidate into a rule result."""
    if config is None:
        config = PostureAnalysisConfig()
    if best is None:
        return build_scored_rule_result(score=None, evaluable=False, notes=[missing_note], config=config)
    return build_scored_rule_result(
        score=best['score'], evaluable=True, raw_distance_ratio=best['raw_distance_ratio'],
        zone_score=best['zone_score'], evidence_quality=best['quality'], source=best['source'],
        closest_side=best['closest_side'], landmarks_used=best['landmarks_used'],
        score_components=best['components'], notes=best['notes'],
        config=config,
    )


def evaluate_summary_contact_rule(
    geometry: Mapping[str, Any],
    summary_builder: Any,
    summary_scorer: Any,
    missing_note: str,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a contact rule that uses a whole-hand summary."""
    if config is None:
        config = PostureAnalysisConfig()
    evidence_candidates, unavailable = contact_rule_precheck(geometry, config=config)
    if unavailable is not None:
        return unavailable
    best: dict[str, Any] | None = None
    for evidence in evidence_candidates or []:
        summary = summary_builder(evidence, geometry)
        if summary is None:
            continue
        final_score, components, notes = summary_scorer(summary, float(evidence.get('quality', 0.0)), str(evidence.get('source') or 'unknown'))
        candidate = {
            'score': final_score, 'raw_distance_ratio': components.get('closest_target_ratio'),
            'zone_score': components.get('vertical_score'), 'source': evidence.get('source'),
            'quality': evidence.get('quality'), 'closest_side': evidence.get('side'),
            'landmarks_used': evidence.get('landmarks_used', []),
            'notes': [*evidence.get('notes', []), *notes], 'components': components,
        }
        if best is None or candidate['score'] > best['score']:
            best = candidate
    return build_contact_rule_result(best, missing_note, config=config)


def evaluate_point_contact_rule(
    geometry: Mapping[str, Any],
    point_scorer: Any,
    missing_note: str,
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a contact rule that scores individual evidence points."""
    if config is None:
        config = PostureAnalysisConfig()
    evidence_candidates, unavailable = contact_rule_precheck(geometry, config=config)
    if unavailable is not None:
        return unavailable
    best: dict[str, Any] | None = None
    per_side_scores: dict[str, float] = {}
    for evidence in evidence_candidates or []:
        for landmark_name, point in evidence.get('named_points', []):
            point_score = point_scorer(point, geometry)
            if point_score is None:
                continue
            final_score = clamp_value(point_score['score'] * float(evidence.get('quality', 0.0)))
            per_side_scores[evidence['side']] = max(per_side_scores.get(evidence['side'], 0.0), final_score)
            candidate = {
                'score': final_score, 'raw_distance_ratio': point_score['raw_distance_ratio'],
                'zone_score': point_score['zone_score'], 'source': evidence.get('source'),
                'quality': evidence.get('quality'), 'closest_side': evidence.get('side'),
                'landmarks_used': list(dict.fromkeys([*evidence.get('landmarks_used', []), landmark_name])),
                'notes': evidence.get('notes', []),
                'components': {**point_score['components'], 'unweighted_score': point_score['score'], 'evidence_quality': evidence.get('quality'), 'target_name': point_score.get('target_name'), 'per_side_scores': per_side_scores},
            }
            if best is None or candidate['score'] > best['score']:
                best = candidate
    return build_contact_rule_result(best, missing_note, config=config)


def evaluate_hand_on_head(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the hand-on-head rule."""
    return evaluate_summary_contact_rule(
        geometry, build_head_contact_summary, score_head_contact_summary,
        'Head contact lacked enough multi-point hand evidence.',
        config=config,
    )


def evaluate_hand_on_neck(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the hand-on-neck rule."""
    return evaluate_point_contact_rule(
        geometry, score_neck_contact_point,
        'Rule had evidence candidates but no valid region geometry.',
        config=config,
    )


def evaluate_hand_on_chest(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the hand-on-chest rule."""
    return evaluate_point_contact_rule(
        geometry, score_chest_contact_point,
        'Rule had evidence candidates but no valid region geometry.',
        config=config,
    )


def evaluate_head_down(
    geometry: Mapping[str, Any],
    contact_rules: Mapping[str, Mapping[str, Any]],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate head-down from static anatomy plus optional hand-supported posture cues.

    Args:
        geometry: Shared frame geometry dictionary.
        contact_rules: Already-evaluated hand contact rules.
        config: Analysis configuration.

    Returns:
        A normalized head-down rule result.
    """
    if config is None:
        config = PostureAnalysisConfig()

    score = geometry.get('head_tilt_score')
    components = dict(geometry.get('head_tilt_components', {}))
    forward_head_2d_components = geometry.get('forward_head_2d_components', {})
    forward_head_2d_score = forward_head_2d_components.get('forward_head_2d_score')
    forward_head_world_ratio = geometry.get('forward_head_world_ratio')
    forward_head_world_support = score_ratio_above(forward_head_world_ratio, 0.24, 0.52) if forward_head_world_ratio is not None else None
    head_drop_score = components.get('head_drop_score')
    static_pose_score = components.get('static_pose_score')

    hand_on_head_score = float(contact_rules.get('hand_on_head', {}).get('score') or 0.0)
    hand_on_neck_score = float(contact_rules.get('hand_on_neck', {}).get('score') or 0.0)
    head_support_contact_score = weighted_mean_score([
        (hand_on_head_score, 0.65),
        (hand_on_neck_score, 0.35),
    ])

    support_static_seed = weighted_mean_score([
        (float(static_pose_score) if isinstance(static_pose_score, (int, float)) else None, 0.55),
        (float(head_drop_score) if isinstance(head_drop_score, (int, float)) else None, 0.25),
        (float(forward_head_2d_score) if isinstance(forward_head_2d_score, (int, float)) else None, 0.20),
    ])
    supported_head_down_score = None
    supported_head_down_passed = False
    if head_support_contact_score is not None and support_static_seed is not None:
        supported_head_down_score = weighted_mean_score([
            (support_static_seed, 0.60),
            (head_support_contact_score, 0.40),
        ])
        supported_head_down_passed = bool(
            head_support_contact_score >= 0.42
            and support_static_seed >= 0.24
            and supported_head_down_score is not None
            and supported_head_down_score >= 0.34
        )

    corroboration_score = weighted_mean_score([
        (float(forward_head_2d_score) if isinstance(forward_head_2d_score, (int, float)) else None, 0.45),
        (forward_head_world_support, 0.20),
        (supported_head_down_score, 0.35),
    ])
    corroboration_passed = bool(
        (isinstance(forward_head_2d_score, (int, float)) and float(forward_head_2d_score) >= 0.28)
        or (forward_head_world_support is not None and forward_head_world_support >= 0.35)
        or supported_head_down_passed
    )

    if score is None and supported_head_down_score is not None:
        score = supported_head_down_score
    if score is not None and not corroboration_passed and head_drop_score is not None and not supported_head_down_passed:
        score = min(float(score), 0.40)
    if score is not None and corroboration_score is not None:
        score = weighted_mean_score([
            (float(score), 0.65),
            (corroboration_score, 0.35),
        ])
    if supported_head_down_passed and supported_head_down_score is not None:
        score = max(float(score or 0.0), float(supported_head_down_score))
    if components.get('face_pitch_direction') == 'up' and not supported_head_down_passed and score is not None:
        score = min(float(score), 0.24)

    components['forward_head_2d_score'] = forward_head_2d_score
    components['forward_head_world_support'] = forward_head_world_support
    components['head_support_contact_score'] = head_support_contact_score
    components['supported_head_down_score'] = supported_head_down_score
    components['supported_head_down_passed'] = supported_head_down_passed
    components['hand_on_head_score'] = hand_on_head_score
    components['hand_on_neck_score'] = hand_on_neck_score
    components['head_down_corroboration_score'] = corroboration_score
    components['head_down_corroboration_passed'] = corroboration_passed

    notes: list[str] = []
    if not components.get('face_pitch_valid', False):
        notes.append('Face pitch was unavailable or implausible; relying on static anatomical cues.')
    if components.get('face_pitch_direction') == 'up' and not supported_head_down_passed:
        notes.append('Face pitch indicates upward motion, so head-down score was suppressed.')
    if not components.get('head_down_gate_passed', False) and not supported_head_down_passed:
        notes.append('Static head-down gate did not pass; score was capped to avoid false positives.')
    if supported_head_down_passed:
        notes.append('Hand contact supports a resting-head-down posture pattern.')
    if not corroboration_passed:
        notes.append('Head-down lacked enough forward-flexion or supported-rest corroboration.')

    return build_scored_rule_result(
        score=score,
        evaluable=score is not None,
        source=components.get('head_tilt_basis'),
        score_components=components,
        notes=notes,
        config=config,
    )


def evaluate_forward_head(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the forward-head rule from hybrid world-space and 2D cues."""
    if config is None:
        config = PostureAnalysisConfig()

    world_ratio = geometry.get('forward_head_world_ratio')
    forward_head_2d_components = geometry.get('forward_head_2d_components', {})
    forward_head_2d_score = forward_head_2d_components.get('forward_head_2d_score')
    world_score = score_ratio_above(world_ratio, 0.20, 0.60) if world_ratio is not None else None
    combined_score = weighted_mean_score([
        (world_score, 0.60),
        (float(forward_head_2d_score) if isinstance(forward_head_2d_score, (int, float)) else None, 0.40),
    ])
    if combined_score is None:
        return build_scored_rule_result(
            score=None,
            evaluable=False,
            source='forward_head_unavailable',
            notes=['Neither world-space nor 2D forward-head cues were available.'],
            config=config,
        )
    return build_scored_rule_result(
        score=combined_score,
        evaluable=True,
        raw_distance_ratio=world_ratio,
        source='pose_world_plus_face_2d',
        score_components={
            'forward_head_world_ratio': world_ratio,
            'forward_head_world_score': world_score,
            **forward_head_2d_components,
            'forward_head_combined_score': combined_score,
        },
        config=config,
    )


def evaluate_rounded_shoulders_or_asymmetry(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Evaluate upper-body tension and asymmetry cues."""
    if config is None:
        config = PostureAnalysisConfig()

    left_ratio = geometry.get('left_shoulder_ear_ratio')
    right_ratio = geometry.get('right_shoulder_ear_ratio')
    asymmetry_ratio = geometry.get('shoulder_asymmetry_ratio')

    ear_ratios = [ratio for ratio in [left_ratio, right_ratio] if ratio is not None]
    shrug_score = None
    if ear_ratios:
        shrug_score = score_ratio_below(float(np.mean(ear_ratios)), 0.30, 0.45)
    asymmetry_score = score_ratio_above(asymmetry_ratio, 0.08, 0.18)
    if shrug_score is None and asymmetry_ratio is None:
        return build_scored_rule_result(
            score=None,
            evaluable=False,
            notes=['Shoulder-ear and shoulder asymmetry cues unavailable.'],
            config=config,
        )

    score = max([value for value in [shrug_score, asymmetry_score] if value is not None], default=0.0)
    return build_scored_rule_result(
        score=score,
        evaluable=True,
        source='pose_2d',
        score_components={
            'shrug_score': shrug_score,
            'asymmetry_score': asymmetry_score,
            'left_shoulder_ear_ratio': left_ratio,
            'right_shoulder_ear_ratio': right_ratio,
            'shoulder_asymmetry_ratio': asymmetry_ratio,
        },
        config=config,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_rules(
    geometry: Mapping[str, Any],
    config: PostureAnalysisConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all final frame-level posture rules in dependency order.

    Args:
        geometry: Shared frame geometry dictionary (output of
            :func:`compute_reference_geometry`).
        config: Analysis configuration.

    Returns:
        A dictionary mapping each rule name to its normalised result.
    """
    if config is None:
        config = PostureAnalysisConfig()

    hand_on_head = evaluate_hand_on_head(geometry, config=config)
    hand_on_neck = evaluate_hand_on_neck(geometry, config=config)
    hand_on_chest = evaluate_hand_on_chest(geometry, config=config)
    head_down = evaluate_head_down(
        geometry,
        {'hand_on_head': hand_on_head, 'hand_on_neck': hand_on_neck},
        config=config,
    )
    return {
        'hand_on_head': hand_on_head,
        'hand_on_neck': hand_on_neck,
        'hand_on_chest': hand_on_chest,
        'head_down': head_down,
        'forward_head': evaluate_forward_head(geometry, config=config),
        'rounded_shoulders_or_asymmetry': evaluate_rounded_shoulders_or_asymmetry(geometry, config=config),
    }


# ---------------------------------------------------------------------------
# Convenience default
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PostureAnalysisConfig()
