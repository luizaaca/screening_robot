# MediaPipe Posture Screening Notebook Specification

## Status
- Approved planning draft for implementation
- Scope: single self-contained notebook
- Reference notebook: `concepts_video/deepface_video_emotion_timeline_v2.ipynb`

## Objective
Create a new notebook for geometric posture screening in interview / teleconsultation videos with upper-body framing. The notebook must follow the same overall workflow as the current DeepFace timeline notebook, but replace emotion inference with objective landmark-based posture and self-touch rules using MediaPipe.

The system must not infer emotion, guilt, aggression, deception, diagnosis, or any mental-state conclusion. It only reports posture / somatic patterns that are directly observable from landmarks.

## Target Notebook
- `concepts_video/mediapipe_video_posture_timeline.ipynb`

## Implementation Decisions
- Everything must live inside a single notebook.
- The notebook must reuse the DeepFace v2 pattern end to end:
  - setup/config cells
  - reusable helper functions
  - `process_video_asset(video_path: str | Path) -> dict[str, Any]`
  - batch execution over a folder
  - JSON output
  - annotated MP4 output
  - final summary table / preview
- Sampling must run at `5 FPS`.
- Backend must be `MediaPipe Holistic` only in v1.
- Partial scoring is allowed when only part of the landmarks is available.
- Final outputs must be in English.
- Failure handling must be partial fail-safe:
  - do not silently ignore low-quality frames
  - do not fail the entire video unless the clip is effectively unusable
  - always emit explicit limitations

## Required Libraries
- `opencv-python`
- `mediapipe`
- `numpy`
- `dataclasses`
- `scipy` optional
- `pandas` optional for notebook summaries

## Scope
The notebook must process ordinary RGB videos and detect objective upper-body signals using MediaPipe Holistic landmarks. It must work on sampled frames, aggregate signals into sliding windows, and produce both frame-level and video-level triage summaries.

## Non-Goals
- No emotion inference
- No deception or aggression inference
- No psychiatric, clinical, or forensic diagnosis
- No multi-person reasoning in v1
- No claim of calibrated model accuracy; this is a heuristic geometric screener

## High-Level Pipeline
1. Load an RGB video with OpenCV.
2. Sample frames at an effective rate of `5 FPS`.
3. Run `MediaPipe Holistic` on sampled frames.
4. Extract upper-body pose and hand landmarks.
5. Smooth coordinates temporally.
6. Compute geometric references normalized by shoulder width.
7. Evaluate six posture / self-touch rules per frame.
8. Compute frame score and frame-level label.
9. Aggregate results into sliding windows.
10. Derive a final video-level summary.
11. Render an annotated MP4 in a second pass.
12. Save a JSON timeline payload.

## Notebook Structure
1. Setup and dependency checks
2. Configuration constants
3. Landmark extraction helpers
4. Temporal smoothing helpers
5. Geometry helpers
6. Rule evaluation
7. Frame scoring
8. Window aggregation
9. JSON serialization helpers
10. Annotated video rendering
11. `process_video_asset(...)`
12. Batch execution over `concepts_video/data/video`
13. Final summary and preview cells

## Video Sampling
- Read source FPS from OpenCV.
- Compute `frame_step = max(1, round(source_fps / 5.0))`.
- Process only sampled frames for inference.
- Keep original FPS and source duration in the annotated MP4 output.
- Record sampled-frame timestamps in seconds.

## Landmark Backend
Use `MediaPipe Holistic` as the only backend in v1.

### Required landmarks per frame
Pose / face anchors:
- `nose`
- `mouth_left`
- `mouth_right`
- `left_ear`
- `right_ear`
- `left_shoulder`
- `right_shoulder`

Left hand:
- `left_wrist`
- `left_index_tip`
- `left_middle_tip`
- `left_thumb_tip`

Right hand:
- `right_wrist`
- `right_index_tip`
- `right_middle_tip`
- `right_thumb_tip`

### Coordinate policy
- Prefer `pose_world_landmarks` when available.
- Fall back to normalized 2D landmarks when world coordinates are not available.
- Use mixed mode if some rules depend on world landmarks while others fall back to 2D.

### Landmark validity
- Pose landmarks are valid only when `visibility >= 0.5`.
- Hand landmarks are considered available when the corresponding hand landmark list is returned by Holistic.
- Invalid landmarks must not be used in geometry calculations.

## Temporal Smoothing
Apply EMA smoothing to each coordinate before rule evaluation.

### Default configuration
- `ema_alpha = 0.35`

### Behavior
- Smooth each tracked landmark independently.
- If a landmark disappears, treat it as unavailable for that frame.
- Reset EMA for a landmark if it remains missing long enough or reappears after a long gap.

## Geometric References
For each valid sampled frame, compute:

- `shoulder_mid = midpoint(left_shoulder, right_shoulder)`
- `shoulder_width = distance(left_shoulder, right_shoulder)`
- `face_points = [nose, mouth_left, mouth_right, left_ear, right_ear]` filtered to valid points
- `left_hand_points = [left_wrist, left_index_tip, left_middle_tip, left_thumb_tip]`
- `right_hand_points = [right_wrist, right_index_tip, right_middle_tip, right_thumb_tip]`

All distances must be normalized by `shoulder_width`.

If `shoulder_width` is invalid or too small, the frame must be marked `insufficient_data`.

## Rule Set
Each rule must return:
- `state`: `true`, `false`, or `unknown`
- `strength`: `none`, `weak`, or `strong`
- supporting numeric metrics

### Rule 1: `hand_on_face`
- Compute the minimum distance between any valid hand point and any valid face point.
- Activate if `min_face_hand_dist / shoulder_width < 0.30`.
- Strong if `< 0.25`.
- If both hands are visible, use the smaller distance.
- This is the highest-priority rule.

### Rule 2: `hand_on_neck`
- Define `neck_center` above `shoulder_mid`.
- In 2D: `(shoulder_mid.x, shoulder_mid.y - 0.12 * shoulder_width)`.
- In world coordinates: use the equivalent vertical offset.
- Activate if `min_neck_hand_dist / shoulder_width < 0.22`.
- Strong if `< 0.18`.
- This rule can only be active when `hand_on_face == false`.

### Rule 3: `hand_on_chest`
- Define `upper_chest_center` below `shoulder_mid`.
- In 2D: `(shoulder_mid.x, shoulder_mid.y + 0.18 * shoulder_width)`.
- Activate if `min_chest_hand_dist / shoulder_width < 0.26`.
- Strong if `< 0.22`.
- This rule can only be active when:
  - `hand_on_face == false`
  - `hand_on_neck == false`

### Rule 4: `head_down`
- Compute `head_height_ratio = (shoulder_mid.y - nose.y) / shoulder_width`.
- Activate if `head_height_ratio < 0.40`.
- If `nose` is unavailable, the rule must return `unknown`.
- v1 uses a fixed threshold only, with no per-person baseline.

### Rule 5: `forward_head`
- Preferred rule: use world-space depth.
- Activate if `(shoulder_mid_world.z - nose_world.z) / shoulder_width_world > 0.10`.
- If world coordinates are unavailable, use a weak 2D fallback combining:
  - `head_down == true`
  - reduced shoulder-ear distance
  - visually closed neck geometry
- This rule has medium weight and must never be used alone as strong evidence.

### Rule 6: `rounded_shoulders_or_asymmetry`
Two submeasures:

- `shrugged_shoulders`
  - compute left and right shoulder-ear distances normalized by shoulder width
  - activate if both are `< 0.33`

- `shoulder_asymmetry`
  - activate if `abs(left_shoulder.y - right_shoulder.y) / shoulder_width > 0.12`

Final rule:
- `rounded_shoulders_or_asymmetry = true` if either submeasure is true

## Rule Weights
- `hand_on_face = 1.0`
- `hand_on_neck = 0.9`
- `hand_on_chest = 0.8`
- `head_down = 0.7`
- `forward_head = 0.6`
- `rounded_shoulders_or_asymmetry = 0.4`

## Frame Scoring
For each sampled frame:

- `frame_score = sum(weights of active true rules)`

### Frame-level labels
- `strong_signal` if:
  - `hand_on_face and head_down`, or
  - `hand_on_neck and forward_head`, or
  - `frame_score >= 2.4`
- `possible_signal` if `frame_score >= 1.8`
- `ok` otherwise
- `insufficient_data` if:
  - `shoulder_width` cannot be computed, or
  - all rules are `unknown`

## Partial-Data Policy
- Evaluate each rule independently with the landmarks available in the frame.
- Unknown rules do not add to the score.
- A frame remains scorable if at least one meaningful rule can be evaluated.
- Mark `partial_data = true` when at least one rule is unknown and at least one rule is evaluable.
- Never silently convert missing evidence into `false`.

## Window Aggregation
Use sliding windows over sampled frames.

### Default configuration
- `window_seconds = 8`
- `window_stride_seconds = 2`

### Metrics per window
- `frames_total`
- `frames_scorable`
- `frames_insufficient`
- `signal_ratio`
  - fraction of scorable frames labeled `possible_signal` or `strong_signal`
- `strong_signal_ratio`
- `max_consecutive_signal_frames`
- `most_common_trigger`
- `window_score_mean`
- `window_score_peak`

### Window-level labels
- `ok` if `signal_ratio < 0.15`
- `possible_signal` if `0.15 <= signal_ratio <= 0.35`
- `strong_signal` if `signal_ratio > 0.35`

## Video-Level Summary
Aggregate all valid windows into a final video summary.

### Metrics
- `video_signal_ratio`
- `video_strong_window_ratio`
- `peak_window_level`
- `most_common_trigger`
- `dominant_explanation`
- `coverage_ratio`

### Video-level labels
- `strong_signal` if:
  - more than `35%` of valid windows are `strong_signal`, or
  - there are at least `2` adjacent strong windows
- `possible_signal` if at least `15%` of valid windows are `possible_signal` or stronger
- `ok` otherwise

## Explanations
Generate short English explanations from the most relevant active rules.

Examples:
- `recurrent hand-to-face contact + head down`
- `recurrent hand-to-neck contact + forward head`
- `persistent upper-body tension / asymmetry`

## JSON Output
### Output path
- `concepts_video/outputs/posture_timelines/<video_id>.signals.json`

### Schema version
- `posture_signal_timeline_v1`

### Top-level fields
- `schema_version`
- `video`
- `pipeline`
- `limitations`
- `frame_signals`
- `window_summaries`
- `video_summary`

### `video`
- `video_id`
- `source_path`
- `source_name`
- `annotated_video_path`
- `source_fps`
- `sample_fps`
- `duration_seconds`
- `frames_total`
- `frames_sampled`

### `pipeline`
- `backend`
- `coordinate_mode`
- `visibility_threshold`
- `ema_alpha`
- `window_seconds`
- `window_stride_seconds`

### `frame_signals`
Per sampled frame:
- `frame_idx`
- `timestamp_s`
- `status`
- `partial_data`
- `metrics`
- `rules`
- `frame_score`
- `frame_level`
- `explanation`

### `window_summaries`
Per sliding window:
- `window_idx`
- `start_s`
- `end_s`
- `frames_total`
- `frames_scorable`
- `signal_ratio`
- `strong_signal_ratio`
- `max_consecutive_signal_frames`
- `most_common_trigger`
- `window_level`
- `explanation`

### `video_summary`
- `level`
- `video_signal_ratio`
- `video_strong_window_ratio`
- `coverage_ratio`
- `most_common_trigger`
- `explanation`

## Annotated MP4 Output
### Output path
- `concepts_video/outputs/annotated_videos/<video_id>.annotated.mp4`

### Rendering behavior
Perform a second pass over the original video and draw:
- upper-body pose landmarks
- hand landmarks
- current frame level
- current frame score
- active rules
- current window level
- short English explanation

If the frame is not scorable, display:
- `insufficient data`

If the video has known coverage limitations, keep rendering normally and expose those limitations in the JSON payload.

## Failure and Limitation Policy
The notebook must use partial fail-safe behavior.

### Expected cases
- upper-body framing only
- hands outside the frame
- temporary face occlusion
- unstable shoulder geometry
- multiple visible people
- low landmark coverage

### Required behavior
- do not crash on partial body videos
- do not silently discard low-quality frames
- do not silently downgrade unknown rules to `false`
- record explicit limitations in the payload
- still emit outputs whenever the clip has enough usable data

### Hard unusable case
If the clip has essentially no usable shoulder / face reference across the video, return a fail-safe payload with:
- no positive posture claims
- explicit limitation text
- no misleading video-level label

## Main Entrypoint
The notebook must expose:

```python
def process_video_asset(video_path: str | Path) -> dict[str, Any]:
    ...
```

Expected responsibilities:
- open the video
- sample frames
- run Holistic inference
- evaluate rules
- aggregate windows
- render annotated MP4
- save JSON
- return the final payload dictionary

## Batch Execution
Provide a batch cell that:
- scans `concepts_video/data/video`
- runs `process_video_asset(...)` for each supported video
- stores results in a list
- builds a summary table for notebook inspection

## Validation Checklist
- Run the notebook on at least one local MP4.
- Confirm JSON is generated.
- Confirm sampled timestamps are monotonic.
- Confirm annotated MP4 opens in OpenCV.
- Confirm annotated output preserves source FPS and source duration.
- Confirm partial-body clips do not crash processing.
- Confirm missing hands / occlusion generate explicit limitations.
- Manually review at least one annotated clip for timing and rule consistency.

## Immediate Implementation Notes
- Start from the output and orchestration pattern in `deepface_video_emotion_timeline_v2.ipynb`.
- Keep implementation inside the notebook for v1.
- Favor small, testable helper functions for each stage.
- Make thresholds notebook-level constants so they can be tuned quickly after visual review.
