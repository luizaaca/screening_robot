import os
import json
from video_pipeline.contracts import VideoAnalysisResult, VideoMeta, PipelineConfig
from video_pipeline.paths import resolve_project_path


def _module_debug_config(config: PipelineConfig, module: str) -> dict[str, object]:
    module_config = getattr(config, module).model_dump()
    return {
        "window_s": config.window_s,
        "stride_s": config.stride_s,
        "debug": config.debug,
        "output_dir": config.output_dir,
        module: module_config,
    }


def write_results(
    result: VideoAnalysisResult,
    video_meta: VideoMeta,
    config: PipelineConfig,
    module_debug_payloads: dict[str, dict[str, object]],
    raw_transcript_segments: list
) -> None:
    """Grava as saídas JSON padrão e de debug do pipeline em disco se output_dir estiver configurado."""
    if not config.output_dir:
        return

    out_dir = resolve_project_path(config.output_dir, field_name="output_dir")
    os.makedirs(out_dir, exist_ok=True)
    video_id = result.video_id

    # 1. Grava JSONs Default
    if result.expression:
        expr_path = out_dir / f"{video_id}.expression.json"
        with open(expr_path, "w", encoding="utf-8") as f:
            f.write(result.expression.model_dump_json(indent=2))

    if result.pose:
        pose_path = out_dir / f"{video_id}.pose.json"
        with open(pose_path, "w", encoding="utf-8") as f:
            f.write(result.pose.model_dump_json(indent=2))

    if result.transcription:
        trans_path = out_dir / f"{video_id}.transcription.json"
        with open(trans_path, "w", encoding="utf-8") as f:
            f.write(result.transcription.model_dump_json(indent=2))

    # 2. Grava JSONs de Debug (se config.debug for True)
    if config.debug:
        meta_dict = video_meta.model_dump()

        # Expression Debug
        if result.expression:
            expr_debug = {
                "metadata": meta_dict,
                "config": _module_debug_config(config, "expression"),
                "module": "expression",
                "processor_payload": module_debug_payloads.get("expression", {}),
                "windows_summary": result.expression.model_dump()["windows"]
            }
            expr_debug_path = out_dir / f"{video_id}.expression.debug.json"
            with open(expr_debug_path, "w", encoding="utf-8") as f:
                json.dump(expr_debug, f, indent=2, ensure_ascii=False)

        # Pose Debug
        if result.pose:
            pose_debug = {
                "metadata": meta_dict,
                "config": _module_debug_config(config, "pose"),
                "module": "pose",
                "processor_payload": module_debug_payloads.get("pose", {}),
                "windows_summary": result.pose.model_dump()["windows"]
            }
            pose_debug_path = out_dir / f"{video_id}.pose.debug.json"
            with open(pose_debug_path, "w", encoding="utf-8") as f:
                json.dump(pose_debug, f, indent=2, ensure_ascii=False)

        # Transcription Debug
        if result.transcription:
            transcription_payload = result.transcription.model_dump()
            trans_debug = {
                "metadata": meta_dict,
                "config": _module_debug_config(config, "transcription"),
                "module": "transcription",
                "processor_payload": module_debug_payloads.get("transcription", {}),
                "text": result.transcription.text,
                "raw_segments": [seg.model_dump() for seg in raw_transcript_segments],
                "result_summary": transcription_payload,
            }
            trans_debug_path = out_dir / f"{video_id}.transcription.debug.json"
            with open(trans_debug_path, "w", encoding="utf-8") as f:
                json.dump(trans_debug, f, indent=2, ensure_ascii=False)
