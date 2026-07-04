import os
import json
from pathlib import Path
from video_pipeline.contracts import VideoAnalysisResult, VideoMeta, PipelineConfig

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

    out_dir = Path(config.output_dir)
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
        config_dict = config.model_dump()

        # Expression Debug
        if result.expression:
            expr_debug = {
                "metadata": meta_dict,
                "config": config_dict,
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
                "config": config_dict,
                "module": "pose",
                "processor_payload": module_debug_payloads.get("pose", {}),
                "windows_summary": result.pose.model_dump()["windows"]
            }
            pose_debug_path = out_dir / f"{video_id}.pose.debug.json"
            with open(pose_debug_path, "w", encoding="utf-8") as f:
                json.dump(pose_debug, f, indent=2, ensure_ascii=False)

        # Transcription Debug
        if result.transcription:
            trans_debug = {
                "metadata": meta_dict,
                "config": config_dict,
                "module": "transcription",
                "processor_payload": module_debug_payloads.get("transcription", {}),
                "raw_segments": [seg.model_dump() for seg in raw_transcript_segments],
                "windows_summary": result.transcription.model_dump()["windows"]
            }
            trans_debug_path = out_dir / f"{video_id}.transcription.debug.json"
            with open(trans_debug_path, "w", encoding="utf-8") as f:
                json.dump(trans_debug, f, indent=2, ensure_ascii=False)
