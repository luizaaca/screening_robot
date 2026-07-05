import os
from pathlib import Path
from video_pipeline.contracts import (
    PipelineConfig,
    VideoMeta,
    VideoAnalysisResult,
    ExpressionResult,
    PoseResult,
    TranscriptionResult
)
from video_pipeline.media_probe import probe_video
from video_pipeline.video_reader import read_frames
from video_pipeline.audio_extractor import extract_audio
from video_pipeline.writers import write_results
from video_pipeline.config import load_pipeline_config
from video_pipeline.processors import (
    ExpressionDeepFaceProcessor,
    PoseMediaPipeProcessor,
    TranscriptionWhisperProcessor
)

def process_video(
    video_path: str,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> VideoAnalysisResult:
    """Processa um único vídeo de forma multimodal (expressões, postura e transcrição).
    
    Orquestra a decodificação dos frames, extração de áudio, processamento
    dos sinais e agrupamento em janelas de tempo.
    """
    if config is not None and config_path is not None:
        raise ValueError("config e config_path são mutuamente exclusivos.")
        
    if config is None:
        config = load_pipeline_config(config_path)

    # 1. Proba metadata comum do vídeo
    video_meta: VideoMeta = probe_video(video_path)
    video_id = video_meta.video_id
    duration = video_meta.duration_s

    # 2. Inicializa os processadores
    expression_proc = ExpressionDeepFaceProcessor()
    pose_proc = PoseMediaPipeProcessor()
    trans_proc = TranscriptionWhisperProcessor()

    expression_proc.setup(video_meta, config)
    pose_proc.setup(video_meta, config)
    trans_proc.setup(video_meta, config)

    # 3. Branch Visual: abre o vídeo uma vez e distribui os frames
    raw_frame_records = {
        "expression": [],
        "pose": []
    }

    # Só roda processadores visuais se a duração for maior que 0
    if duration > 0:
        for frame_packet, frame_bgr in read_frames(video_meta):
            # Repassa o frame se o processador desejar
            if expression_proc.wants_frame(frame_packet):
                rec = expression_proc.process_frame(frame_packet, frame_bgr)
                raw_frame_records["expression"].append(rec)
                
            if pose_proc.wants_frame(frame_packet):
                rec = pose_proc.process_frame(frame_packet, frame_bgr)
                raw_frame_records["pose"].append(rec)

    # Agregação Visual por Window (executada internamente em cada processador)
    expression_windows = expression_proc.aggregate(
        raw_frame_records["expression"]
    )
    pose_windows = pose_proc.aggregate(
        raw_frame_records["pose"]
    )

    expression_res = ExpressionResult(
        video_id=video_id,
        duration_s=duration,
        window_s=config.window_s,
        stride_s=config.stride_s,
        windows=expression_windows
    )

    pose_res = PoseResult(
        video_id=video_id,
        duration_s=duration,
        window_s=config.window_s,
        stride_s=config.stride_s,
        windows=pose_windows
    )

    # 4. Branch de Áudio: extrai áudio e faz transcrição
    raw_transcript_segments = []
    trans_windows = []

    # Extracao de audio: guarda o WAV no output apenas em debug.
    audio_output_dir = config.output_dir if config.debug else None
    audio_packet = extract_audio(video_meta, config.audio, audio_output_dir)
    raw_transcript_segments = trans_proc.process_audio(audio_packet)

    # Agregação de áudio por window
    trans_windows = trans_proc.aggregate(
        raw_transcript_segments
    )

    trans_res = TranscriptionResult(
        video_id=video_id,
        duration_s=duration,
        window_s=config.window_s,
        stride_s=config.stride_s,
        language=config.transcription.language,
        windows=trans_windows,
        has_audio=video_meta.has_audio
    )

    # 5. Monta o resultado combinado em memória
    analysis_result = VideoAnalysisResult(
        video_id=video_id,
        duration_s=duration,
        window_s=config.window_s,
        stride_s=config.stride_s,
        expression=expression_res,
        pose=pose_res,
        transcription=trans_res
    )

    # Coleta payloads de debug
    module_debug_payloads = {
        "expression": expression_proc.debug_payload(),
        "pose": pose_proc.debug_payload(),
        "transcription": trans_proc.debug_payload()
    }

    # 6. Gravação opcional em disco dos JSONs padrão/debug
    write_results(
        result=analysis_result,
        video_meta=video_meta,
        config=config,
        module_debug_payloads=module_debug_payloads,
        raw_transcript_segments=raw_transcript_segments
    )

    # Limpeza opcional do áudio temporário caso não queira salvá-lo no output
    # (Só apagamos se não for modo debug e se tiver sido gerado na pasta temporária padrão)
    if (
        not config.debug
        and audio_packet.available
        and audio_packet.audio_path
        and ".tmp_audio" in audio_packet.audio_path
    ):
        try:
            if os.path.exists(audio_packet.audio_path):
                os.remove(audio_packet.audio_path)
        except Exception:
            pass

    return analysis_result
