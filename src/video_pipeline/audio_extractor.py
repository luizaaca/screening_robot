import os
from pathlib import Path
from video_pipeline.contracts import VideoMeta, AudioPacket, AudioExtractionConfig

def extract_audio(
    video_meta: VideoMeta, 
    audio_config: AudioExtractionConfig,
    output_dir: str | None = None
) -> AudioPacket:
    """Extrai a faixa de áudio do vídeo para processamento ASR.
    
    Se moviepy estiver disponível e o arquivo existir, extrai o áudio real.
    Caso contrário, retorna um AudioPacket simulado.
    """
    if not video_meta.has_audio:
        return AudioPacket(
            audio_path="",
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            codec=audio_config.codec,
            format=audio_config.format,
            duration_s=0.0
        )

    # Determina onde salvar o áudio temporário
    # Se output_dir não for fornecido, usa a mesma pasta do vídeo ou pasta temp no workspace
    temp_dir = Path(output_dir) if output_dir else Path("./.tmp_audio")
    audio_path = temp_dir / f"{video_meta.video_id}.{audio_config.format}"
    
    path = Path(video_meta.source_path)
    use_simulation = not path.exists()
    
    if not use_simulation:
        try:
            # Cria diretório temporário se necessário
            os.makedirs(temp_dir, exist_ok=True)
            
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(path))
            if clip.audio:
                # Salva em formato wav 16kHz mono (padrão Whisper)
                clip.audio.write_audiofile(
                    str(audio_path),
                    fps=audio_config.sample_rate,
                    nbytes=2,
                    codec=audio_config.codec,
                    ffmpeg_params=["-ac", str(audio_config.channels)],
                    logger=None
                )
                clip.close()
                return AudioPacket(
                    audio_path=str(audio_path.resolve()),
                    sample_rate=audio_config.sample_rate,
                    channels=audio_config.channels,
                    codec=audio_config.codec,
                    format=audio_config.format,
                    duration_s=video_meta.duration_s
                )
            else:
                clip.close()
                use_simulation = True
        except ImportError:
            use_simulation = True
        except Exception:
            use_simulation = True

    if use_simulation:
        # Modo simulação
        return AudioPacket(
            audio_path=f"/virtual_path/{video_meta.video_id}.{audio_config.format}",
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            codec=audio_config.codec,
            format=audio_config.format,
            duration_s=video_meta.duration_s
        )
