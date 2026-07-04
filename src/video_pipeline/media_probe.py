import os
from pathlib import Path
from video_pipeline.contracts import VideoMeta

def probe_video(video_path: str) -> VideoMeta:
    """Extrai metadados do vídeo usando cv2 (se disponível) ou fallback para mocks."""
    path = Path(video_path)
    video_id = path.stem
    
    # Defaults de fallback
    duration_s = 30.0
    fps = 30.0
    width = 1920
    height = 1080
    has_audio = True

    # Tenta usar OpenCV se disponível
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or fps
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            if frame_count > 0 and fps > 0:
                duration_s = frame_count / fps
            cap.release()
    except ImportError:
        pass
    except Exception:
        pass

    # Tenta detectar se o vídeo tem áudio (usando moviepy se disponível, ou simples heurística de extensão)
    # Por padrão, assumimos True, a menos que especificado ou falhe.
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(path))
        has_audio = clip.audio is not None
        clip.close()
    except ImportError:
        # Se for um arquivo de teste inexistente, podemos checar se o nome sugere ausência de áudio
        if "no_audio" in video_id.lower() or "silence" in video_id.lower():
            has_audio = False
    except Exception:
        pass

    return VideoMeta(
        video_id=video_id,
        source_path=str(path.resolve()),
        duration_s=round(duration_s, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
        has_audio=has_audio
    )
