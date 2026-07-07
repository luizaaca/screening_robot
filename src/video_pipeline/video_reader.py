from typing import Iterator, Optional, Tuple, Any
from pathlib import Path
from video_pipeline.contracts import VideoMeta, FramePacket

def read_frames(video_meta: VideoMeta) -> Iterator[Tuple[FramePacket, Optional[Any]]]:
    """Lê todos os frames do vídeo em ordem sequencial.
    
    Se OpenCV estiver disponível e o arquivo existir, lê os frames reais.
    Caso contrário, simula a emissão dos packets correspondentes.
    """
    path = Path(video_meta.source_path)
    
    # Se o arquivo não existir ou falhar a leitura real, entra no modo de simulação
    use_simulation = not path.exists()
    
    if not use_simulation:
        cap = None
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                fps = video_meta.fps if video_meta.fps > 0 else cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Redimensiona frame para no máximo 480px mantendo o aspect ratio
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

                return
            use_simulation = True
        except ImportError:
            use_simulation = True
        except Exception:
            use_simulation = True
        finally:
            if cap is not None:
                cap.release()

    if use_simulation:
        # Modo de simulação: emite pacotes fictícios com frame=None
        fps = video_meta.fps if video_meta.fps > 0 else 30.0
        step = 1.0 / fps
        
        current_time = 0.0
        frame_idx = 0
        while current_time < video_meta.duration_s:
            packet = FramePacket(timestamp_s=round(current_time, 3), frame_index=frame_idx)
            yield packet, None
            current_time += step
            frame_idx += 1
