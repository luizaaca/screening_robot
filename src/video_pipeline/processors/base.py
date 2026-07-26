from typing import Protocol
from video_pipeline.contracts import (
    FramePacket,
    FrameDetection,
    VideoMeta,
    PipelineConfig,
    DetectionWindow,
    AudioPacket,
    TranscriptionResult,
    FrameAnalysisRecord
)

class FrameProcessor(Protocol):
    name: str
    sample_fps: float

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        """Inicializa modelos e recursos necessários para o processamento de imagem."""
        ...

    def wants_frame(self, frame_packet: FramePacket) -> bool:
        """Determina se este processador precisa do frame no timestamp dado."""
        ...

    def process_frame(self, frame_packet: FramePacket, frame_bgr: object | None = None) -> FrameAnalysisRecord:
        """Processa um frame e retorna o registro interno de analise."""
        ...

    def aggregate(self, frame_records: list[FrameAnalysisRecord]) -> list[DetectionWindow]:
        """Agrega as detecções brutas por frames em janelas temporais."""
        ...

    def debug_payload(self) -> dict[str, object]:
        """Retorna dados ricos do modulo para o JSON debug."""
        ...


class AudioProcessor(Protocol):
    name: str

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        """Inicializa os modelos e recursos para transcrição/processamento de áudio."""
        ...

    def process_audio(self, audio_packet: AudioPacket) -> TranscriptionResult:
        """Processa a faixa de áudio e retorna texto + segmentos com timestamps."""
        ...

    def debug_payload(self) -> dict[str, object]:
        """Retorna dados ricos do modulo para o JSON debug."""
        ...
