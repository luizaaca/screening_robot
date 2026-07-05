from typing import Literal, Optional
from pydantic import BaseModel, Field

class ExpressionDeepFaceConfig(BaseModel):
    """Configuracao do processador de expressoes faciais."""
    label_map: dict[str, str] = Field(default_factory=lambda: {
        "angry": "anger_expression",
        "disgust": "disgust_expression",
        "fear": "fear_expression",
        "happy": "joy_expression",
        "neutral": "neutral_expression",
        "sad": "sad_expression",
        "surprise": "surprise_expression",
    })
    min_confidence: float = 0.30
    min_segment_duration_s: float = 0.20
    detector_backend: str = "opencv"
    enforce_detection: bool = False
    align: bool = True
    silent: bool = True
    actions: tuple[str, ...] = ("emotion",)
    build_model_name: str = "Emotion"
    build_model_task: str = "facial_attribute"


class PoseMediaPipeConfig(BaseModel):
    """Configuracao do processador de sinais posturais."""
    target_sample_fps: float = 5.0
    visibility_threshold: float = 0.50
    ema_alpha: float = 0.35
    ema_max_gap_frames: int = 3

    min_shoulder_width_norm: float = 0.02
    display_score_threshold: float = 0.35
    rule_score_threshold: float = 0.35
    frame_strong_score_threshold: float = 0.70

    base_neck_fraction: float = 0.20
    max_neck_tilt_bonus: float = 0.25
    max_neck_fraction: float = 0.45
    chest_overlap_fraction: float = 0.08
    head_drop_neutral_ratio: float = 0.42
    head_drop_strong_ratio: float = 0.26

    hand_source_quality: dict[str, float] = Field(default_factory=lambda: {
        "holistic_hand": 1.00,
        "pose_wrist_fallback": 0.75,
        "arm_proxy": 0.45,
        "unknown": 0.00,
    })

    rule_order: tuple[str, ...] = (
        "hand_on_head",
        "hand_on_neck",
        "hand_on_chest",
        "head_down",
        "forward_head",
        "rounded_shoulders_or_asymmetry",
    )

    model_asset_path: Optional[str] = None
    model_asset_url: str = (
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
        "holistic_landmarker/float16/1/holistic_landmarker.task"
    )
    min_face_detection_confidence: float = 0.5
    min_face_suppression_threshold: float = 0.5
    min_face_landmarks_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_pose_suppression_threshold: float = 0.5
    min_pose_landmarks_confidence: float = 0.5
    min_hand_landmarks_confidence: float = 0.5
    output_face_blendshapes: bool = False
    output_segmentation_mask: bool = False


class AudioExtractionConfig(BaseModel):
    """Configuracao do extrator de audio."""
    sample_rate: int = 16000
    channels: int = 1
    codec: str = "pcm_s16le"
    format: str = "wav"


class TranscriptionWhisperConfig(BaseModel):
    """Configuracao do processador de transcricao."""
    model_name: str = "base"
    language: str = "pt"                     # "pt", "en", "auto", etc.
    fp16: Optional[bool] = None                 # None = autodetectar conforme acelerador disponivel
    generate_debug_csv: bool = True
    generate_debug_srt: bool = True


class PipelineConfig(BaseModel):
    """Configuração global do pipeline."""
    window_s: float = Field(default=8.0, description="Duração da janela em segundos")
    stride_s: float = Field(default=2.0, description="Stride/deslocamento da janela em segundos")
    debug: bool = Field(default=False, description="Ativar modo debug para logs e outputs adicionais")
    output_dir: Optional[str] = Field(default=None, description="Diretório de saída para salvar JSONs. Se None, não salva")
    
    audio: AudioExtractionConfig = Field(default_factory=AudioExtractionConfig)
    expression: ExpressionDeepFaceConfig = Field(default_factory=ExpressionDeepFaceConfig)
    pose: PoseMediaPipeConfig = Field(default_factory=PoseMediaPipeConfig)
    transcription: TranscriptionWhisperConfig = Field(default_factory=TranscriptionWhisperConfig)


class VideoMeta(BaseModel):
    """Metadata extraída pelo media_probe."""
    video_id: str
    source_path: str
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool


class FramePacket(BaseModel):
    """Frame decodificado com timestamp."""
    timestamp_s: float
    frame_index: int
    # O frame em si (numpy.ndarray) é passado fora do modelo


class AudioPacket(BaseModel):
    """Referência ao áudio extraído."""
    audio_path: str
    sample_rate: int
    channels: int
    codec: str
    format: str
    duration_s: float
    available: bool = True
    extraction_status: Literal["extracted", "no_audio", "failed"] = "extracted"
    error: Optional[str] = None


class FrameDetection(BaseModel):
    """Detecção bruta de um frame individual."""
    timestamp_s: float
    label: str
    score: float  # expression: score ponderado do label; pose: score da regra no frame


class FrameAnalysisRecord(BaseModel):
    """Registro interno de análise visual de um frame."""
    timestamp_s: float
    frame_index: int
    status: Literal["scorable", "insufficient_data", "no_detection", "uncertain", "skipped"]
    detections: list[FrameDetection] = Field(default_factory=list)
    debug: dict[str, object] = Field(default_factory=dict)


class TranscriptSegment(BaseModel):
    """Segmento de fala retornado pelo ASR."""
    start_s: float
    end_s: float
    text: str


class Detection(BaseModel):
    """Detecção agregada dentro de uma window."""
    label: str
    score: float  # média ponderada internamente pelo processador
    support: float  # proporção de frames válidos/scorable


class DominantDetection(BaseModel):
    """Detecção dominante da window."""
    label: str
    score: float


class DetectionWindow(BaseModel):
    """Window com detecções visuais (expression ou pose)."""
    start_s: float
    end_s: float
    detections: list[Detection]  # ordenadas por score decrescente, score >= score_threshold e support >= support_threshold
    dominant: Optional[DominantDetection] = None


class TranscriptionWindow(BaseModel):
    """Window com transcrição de áudio."""
    start_s: float
    end_s: float
    text: str
    segments: list[TranscriptSegment]  # segmentos podem aparecer em mais de uma window (duplicação em bordas)
    coverage_s: float


class ModuleResult(BaseModel):
    """Base comum dos resultados por módulo."""
    video_id: str
    module: str
    duration_s: float
    window_s: float
    stride_s: float


class ExpressionResult(ModuleResult):
    module: str = "expression"
    windows: list[DetectionWindow]


class PoseResult(ModuleResult):
    module: str = "pose"
    windows: list[DetectionWindow]


class TranscriptionResult(ModuleResult):
    module: str = "transcription"
    language: str
    windows: list[TranscriptionWindow]
    has_audio: bool  # False quando o vídeo não tem faixa de áudio


class VideoAnalysisResult(BaseModel):
    """Resultado completo do processamento de um vídeo.
    
    Combina os três módulos alinhados pela mesma timeline de windows.
    Este é o contrato de consumo pelo screening_agent.
    """
    video_id: str
    duration_s: float
    window_s: float
    stride_s: float
    expression: Optional[ExpressionResult] = None
    pose: Optional[PoseResult] = None
    transcription: Optional[TranscriptionResult] = None
