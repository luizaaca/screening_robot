from video_pipeline.contracts import (
    AudioPacket,
    TranscriptSegment,
    VideoMeta,
    PipelineConfig,
    TranscriptionWindow
)

class TranscriptionWhisperProcessor:
    """Mock/Stub de processador de transcrição usando Whisper ASR."""
    name: str = "transcription_whisper"

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        """Stub: Carregamento do modelo Whisper (não executado no mock)."""
        self.config = config.transcription
        self.pipeline_config = config
        self.video_meta = video_meta

    def process_audio(self, audio_packet: AudioPacket) -> list[TranscriptSegment]:
        """Stub: Retorna segmentos de áudio simulados baseados no arquivo."""
        # Se a duração for extremamente curta ou simulado sem áudio, retorna vazio
        if audio_packet.duration_s <= 0:
            return []
            
        return [
            TranscriptSegment(
                start_s=1.5,
                end_s=4.5,
                text="Eu tenho sentido bastante dor de cabeça."
            ),
            TranscriptSegment(
                start_s=6.0,
                end_s=9.5,
                text="E sinto uma tensão constante no pescoço."
            ),
            TranscriptSegment(
                start_s=13.0,
                end_s=17.0,
                text="Às vezes não consigo dormir direito de tanta preocupação."
            )
        ]

    def aggregate(self, transcript_segments: list[TranscriptSegment]) -> list[TranscriptionWindow]:
        """Agrega segmentos de fala em janelas temporais, duplicando se cruzarem as bordas."""
        duration = self.video_meta.duration_s
        window_s = self.pipeline_config.window_s
        stride_s = self.pipeline_config.stride_s
        
        windows = []
        
        # Se o vídeo não possui áudio, retorna janelas vazias
        if not self.video_meta.has_audio:
            start = 0.0
            while start < duration:
                end = min(start + window_s, duration)
                windows.append(
                    TranscriptionWindow(
                        start_s=start,
                        end_s=end,
                        text="",
                        segments=[],
                        coverage_s=0.0
                    )
                )
                if start + window_s >= duration:
                    break
                start += stride_s
            return windows

        # Se possui áudio, faz a agregação real
        start = 0.0
        while start < duration:
            end = min(start + window_s, duration)
            
            # Um segmento entra na window se end_s > window.start_s E start_s < window.end_s
            win_segments = []
            coverage_s = 0.0
            for seg in transcript_segments:
                if seg.end_s > start and seg.start_s < end:
                    win_segments.append(seg)
                    # Calcula interseção para coverage
                    intersect_start = max(start, seg.start_s)
                    intersect_end = min(end, seg.end_s)
                    coverage_s += max(0.0, intersect_end - intersect_start)
            
            # Concatena texto em ordem temporal
            win_text = " ".join([seg.text for seg in win_segments])
            
            windows.append(
                TranscriptionWindow(
                    start_s=start,
                    end_s=end,
                    text=win_text,
                    segments=win_segments,
                    coverage_s=round(coverage_s, 2)
                )
            )
            
            if start + window_s >= duration:
                break
            start += stride_s
            
        return windows

    def debug_payload(self) -> dict[str, object]:
        """Retorna dados ricos do modulo para o JSON debug."""
        return {"backend": "whisper", "simulated": True}
