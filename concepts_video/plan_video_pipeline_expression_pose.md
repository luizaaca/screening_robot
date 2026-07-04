# Planejamento do pipeline multimodal: expression + pose + transcription

Atualizado em: 2026-07-04

Este documento define a arquitetura, os contratos, as configuracoes e os JSONs de saida do pipeline multimodal de processamento de videos.

## Objetivo

Receber uma lista de videos e executar, de forma eficiente, processamentos independentes de:

- `expression`: deteccao de expressoes faciais por DeepFace ou backend equivalente.
- `pose`: deteccao de sinais posturais por MediaPipe Holistic ou backend equivalente.
- `transcription`: transcricao da faixa de audio por Whisper ou backend equivalente.

Cada modulo deve produzir um JSON proprio por video, contendo resultados agregados por janela de tempo (`window`). O JSON default deve ser o mais enxuto possivel, mantendo a semantica analitica. Informacoes internas, metricas intermediarias, caminhos locais e metadados operacionais devem ir para um JSON de debug.

## Principio de arquitetura

O pipeline deve ter uma orquestracao unica por video, mas com branches independentes por modalidade.

Modelo desejado:

```text
lista de videos
  -> orchestrator
    -> media probe comum
    -> video branch
      -> video reader compartilhado
      -> expression processor
      -> pose processor
    -> audio branch
      -> audio extractor
      -> transcription processor
    -> aggregation por window
    -> json default por modulo
    -> json debug opcional por modulo
```

O video deve ser decodificado uma vez para os modulos baseados em frames. O audio deve ser extraido uma vez para os modulos baseados em fala. As duas branches compartilham `video_id`, duracao e timeline, mas nao precisam compartilhar o mesmo decoder.

A branch visual nao deve fazer sampling global antes dos processadores. O `video_reader` compartilhado deve ler o video sequencialmente e entregar todos os frames aos processadores visuais. Cada processador decide internamente se analisa todos os frames ou se aplica sampling proprio. Essa regra preserva a semantica analitica de cada modulo e evita que uma otimizacao global altere resultados de deteccao.

## Decisao sobre transcricao

A transcricao deve entrar no mesmo pipeline de orquestracao, mas como uma branch propria de audio.

Nao e ideal misturar transcricao com o `video_reader` de frames, porque ela nao depende de frames. Ela depende da faixa de audio do arquivo. Tambem nao e ideal fazer um pipeline totalmente separado neste momento, porque os resultados precisam ficar alinhados por tempo com `expression` e `pose`.

Decisao:

```text
Mesmo orchestrator.
Branch de audio independente.
JSON proprio de transcription.
Agregacao por window para alinhar com expression e pose.
```

Um pipeline separado para transcricao so deve ser considerado se:

- o ASR rodar em infraestrutura diferente;
- o custo/tempo de transcricao exigir fila propria;
- a transcricao for reaproveitada por muitos fluxos externos ao pipeline de video;
- houver necessidade de retry, cache e versionamento independentes.

Mesmo nesse caso, o contrato deve preservar `video_id`, `duration_s`, `start_s` e `end_s` para permitir merge posterior.

## Estrutura proposta de modulos

```text
video_pipeline/
  orchestrator.py
  media_probe.py
  video_reader.py
  audio_extractor.py
  config.py
  contracts.py
  writers.py
  configs/
    default.yaml

  processors/
    base.py
    expression_deepface.py
    pose_mediapipe.py
    transcription_whisper.py
```

Responsabilidades:

- `orchestrator.py`: recebe o caminho do video, instancia os processadores e coordena a execucao. Deriva `video_id` do nome do arquivo sem extensao (ex: `domestic_abuse1.mp4` -> `domestic_abuse1`).
- `media_probe.py`: le metadata comum do arquivo, como duracao, fps, dimensoes e presenca de audio.
- `video_reader.py`: abre o video, decodifica todos os frames em ordem sequencial, calcula timestamps por `frame_index / fps` e entrega os frames para os processadores visuais.
- `audio_extractor.py`: extrai a faixa de audio uma vez, aplicando formato normalizado exigido pela transcricao (ex: sample rate e codec extraidos do `TranscriptionWhisperConfig`).
- `config.py`: carrega e valida arquivos de configuracao do pipeline.
- `configs/default.yaml`: declara os parametros padrao compativeis com a semantica analitica dos processadores.
- `contracts.py`: define contratos comuns como `VideoMeta`, `FramePacket`, `AudioPacket`, `FrameAnalysisRecord`, `FrameDetection`, `TranscriptSegment` e os modelos de resultado.
- `writers.py`: grava JSON default e JSON debug em disco (ativado apenas quando `output_dir` esta configurado).
- `expression_deepface.py`: encapsula a logica de expressao facial com DeepFace. Inclui agregacao por window internamente.
- `pose_mediapipe.py`: encapsula a logica de postura com MediaPipe Holistic. Inclui agregacao por window internamente.
- `transcription_whisper.py`: encapsula a logica de transcricao com Whisper. Inclui agregacao por window internamente.

## Interfaces dos processadores (Protocols)

Os modulos de frame e de audio tem naturezas diferentes. A interface e separada por modalidade, usando `Protocol` (consistente com o padrao do `screening_agent`, que usa Protocol para `ControlModel`).

Cada processador e responsavel por sua propria agregacao por window. Nao existe `aggregation.py` centralizado.

Processadores visuais:

```python
class FrameProcessor(Protocol):
    name: str

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        ...

    def wants_frame(self, frame_packet: FramePacket) -> bool:
        """Decisao interna do processador. O reader/orchestrator nao faz sampling global."""
        ...

    def process_frame(self, frame_packet: FramePacket, frame_bgr: object | None = None) -> FrameAnalysisRecord:
        """Processa um frame e retorna o registro interno de analise."""
        ...

    def aggregate(self, frame_records: list[FrameAnalysisRecord]) -> list[DetectionWindow]:
        """Agrega registros internos em windows."""
        ...

    def debug_payload(self) -> dict[str, object]:
        """Retorna dados ricos do modulo para o JSON debug."""
        ...
```

Processadores de audio:

```python
class AudioProcessor(Protocol):
    name: str

    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        ...

    def process_audio(self, audio_packet: AudioPacket) -> list[TranscriptSegment]:
        ...

    def aggregate(self, transcript_segments: list[TranscriptSegment]) -> list[TranscriptionWindow]:
        """Agrega segmentos em windows alinhadas com a timeline."""
        ...

    def debug_payload(self) -> dict[str, object]:
        """Retorna dados ricos do modulo para o JSON debug."""
        ...
```

## Fluxo por video

```text
1. Resolver caminho do video.
2. Derivar video_id do nome do arquivo sem extensao.
3. Ler metadata comum com media_probe:
   - video_id
   - duracao
   - fps
   - largura
   - altura
   - presenca de audio
4. Instanciar processadores sem config no construtor e chamar `setup(video_meta, config)`.
5. Executar branch visual:
   - abrir video uma vez.
   - decodificar todos os frames sequencialmente.
   - calcular timestamp por `frame_index / fps`, preservando a base temporal do video.
   - oferecer cada frame a todos os processadores visuais.
   - cada processador decide internamente, via `wants_frame()` ou logica equivalente, se processa ou ignora o frame.
   - coletar registros internos de analise por frame.
   - cada processador agrega seus registros em windows internamente.
6. Executar branch de audio:
   - extrair audio uma vez.
   - normalizar audio para ASR.
   - transcrever com Whisper ou backend equivalente.
   - coletar segmentos com timestamps.
   - o processador agrega segmentos em windows internamente.
7. Montar VideoAnalysisResult em memoria.
8. Se output_dir configurado: gravar JSONs default por modulo.
9. Se output_dir configurado e debug=True: gravar JSONs debug por modulo.
10. Retornar VideoAnalysisResult ao chamador.
```

As branches visual e audio podem rodar em paralelo no futuro, mas o desenho inicial pode mante-las sequenciais para reduzir complexidade.

## Contrato do JSON default

O JSON default deve responder:

```text
Neste video, nesta janela de tempo, quais sinais ou conteudos foram detectados?
```

Ele nao deve carregar detalhes de execucao, caminhos locais irrelevantes, landmarks, metricas geometricas internas, evidencias detalhadas ou parametros de modelo.

Campos comuns:

```json
{
  "video_id": "domestic_abuse1",
  "module": "pose",
  "duration_s": 42.3,
  "window_s": 8.0,
  "stride_s": 2.0,
  "windows": []
}
```

Campos minimos comuns:

- `video_id`
- `module`
- `duration_s`
- `window_s`
- `stride_s`
- `windows`

## JSON default de expression

O modulo `expression` deve emitir deteccoes por window.

Exemplo:

```json
{
  "video_id": "domestic_abuse1",
  "module": "expression",
  "duration_s": 42.3,
  "window_s": 8.0,
  "stride_s": 2.0,
  "windows": [
    {
      "start_s": 0.0,
      "end_s": 8.0,
      "detections": [
        {
          "label": "sad_expression",
          "score": 0.81,
          "support": 0.58
        },
        {
          "label": "neutral_expression",
          "score": 0.44,
          "support": 0.21
        }
      ],
      "dominant": {
        "label": "sad_expression",
        "score": 0.81
      }
    }
  ]
}
```

Notas:

- `score`: confianca ou intensidade agregada da expressao na window.
- `support`: proporcao de frames validos da window em que a expressao apareceu.
- `dominant`: deteccao de maior score na window.
- `detections`: todas as expressoes relevantes detectadas na window, nao apenas a dominante.

## JSON default de pose

O ajuste principal do modulo `pose` e abandonar uma saida centrada apenas em `window_level` e passar a emitir todas as poses/sinais detectados por window, com score, no mesmo estilo do JSON de expression.

Exemplo:

```json
{
  "video_id": "domestic_abuse1",
  "module": "pose",
  "duration_s": 42.3,
  "window_s": 8.0,
  "stride_s": 2.0,
  "windows": [
    {
      "start_s": 0.0,
      "end_s": 8.0,
      "detections": [
        {
          "label": "hand_on_neck",
          "score": 0.76,
          "support": 0.42
        },
        {
          "label": "head_down",
          "score": 0.61,
          "support": 0.28
        },
        {
          "label": "hand_on_head",
          "score": 0.54,
          "support": 0.17
        }
      ],
      "dominant": {
        "label": "hand_on_neck",
        "score": 0.76
      }
    }
  ]
}
```

Notas:

- `detections` deve listar todos os sinais posturais relevantes da window.
- Cada item deve ter `label`, `score` e `support`.
- Em pose, `score` representa o pico (`max`) do score da regra dentro da window, e `support` representa `detection_ratio`.
- `window_level`, `signal_ratio`, `strong_signal_ratio`, `most_common_trigger` e explicacoes textuais podem continuar existindo, mas devem ficar no JSON debug, nao no default.
- A semantica principal do default deve ser a deteccao por label, alinhada ao contrato de expression.

## JSON default de transcription

O modulo `transcription` tem uma natureza diferente: ele nao detecta uma classe visual, ele transforma fala em texto com timestamps. Ainda assim, o JSON default deve ser agregado por window para alinhar com `expression` e `pose`.

Exemplo:

```json
{
  "video_id": "domestic_abuse1",
  "module": "transcription",
  "duration_s": 42.3,
  "window_s": 8.0,
  "stride_s": 2.0,
  "language": "pt",
  "has_audio": true,
  "windows": [
    {
      "start_s": 0.0,
      "end_s": 8.0,
      "text": "texto transcrito dentro desta janela",
      "segments": [
        {
          "start_s": 1.2,
          "end_s": 4.8,
          "text": "texto transcrito dentro desta janela"
        }
      ],
      "coverage_s": 3.6
    }
  ]
}
```

Notas:

- `text`: texto concatenado dos segmentos que intersectam a window.
- `segments`: segmentos de fala que caem total ou parcialmente dentro da window.
- `coverage_s`: duracao total aproximada de fala coberta pela window.
- `language`: idioma detectado ou configurado.
- O default nao deve incluir caminho do audio extraido, modelo, CSV, SRT, parametros de audio ou paths locais.

Se no futuro quisermos detectar sinais semanticos no texto, isso deve ser outro modulo, por exemplo:

```text
speech_analysis
```

Esse modulo poderia consumir o JSON de transcricao e emitir deteccoes por window, como:

- `aggressive_language`
- `distress_language`
- `request_for_help`
- `silence`
- `high_speech_density`

Mas a transcricao em si deve permanecer responsavel apenas por texto e timestamps.

## Saidas esperadas por video

O retorno principal e o `VideoAnalysisResult` em memoria. Se `output_dir` estiver configurado, os seguintes arquivos sao gravados:

```text
domestic_abuse1.expression.json
domestic_abuse1.pose.json
domestic_abuse1.transcription.json
```

Todos os modulos usam a mesma timeline de windows:

```text
window_s = 8.0
stride_s = 2.0
```

Essa decisao facilita analises posteriores por alinhamento temporal.

## Campos removidos do JSON default

Remover do JSON default e manter apenas em debug, quando necessario:

- `source_path`
- `video_path`
- `audio_path`
- `annotated_video_path`
- `json_output_path`
- `csv_path`
- `srt_path`
- `source_fps`
- `frames_total`
- politica de sampling interna de cada processador
- contadores operacionais como `frames_total`, `frames_delivered`, `frames_processed` e `frames_skipped`
- backend detalhado do pipeline
- caminho do modelo
- URL do modelo
- nome do modelo ASR
- parametros de audio
- thresholds internos
- landmarks
- metricas geometricas
- evidencias detalhadas
- limitacoes extensas
- notas de debug
- `frame_signals` completos
- transcricao bruta completa fora das windows
- imagens ou caminhos de artefatos anotados

## JSON debug

O JSON debug deve responder:

```text
Por que o modulo decidiu isso e quais dados intermediarios sustentam a decisao?
```

Exemplos de arquivos:

```text
domestic_abuse1.expression.debug.json
domestic_abuse1.pose.debug.json
domestic_abuse1.transcription.debug.json
```

Conteudo esperado no debug:

- metadata completa do video.
- parametros do pipeline.
- backend e versao/modelo usado.
- `source_fps`, `frames_total`, `frames_delivered`, `frames_processed`, `frames_skipped`.
- politica de sampling interna por processador, quando houver.
- `frame_signals` ou deteccoes por frame.
- landmarks e metricas internas, quando uteis.
- evidencias por regra de pose.
- thresholds usados.
- warnings, falhas e limitacoes.
- caminho do audio extraido, quando preservado.
- resultado bruto do ASR, quando util.
- segmentos ASR originais.
- caminhos para CSV/SRT.
- caminhos para videos anotados ou artefatos auxiliares.

## Agregacao por window

Cada processador e responsavel por sua propria agregacao. As regras abaixo sao o contrato que cada processador deve seguir.

`FrameAnalysisRecord.detections` e a fonte tipada para a agregacao do JSON default.
`FrameAnalysisRecord.debug` e diagnostico/auditoria e nao deve ser usado pelo agregador para descobrir scores, labels ou flags de deteccao.

Para `expression`:

- Coletar frames validos dentro de `[start_s, end_s]`.
- Agrupar deteccoes por `label`.
- Calcular `support` como proporcao dos frames validos (scorable) em que o label apareceu.
- Calcular `score` como media ponderada dos scores do label. A ponderacao e interna ao processador (cada processador ja emite `FrameDetection.score` ponderado).
- Ordenar `detections` por `score` decrescente.
- Definir `dominant` como a primeira deteccao da lista.
- Expor as deteccoes agregadas sem threshold adicional de apresentacao.

Regra:

```text
score = media dos FrameDetection.score do label (ja ponderados internamente pelo processador)
support = frames_com_label / frames_validos_da_window
dominant = deteccao com maior score
```

Para `pose`, a agregacao deve preservar a semantica dos resumos por regra:

- Coletar frames amostrados internamente pelo processador dentro de `[start_s, end_s]`.
- Considerar `scorable_frames` como records com `status == "scorable"`.
- O processador de pose deve emitir em `FrameAnalysisRecord.detections` apenas regras ativas no frame, isto e, regras com `state != "unknown"` e `passed_threshold == True`.
- Cada item de `FrameDetection` de pose representa uma regra ativa naquele frame: `label = rule_name` e `score = rule.score`.
- Garantir no maximo uma `FrameDetection` por `label` em cada frame.
- Iterar em `config.pose.rule_order`.
- Incluir uma deteccao de window quando `frames_detected > 0`.
- Mapear `Detection.score` para o pico dos `FrameDetection.score` daquela regra na window (`max(scores)`), nao para a media.
- Mapear `Detection.support` para `detection_ratio = frames_detected / len(scorable_frames)`.
- Manter `score_mean`, `frames_detected`, `window_level`, `signal_ratio`, `strong_signal_ratio`, `most_common_trigger`, `window_score_mean`, `window_score_peak` e explicacoes no JSON debug.
- Ordenar as deteccoes por `(score, support)` decrescente.

Regra de mapeamento para o model default de pose:

```text
score = max(FrameDetection.score do label nos records scorable da window)
support = frames_scorable_com_FrameDetection_do_label / frames_scorable_da_window
dominant = primeira deteccao ordenada por (score, support)
```

O JSON default de pose nao deve aplicar threshold adicional de apresentacao. O `rule_score_threshold = 0.35` da config de pose controla a deteccao por frame via `passed_threshold`. Se for desejado um JSON operacional mais compacto no futuro, isso deve ser outro parametro explicito de pose.

Para `transcription`:

- Coletar segmentos ASR que intersectam `[start_s, end_s]`.
- Concatenar texto em ordem temporal.
- Preservar segmentos com `start_s`, `end_s` e `text`.
- Calcular `coverage_s` como soma aproximada das intersecoes entre segmentos e a window.
- Nao inventar deteccoes emocionais ou semanticas no modulo de transcricao.

Regra sugerida inicial:

```text
segmento entra na window se segment.end_s > window.start_s
e segment.start_s < window.end_s
```

Se um segmento atravessar bordas de window, ele pode aparecer em mais de uma window. Isso e aceitavel para alinhamento temporal. Uma alternativa futura e dividir segmentos nas bordas, mas isso pode prejudicar legibilidade do texto.

## Semantica analitica dos modulos

`expression_deepface`:

- Usa DeepFace para analisar emocao facial por frame.
- Processa todos os frames sequenciais entregues pelo `video_reader`.
- Seleciona a maior face valida quando houver mais de uma face.
- Normaliza scores para o intervalo `[0.0, 1.0]`.
- Aplica `min_confidence = 0.30` para aceitar a emocao por frame.
- Mantem segmentos e percentuais globais no debug para auditoria.
- Projeta a saida default para `windows[].detections[]`.

`pose_mediapipe`:

- Usa MediaPipe Tasks Holistic em modo `VIDEO`.
- Recebe todos os frames da branch visual, mas aplica sampling interno por `target_sample_fps = 5.0`.
- Usa `frame_step = max(1, round(source_fps / target_sample_fps))`.
- Usa `timestamp_s = frame_index / source_fps` e `timestamp_ms = round(timestamp_s * 1000.0)` para inferencia em video.
- Extrai landmarks de pose, pose world, face, maos 2D e maos world.
- Aplica suavizacao EMA por grupo de landmarks antes de calcular geometria.
- Calcula geometria de referencia e avalia regras em ordem de dependencia.
- Mantem `frame_signals`, `window_summaries`, `video_summary`, metricas, regras e evidencias no debug.
- Projeta a saida default para `windows[].detections[]`, usando `score=max(rule.score)` e `support=detection_ratio`.

`transcription_whisper`:

- Consome audio ja normalizado pelo `audio_extractor` em mono, 16 kHz e codec `pcm_s16le`.
- Usa Whisper com modelo `base` e idioma `pt` por padrao.
- Define `fp16` automaticamente conforme disponibilidade de acelerador, salvo override de config.
- Normaliza segmentos com timestamps.
- Agrega segmentos em windows alinhadas com `expression` e `pose`.
- Mantem CSV, SRT, audio extraido, nome do modelo e payload bruto do ASR no debug ou em artefatos auxiliares.

## Decisoes

- O novo pipeline sera planejado com orquestracao unica por video.
- `expression`, `pose` e `transcription` serao modulos dedicados.
- `expression` e `pose` usam a branch visual com leitor de video compartilhado.
- `transcription` usa branch de audio propria com extracao de audio.
- O retorno principal e o `VideoAnalysisResult` em memoria.
- Gravacao de JSONs em disco e opcional, controlada por `output_dir`.
- Cada processador faz sua propria agregacao por window internamente.
- O `video_id` e derivado do nome do arquivo sem extensao.
- O JSON default sera enxuto e orientado a windows.
- O JSON de pose default deve listar todas as poses/sinais detectados por window com score, igual ao estilo de expression.
- O JSON de transcription default deve agregar texto e segmentos por window para alinhar com expression e pose.
- Metadados irrelevantes para analise de deteccao de emocao, postura e transcricao saem do default e ficam somente em debug.
- A branch visual nao faz sampling global. Todos os frames sao entregues aos processadores visuais; qualquer sampling deve ser decisao interna do processador.

## Decisoes de desenho resolvidas

Resolvidas em: 2026-07-04

### 1. Calculo do `score` por label na window

**Decisao: media ponderada para `expression`; mapeamento especifico para `pose`.**

Em `expression`, frames com deteccao mais confiavel pesam mais no calculo do score agregado. Reduz ruido de frames com deteccao marginal.

Em `pose`, `Detection.score` representa o pico da regra na window (`max(rule.score)`), e `score_mean` fica no debug quando necessario.

### 2. Base de calculo do `support`

**Decisao: frames validos/scorable.**

O denominador do support considera apenas frames onde o detector conseguiu operar (face detectada para expression, corpo detectado para pose). Reflete a proporcao real do sinal nos frames aproveitaveis.

### 3. Inclusao de deteccoes no JSON default

**Decisao: o JSON default nao aplica thresholds adicionais de apresentacao.**

Cada processador e responsavel por decidir suas deteccoes internamente. O JSON default apenas projeta as deteccoes agregadas para o contrato comum `Detection(label, score, support)`.

Para `expression`, a aceitacao por frame e controlada por `ExpressionDeepFaceConfig.min_confidence`.

Para `pose`, a aceitacao por frame e controlada por `PoseMediaPipeConfig.rule_score_threshold` e por `passed_threshold`.

### 4. Gravacao do JSON de debug

**Decisao: apenas quando `debug=True`.**

Reduz I/O e armazenamento por padrao. O flag `debug` deve ser parametro do orchestrator.

### 5. Convencao de nomes dos arquivos de saida

**Decisao: flat com dots.**

```text
{video_id}.{module}.json
{video_id}.{module}.debug.json
```

Exemplos:

```text
domestic_abuse1.expression.json
domestic_abuse1.pose.json
domestic_abuse1.transcription.json
domestic_abuse1.pose.debug.json
```

### 6. Nomenclatura final dos labels de pose

**Decisao: usar os labels definidos pelo `rule_order` da config de pose.**

Labels padrao:

- `hand_on_head`
- `hand_on_neck`
- `hand_on_chest`
- `head_down`
- `forward_head`
- `rounded_shoulders_or_asymmetry`

Labels como `arms_crossed`, `shoulders_raised` e `hand_on_face` nao pertencem ao conjunto padrao de regras de pose e nao devem ser introduzidos sem nova regra analitica e configuracao correspondente.

### 7. Geracao de SRT/CSV na transcricao

**Decisao: apenas em debug ou como artefatos auxiliares.**

O JSON default por windows e suficiente para analise. SRT e CSV sao uteis para revisao manual e devem ser gerados apenas quando `debug=True`.

### 8. Segmentos de transcricao que cruzam bordas de window

**Decisao: duplicar.**

O segmento aparece em todas as windows que intersecta, preservando o texto integral. Isso garante que cada window tenha contexto completo de fala.

### 9. Estrategia para videos sem audio

**Decisao: JSON de transcricao com windows vazias e mensagem informativa.**

O pipeline nao deve falhar. O JSON de transcricao deve ser gerado normalmente, com windows vazias e um campo informativo indicando ausencia de faixa de audio no video.

### 10. Idioma padrao e autodeteccao

**Decisao: `pt` como padrao, com parametro `language` configuravel.**

O parametro `language` aceita codigos de idioma (ex: `pt`, `en`, `es`) e o valor especial `auto` para autodeteccao pelo Whisper. Default: `pt`.

### 11. Configuracao da extracao de audio

**Decisao: separar configuracao de extracao de audio e configuracao do Whisper.**

O `audio_extractor.py` e responsavel por extrair e normalizar a faixa de audio antes do processador de transcricao. Portanto, `sample_rate`, `channels`, `codec` e `format` pertencem a `AudioExtractionConfig`, injetada pelo `orchestrator` no extrator. O `TranscriptionWhisperConfig` deve conter apenas parametros do ASR e dos artefatos de transcricao.

O fluxo esperado e:

```python
audio_packet = extract_audio(video_path, config.audio)
transcription_processor = TranscriptionWhisperProcessor()
transcription_processor.setup(video_meta, config)
segments = transcription_processor.process_audio(audio_packet)
```

O `AudioPacket` deve registrar os parametros efetivos produzidos pelo extrator. O processador Whisper pode validar ou registrar esses parametros no debug, mas nao deve depender de constantes hardcoded para combinar com o formato esperado.

### 12. Cache de audio extraido e transcricao

**Decisao: sem cache na versao inicial.**

Sempre re-executar. Mais simples de implementar inicialmente. Cache por hash do arquivo + parametros pode ser adicionado no futuro se o custo de re-execucao se tornar problema.

### 13. Sampling de frames na branch visual

**Decisao: sem sampling global no `video_reader` ou no `orchestrator`.**

O `video_reader` deve decodificar todos os frames do video em ordem sequencial e o `orchestrator` deve oferecer todos os frames aos processadores visuais. Cada processador decide internamente se processa todos os frames ou se ignora parte deles.

Motivacao:

- `expression_deepface` precisa analisar todos os frames para manter seus segmentos, percentuais, transicoes, contagem de frames incertos/sem face, `support` e dominante por window.
- Sampling global muda resultados de deteccao e impede que modulos com necessidades diferentes coexistam na mesma branch visual.
- `pose_mediapipe` preserva seu `target_sample_fps` como sampling interno, sem afetar os demais processadores.

Consequencia de contrato:

- `PipelineConfig` nao deve ter `sample_fps` visual global.
- Parametros de sampling pertencem a configs especificas de processadores, quando existirem.
- JSON default nao deve expor sampling. JSON debug pode registrar `frames_total`, `frames_delivered`, `frames_processed`, `frames_skipped` e a politica interna de cada processador.

### 14. Semantica do modulo `pose_mediapipe`

**Decisao: preservar a deteccao do modulo e adaptar apenas o envelope de saida para models.**

O modulo `pose_mediapipe` deve manter os parametros e a sequencia analitica definidos em config:

- `target_sample_fps = 5.0` como sampling interno do processador.
- `frame_step = max(1, round(source_fps / target_sample_fps))`.
- `effective_sample_fps = source_fps / frame_step` apenas como metrica de debug.
- `timestamp_s = frame_idx / source_fps` e `timestamp_ms = round(timestamp_s * 1000.0)` para `detect_for_video`.
- `VISIBILITY_THRESHOLD = 0.50`.
- `EMA_ALPHA = 0.35` e `EMA_MAX_GAP_FRAMES = 3`, com gap medido pelo `frame_idx` original do video.
- `WINDOW_SECONDS = 8.0` e `WINDOW_STRIDE_SECONDS = 2.0`.
- `MIN_SHOULDER_WIDTH_NORM = 0.02`.
- `DISPLAY_SCORE_THRESHOLD = 0.35`, `RULE_SCORE_THRESHOLD = 0.35` e `FRAME_STRONG_SCORE_THRESHOLD = 0.70`.
- Parametros geometricos: `BASE_NECK_FRACTION = 0.20`, `MAX_NECK_TILT_BONUS = 0.25`, `MAX_NECK_FRACTION = 0.45`, `CHEST_OVERLAP_FRACTION = 0.08`, `HEAD_DROP_NEUTRAL_RATIO = 0.42`, `HEAD_DROP_STRONG_RATIO = 0.26`.
- `HAND_SOURCE_QUALITY` com pesos `holistic_hand = 1.00`, `pose_wrist_fallback = 0.75`, `arm_proxy = 0.45`, `unknown = 0.00`.
- MediaPipe Tasks `HolisticLandmarker` em `RunningMode.VIDEO`, com confianças `0.5` para deteccao/supressao/landmarks de face, pose e maos, `output_face_blendshapes=False` e `output_segmentation_mask=False`.
- Modelo oficial `holistic_landmarker.task`, com URL registrada apenas no debug.
- Extracao dos mesmos grupos de landmarks: `pose_2d`, `pose_world`, `face_2d`, `left_hand_2d`, `right_hand_2d`, `left_hand_world`, `right_hand_world`.
- Suavizacao EMA por grupo de landmarks antes da geometria.
- `compute_reference_geometry()` como nucleo de metricas.
- `evaluate_rules()` em ordem de dependencia: contatos de mao, depois `head_down`, depois `forward_head` e `rounded_shoulders_or_asymmetry`.

O model default de pose deve ser uma projecao enxuta dos resultados internos:

```text
frame_signals + regras por frame -> windows[].detections[]
```

Campos ricos como `frame_signals`, `window_summaries`, `video_summary`, `metrics`, `rules`, `evidence`, `limitations`, `pipeline`, paths de modelo e caminho de video anotado pertencem ao JSON debug.

### 15. Inicializacao dos processadores

**Decisao: processadores com construtor simples e injecao do `PipelineConfig` global em `setup()`.**

Os processadores devem poder ser tratados pelo `orchestrator` pelas abstracoes `FrameProcessor` e `AudioProcessor`. Para isso, o construtor padrao dos processadores nao deve exigir configs especificas. A configuracao entra por `setup(video_meta, config)`, e cada processador extrai internamente apenas a parte que usa.

Exemplo esperado:

```python
visual_processors: list[FrameProcessor] = [
    ExpressionDeepFaceProcessor(),
    PoseMediaPipeProcessor(),
]

for processor in visual_processors:
    processor.setup(video_meta, config)
```

Dentro do processador:

```python
class ExpressionDeepFaceProcessor:
    def setup(self, video_meta: VideoMeta, config: PipelineConfig) -> None:
        self.video_meta = video_meta
        self.config = config.expression
        self.window_s = config.window_s
        self.stride_s = config.stride_s
```

Como o config ja foi injetado no `setup()`, `aggregate()` nao deve receber `PipelineConfig` novamente. Isso evita dois canais de configuracao e reduz risco de inconsistencia entre inicializacao e agregacao.

Factory/Registry fica fora do desenho inicial. Esse padrao so deve ser introduzido se houver necessidade real de habilitar/desabilitar modulos dinamicamente, trocar backends por config ou carregar plugins.

---

## Contrato do modulo video_pipeline

Definido em: 2026-07-04

### Localizacao

`src/video_pipeline/` — pacote irmao de `screening_agent`, nao aninhado dentro dele.

```text
src/
├── screening_agent/          # existente — NAO modificado
│   └── ...
└── video_pipeline/
    ├── __init__.py
    ├── config.py
    ├── contracts.py
    ├── orchestrator.py
    ├── media_probe.py
    ├── video_reader.py
    ├── audio_extractor.py
    ├── writers.py
    ├── configs/
    │   └── default.yaml
    └── processors/
        ├── __init__.py
        ├── base.py
        ├── expression_deepface.py
        ├── pose_mediapipe.py
        └── transcription_whisper.py
```

Justificativa:

- Responsabilidades distintas: pipeline de processamento offline vs. agente conversacional LangGraph.
- O `setuptools` ja descobre pacotes em `src/` automaticamente (`[tool.setuptools.packages.find] where = ["src"]`).
- Sem acoplamento: `video_pipeline` nao importa `screening_agent` e vice-versa.
- O canal principal de dados e o `VideoAnalysisResult` retornado em memoria pelo `process_video()`, nao o filesystem.

### Modelos de dados (`contracts.py`)

Todos os modelos usam Pydantic BaseModel, seguindo o padrao do projeto (`ClinicalScreeningOutput`, `RouteDecision`).

#### Configuracao

```python
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

    model_asset_path: str | None = None
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
    """Configuracao da extracao e normalizacao de audio."""
    sample_rate: int = 16000
    channels: int = 1
    codec: str = "pcm_s16le"
    format: str = "wav"


class TranscriptionWhisperConfig(BaseModel):
    """Configuracao do processador de transcricao."""
    model_name: str = "base"
    language: str = "pt"                     # "pt", "en", "auto", etc.
    fp16: bool | None = None                 # None = autodetectar conforme acelerador disponivel
    generate_debug_csv: bool = True
    generate_debug_srt: bool = True


class PipelineConfig(BaseModel):
    """Configuracao global do pipeline."""
    window_s: float = 8.0
    stride_s: float = 2.0
    debug: bool = False
    output_dir: str | None = None            # None = nao gravar JSONs; str = diretorio de saida

    expression: ExpressionDeepFaceConfig = Field(default_factory=ExpressionDeepFaceConfig)
    pose: PoseMediaPipeConfig = Field(default_factory=PoseMediaPipeConfig)
    audio: AudioExtractionConfig = Field(default_factory=AudioExtractionConfig)
    transcription: TranscriptionWhisperConfig = Field(default_factory=TranscriptionWhisperConfig)
```

Arquivos de configuracao devem ser carregados por `video_pipeline.config.load_pipeline_config()`.
O arquivo padrao (`configs/default.yaml`) declara os mesmos valores dos models acima. Overrides por ambiente ou chamada devem passar pela validacao Pydantic antes de chegar aos processadores.

Exemplo de `configs/default.yaml`:

```yaml
window_s: 8.0
stride_s: 2.0
debug: false
output_dir: null

expression:
  label_map:
    angry: anger_expression
    disgust: disgust_expression
    fear: fear_expression
    happy: joy_expression
    neutral: neutral_expression
    sad: sad_expression
    surprise: surprise_expression
  min_confidence: 0.30
  min_segment_duration_s: 0.20
  detector_backend: opencv
  enforce_detection: false
  align: true
  silent: true
  actions: [emotion]
  build_model_name: Emotion
  build_model_task: facial_attribute

pose:
  target_sample_fps: 5.0
  visibility_threshold: 0.50
  ema_alpha: 0.35
  ema_max_gap_frames: 3
  min_shoulder_width_norm: 0.02
  display_score_threshold: 0.35
  rule_score_threshold: 0.35
  frame_strong_score_threshold: 0.70
  base_neck_fraction: 0.20
  max_neck_tilt_bonus: 0.25
  max_neck_fraction: 0.45
  chest_overlap_fraction: 0.08
  head_drop_neutral_ratio: 0.42
  head_drop_strong_ratio: 0.26
  hand_source_quality:
    holistic_hand: 1.00
    pose_wrist_fallback: 0.75
    arm_proxy: 0.45
    unknown: 0.00
  rule_order:
    - hand_on_head
    - hand_on_neck
    - hand_on_chest
    - head_down
    - forward_head
    - rounded_shoulders_or_asymmetry
  model_asset_path: null
  min_face_detection_confidence: 0.5
  min_face_suppression_threshold: 0.5
  min_face_landmarks_confidence: 0.5
  min_pose_detection_confidence: 0.5
  min_pose_suppression_threshold: 0.5
  min_pose_landmarks_confidence: 0.5
  min_hand_landmarks_confidence: 0.5
  output_face_blendshapes: false
  output_segmentation_mask: false

audio:
  sample_rate: 16000
  channels: 1
  codec: pcm_s16le
  format: wav

transcription:
  model_name: base
  language: pt
  fp16: null
  generate_debug_csv: true
  generate_debug_srt: true
```

#### Metadata do video

```python
class VideoMeta(BaseModel):
    """Metadata extraida pelo media_probe."""
    video_id: str
    source_path: str
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
```

#### Pacotes internos (nao exportados nos JSONs default)

```python
class FramePacket(BaseModel):
    """Frame decodificado com timestamp."""
    timestamp_s: float
    frame_index: int
    # frame: np.ndarray e passado fora do modelo (nao serializavel)

class AudioPacket(BaseModel):
    """Referencia ao audio extraido."""
    audio_path: str
    sample_rate: int
    channels: int
    codec: str
    format: str
    duration_s: float
```

#### Deteccoes por frame (internas)

```python
class FrameDetection(BaseModel):
    """Deteccao bruta de um frame individual."""
    timestamp_s: float
    label: str
    score: float    # expression: score ponderado do label; pose: score da regra no frame

class FrameAnalysisRecord(BaseModel):
    """Registro interno de analise visual de um frame."""
    timestamp_s: float
    frame_index: int
    status: Literal["scorable", "insufficient_data", "no_detection", "uncertain", "skipped"]
    detections: list[FrameDetection] = Field(default_factory=list)
    debug: dict[str, object] = Field(default_factory=dict)
```

`FrameAnalysisRecord.detections` alimenta a agregacao default de forma tipada.
`FrameAnalysisRecord.debug` deve carregar payload rico de auditoria, mas nao deve ser parseado para produzir `DetectionWindow`.

#### Segmento de transcricao

```python
class TranscriptSegment(BaseModel):
    """Segmento de fala retornado pelo ASR."""
    start_s: float
    end_s: float
    text: str
```

#### Resultado agregado por window (saida)

```python
class Detection(BaseModel):
    """Deteccao agregada dentro de uma window."""
    label: str
    score: float                             # expression: media agregada; pose: pico da regra na window
    support: float                           # proporcao de frames validos/scorable

class DominantDetection(BaseModel):
    """Deteccao dominante da window."""
    label: str
    score: float

class DetectionWindow(BaseModel):
    """Window com deteccoes visuais (expression ou pose)."""
    start_s: float
    end_s: float
    detections: list[Detection]              # ordenadas conforme regra do processador
    dominant: DominantDetection | None

class TranscriptionWindow(BaseModel):
    """Window com transcricao de audio."""
    start_s: float
    end_s: float
    text: str
    segments: list[TranscriptSegment]        # segmentos podem aparecer em mais de uma window (duplicacao em bordas)
    coverage_s: float
```

#### Resultado por modulo

Esses modelos representam os dados em memoria retornados por cada processador.
Se `output_dir` estiver configurado, o `writers.py` os serializa como JSON em disco.

```python
class ModuleResult(BaseModel):
    """Base comum dos resultados por modulo."""
    module: str
    video_id: str
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
    has_audio: bool                          # False quando o video nao tem faixa de audio
```

#### Resultado combinado por video (contrato principal para o agente)

```python
class VideoAnalysisResult(BaseModel):
    """Resultado completo do processamento de um video.

    Combina os tres modulos alinhados pela mesma timeline de windows.
    Este e o contrato de consumo pelo screening_agent.
    """
    video_id: str
    duration_s: float
    window_s: float
    stride_s: float
    expression: ExpressionResult | None
    pose: PoseResult | None
    transcription: TranscriptionResult | None
```

### API publica (`__init__.py`)

```python
def load_pipeline_config(config_path: str | None = None) -> PipelineConfig:
    """Carrega um arquivo YAML/JSON de configuracao e retorna um PipelineConfig validado.

    Se config_path for None, carrega `video_pipeline/configs/default.yaml`.
    """
    ...


def process_video(
    video_path: str,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> VideoAnalysisResult:
    """Processa um video e retorna o VideoAnalysisResult completo em memoria.

    `config` e `config_path` sao mutuamente exclusivos. Quando nenhum dos dois e
    informado, o pipeline usa `load_pipeline_config()`.

    Comportamento de gravacao:
    - Se config.output_dir for None (padrao), nenhum arquivo e gravado.
    - Se config.output_dir for um caminho valido, os resultados de cada modulo
      sao serializados como JSON em disco no formato:
        {output_dir}/{video_id}.{module}.json
        {output_dir}/{video_id}.{module}.debug.json  (apenas se config.debug=True)

    O retorno e sempre o VideoAnalysisResult completo, independente de output_dir.
    """
    ...
```

### Estrategia de integracao com o screening_agent

O `screening_agent` NAO sera modificado agora.

O canal principal de dados e o objeto `VideoAnalysisResult` retornado em memoria por `process_video()`.
O agente pode receber esse objeto diretamente de quem chamar o pipeline, sem depender de arquivos em disco.

Caminho de integracao futura:

- Quem orquestra (ex: `app_chainlit.py` ou uma tool do agente) chama `process_video()` e recebe o `VideoAnalysisResult`.
- O resultado pode ser serializado com `.model_dump_json()` e injetado no estado do agente.
- **Integração**: novo `RouteIntent` (`video_analysis`), novo campo em `AssistantState` (`video_analysis: VideoAnalysisResult | None`) e novo no dedicado.

A gravacao em disco via `output_dir` e util para:

- Inspecao manual e debug.
- Integracao com sistemas externos que consomem arquivos.

Compatibilidade garantida por design:

| Aspecto | video_pipeline | screening_agent |
|---|---|---|
| Canal de dados | `VideoAnalysisResult` em memoria | `.model_dump_json()` |
| Gravacao em disco | Opcional, via `output_dir` | Nao depende de arquivos em disco |

---

## Alteracoes para orientar a refatoracao

Esta secao lista as mudancas necessarias no codigo existente para alinhar a implementacao ao contrato final.

### 1. Configuracao por arquivo

- Criar `src/video_pipeline/config.py`.
- Criar `src/video_pipeline/configs/default.yaml`.
- Implementar `load_pipeline_config(config_path: str | None = None) -> PipelineConfig`.
- Validar YAML/JSON via Pydantic antes de instanciar processadores.
- Remover `sample_fps` global de `PipelineConfig`.
- Remover de `PipelineConfig` qualquer threshold global de apresentacao do JSON default.
- Nao aplicar thresholds globais de apresentacao no JSON default; cada processador decide suas deteccoes internamente.
- Adicionar `ExpressionDeepFaceConfig`, `PoseMediaPipeConfig`, `AudioExtractionConfig` e `TranscriptionWhisperConfig` em `contracts.py` ou em modulo dedicado reexportado por `contracts.py`.

### 2. Contratos internos

- Adicionar `FrameAnalysisRecord`.
- Alterar `FrameProcessor.setup()` e `AudioProcessor.setup()` para receber `(video_meta, config)`.
- Alterar `FrameProcessor.wants_frame()` para receber `FramePacket`, nao apenas `timestamp_s`.
- Alterar `FrameProcessor.process_frame()` para receber `(frame_packet, frame_bgr)` e retornar `FrameAnalysisRecord`.
- Alterar `FrameProcessor.aggregate()` para receber `list[FrameAnalysisRecord]`.
- Alterar `AudioProcessor.aggregate()` para receber apenas `list[TranscriptSegment]`.
- Remover `PipelineConfig` e `VideoMeta` dos metodos `aggregate()`, pois esses dados ja entram pelo `setup()`.
- Adicionar `debug_payload()` aos processadores visuais e de audio.
- Usar `FrameAnalysisRecord.detections` como fonte tipada para agregacao do JSON default.
- Manter `FrameAnalysisRecord.debug` apenas para diagnostico/auditoria; agregadores nao devem fazer parsing de `debug` para descobrir scores, labels ou flags de deteccao.
- Manter `FrameDetection` como item compacto dentro de `FrameAnalysisRecord.detections`. Para `pose`, cada regra analisada no frame deve ser mapeada para um `FrameDetection` (label=rule_name, score=rule.score) para que o agregador consiga ler o score sem precisar fazer parsing do campo `debug`.
- Expandir `AudioPacket` para incluir `sample_rate`, `channels`, `codec`, `format` e `duration_s` efetivos.

### 3. Reader visual

- Alterar `read_frames(video_meta)` para nao receber `sample_fps`.
- Remover seek por `CAP_PROP_POS_MSEC` no fluxo principal.
- Ler frames sequencialmente com `capture.read()`.
- Emitir `FramePacket(frame_index=frame_idx, timestamp_s=frame_idx / source_fps)`.
- Entregar todos os frames ao `orchestrator`.
- Manter fallback de simulacao apenas para testes e cenarios sem arquivo real.

### 4. Orchestrator

- Carregar config via `load_pipeline_config()` quando `config` nao for fornecido.
- Rejeitar chamadas com `config` e `config_path` simultaneamente.
- Instanciar processadores sem passar config no construtor.
- Chamar `setup(video_meta, config)` em cada processador. O processador deve extrair sua propria sub-configuracao internamente.
- O orquestrador injeta os parametros de `config.transcription` no `audio_extractor` ao inves de hardcodar o formato.
- Executar `extract_audio(video_path, config.audio)` na branch de audio quando `video_meta.has_audio=True`.
- Passar o `AudioPacket` retornado pelo extrator para o processador de transcricao.
- Chamar `setup(video_meta, config)` em cada processador.
- Enviar `frame_bgr` real para `process_frame()`.
- Guardar `FrameAnalysisRecord` por modulo.
- Agregar por modulo usando os records internos.
- Coletar `debug_payload()` por modulo e passar ao writer.

### 5. Expression DeepFace

- Substituir a simulacao por chamada real a DeepFace.
- Usar exclusivamente parametros de `ExpressionDeepFaceConfig`.
- Processar todos os frames entregues pela branch visual.
- Implementar selecao da maior face, normalizacao de score, threshold `min_confidence` e contadores internos.
- Gerar deteccoes por frame aceitas e records `uncertain`/`no_detection` quando aplicavel.
- Agregar windows a partir de `FrameAnalysisRecord.detections`, sem threshold global adicional de apresentacao.
- Expor no debug segmentos globais, percentuais, contadores e parametros efetivos.

### 6. Pose MediaPipe

- Substituir a simulacao por MediaPipe Tasks Holistic.
- Usar exclusivamente parametros de `PoseMediaPipeConfig`.
- Aplicar sampling interno por `frame_step = max(1, round(source_fps / target_sample_fps))`.
- Manter `frame_index` original para timestamp, sampling e EMA gap.
- Implementar extracao de landmarks, suavizacao EMA, geometria de referencia, regras e records por frame.
- Usar labels definidos por `config.pose.rule_order`.
- Emitir em `FrameAnalysisRecord.detections` apenas regras ativas no frame, com `label = rule_name` e `score = rule.score`.
- Garantir no maximo uma `FrameDetection` por label em cada frame de pose.
- Agregar windows por regra:
  - `score = max(FrameDetection.score do label)`.
  - `support = frames_detected / frames_scorable`.
  - incluir deteccao quando `frames_detected > 0`.
  - ordenar por `(score, support)` decrescente.
- Nao aplicar thresholds globais de apresentacao na pose.
- Expor no debug `frame_signals`, `window_summaries`, `video_summary`, metricas, regras, evidencias, limitations, warnings e parametros efetivos.

### 7. Audio Extractor

- Injetar `AudioExtractionConfig` em `audio_extractor.py`.
- Usar `sample_rate=16000`, `channels=1`, `codec="pcm_s16le"` e `format="wav"` como defaults de config, nao como constantes soltas.
- Remover hardcodes de formato de audio fora dos defaults de config.
- Retornar `AudioPacket` com os parametros efetivos do audio normalizado.
- Registrar caminho do audio extraido e parametros efetivos apenas no debug ou em artefatos auxiliares.

### 8. Transcription Whisper

- Injetar `TranscriptionWhisperConfig`.
- Usar `model_name`, `language` e `fp16` da config.
- Consumir `AudioPacket` ja normalizado pelo `audio_extractor`.
- Validar ou registrar `AudioPacket.sample_rate`, `channels`, `codec` e `format` no debug quando util, sem reconfigurar a extracao.
- Gerar CSV/SRT apenas quando `debug=True` e flags de config estiverem ativas.
- Manter JSON default apenas com windows, texto, segmentos e cobertura.

### 9. Writers

- Alterar `write_results()` para receber `module_debug_payloads: dict[str, dict[str, object]]`.
- Continuar gravando JSON default apenas a partir dos models de resultado.
- Gravar debug por modulo apenas quando `config.debug=True`.
- Nao reconstruir debug a partir de `FrameDetection` compacto; usar `debug_payload()` dos processadores.

### 10. Testes

- Atualizar testes que dependem de simulacoes fixas.
- Adicionar testes unitarios para carregamento de config e defaults.
- Adicionar testes garantindo que processadores recebem `PipelineConfig` via `setup(video_meta, config)` e que `aggregate()` nao recebe config.
- Adicionar testes garantindo que `audio_extractor` usa `AudioExtractionConfig` e retorna `AudioPacket` com parametros efetivos.
- Adicionar testes de `video_reader` garantindo leitura sequencial e timestamps por `frame_index / fps`.
- Adicionar testes de agregacao de `expression` e `pose` com records sintéticos.
- Adicionar testes garantindo que o JSON default nao aplica thresholds globais de apresentacao.
- Adicionar testes garantindo que agregacao de pose usa `FrameAnalysisRecord.detections`, nao parsing de `debug`.
