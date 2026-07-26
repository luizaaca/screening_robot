# Plano: Integracao De Video No `screening_agent`

## Objetivo

Integrar ao app `app_chainlit.py` o fluxo validado no notebook
`langgraph_router_specialists_video_v2.ipynb`: processar video real com
`src/video_pipeline`, manter resultado cumulativo no estado LangGraph e permitir
QA clinico sobre o video com um backend proprio.

## Escopo

- Aceitar video por upload Chainlit e por texto com `video_path=...` ou
  `path: ...`.
- Solicitar upload quando a pergunta for sobre video e nenhum arquivo/path for
  fornecido.
- Processar video em um no deterministico e sincrono no v1.
- Persistir no estado apenas JSON serializavel, resumo textual, caminho do video
  e diretorio de artefatos.
- Usar paciente ativo como contexto opcional no QA; a ausencia de paciente nao
  bloqueia a analise do video.
- Usar backend dedicado `SCREENING_AGENT_VIDEO_ANALYST_*` para respostas sobre o
  video.
- Falhar fechado quando o pipeline ou dependencias opcionais falharem; nunca
  fabricar resultado valido nem reutilizar resultado antigo para um novo video.

## Contratos De Configuracao

Novas variaveis de ambiente:

- `SCREENING_AGENT_VIDEO_ANALYST_BACKEND`
- `SCREENING_AGENT_VIDEO_ANALYST_MODEL`
- `SCREENING_AGENT_VIDEO_ANALYST_BASE_URL`
- `SCREENING_AGENT_VIDEO_ANALYST_API_KEY`
- `SCREENING_AGENT_VIDEO_ANALYST_TEMPERATURE`
- `SCREENING_AGENT_VIDEO_PIPELINE_OUTPUT_DIR=outputs/videos`
- `SCREENING_AGENT_VIDEO_PIPELINE_DEBUG=false`
- `SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB=100`
- `SCREENING_AGENT_VIDEO_PIPELINE_WINDOW_S=8.0`
- `SCREENING_AGENT_VIDEO_PIPELINE_STRIDE_S=5.0`

O extra `video` do `pyproject.toml` deve incluir `mediapipe` e preservar o
package-data ja existente para configs e assets do `video_pipeline`.

## Estado LangGraph

Expandir `AssistantState` com:

- `video_path`
- `video_artifact_dir`
- `video_analysis_summary`
- `video_analysis_json`
- `video_analysis_status`
- `video_analysis_error`

Expandir `RouteIntent` com:

- `video_analysis`
- `video_qa`

## Nos

### `video_analysis`

- Monta `PipelineConfig` com window/stride/debug/output_dir.
- Usa `outputs/videos/{thread_id}` como diretorio de artefatos.
- Chama `process_video(video_path, config=config)` ou processador injetado.
- Salva JSON compacto serializavel e resumo textual no estado.
- Ao iniciar novo video, limpa resultado/erro anterior para evitar reuso indevido.

### `video_qa`

- Se ja houver resultado, envia resumo, JSON completo e paciente ativo opcional
  para o video analyst.
- Se nao houver resultado mas houver `video_path`, processa primeiro e entao
  responde.
- Se nao houver video, responde solicitando upload ou `video_path=...`.
- Se processamento falhar, responde fail-closed.

## Chainlit

- Extrair caminho de arquivo de `message.elements`.
- Validar tamanho maximo do upload com `SCREENING_AGENT_VIDEO_UPLOAD_MAX_MB`.
- Aceitar path explicito no texto.
- Passar `video_path` como entrada adicional para `_stream_graph_turn`.
- Quando houver pergunta sobre video sem arquivo/path, usar `AskFileMessage`.

## Documentacao E Testes

Atualizar README em ingles e portugues com arquitetura, backends, extras de
instalacao, upload/path, variaveis de ambiente e limitacoes. Cobrir com testes:

- parsing de `video_path=...` e `path: ...`;
- upload Chainlit fake e limite de tamanho;
- defaults e leitura de `SCREENING_AGENT_VIDEO_*`;
- grafo com processador fake;
- QA com resultado previo e paciente ativo opcional;
- QA sem video;
- falha de pipeline sem fabricacao de resultado nem reuso antigo.

