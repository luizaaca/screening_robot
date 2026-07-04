# Analise do notebook `mediapipe_video_posture_timeline_v3.ipynb`

Arquivo analisado: `concepts_video/mediapipe_video_posture_timeline_v3.ipynb`.

O notebook v3 e uma pipeline funcional para processar videos com MediaPipe, extrair landmarks de corpo/face/maos, suavizar a serie temporal, derivar geometrias anatomicas, avaliar regras posturais por frame, agregar os sinais em janelas temporais, gerar um payload JSON e renderizar um MP4 anotado.

## Visao geral por bloco

| Bloco | Celula | Papel |
|---|---:|---|
| Ambiente/imports/config | 1-8 | Instala dependencias opcionais, importa bibliotecas e define constantes globais, paths, thresholds, fps de amostragem, versao de schema e assets do modelo. |
| Assets do modelo | 9 | Baixa e valida o `.task` usado pelo MediaPipe. |
| Dados e landmarks | 11 | Define `PointData` e normaliza landmarks em dicionarios nomeados. |
| Suavizacao temporal | 13 | Aplica EMA por grupo de landmarks para reduzir jitter entre frames. |
| Geometria e metricas | 15 | Calcula distancias, centros, limites verticais, regioes de cabeca/pescoco/torax e componentes posturais. |
| Contato e score | 17 | Converte evidencia de contato mao-corpo em scores normalizados. |
| Regras concorrentes | 19 | Avalia regras finais por frame: mao na cabeca/pescoco/peito, cabeca baixa, cabeca anteriorizada e ombros arredondados/assimetria. |
| Frame, janelas e JSON | 21 | Serializa registros por frame, agrega janelas deslizantes e monta resumo do video. |
| Renderizacao | 23 | Desenha landmarks, regioes de debug, painel de status e salva video anotado. |
| Pipeline principal | 25 | Orquestra a leitura do video, inferencia MediaPipe, scoring, agregacao, renderizacao e gravacao do JSON. |
| Execucao/inspecao | 26-29 | Executa lote de videos e carrega payloads gerados para inspecao. |

## Tabela de metodos e responsabilidades

| Metodo / classe | Celula | Responsabilidade |
|---|---:|---|
| `download_file_with_progress(url, destination_path)` | 9 | Baixar um arquivo remoto exibindo progresso e gravar no destino local. |
| `download_required_model_assets(force_download=False)` | 9 | Garantir que os assets obrigatorios do modelo existam localmente, baixando quando necessario. |
| `assert_model_assets_available()` | 9 | Validar a existencia dos assets antes da execucao da pipeline. |
| `PointData` | 11 | Modelo de ponto normalizado com `x`, `y`, `z`, `visibility` e `space`. E a unidade de dados usada em toda a geometria. |
| `make_point(landmark, space, visibility=None)` | 11 | Converter um landmark bruto do MediaPipe para `PointData`. |
| `normalize_landmark_container(raw_landmarks)` | 11 | Aceitar formatos variados de container do MediaPipe e devolver uma lista uniforme de landmarks. |
| `landmark_list_to_points(landmark_list, name_to_index, space, visibility_min=None)` | 11 | Mapear landmarks indexados para um dicionario nomeado de `PointData`, filtrando por visibilidade quando aplicavel. |
| `extract_landmark_sets(holistic_result)` | 11 | Extrair grupos `pose_2d`, `pose_world`, `face_2d`, `left_hand_2d` e `right_hand_2d` do resultado MediaPipe. |
| `build_empty_landmark_sets()` | 11 | Criar a estrutura vazia padrao quando a inferencia falha ou nao detecta landmarks. |
| `clone_drawing_points(landmark_sets)` | 11 | Copiar pontos suavizados para uso posterior na renderizacao do frame. |
| `blend_optional(current_value, previous_value, alpha)` | 13 | Fazer blend EMA de valores opcionais, principalmente `z`. |
| `EMASmoother.__init__(alpha, max_gap_frames)` | 13 | Inicializar estado, fator EMA e tolerancia de lacuna para um grupo de landmarks. |
| `EMASmoother.update(frame_idx, points)` | 13 | Atualizar landmarks suavizados; reinicia o estado quando um ponto some por muitos frames. |
| `build_smoothers()` | 13 | Criar um `EMASmoother` para cada grupo de landmarks, preservando escalas e identidades. |
| `smooth_landmark_sets(frame_idx, raw_sets, smoothers)` | 13 | Aplicar suavizacao EMA a todos os grupos do frame. |
| `optional_round(value, digits=4)` | 15 | Arredondar floats opcionais para JSON/debug. |
| `clamp_value(value, low=0.0, high=1.0)` | 15 | Limitar scores e razoes ao intervalo esperado. |
| `point_distance(point_a, point_b)` | 15 | Calcular distancia euclidiana entre dois pontos. |
| `midpoint(point_a, point_b)` | 15 | Calcular ponto medio entre dois pontos. |
| `offset_point(point, dx=0.0, dy=0.0)` | 15 | Criar ponto deslocado a partir de um ponto base. |
| `centroid(points, fallback_space='image')` | 15 | Calcular centroide dos pontos validos. |
| `valid_points(points)` | 15 | Filtrar valores `None` de uma colecao de pontos. |
| `min_distance_to_named_points(point, named_points)` | 15 | Encontrar o alvo nomeado mais proximo de um ponto. |
| `score_ratio_below(ratio, strong_ratio, weak_ratio)` | 15 | Pontuar razoes em que valores menores indicam evidencia mais forte. |
| `score_ratio_above(ratio, weak_ratio, strong_ratio)` | 15 | Pontuar razoes em que valores maiores indicam evidencia mais forte. |
| `weighted_mean_score(components)` | 15 | Combinar componentes de score com pesos, ignorando ausentes. |
| `estimate_face_pitch_degrees(face_2d)` | 15 | Estimar pitch facial em graus usando landmarks faciais estaveis. |
| `classify_face_pitch_direction(face_pitch_degrees)` | 15 | Classificar direcao do pitch para raciocinio de cabeca baixa. |
| `compute_directional_face_pitch_score(face_pitch_degrees)` | 15 | Transformar pitch facial bruto em score direcional e flag de validade. |
| `compute_neck_height_from_shoulders(...)` | 15 | Estimar altura compacta do pescoco a partir de largura de ombros e sinais de inclinacao/queda da cabeca. |
| `estimate_head_width(face_2d, pose_2d, shoulder_width)` | 15 | Estimar largura da cabeca por ancoras faciais e fallback proporcional aos ombros. |
| `infer_neck_pose_mode(...)` | 15 | Inferir se a pose pede perfil de pescoco compacto ou posterior. |
| `compute_face_vertical_limits(...)` | 15 | Calcular limites verticais estaveis para rosto, cabeca, pescoco e nuca. |
| `compute_head_tilt_components(...)` | 15 | Combinar pistas anatomicas e pitch facial para score de cabeca baixa. |
| `estimate_arm_proxy_point(shoulder, elbow)` | 15 | Estimar um ponto proxy de punho quando o punho nao esta detectado. |
| `build_side_contact_evidence(side, pose_2d, hand_2d)` | 15 | Montar evidencia de contato para um lado do corpo/mao. |
| `build_neck_lateral_anchor_summary(...)` | 15 | Inferir ancoras laterais para alargar regioes de pescoco/nuca. |
| `build_neck_regions(...)` | 15 | Construir regioes anterior do pescoco e posterior da nuca, adaptadas a pose. |
| `build_head_reference_points(...)` | 15 | Gerar ancoras sinteticas e landmarks de cabeca para contato na cabeca. |
| `compute_forward_head_2d_components(...)` | 15 | Estimar evidencia 2D de cabeca anteriorizada a partir de face e ombros. |
| `compute_reference_geometry(landmark_sets)` | 15 | Consolidar toda a geometria usada pelas regras: landmarks, medidas, regioes, contato, scores intermediarios e modo de coordenadas. |
| `serialize_metric_value(value)` | 17 | Converter estruturas aninhadas de metricas em tipos seguros para JSON. |
| `build_scored_rule_result(...)` | 17 | Criar payload padronizado de uma regra com score, ativo/inativo, evidencia e notas. |
| `face_vertical_score(point, geometry)` | 17 | Medir compatibilidade vertical de um ponto com a regiao do rosto. |
| `head_vertical_score(point, geometry)` | 17 | Medir compatibilidade vertical de um ponto com a regiao da cabeca. |
| `neck_zone_score(point, geometry, zone_name=None)` | 17 | Medir se o ponto cai nas regioes explicitas de pescoco/nuca. |
| `chest_vertical_score(point, geometry)` | 17 | Medir compatibilidade vertical com a zona superior do torax. |
| `build_face_contact_targets(geometry)` | 17 | Selecionar ancoras faciais usadas como referencia de overlap nas regras de contato adjacente ao pescoco. |
| `build_head_contact_targets(geometry)` | 17 | Selecionar ancoras sinteticas e reais para scoring de mao na cabeca. |
| `build_head_contact_summary(evidence, geometry)` | 17 | Agregar evidencia de mao para contato na cabeca sem supressao por espalhamento. |
| `score_head_contact_summary(summary, evidence_quality, source)` | 17 | Converter o resumo de contato na cabeca em score final e metadados. |
| `point_target_overlap_score(point, targets, shoulder_width)` | 17 | Calcular melhor sobreposicao/proximidade entre ponto e alvos nomeados. |
| `score_neck_contact_point(point, geometry)` | 17 | Pontuar um ponto individual contra regioes de pescoco/nuca e alvos adjacentes. |
| `score_chest_contact_point(point, geometry)` | 17 | Pontuar um ponto individual contra a zona superior do peito. |
| `contact_rule_precheck(geometry)` | 19 | Validar se ha evidencia de contato suficiente ou retornar resultado padrao nao avaliavel. |
| `build_contact_rule_result(best, missing_note)` | 19 | Normalizar o melhor candidato de contato em payload de regra. |
| `evaluate_summary_contact_rule(...)` | 19 | Avaliar regras de contato que dependem de um resumo agregado da mao inteira. No estado atual, serve a regra de contato na cabeca. |
| `evaluate_point_contact_rule(...)` | 19 | Avaliar regras de contato que pontuam pontos individuais. |
| `evaluate_hand_on_head(geometry)` | 19 | Avaliar regra "mao na cabeca". |
| `evaluate_hand_on_neck(geometry)` | 19 | Avaliar regra "mao no pescoco/nuca". |
| `evaluate_hand_on_chest(geometry)` | 19 | Avaliar regra "mao no peito". |
| `evaluate_head_down(geometry, contact_rules)` | 19 | Avaliar cabeca baixa combinando anatomia e possivel suporte de contato. |
| `evaluate_forward_head(geometry)` | 19 | Avaliar cabeca anteriorizada com pistas 2D e world-space. |
| `evaluate_rounded_shoulders_or_asymmetry(geometry)` | 19 | Avaliar tensao/assimetria de tronco superior e ombros arredondados. |
| `evaluate_rules(geometry)` | 19 | Executar todas as regras finais em ordem de dependencia. |
| `pretty_rule_name(rule_name)` | 21 | Converter identificador tecnico de regra em label amigavel. |
| `rule_score(rule)` | 21 | Extrair score numerico de uma regra. |
| `rule_is_active(rule)` | 21 | Determinar se a regra passou o limiar de ativacao. |
| `choose_primary_trigger(rules)` | 21 | Escolher regra ativa prioritaria para explicacao do frame. |
| `display_rules_for_overlay(rules)` | 21 | Montar lista compacta de regras exibidas no overlay. |
| `build_frame_explanation(rules, frame_level)` | 21 | Criar explicacao textual curta do frame. |
| `evidence_summary(evidence)` | 21 | Resumir evidencias para o JSON de frame. |
| `build_frame_signal_record(...)` | 21 | Serializar o resultado completo de um frame: score, nivel, gatilho, regras, geometria resumida e evidencias. |
| `window_ranges(duration_seconds, window_seconds, stride_seconds)` | 21 | Gerar janelas deslizantes no intervalo do video. |
| `count_max_consecutive_signal_frames(frame_records)` | 21 | Medir a maior sequencia continua de frames com sinal ativo. |
| `label_window(signal_ratio)` | 21 | Converter proporcao de sinal em rotulo de severidade da janela. |
| `aggregate_windows(frame_records, duration_seconds)` | 21 | Agregar frames em janelas temporais com contagens, proporcoes, top rules e nivel. |
| `peak_window_level(window_summaries)` | 21 | Encontrar o maior nivel observado entre janelas. |
| `has_adjacent_strong_windows(valid_windows)` | 21 | Detectar persistencia por janelas fortes adjacentes. |
| `build_video_summary(frame_records, window_summaries)` | 21 | Consolidar resumo final do video a partir de frames e janelas. |
| `percentage(count, total)` | 21 | Calcular percentual protegido contra divisao por zero. |
| `build_limitations(frame_records)` | 21 | Gerar lista de limitacoes/caveats observadas durante o processamento. |
| `sanitize_for_json(value)` | 21 | Converter recursivamente valores em primitivas serializaveis. |
| `write_payload_json(payload, json_output_path)` | 21 | Gravar payload JSON UTF-8 no disco. |
| `point_to_pixel(point, frame_shape)` | 23 | Converter ponto normalizado para coordenada de pixel. |
| `normalized_to_pixel(x, y, frame_shape)` | 23 | Converter coordenadas normalizadas soltas para pixel. |
| `draw_segment(frame, point_a, point_b, color, thickness)` | 23 | Desenhar segmento entre dois pontos normalizados. |
| `draw_points(frame, points, color, radius=4)` | 23 | Desenhar colecao de pontos no frame. |
| `draw_landmarks_overlay(frame, drawing_points)` | 23 | Desenhar landmarks de pose, face e maos. |
| `draw_neck_debug_region(frame, frame_record)` | 23 | Desenhar regioes refinadas de pescoco/nuca usadas no debug. |
| `select_window_for_timestamp(timestamp_s, window_summaries)` | 23 | Selecionar janela ativa/mais forte para o timestamp do frame. |
| `format_display_rules(frame_record)` | 23 | Formatar regras do overlay em uma string compacta. |
| `draw_status_panel(frame, frame_record, window_record)` | 23 | Desenhar painel de status com resumo de frame e janela. |
| `render_annotated_video(...)` | 23 | Reabrir o video fonte e renderizar MP4 anotado com overlays e painel. |
| `ensure_runtime_ready()` | 25 | Validar dependencias/objetos essenciais antes da execucao. |
| `resolve_video_input_path(video_path)` | 25 | Resolver caminho de entrada relativo ou absoluto. |
| `create_holistic_landmarker()` | 25 | Instanciar o landmarker do MediaPipe para inferencia em video. |
| `frame_to_mp_image(frame_rgb)` | 25 | Converter frame RGB do OpenCV para `mp.Image`. |
| `build_pipeline_metadata(coordinate_mode)` | 25 | Montar metadados da pipeline: versao, modelo, modo de coordenada e parametros. |
| `build_fail_safe_payload(...)` | 25 | Gerar payload de erro/sem sinal e gravar JSON quando a pipeline nao consegue processar. |
| `process_video_asset(video_path)` | 25 | Pipeline principal: valida, le video, amostra frames, roda MediaPipe, calcula geometria, avalia regras, agrega, renderiza e salva JSON. |
| `load_payload(json_path)` | 29 | Carregar um payload JSON gerado para inspecao posterior. |

## Arvore de chamadas principal

```text
process_video_asset(video_path)
|- ensure_runtime_ready()
|- assert_model_assets_available()
|- resolve_video_input_path(video_path)
|- abre cv2.VideoCapture
|- build_smoothers()
|  `- EMASmoother() por grupo: pose_2d, pose_world, face_2d, left_hand_2d, right_hand_2d
|- create_holistic_landmarker()
|- para cada frame amostrado
|  |- frame_to_mp_image(frame_rgb)
|  |- holistic_landmarker.detect_for_video(...)
|  |- extract_landmark_sets(...)
|  |  |- landmark_list_to_points(...)
|  |  |  |- normalize_landmark_container(...)
|  |  |  `- make_point(...)
|  |  `- build_empty_landmark_sets() em fallback
|  |- smooth_landmark_sets(...)
|  |  `- EMASmoother.update(...)
|  |     `- blend_optional(...)
|  |- compute_reference_geometry(...)
|  |  |- primitivas: midpoint, centroid, point_distance, offset_point
|  |  |- compute_face_vertical_limits(...)
|  |  |- estimate_head_width(...)
|  |  |- compute_head_tilt_components(...)
|  |  |  |- estimate_face_pitch_degrees(...)
|  |  |  |- compute_directional_face_pitch_score(...)
|  |  |  |  `- classify_face_pitch_direction(...)
|  |  |  `- weighted_mean_score(...)
|  |  |- infer_neck_pose_mode(...)
|  |  |- build_neck_lateral_anchor_summary(...)
|  |  |- build_neck_regions(...)
|  |  |  `- compute_neck_height_from_shoulders(...)
|  |  |- build_head_reference_points(...)
|  |  |- build_side_contact_evidence(left/right)
|  |  |  `- estimate_arm_proxy_point(...)
|  |  `- compute_forward_head_2d_components(...)
|  |- evaluate_rules(geometry)
|  |  |- evaluate_hand_on_head(...)
|  |  |  `- evaluate_summary_contact_rule(...)
|  |  |     |- contact_rule_precheck(...)
|  |  |     |- build_head_contact_summary(...)
|  |  |     |  `- build_head_contact_targets(...)
|  |  |     |- score_head_contact_summary(...)
|  |  |     `- build_contact_rule_result(...)
|  |  |- evaluate_hand_on_neck(...)
|  |  |  `- evaluate_point_contact_rule(...)
|  |  |     `- score_neck_contact_point(...)
|  |  |        |- neck_zone_score(...)
|  |  |        |- point_target_overlap_score(...)
|  |  |        |- build_face_contact_targets(...)
|  |  |        |- build_head_contact_targets(...)
|  |  |        |- face_vertical_score(...)
|  |  |        `- head_vertical_score(...)
|  |  |- evaluate_hand_on_chest(...)
|  |  |  `- evaluate_point_contact_rule(...)
|  |  |     `- score_chest_contact_point(...)
|  |  |        `- chest_vertical_score(...)
|  |  |- evaluate_head_down(...)
|  |  |- evaluate_forward_head(...)
|  |  `- evaluate_rounded_shoulders_or_asymmetry(...)
|  |- build_frame_signal_record(...)
|  |  |- choose_primary_trigger(...)
|  |  |- display_rules_for_overlay(...)
|  |  |- build_frame_explanation(...)
|  |  `- evidence_summary(...)
|  `- clone_drawing_points(...)
|- aggregate_windows(...)
|  |- window_ranges(...)
|  |- count_max_consecutive_signal_frames(...)
|  |- label_window(...)
|  `- pretty_rule_name(...)
|- build_video_summary(...)
|  |- peak_window_level(...)
|  |- has_adjacent_strong_windows(...)
|  `- pretty_rule_name(...)
|- build_limitations(...)
|- render_annotated_video(...)
|  |- select_window_for_timestamp(...)
|  |- draw_landmarks_overlay(...)
|  |  |- draw_points(...)
|  |  `- draw_segment(...)
|  |- draw_neck_debug_region(...)
|  |  `- normalized_to_pixel(...)
|  `- draw_status_panel(...)
|     `- format_display_rules(...)
|- build_pipeline_metadata(...)
`- write_payload_json(...)
   `- sanitize_for_json(...)
```

## Sequencia operacional resumida

```text
Entrada de video
  -> validacao de runtime e modelo
  -> abertura OpenCV e coleta de FPS/dimensoes/duracao
  -> amostragem por TARGET_SAMPLE_FPS
  -> inferencia MediaPipe por frame
  -> normalizacao de landmarks
  -> suavizacao EMA temporal
  -> geometria anatomica de referencia
  -> regras posturais concorrentes
  -> registro serializado do frame
  -> agregacao em janelas deslizantes
  -> resumo final do video
  -> renderizacao de MP4 anotado
  -> gravacao do JSON final
```

## Pontos arquiteturais importantes

- `process_video_asset` e o unico orquestrador real. Quase todo o restante e funcao pura ou quase pura, o que facilita depuracao por etapa.
- `compute_reference_geometry` e o nucleo semantico da pipeline: ele traduz landmarks crus/suavizados em medidas interpretaveis pelas regras.
- `evaluate_rules` depende de `compute_reference_geometry`, nao de landmarks crus. Essa separacao reduz acoplamento entre MediaPipe e regras de negocio.
- Regras de contato continuam separadas em dois estilos: resumo da mao inteira (`evaluate_summary_contact_rule`, hoje aplicado ao contato na cabeca) e ponto individual (`evaluate_point_contact_rule`, usado em pescoco e peito).
- `build_frame_signal_record`, `aggregate_windows` e `build_video_summary` formam a camada de decisao temporal: frame -> janela -> video.
- A renderizacao (`render_annotated_video`) e posterior ao scoring: ela reutiliza `frame_records`, `drawings_by_frame` e `window_summaries`, sem recalcular MediaPipe.
- O fluxo de erro usa `build_fail_safe_payload` para produzir JSON mesmo em falhas de arquivo, abertura ou ausencia de frames amostrados.
