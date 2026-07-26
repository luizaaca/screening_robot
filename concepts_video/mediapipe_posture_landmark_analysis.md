# Analise de landmarks e regras no `mediapipe_video_posture_timeline.ipynb`

Data da analise: 2026-06-29.

Atualizacao de decisoes: 2026-06-29.

## Escopo

Esta analise foi feita sem alterar o notebook ou o codigo do pipeline. Foram inspecionados:

- `concepts_video/mediapipe_video_posture_timeline.ipynb`
- `concepts_video/outputs/posture_timelines/sad_woman1.signals.json`
- `concepts_video/outputs/posture_timelines/sad_woman2.signals.json`
- anotacoes geradas em `concepts_video/outputs/annotated_videos/`
- documentacao atual do MediaPipe Pose, Hand e Holistic Landmarker.

## Conclusao curta

O comportamento observado parece vir de quatro causas combinadas, e as decisoes de correcao ja fechadas apontam para uma refatoracao controlada do pipeline:

1. O triangulo visual nao e uma conexao padrao do MediaPipe. Ele e criado manualmente pelo notebook em `POSE_DRAW_SEGMENTS`, conectando orelhas-ombros, ombro-ombro e nariz-ombros. Por isso os bracos nao aparecem no overlay.
2. O pipeline extrai somente 7 landmarks de pose: nariz, orelhas, cantos da boca e ombros. Cotovelos, punhos e quadris existem no Pose Landmarker, mas sao descartados antes das regras e antes do desenho.
3. As regras de contato dependem quase totalmente do Hand Landmarker. Quando a mao fica coberta ou pouco visivel, como em `sad_woman1`, nao ha fallback por punho/cotovelo da pose. Em `sad_woman2`, a regra `hand_on_face` tem prioridade absoluta sobre `hand_on_neck`, entao varios frames proximos ao pescoco sao classificados como rosto.
4. A regra atual de `head_down` gera falso positivo em `sad_woman1`: o video tem varios frames com cabeca frontal/ereta, mas o JSON marca `head_down` em 100% dos frames amostrados. O problema vem do uso de `head_height_ratio` absoluto com limiares fixos, normalizado pela largura dos ombros.

Decisoes fechadas para a proxima versao:

- usar `HolisticLandmarker` da API nova de MediaPipe Tasks;
- incluir `pose wrist/elbow` como fallback de contato quando a mao nao for detectada;
- tornar as regras concorrentes por score, sem supressao exclusiva entre rosto, pescoco e peito;
- gravar os scores concorrentes no JSON;
- no video anotado, mostrar regras e scores quando passarem seus limiares;
- incluir os landmarks necessarios para um calculo mais preciso de cabeca, pescoco, tronco, bracos e maos.
- recalibrar `head_down`/inclinacao da cabeca antes de usar essa medida para expandir a zona de pescoco.

## Evidencias locais

### 1. O triangulo vem do overlay, nao do MediaPipe

No notebook, o desenho de pose usa uma lista manual:

```python
POSE_DRAW_SEGMENTS = [
    ('left_ear', 'left_shoulder'),
    ('right_ear', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('nose', 'left_shoulder'),
    ('nose', 'right_shoulder'),
]
```

Isso explica exatamente o padrao visto: duas arestas de ombros para orelhas e duas de ombros para nariz, formando um triangulo/losango na cabeca e tronco superior. Nao ha segmentos de braco porque `left_elbow`, `right_elbow`, `left_wrist` e `right_wrist` nao sao extraidos nem desenhados.

O desenho de mao tambem e reduzido:

```python
HAND_DRAW_SEGMENTS = [
    ('wrist', 'thumb_tip'),
    ('wrist', 'index_tip'),
    ('wrist', 'middle_tip'),
]
```

Ou seja, mesmo quando a mao aparece, o overlay mostra so tres raios simplificados, nao a malha completa da mao.

### 2. O notebook descarta landmarks de braco

O `POSE_LANDMARK_INDEX` atual contem apenas:

```python
{
    'nose': 0,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
}
```

O Pose Landmarker retorna 33 landmarks por pessoa, incluindo cotovelos e punhos. Como esses pontos nao entram no dicionario local, o restante do pipeline nao tem como:

- desenhar os bracos;
- usar punho/cotovelo como fallback quando a mao nao e detectada;
- distinguir "mao no pescoco" de "mao no rosto" por posicao do antebraco;
- diagnosticar se a mao sumiu por oclusao, por corte de enquadramento ou por baixa confianca.

### 3. `sad_woman1`: a mao nao foi detectada em nenhum frame amostrado

Resumo calculado a partir de `sad_woman1.signals.json`:

- frames amostrados: 176
- frames com mao esquerda visivel: 0
- frames com mao direita visivel: 0
- frames com qualquer mao visivel: 0
- `hand_on_face`: 176 frames `unknown`
- `hand_on_neck`: 176 frames `unknown`
- `hand_on_chest`: 176 frames `unknown`
- `head_down`: 176 frames `true`
- `forward_head`: 176 frames `true`

O JSON ja registra a limitacao: `Hand landmarks were unavailable in 100.0% of sampled frames`. Como as regras de contato so usam pontos de mao, uma mao coberta nao pode ser inferida. O pipeline ate sabe que ha cabeca baixa e cabeca projetada a frente, mas nao consegue associar contato sem landmarks de mao ou fallback de braco.

Revisao adicional: a inspeccao visual de frames aos 0s, 5s, 10s, 15s, 20s, 25s, 30s e 35s mostra que `sad_woman1` nao deve ser tratado como cabeca inclinada de forma persistente. Ha momentos com olhar para baixo, expressao triste e oclusao por manga/mao, mas varios frames sao frontais e eretos. Portanto o resultado `head_down = true` em 176/176 frames e um falso positivo sistematico da regra atual.

Distribuicao observada no JSON atual:

- `head_height_ratio`: minimo 0.2332, media 0.2885, maximo 0.3444;
- `head_down`: 176/176 frames `true`;
- `head_down` forte: 105/176 frames;
- `forward_head_world_ratio`: media 0.8703, tambem sempre ativando `forward_head`.

Conclusao: `head_height_ratio = (shoulder_mid.y - nose.y) / shoulder_width` nao deve ser usado sozinho para inferir inclinacao da cabeca. Em enquadramentos verticais, ombros largos no frame reduzem artificialmente a razao e fazem uma cabeca ereta parecer "baixa". Essa metrica pode continuar como componente fraco e relativo, mas nao como decisor absoluto.

### 4. `sad_woman2`: pescoco perde para rosto por prioridade e limiar

Resumo calculado a partir de `sad_woman2.signals.json`:

- frames amostrados: 184
- frames com qualquer mao visivel: 78
- frames com `hand_on_face = true`: 74
- frames com `hand_on_neck = true`: 0
- frames em que `hand_on_neck` foi suprimido por `hand_on_face`: 74
- frames sem landmarks de mao: 106

O ponto critico e a regra:

```python
if hand_on_face_rule.get('state') == 'true':
    return build_rule_result('false', 'none', ..., suppressed_by='hand_on_face')
```

Na pratica, qualquer contato considerado "rosto" impede "pescoco". Isso e fragil para videos em que a mao esta no pescoco, queixo, mandibula ou lateral da cabeca, porque os anchors de face sao apenas nariz, boca e orelhas. Orelhas e cantos da boca podem ficar geometricamente proximos do pescoco, principalmente com cabeca baixa.

Nos 78 frames de `sad_woman2` com ratios de mao:

- `hand_on_face` cru ficaria forte em 59 frames e fraco em 15;
- `hand_on_neck` cru ficaria forte em 32 frames e fraco em 22;
- em 25 frames, o ratio do pescoco e menor que o ratio do rosto, mas a prioridade atual ainda favorece rosto quando `hand_on_face` passa no limiar.

Isso confirma que o problema nao e apenas deteccao de mao. Ha uma ambiguidade de regra: pescoco e rosto competem na mesma regiao e a decisao atual e unilateral.

## O que a documentacao do MediaPipe sugere

Referencias consultadas:

- Pose Landmarker Python: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
- Hand Landmarker Python: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- Holistic Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
- MediaPipe Holistic legado, com pipeline e exemplos de desenho: https://chuoling.github.io/mediapipe/solutions/holistic.html

Pontos relevantes:

- O Pose Landmarker e indicado para detectar landmarks corporais em imagem/video, analisar postura e categorizar movimentos. Ele retorna landmarks 2D normalizados e world landmarks 3D, com 33 pontos por pose.
- O Hand Landmarker retorna 21 landmarks por mao detectada, alem de handedness e world landmarks. Em modo video, ele usa tracking para reduzir custo e reexecuta deteccao quando presenca/tracking caem abaixo dos limiares configurados.
- O Holistic Landmarker combina pose, face e maos e retorna 543 landmarks: 33 de pose, 468 de face e 21 por mao. A documentacao legada explica um detalhe importante para este caso: o pipeline usa pose para derivar ROIs de face e maos em alta resolucao, aplica modelos especificos nessas regioes e depois junta tudo. Isso costuma ser mais adequado do que rodar hand detection independente no frame inteiro quando a mao e pequena, parcialmente visivel ou perto da cabeca.
- Os exemplos oficiais desenham `POSE_CONNECTIONS` e `HAND_CONNECTIONS`, nao um conjunto manual nariz-ombros. Isso facilita validar visualmente se cotovelos, punhos e maos seguem o corpo.

No ambiente local, `mediapipe 0.10.35` expoe `mp.tasks.vision.HolisticLandmarker`, `HolisticLandmarkerOptions` e `HolisticLandmarkerResult`. Portanto a migracao pode ficar dentro da API nova de Tasks, sem retornar para `mp.solutions.holistic` legado.

## Desenho de correcao decidido

### 1. Migrar para Holistic Tasks

O notebook deve passar a usar `mp.tasks.vision.HolisticLandmarker` em modo `VIDEO`. O resultado esperado passa a ser:

- `pose_landmarks`
- `pose_world_landmarks`
- `face_landmarks`
- `left_hand_landmarks`
- `left_hand_world_landmarks`
- `right_hand_landmarks`
- `right_hand_world_landmarks`

Isso elimina a duplicidade atual entre `PoseLandmarker` e `HandLandmarker` separados e disponibiliza face mesh denso para separar melhor rosto, mandibula, queixo e pescoco.

### 2. Landmarks necessarios

O pipeline deve extrair, no minimo:

- pose: `nose`, olhos, orelhas, boca quando disponivel, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`;
- maos: punho, polegar, indicador, medio, anelar e minimo, preferencialmente todos os 21 pontos;
- face: pontos de nariz, olhos, boca, queixo/mandibula e contorno lateral da face; os indices exatos do face mesh devem ser definidos na implementacao para evitar anchors instaveis;
- world landmarks: ombros, nariz, cotovelos e punhos quando disponiveis, para diagnostico e possivel score 3D.

Os landmarks de braco sao necessarios mesmo usando Holistic, porque mao coberta ou mao fora do ROI ainda pode falhar.

### 3. Fallback por wrist/elbow da pose

Cada lado do corpo deve montar uma evidencia de contato com prioridade de qualidade:

- `holistic_hand`: usa landmarks de mao do Holistic, confianca alta;
- `pose_wrist_fallback`: usa `left_wrist` ou `right_wrist` da pose quando a mao nao foi detectada, confianca media;
- `arm_proxy`: usa cotovelo + direcao do antebraco quando o punho esta ausente ou pouco confiavel, confianca baixa;
- `unknown`: sem evidencia suficiente.

O fallback nao deve fingir precisao de mao. Ele deve entrar no JSON como fonte de evidencia e reduzir o score maximo possivel da regra. Exemplo conceitual:

- mao detectada: score maximo 1.00;
- punho da pose: score maximo 0.75;
- proxy por cotovelo/antebraco: score maximo 0.45;
- ausente: score 0 ou `unknown`, conforme a regra.

### 4. Regras concorrentes, nao excludentes

As regras `hand_on_face`, `hand_on_neck` e `hand_on_chest` devem ser calculadas de forma independente. Nenhuma deve zerar a outra por prioridade fixa.

Cada regra deve gravar no JSON:

- `raw_distance_ratio`;
- `zone_score`;
- `evidence_quality`;
- `source`;
- `final_score`;
- `passed_threshold`;
- `closest_side`;
- `landmarks_used`;
- `notes`, quando houver ambiguidade ou dado parcial.

O termo "probabilidade" deve ser tratado como score heuristico entre 0 e 1, nao como probabilidade estatistica calibrada, a menos que seja feita calibracao supervisionada depois.

No video anotado, mostrar apenas regras acima do limiar de exibicao. Exemplo:

```text
hand_on_neck 0.82
hand_on_face 0.41
head_down 0.76
```

Isso permite ver competicao entre regioes sem esconder sinais secundarios.

### 5. Regra matematica para pescoco

Coordenadas normalizadas 2D do MediaPipe usam `y` crescente para baixo. A proposta inicial para a zona de pescoco e:

```text
shoulder_mid = midpoint(left_shoulder, right_shoulder)
shoulder_to_nose = max(epsilon, shoulder_mid.y - nose.y)
head_height_ratio = shoulder_to_nose / shoulder_width
```

Com cabeca neutra, considerar pescoco ate 20% acima da linha dos ombros, usando a distancia ombros-nariz como escala:

```text
base_neck_fraction = 0.20
```

Como a area visual do pescoco tende a se deslocar/expandir quando a cabeca inclina para baixo, aplicar um bonus de inclinacao. A revisao de `sad_woman1` mostrou que esse bonus nao pode usar `head_height_ratio` absoluto sozinho. A proposta ajustada e calcular um `head_tilt_score` composto:

```text
head_drop_score = clamp(
    (neutral_head_height_ratio - head_height_ratio)
    / (neutral_head_height_ratio - strong_head_down_ratio),
    0,
    1
)

face_pitch_score = score derivado de face mesh/head pose

head_tilt_score = weighted_mean(
    face_pitch_score com peso alto,
    head_drop_score com peso baixo,
    shoulder_ear_compression_score com peso baixo,
    ignorando componentes sem confianca
)

neck_fraction = clamp(
    base_neck_fraction + max_neck_tilt_bonus * head_tilt_score,
    0.20,
    max_neck_fraction
)

neck_top_y = shoulder_mid.y - (neck_fraction * shoulder_to_nose)
neck_bottom_y = shoulder_mid.y + chest_overlap_fraction * shoulder_width
```

Interpretacao:

- pontos entre `neck_top_y` e `neck_bottom_y` favorecem `hand_on_neck`;
- pontos acima de `neck_top_y` favorecem `hand_on_face/head`;
- pontos abaixo de `neck_bottom_y` favorecem `hand_on_chest`;
- `hand_on_neck` e `hand_on_face` continuam concorrentes: ambos podem receber score, mas a posicao vertical, distancia aos anchors e qualidade da evidencia determinam a intensidade.
- em videos como `sad_woman1`, com cabeca visualmente ereta, `head_tilt_score` deve ficar baixo mesmo que `head_height_ratio` absoluto pareca baixo.

Parametros iniciais sugeridos para teste:

```text
neutral_head_height_ratio = estimado por video quando houver frames frontais confiaveis, ou valor calibrado
strong_head_down_ratio = 0.30
base_neck_fraction = 0.20
max_neck_tilt_bonus = 0.25
max_neck_fraction = 0.45
chest_overlap_fraction = 0.08
```

Essa regra ainda precisa de calibracao visual, mas atende a decisao: a referencia principal e a distancia ombros-nariz, a zona base de pescoco e 20% acima dos ombros, e a zona aumenta quando a cabeca esta mais inclinada.

### 6. Score por regiao

Uma forma pratica de transformar distancia em score:

```text
distance_score = 1 - clamp(distance_to_region / region_radius, 0, 1)
vertical_score = 1 se dentro da faixa principal; decai fora da faixa
evidence_score = 1.00, 0.75 ou 0.45 conforme a fonte
final_score = distance_score * vertical_score * evidence_score
```

Para `hand_on_face`, a regiao deve usar anchors de face mesh: nariz, boca, queixo/mandibula e lateral da face, com pesos menores para pontos muito proximos ao pescoco.

Para `hand_on_neck`, a regiao deve usar a faixa dinamica descrita acima, mais centro do pescoco derivado de ombros, nariz, queixo e inclinacao.

Para `hand_on_chest`, a regiao deve usar ombros, centro do peito e quadris quando disponiveis.

### 7. Overlay de validacao

O overlay deve deixar de desenhar o triangulo nariz-ombros como conexao principal. A visualizacao deve:

- trocar `POSE_DRAW_SEGMENTS` manual por conexoes anatomicas, pelo menos:
  - `left_shoulder -> left_elbow -> left_wrist`
  - `right_shoulder -> right_elbow -> right_wrist`
  - `left_shoulder -> right_shoulder`
  - opcionalmente ombros-quadris para estabilizar tronco;
- remover ou marcar como debug as arestas `nose -> shoulder`, porque elas sugerem que o modelo esta errando quando na verdade o notebook esta desenhando esse triangulo;
- desenhar `HAND_CONNECTIONS` completo ou uma lista local equivalente com os 21 pontos da mao.

Esse ajuste nao altera a classificacao, mas torna os erros reais visiveis.

## Calibracao recomendada

Antes de escolher novos limiares fixos, separar alguns frames de `sad_woman1` e `sad_woman2` com rotulos simples:

- `head_contact`
- `neck_contact`
- `chest_contact`
- `covered_or_unknown`
- `no_contact`

Depois, plotar por frame:

- `min_face_hand_ratio`
- `min_neck_hand_ratio`
- `min_chest_hand_ratio`
- fonte da evidencia (`hand_landmarker`, `pose_wrist_fallback`, `arm_proxy`)
- landmarks ausentes.

Isso evita ajustar thresholds apenas por observacao visual de um video.

## JSON e diagnosticos

Campos uteis para depurar os proximos testes:

- disponibilidade por landmark: `left_wrist_visible`, `right_wrist_visible`, `left_elbow_visible`, `right_elbow_visible`;
- `hand_detection_source` por lado;
- scores crus de todas as regras, mesmo quando outras regras tem score maior;
- `score_components`, com distancia, faixa vertical e qualidade da evidencia;
- `ambiguous_regions`, quando face e pescoco passam simultaneamente com scores proximos;
- contagem por video de frames com mao via Holistic versus punho/cotovelo via Pose;
- `display_rules`, contendo apenas regras acima do limiar de exibicao para o overlay.

## Riscos e limites

- Baixar limiares de `min_hand_detection_confidence`, `min_hand_presence_confidence` ou `min_tracking_confidence` pode aumentar recall, mas tambem pode criar falsos positivos. Eu trataria isso como experimento controlado, nao como primeira correcao.
- Se a mao esta realmente coberta, o Holistic tambem pode nao detectar a mao. A solucao correta e classificar como `unknown` ou usar evidencia indireta de braco com menor confianca.
- `head_down` e `forward_head_world_ratio` estao sempre ativando em `sad_woman1` no JSON analisado. Pelo video bruto, `head_down` e falso positivo persistente; `forward_head` tambem merece revisao antes de influenciar o score final.
- A palavra "probabilidade" pode induzir leitura estatistica. Sem calibracao supervisionada, o nome tecnicamente mais correto e `score`.

## Decisoes pendentes resolvidas

1. Asset do Holistic Tasks: baixar o modelo a partir do notebook para `concepts_video/artifacts`.
   - URL validada com `HEAD 200 OK`: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task`
   - tamanho observado: 13.683.609 bytes;
   - caminho local recomendado: `concepts_video/artifacts/holistic_landmarker.task`;
   - usar a versao fixa `float16/1` em vez de `latest` para manter reprodutibilidade.
2. Indices de face mesh: sim, escolher explicitamente os landmarks de queixo, mandibula, boca e contorno lateral usados em `hand_on_face` e `hand_on_neck`.
3. Escala da zona de pescoco: sim, usar a expansao proposta inicialmente, com `max_neck_fraction = 0.45` para o primeiro teste.
4. Nome no JSON: usar `score`, nao `probability`.
5. Agregacao final do frame: usar o maior score individual como criterio primario.
6. Overlay: usar limiar unico para exibicao de regras no video anotado.

## Ambiguidades remanescentes

1. Os indices numericos finais do face mesh ainda precisam ser enumerados no codigo, mesmo com a decisao de usa-los explicitamente.
2. O valor do limiar unico de overlay ainda precisa ser escolhido. Sugestao inicial para teste: `display_score_threshold = 0.35`.
3. A formula exata de `face_pitch_score` precisa ser definida com os landmarks de face mesh escolhidos. A direcao esta clara: ela deve ter peso maior que `head_height_ratio` para evitar o falso positivo visto em `sad_woman1`.

## Plano de acao com checks

Status: implementação inicial criada em `concepts_video/mediapipe_video_posture_timeline_v2.ipynb`. Checks marcados indicam código implementado no notebook v2; a validação visual e os ajustes finos dependem da execução manual dos vídeos.

- [x] Baixar pelo notebook o asset `holistic_landmarker.task` de `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task`.
- [x] Substituir `create_pose_landmarker` e `create_hand_landmarker` por `create_holistic_landmarker`.
- [x] Adaptar `extract_landmark_sets` para ler `HolisticLandmarkerResult`.
- [x] Extrair pose completa necessaria: ombros, cotovelos, punhos, quadris, nariz, olhos, orelhas e boca.
- [x] Extrair maos completas com 21 landmarks por lado.
- [x] Extrair anchors de face mesh para rosto, mandibula e queixo.
- [x] Redesenhar overlay anatomico com bracos, maos e regioes de debug.
- [x] Implementar fonte de evidencia por lado: `holistic_hand`, `pose_wrist_fallback`, `arm_proxy`, `unknown`.
- [x] Implementar regra dinamica de zona do pescoco baseada em ombros-nariz e inclinacao da cabeca.
- [x] Recalibrar `head_down` para usar score composto de inclinacao, evitando falso positivo em `sad_woman1`.
- [x] Implementar scoring independente para `hand_on_face`, `hand_on_neck` e `hand_on_chest`.
- [x] Remover supressoes exclusivas entre regras de contato.
- [x] Atualizar JSON para salvar scores, componentes, fonte de evidencia e regras exibiveis.
- [x] Atualizar painel do video anotado para mostrar regras acima do limiar com scores.
- [ ] Reprocessar manualmente `sad_woman1` e `sad_woman2` no notebook.
- [ ] Comparar cobertura de maos/punhos, reducao de `unknown`, scores de face versus pescoco e resumo final.
