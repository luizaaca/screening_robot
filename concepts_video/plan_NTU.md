Segue um plano direto para implementar primeiro em **NTU RGB+D 60 com skeleton 3D**, porque ele já contém classes úteis para o seu tema — como **bow (A35)**, **wipe face (A37)**, **staggering (A42)**, **touch head/headache (A44)**, **touch chest (A45)**, **touch neck (A47)** e **nausea/vomiting (A48)** — e a modalidade de skeleton é muito menor que RGB: **5,8 GB** contra **136 GB** para os vídeos RGB. Para um protótipo leve, isso é o melhor ponto de partida.

Também é o caminho mais simples para notebook porque o **PYSKL** já fornece **anotações 3D pré-processadas** para **NTU60** e **NTU120** em `.pkl`, além de suportar **ST-GCN** e **ST-GCN++** e disponibilizar checkpoints para esses modelos. O formato das anotações já vem pronto para treino, com campos como `split`, `annotations`, `label` e `keypoint`, e para NTU 3D o array de keypoints usa **25 juntas** e **3 coordenadas por junta**.

## Decisão de implementação

Eu recomendo este recorte inicial:

* **Dataset:** NTU RGB+D 60, **somente skeleton 3D**.
* **Tarefa inicial:** **multiclasse** com um subconjunto pequeno e útil das classes do NTU.
* **Modelo inicial:** um **GRU leve** ou **TCN leve** em PyTorch, treinado diretamente sobre o `.pkl` do PYSKL.
* **Upgrade posterior:** fine-tuning de **ST-GCN** ou **ST-GCN++** via PYSKL, reaproveitando exatamente o mesmo subconjunto.

## Referências de download

### 1) Página oficial do dataset

A página oficial da ROSE/NTU informa que o download completo requer **cadastro, request form e aceite do release agreement**, e que o uso é **acadêmico, não comercial**. Ela também descreve tamanhos por modalidade e a lista de classes.

### 2) Repositório oficial do NTU RGB+D

O repositório oficial no GitHub reúne a descrição do dataset, a lista completa de ações, links para download e observações importantes como a lista de amostras com skeleton ausente/incompleto, que devem ser ignoradas em pipelines baseados em skeleton.

### 3) Anotações pré-processadas do PYSKL

O PYSKL fornece diretamente os arquivos:

* `ntu60_3danno.pkl`
* `ntu120_3danno.pkl`

Esses arquivos já estão no formato de treino/teste do toolbox e são o melhor atalho para começar sem baixar tudo do zero.

### 4) Toolbox para upgrade de fine-tuning

O PYSKL suporta **ST-GCN**, **ST-GCN++**, **AAGCN**, **CTR-GCN** e outros modelos de skeleton action recognition, além de informar que possui modelos treinados/checkpoints para NTU.

## Recorte multiclasses recomendado

Para um primeiro experimento, eu usaria **8 classes**: 6 sinais comportamentais + 2 neutras.

As ações abaixo existem no NTU RGB+D oficial:
**A11 reading**, **A12 writing**, **A35 nod head/bow**, **A37 wipe face**, **A42 staggering**, **A44 touch head (headache)**, **A47 touch neck (neckache)** e **A48 nausea/vomiting**.

Minha sugestão de mapeamento:

* `reading`
* `writing`
* `bow`
* `wipe_face`
* `staggering`
* `touch_head`
* `touch_neck`
* `nausea_vomiting`

Esse conjunto é bom porque:

* mantém o problema pequeno;
* já cobre sinais corporais úteis;
* evita classes muito dependentes de interação com outra pessoa;
* fica fácil depois agrupar em binário `ok / not_ok`.

## Plano de notebook pronto para o Codex

### Notebook 01 — Setup e download

Objetivo: criar ambiente, baixar o `.pkl` do NTU60 e confirmar a estrutura.

```bash
# célula 1
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib tqdm jupyter
```

```bash
# célula 2
mkdir -p data/nturgbd
wget -O data/nturgbd/ntu60_3danno.pkl \
  https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
```

Esse arquivo é a anotação 3D pré-processada do NTU60 fornecida pelo PYSKL.

Se depois você quiser o dataset oficial bruto, use a página da ROSE ou o repositório oficial do NTU. O próprio repositório aponta alternativas para obter os arquivos de skeleton.

---

### Notebook 02 — Carregar e inspecionar o `.pkl`

Objetivo: entender `split`, `annotations`, contagem de classes e formato dos keypoints.

```python
# célula 3
import pickle
from pathlib import Path

pkl_path = Path("data/nturgbd/ntu60_3danno.pkl")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())
print(data["split"].keys())
print(type(data["annotations"]), len(data["annotations"]))
print(data["annotations"][0].keys())
print(data["annotations"][0]["label"])
print(data["annotations"][0]["keypoint"].shape)
```

O PYSKL documenta que o `.pkl` contém um dicionário com `split` e `annotations`, e que cada anotação traz `label` e `keypoint` com shape `[M x T x V x C]`. Para NTU 3D, `V=25` juntas e `C=3` coordenadas.

---

### Notebook 03 — Selecionar classes e remapear rótulos

Objetivo: criar um subconjunto menor e mais aderente ao problema.

Observação importante: o script oficial de geração do ST-GCN converte o rótulo do arquivo NTU para **zero-based label** usando `action_class - 1`. Então, por exemplo, **A35** vira **34** no `.pkl`.

```python
# célula 4
SELECTED_CLASSES = {
    10: "reading",             # A11
    11: "writing",             # A12
    34: "bow",                 # A35
    36: "wipe_face",           # A37
    41: "staggering",          # A42
    43: "touch_head",          # A44
    46: "touch_neck",          # A47
    47: "nausea_vomiting",     # A48
}

label_to_new = {old_label: i for i, old_label in enumerate(SELECTED_CLASSES.keys())}
id_to_name = {i: name for i, name in enumerate(SELECTED_CLASSES.values())}
print(label_to_new)
print(id_to_name)
```

```python
# célula 5
selected_annotations = [
    ann for ann in data["annotations"]
    if ann["label"] in SELECTED_CLASSES
]

for ann in selected_annotations:
    ann["orig_label"] = ann["label"]
    ann["label"] = label_to_new[ann["label"]]

print("total subset:", len(selected_annotations))
```

---

### Notebook 04 — Preservar o split oficial e filtrar só o subset

Objetivo: usar um split padrão do dataset.

O NTU foi proposto com critérios de avaliação **cross-subject** e **cross-view**. O pipeline clássico do ST-GCN também materializa esses splits (`xsub` e `xview`). Para o primeiro experimento, eu começaria com **cross-subject**, porque ele generaliza melhor para pessoas novas.

```python
# célula 6
available_splits = list(data["split"].keys())
print("splits:", available_splits)

# ajuste depois de inspecionar os nomes reais
train_split_name = [k for k in available_splits if "xsub" in k.lower() and "train" in k.lower()][0]
val_split_name   = [k for k in available_splits if "xsub" in k.lower() and ("val" in k.lower() or "test" in k.lower())][0]

train_ids = set(data["split"][train_split_name])
val_ids   = set(data["split"][val_split_name])

train_annotations = [a for a in selected_annotations if a["frame_dir"] in train_ids]
val_annotations   = [a for a in selected_annotations if a["frame_dir"] in val_ids]

print(len(train_annotations), len(val_annotations))
```

---

### Notebook 05 — Pré-processamento leve

Objetivo: transformar skeleton variável em tensor fixo.

Estratégia simples:

1. manter **a pessoa principal**;
2. normalizar em relação à junta raiz;
3. reamostrar temporalmente para um tamanho fixo, por exemplo **64 frames**;
4. opcionalmente adicionar **velocidade temporal**.

```python
# célula 7
import numpy as np

TARGET_FRAMES = 64

def select_main_person(kp):
    # kp shape: [M, T, V, C]
    # escolhe a pessoa com maior energia de movimento
    energies = []
    for m in range(kp.shape[0]):
        person = kp[m]
        energy = np.nan_to_num(np.abs(np.diff(person, axis=0))).sum()
        energies.append(energy)
    main_idx = int(np.argmax(energies))
    return kp[main_idx]  # [T, V, C]

def center_on_root(person_kp, root_idx=1):
    # root_idx ajustar depois de inspecionar o esqueleto; a ideia é centralizar numa junta do tronco
    root = person_kp[:, root_idx:root_idx+1, :]
    return person_kp - root

def temporal_resample(person_kp, target_frames=64):
    T = person_kp.shape[0]
    if T == target_frames:
        return person_kp
    idx = np.linspace(0, T - 1, target_frames).astype(np.int32)
    return person_kp[idx]

def build_feature(person_kp):
    # [T, V, C] -> [T, V*C]
    pos = person_kp.reshape(person_kp.shape[0], -1)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    feat = np.concatenate([pos, vel], axis=-1)  # leve e útil
    return feat.astype(np.float32)
```

```python
# célula 8
def preprocess_ann(ann):
    kp = ann["keypoint"]               # [M, T, V, C]
    person = select_main_person(kp)
    person = center_on_root(person)
    person = temporal_resample(person, TARGET_FRAMES)
    feat = build_feature(person)       # [T, 2*V*C]
    label = ann["label"]
    return feat, label
```

---

### Notebook 06 — Dataset PyTorch

Objetivo: gerar `DataLoader` simples.

```python
# célula 9
import torch
from torch.utils.data import Dataset, DataLoader

class NTUSubsetDataset(Dataset):
    def __init__(self, annotations):
        self.items = [preprocess_ann(a) for a in annotations]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.tensor(x), torch.tensor(y)

train_ds = NTUSubsetDataset(train_annotations)
val_ds = NTUSubsetDataset(val_annotations)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)
```

---

### Notebook 07 — Modelo leve

Objetivo: começar com um baseline rápido e treinável em notebook.

Eu usaria primeiro um **BiGRU pequeno**. É simples, leve e suficiente para validar o pipeline. Se o baseline funcionar, aí sim vale migrar para ST-GCN. O input aqui será `[batch, T, F]`, onde `F = 2 * 25 * 3 = 150` se você usar posição + velocidade. A dimensão de **25 juntas** e **3 coordenadas** vem do NTU 3D skeleton.

```python
# célula 10
import torch.nn as nn

NUM_CLASSES = len(SELECTED_CLASSES)

class TinyBiGRU(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)

model = TinyBiGRU(num_classes=NUM_CLASSES)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
```

```python
# célula 11
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

### Notebook 08 — Loop de treino e validação

Objetivo: treinar e medir.

```python
# célula 12
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

def run_epoch(model, dl, train=True):
    model.train(train)
    losses = []
    y_true, y_pred = [], []

    for xb, yb in tqdm(dl):
        xb, yb = xb.to(device), yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds)

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }

best_f1 = -1
for epoch in range(1, 21):
    train_metrics = run_epoch(model, train_dl, train=True)
    val_metrics = run_epoch(model, val_dl, train=False)

    print(f"Epoch {epoch}")
    print("train:", train_metrics)
    print("val  :", val_metrics)

    if val_metrics["f1_macro"] > best_f1:
        best_f1 = val_metrics["f1_macro"]
        torch.save(model.state_dict(), "artifacts/tiny_bigru_ntu60_subset.pt")
```

---

### Notebook 09 — Matriz de confusão e análise por classe

Objetivo: entender quais classes fazem sentido no domínio do depoimento.

```python
# célula 13
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def predict_all(model, dl):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = predict_all(model, val_dl)
print(classification_report(y_true, y_pred, target_names=[id_to_name[i] for i in range(NUM_CLASSES)]))
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(NUM_CLASSES), [id_to_name[i] for i in range(NUM_CLASSES)], rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), [id_to_name[i] for i in range(NUM_CLASSES)])
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()
```

---

### Notebook 10 — Exportar artefatos

Objetivo: salvar tudo que o agente de triagem precisará depois.

```python
# célula 14
import json, os
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/id_to_name.json", "w") as f:
    json.dump(id_to_name, f, indent=2)

with open("artifacts/config.json", "w") as f:
    json.dump({
        "target_frames": TARGET_FRAMES,
        "selected_classes": SELECTED_CLASSES,
        "label_to_new": label_to_new
    }, f, indent=2)
```

---

### Notebook 11 — Agente de triagem

Objetivo: transformar multiclasses em um alerta interpretável.

A ideia aqui é simples:

* o modelo continua **multiclasse**;
* o agente agrupa as classes em:

  * `neutral`
  * `mild_signal`
  * `strong_signal`
* ele roda em janelas deslizantes de 2–4 segundos e gera um score por vídeo.

Exemplo de agrupamento sugerido:

* `neutral`: reading, writing
* `mild_signal`: bow, wipe_face, touch_head, touch_neck
* `strong_signal`: staggering, nausea_vomiting

```python
# célula 15
TRIAGE_GROUPS = {
    "neutral": {"reading", "writing"},
    "mild_signal": {"bow", "wipe_face", "touch_head", "touch_neck"},
    "strong_signal": {"staggering", "nausea_vomiting"},
}

def triage_from_logits(logits, id_to_name):
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred_id = int(probs.argmax())
    pred_name = id_to_name[pred_id]
    conf = float(probs.max())

    if pred_name in TRIAGE_GROUPS["strong_signal"]:
        level = "not_ok_strong"
    elif pred_name in TRIAGE_GROUPS["mild_signal"]:
        level = "not_ok_mild"
    else:
        level = "ok"

    return {"pred_class": pred_name, "confidence": conf, "triage": level}
```

## Caminho de upgrade com transfer learning

Depois que esse baseline funcionar, o próximo passo natural é **fine-tunar ST-GCN ou ST-GCN++** no mesmo subconjunto. O PYSKL suporta esses modelos e informa que disponibiliza modelos/checkpoints para NTU60 e NTU120.

A vantagem é que você:

* mantém o mesmo dataset;
* mantém o mesmo mapeamento de classes;
* só troca o backbone.

## Como aplicar depois em vídeos de depoimento

Quando o classificador multiclasses estiver bom no NTU subset, você parte para inferência em vídeos reais. O PYSKL informa que também fornece scripts para **extrair skeletons 2D de datasets RGB arbitrários**, o que é útil para a etapa futura de adaptar o pipeline aos seus próprios vídeos.

Na prática, o pipeline final fica:

1. vídeo de depoimento
2. extração de pose/skeleton
3. janelas temporais fixas
4. classificador multiclasse
5. agente de triagem agregando eventos ao longo do vídeo

