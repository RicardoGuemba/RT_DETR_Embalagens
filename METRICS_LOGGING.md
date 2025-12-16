# Sistema de Logging de Métricas - Guia de Uso

Este documento descreve o sistema completo de logging de métricas implementado no pipeline de treinamento.

## 📊 Funcionalidades

O sistema agora exibe e salva métricas detalhadas durante o treinamento:

### Durante o Treinamento (por Step)
- **Loss total** e **losses individuais** (loss_ce, loss_bbox, loss_giou)
- **Learning rate** atual
- **Tempo por iteração**
- **Época** e **step** atual

### Durante a Validação (por Época)
- **Loss de validação**
- **mAP@[0.5:0.95]** (média de Average Precision)
- **mAP@0.5** (AP50)
- **mAP@0.75** (AP75)
- **Precision** e **Recall** @ IoU=0.5

## 💾 Arquivos Gerados

Todos os arquivos são salvos no diretório de saída (`runs_rtdetr` por padrão):

### 1. `train_metrics.csv`
Métricas de treinamento registradas a cada `logging_steps` (padrão: 50 steps).

**Colunas:**
- `epoch`: Época atual (float)
- `step`: Step global
- `loss`: Loss total
- `loss_ce`: Loss de classificação
- `loss_bbox`: Loss de coordenadas de bbox
- `loss_giou`: Loss GIoU
- `lr`: Learning rate
- `time_per_iter`: Tempo por iteração (segundos)
- `timestamp`: Data/hora ISO

### 2. `val_metrics.csv`
Métricas de validação registradas a cada `eval_steps` (padrão: 500 steps).

**Colunas:**
- `epoch`: Época atual (float)
- `step`: Step global
- `eval_loss`: Loss de validação
- `mAP`: mAP@[0.5:0.95]
- `AP50`: mAP@0.5
- `AP75`: mAP@0.75
- `precision`: Precision @ IoU=0.5
- `recall`: Recall @ IoU=0.5
- `timestamp`: Data/hora ISO

### 3. `metrics.jsonl`
Métricas de validação em formato JSON Lines (1 linha por avaliação).

**Formato:**
```json
{"epoch": 0.5, "step": 500, "timestamp": "2024-01-01T12:00:00", "eval_loss": 0.123, "mAP": 0.45, "AP50": 0.67, ...}
```

### 4. `tb/` (TensorBoard)
Logs do TensorBoard para visualização gráfica das métricas.

## 🚀 Como Usar

### 1. Treinar o Modelo

Execute o treinamento normalmente:

```bash
python src/train_rtdetr.py \
    --dataset_dir dataset \
    --out_dir runs_rtdetr \
    --epochs 50 \
    --batch_size 1 \
    --logging_steps 50 \
    --eval_steps 500
```

### 2. Visualizar Métricas no Console

Durante o treinamento, você verá:

**A cada logging step (50 steps por padrão):**
```
📊 Step 50 | Época 0.12 | Loss: 0.123456 | Loss_CE: 0.045 | Loss_Bbox: 0.034 | Loss_GIoU: 0.044 | LR: 1.00e-05 | Time/iter: 0.234s
```

**A cada avaliação (500 steps por padrão):**
```
======================================================================
📊 MÉTRICAS DE VALIDAÇÃO - Step 500 | Época 1.25
======================================================================
  Loss:              0.098765
  mAP@0.5:0.95:      0.4523 (45.23%)
  mAP@0.5:           0.6789 (67.89%)
  mAP@0.75:          0.3456 (34.56%)
  Precision:         0.7123 (71.23%)
  Recall:            0.6543 (65.43%)
======================================================================
💾 Métricas salvas em:
   - runs_rtdetr/val_metrics.csv
   - runs_rtdetr/metrics.jsonl
   - TensorBoard: runs_rtdetr/tb
======================================================================
```

### 3. Visualizar no TensorBoard

**Instalar TensorBoard (se ainda não instalado):**
```bash
pip install tensorboard
```

**Iniciar TensorBoard:**
```bash
tensorboard --logdir runs_rtdetr/tb
```

Acesse `http://localhost:6006` no navegador para ver gráficos interativos das métricas.

### 4. Analisar Métricas em Python

```python
import pandas as pd
import json

# Ler métricas de treinamento
train_df = pd.read_csv('runs_rtdetr/train_metrics.csv')
print(train_df.head())

# Ler métricas de validação
val_df = pd.read_csv('runs_rtdetr/val_metrics.csv')
print(val_df.head())

# Ler JSONL
metrics = []
with open('runs_rtdetr/metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))
```

## ⚙️ Configuração

### Parâmetros Importantes

- `--logging_steps 50`: Frequência de logging de métricas de treinamento
- `--eval_steps 500`: Frequência de avaliação e cálculo de métricas COCO
- `--save_steps 500`: Frequência de salvamento de checkpoints

### Desabilitar TensorBoard

O TensorBoard é habilitado automaticamente se disponível. Para desabilitar, edite `src/train_rtdetr.py`:

```python
metrics_callback = MetricsCallback(
    output_dir=out_dir,
    use_tensorboard=False  # Desabilitar TensorBoard
)
```

## 🔍 Troubleshooting

### Métricas não aparecem no console

- Verifique se `logging_steps` está configurado corretamente
- Certifique-se de que há um dataset de validação (`valid/_annotations.coco.json`)
- Verifique os logs para erros de cálculo de métricas COCO

### TensorBoard não inicia

- Instale TensorBoard: `pip install tensorboard`
- Verifique se o diretório `runs_rtdetr/tb` existe
- Tente usar uma porta diferente: `tensorboard --logdir runs_rtdetr/tb --port 6007`

### Losses individuais aparecem como vazias

- Isso é normal se o modelo DETR não retornar `loss_dict`
- Apenas a loss total será registrada
- O sistema funciona normalmente mesmo sem losses individuais

## 📝 Notas

- As métricas são salvas **durante** o treinamento, não apenas no final
- O sistema é compatível com **MPS (Mac)** e **CUDA (Windows/Linux)**
- Métricas COCO podem ser zeradas nas primeiras épocas (normal quando o modelo ainda está aprendendo)
- O TensorBoard é opcional mas recomendado para visualização gráfica

