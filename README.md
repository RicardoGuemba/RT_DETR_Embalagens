# Aplicação de detecção voltada a embalagens

Sistema completo para treinar e avaliar modelos DETR/RT-DETR usando datasets COCO JSON do Roboflow, otimizado para MacBook com Apple Silicon (MPS).

**Arquivos de referência**

- [AVALIACAO_SISTEMA.md](AVALIACAO_SISTEMA.md)
- [IMPLEMENTACOES_CENTROIDE.md](IMPLEMENTACOES_CENTROIDE.md)

## 🚀 Início Rápido

```bash
# 1. Setup do ambiente
./scripts/bootstrap_mac.sh
source .venv/bin/activate

# 2. Executar aplicação principal
python app.py
```

O `app.py` é o ponto de entrada principal que inicia a interface Tkinter.

## 🎯 Interface Gráfica

O projeto oferece uma **interface gráfica Tkinter totalmente visual** para facilitar o uso:

### Interface Tkinter (Design Moderno) ⭐

Interface gráfica desktop moderna com design estado da arte:

```bash
python app.py
```

**Funcionalidades:**
- 🎨 **Design moderno** com interface totalmente gráfica
- 📥 **Download de dataset** com seleção de versão (spinbox)
- 🏋️ **Treinamento completo** com todos os hiperparâmetros editáveis:
  - Épocas, Batch Size, Tamanho da Imagem (dropdown)
  - Learning Rate, Gradient Accumulation
  - Save Steps, Eval Steps
- 🔮 **Predição avançada** com controles gráficos:
  - Score Threshold (slider 0.0-1.0 com valor em tempo real)
  - IOU Threshold (slider 0.0-1.0 com valor em tempo real)
  - Max Detections
- 📊 **Avaliação** de modelos treinados
- ✅ **Status do sistema** em tempo real
- 📝 **Logs coloridos** em tempo real (tema escuro)
- 🎯 **Interface responsiva** e moderna
- 📁 **Seleção de diretórios** com botão de navegação

**Alternativa CLI:**
```bash
python interface.py
```

## 🚀 Início Rápido

### 1. Setup do Ambiente

```bash
chmod +x scripts/bootstrap_mac.sh
./scripts/bootstrap_mac.sh
source .venv/bin/activate
```

### 2. Executar Aplicação

```bash
python app.py
```

Isso iniciará a interface gráfica Tkinter automaticamente.

Ou execute diretamente:
```bash
# Interface Tkinter (Recomendada)
python interface_tkinter.py

# Interface CLI (Linha de Comando)
python interface.py
```

### 4. Configurar Variáveis de Ambiente

### 2. Configurar Variáveis de Ambiente

**⚠️ IMPORTANTE: Proteção de Credenciais**

O projeto usa variáveis de ambiente para proteger sua API key. **NUNCA** commite o arquivo `.env` com credenciais reais!

**Opção 1: Setup Interativo (Recomendado)**
```bash
python scripts/setup_env.py
```

**Opção 2: Manual**
```bash
cp .env.example .env
# Edite .env com suas credenciais do Roboflow
# O arquivo .env está no .gitignore e não será commitado
```

**Verificação de Segurança:**
```bash
# Antes de fazer commit, verifique se não há credenciais expostas:
python scripts/check_security.py

# Opcional: Instalar hook pré-commit automático
./scripts/install_pre_commit_hook.sh
```

### 5. Baixar Dataset

**Usando Interface:**
```bash
python interface.py
# Escolha opção 1: Baixar Dataset do Roboflow
```

**Ou via linha de comando:**
```bash
python scripts/download_roboflow_coco.py
python scripts/sanity_coco.py --dataset_dir dataset
```

> 💡 **Dica**: Se você ainda não fez split de valid/test no Roboflow, adicione as anotações e configure os splits antes de baixar. Depois use a interface para baixar novamente.

### 6. Treinar Modelo

**Usando Interface (Recomendado):**
```bash
python interface.py
# Escolha opção 3: Treinar Modelo
# Configure os parâmetros interativamente
```

**Ou via linha de comando:**
```bash
python src/train_rtdetr.py --dataset_dir dataset --out_dir runs_rtdetr --epochs 50 --img_size 640
```

### 7. Avaliar Modelo

**Usando Interface:**
```bash
python interface.py
# Escolha opção 4: Avaliar Modelo
# Selecione o modelo e o split
```

**Ou via linha de comando:**
```bash
# Valid
python src/eval_coco.py --dataset_dir dataset --model_dir runs_rtdetr/model_best --split valid

# Test
python src/eval_coco.py --dataset_dir dataset --model_dir runs_rtdetr/model_best --split test
```

### 8. Inferência

**Usando Interface:**
```bash
python interface.py
# Escolha opção 5: Fazer Predição/Inferência
# Configure diretórios e threshold
```

**Ou via linha de comando:**
```bash
python src/infer_images.py --model_dir runs_rtdetr/model_best --input_dir dataset/test --out_dir runs_rtdetr/infer_test --score_threshold 0.3
```

## 📁 Estrutura do Projeto

```
ObjectDetection_DETR/
  README.md
  requirements.txt
  .env.example
  app.py                          # Ponto de entrada principal (Tkinter)
  interface_tkinter.py            # Interface gráfica Tkinter
  interface.py                    # Interface CLI (alternativa)
  dataset/                        # gerado pelo download
  runs_rtdetr/                    # outputs (checkpoints, logs, modelos)
  scripts/
    bootstrap_mac.sh
    download_roboflow_coco.py
    sanity_coco.py
  src/
    train_rtdetr.py
    eval_coco.py
    infer_images.py
    infer_video.py
    coco_utils.py
```

## 🔧 Requisitos

- Python 3.10+ (recomendado 3.11)
- MacBook com Apple Silicon (M4 ou superior)
- PyTorch com suporte MPS

## 📊 Métricas de Treinamento

O sistema agora exibe e salva métricas detalhadas durante o treinamento:

### Métricas por Step (durante treinamento)
- **Loss Total**, **Loss CE**, **Loss BBox**, **Loss GIoU**
- **Learning Rate**
- **Tempo por iteração**

### Métricas de Validação (ao final de cada época)
- **mAP@[0.5:0.95]**, **mAP@0.5**, **mAP@0.75**
- **Precision**, **Recall**, **AR (Average Recall)**

### Arquivos de Log Gerados

Os logs são salvos automaticamente em `runs_rtdetr/runs/`:

- **`train_metrics.csv`**: Métricas de treinamento por step
  - Colunas: `epoch`, `step`, `loss_total`, `loss_ce`, `loss_bbox`, `loss_giou`, `lr`, `time_per_iter`
  
- **`val_metrics.csv`**: Métricas de validação por época
  - Colunas: `epoch`, `step`, `loss`, `mAP_0.5_0.95`, `mAP_0.5`, `mAP_0.75`, `precision`, `recall`, `AR`
  
- **`metrics.jsonl`**: Histórico completo em formato JSONL (1 linha por época)

### TensorBoard

O sistema também gera logs para TensorBoard:

```bash
# Instalar TensorBoard (se ainda não instalado)
pip install tensorboard

# Visualizar métricas durante/após treinamento
tensorboard --logdir runs_rtdetr/tb
```

Acesse `http://localhost:6006` no navegador para visualizar gráficos interativos das métricas.

### Exemplo de Saída no Console

Durante o treinamento, você verá:

```
======================================================================
📊 TREINAMENTO - Step 50 | Época 0.1250
======================================================================
  Loss Total:        2.345678 (avg: 2.456789)
  Loss CE:           1.234567 (avg: 1.345678)
  Loss BBox:         0.567890 (avg: 0.678901)
  Loss GIoU:         0.543210 (avg: 0.432109)
  Learning Rate:     0.00001000
  Tempo/Iter:        0.1234s
======================================================================

======================================================================
📊 VALIDAÇÃO - Step 500 | Época 1.2500
======================================================================
  Loss:              2.123456
  mAP@0.5:0.95:      0.3456 (34.56%)
  mAP@0.5:           0.4567 (45.67%)
  mAP@0.75:          0.2345 (23.45%)
  Precision:         0.5678 (56.78%)
  Recall:            0.4321 (43.21%)
  AR (Average Recall): 0.4890 (48.90%)
======================================================================
```

## 🎯 Parâmetros Recomendados (Mac M4)

- `batch_size`: 1 ou 2
- `gradient_accumulation_steps`: 4 a 8
- `learning_rate`: 1e-5
- `img_size`: 640 (ou 832/960 para objetos menores)

## 📝 Notas Técnicas

- Usa `PYTORCH_ENABLE_MPS_FALLBACK=1` para fallback automático quando MPS não suporta operações
- Modelo base: `PekingU/rtdetr_r50vd` do Hugging Face Transformers
- Dataset format: COCO JSON exportado do Roboflow

## 🔒 Segurança

### Proteção de API Keys

Este projeto implementa várias camadas de proteção para suas credenciais:

1. **`.env` no `.gitignore`**: O arquivo `.env` está automaticamente ignorado pelo Git
2. **`.env.example`**: Template sem credenciais reais para referência
3. **Script de verificação**: `scripts/check_security.py` verifica commits antes de enviar
4. **Setup seguro**: `scripts/setup_env.py` cria `.env` com permissões restritivas (600)

### Boas Práticas

- ✅ Use `python scripts/setup_env.py` para configurar credenciais de forma segura
- ✅ Execute `python scripts/check_security.py` antes de fazer commit
- ✅ Nunca commite arquivos com `ROBOFLOW_API_KEY` real
- ✅ Revise sempre o que está sendo commitado: `git status` e `git diff`

### Se você acidentalmente commitou uma API key:

1. **IMEDIATAMENTE** revogue a chave no Roboflow
2. Remova do histórico do Git (se necessário, use `git filter-branch` ou ferramentas similares)
3. Gere uma nova API key
4. Configure novamente usando `scripts/setup_env.py`

Veja `SECURITY.md` para mais detalhes.
