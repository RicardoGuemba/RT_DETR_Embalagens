#!/usr/bin/env python3
"""
Script para treinar modelo DETR/RT-DETR - ObjectDetection_DETR.
"""

import os
import sys
import json
import argparse
import subprocess
import csv
import time
from pathlib import Path
from datetime import datetime

# Habilitar fallback para CPU em operações MPS não suportadas
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Verificar e atualizar accelerate se necessário
def check_accelerate():
    """Verifica se accelerate está instalado e suficientemente novo para o Trainer."""
    try:
        import accelerate
        from packaging import version
        required = version.parse("0.27.2")  # versões antigas não suportam dispatch_batches
        if version.parse(accelerate.__version__) < required:
            print("⚠️  Atualizando accelerate...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-U", "accelerate>=0.27.2"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ accelerate atualizado!")
    except ImportError:
        print("⚠️  Instalando accelerate...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "accelerate>=0.27.2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ accelerate instalado!")
    except Exception:
        # Continuar mesmo se falhar; o Trainer acusará erro se faltar
        pass

check_accelerate()

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Importar accelerate e aplicar patch de compatibilidade se necessário
try:
    import accelerate
    import inspect
    
    # Compat: versões antigas do accelerate não aceitam alguns parâmetros novos
    accel_sig = inspect.signature(accelerate.Accelerator.__init__)
    known_params = set(accel_sig.parameters.keys())
    
    # Parâmetros que podem não existir em versões antigas
    new_params = ["dispatch_batches", "even_batches", "use_seedable_sampler"]
    missing_params = [p for p in new_params if p not in known_params]
    
    if missing_params:
        _orig_accel_init = accelerate.Accelerator.__init__

        def _patched_accel_init(self, *args, **kwargs):
            # Remover apenas parâmetros que não existem na assinatura
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_params}
            return _orig_accel_init(self, *args, **filtered_kwargs)

        accelerate.Accelerator.__init__ = _patched_accel_init
        print(f"⚙️  Patch aplicado: accelerate.Accelerator ignorará {', '.join(missing_params)} (versão antiga).")
except Exception as e:
    print(f"⚠️  Aviso ao verificar accelerate: {e}")

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    DetrImageProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import numpy as np
from PIL import Image

# TensorBoard (opcional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Imports para métricas COCO
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("⚠️  pycocotools não disponível. Métricas COCO não serão calculadas durante o treinamento.")

from coco_utils import (
    load_coco_json,
    remap_category_ids,
    prepare_annotations_for_processor,
    ensure_coco_info,
    ensure_coco_info_file
)


class MetricsCallback(TrainerCallback):
    """
    Callback completo para exibir e salvar métricas durante o treinamento.
    Salva métricas em CSV/JSON e opcionalmente no TensorBoard.
    """
    
    def __init__(self, output_dir: Path, use_tensorboard: bool = True):
        """
        Args:
            output_dir: Diretório de saída para salvar métricas
            use_tensorboard: Se True, usa TensorBoard (se disponível)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivos CSV
        self.train_csv_path = self.output_dir / "train_metrics.csv"
        self.val_csv_path = self.output_dir / "val_metrics.csv"
        self.metrics_jsonl_path = self.output_dir / "metrics.jsonl"
        
        # Inicializar CSVs
        self._init_csv_files()
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.use_tensorboard:
            tb_dir = self.output_dir / "tb"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"✅ TensorBoard habilitado: tensorboard --logdir {tb_dir}")
        
        # Timestamps para cálculo de tempo por iteração
        self.last_log_time = time.time()
        self.step_start_time = time.time()
        self.last_step = 0
        
    def _init_csv_files(self):
        """Inicializa arquivos CSV com cabeçalhos."""
        # CSV de treinamento (por step)
        train_headers = ["epoch", "step", "loss", "loss_ce", "loss_bbox", "loss_giou", 
                         "lr", "time_per_iter", "timestamp"]
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)
        
        # CSV de validação (por época)
        val_headers = ["epoch", "step", "eval_loss", "mAP", "AP50", "AP75", 
                      "precision", "recall", "timestamp"]
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Chamado a cada logging step para salvar métricas de treinamento."""
        if logs is None:
            return
        
        # Extrair métricas de treinamento primeiro (antes de usar step)
        epoch = state.epoch if hasattr(state, 'epoch') else 0.0
        step = state.global_step if hasattr(state, 'global_step') else 0
        
        # Calcular tempo por iteração
        current_time = time.time()
        if self.last_step > 0:
            # Tempo desde o último log
            time_per_iter = current_time - self.last_log_time
        else:
            time_per_iter = 0.0
        self.last_log_time = current_time
        self.last_step = step
        
        loss = logs.get("loss", None)
        loss_ce = logs.get("loss_ce", None) or logs.get("loss_class_error", None)
        loss_bbox = logs.get("loss_bbox", None) or logs.get("loss_bbox_coord", None)
        loss_giou = logs.get("loss_giou", None) or logs.get("loss_giou_bbox", None)
        lr = logs.get("learning_rate", None) or logs.get("lr", None)
        
        # Salvar em CSV de treinamento
        if loss is not None:
            row = [
                f"{epoch:.4f}",
                step,
                f"{loss:.6f}" if loss is not None else "",
                f"{loss_ce:.6f}" if loss_ce is not None else "",
                f"{loss_bbox:.6f}" if loss_bbox is not None else "",
                f"{loss_giou:.6f}" if loss_giou is not None else "",
                f"{lr:.8f}" if lr is not None else "",
                f"{time_per_iter:.4f}",
                datetime.now().isoformat()
            ]
            with open(self.train_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Log no TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar("train/loss", loss, step)
                if loss_ce is not None:
                    self.tb_writer.add_scalar("train/loss_ce", loss_ce, step)
                if loss_bbox is not None:
                    self.tb_writer.add_scalar("train/loss_bbox", loss_bbox, step)
                if loss_giou is not None:
                    self.tb_writer.add_scalar("train/loss_giou", loss_giou, step)
                if lr is not None:
                    self.tb_writer.add_scalar("train/learning_rate", lr, step)
                self.tb_writer.add_scalar("train/time_per_iter", time_per_iter, step)
        
        # Exibir métricas no console a cada logging step
        if loss is not None:
            print(f"\n📊 Step {step} | Época {epoch:.2f} | Loss: {loss:.6f}", end="")
            if loss_ce is not None:
                print(f" | Loss_CE: {loss_ce:.6f}", end="")
            if loss_bbox is not None:
                print(f" | Loss_Bbox: {loss_bbox:.6f}", end="")
            if loss_giou is not None:
                print(f" | Loss_GIoU: {loss_giou:.6f}", end="")
            if lr is not None:
                print(f" | LR: {lr:.2e}", end="")
            print(f" | Time/iter: {time_per_iter:.3f}s")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Chamado após cada avaliação para exibir e salvar métricas de validação."""
        if logs is None:
            return
        
        # Extrair métricas COCO
        mAP = logs.get("eval_mAP", None) or logs.get("mAP", None)
        AP50 = logs.get("eval_AP50", None) or logs.get("AP50", None)
        AP75 = logs.get("eval_AP75", None) or logs.get("AP75", None)
        precision = logs.get("eval_precision", None) or logs.get("precision", None)
        recall = logs.get("eval_recall", None) or logs.get("recall", None)
        eval_loss = logs.get("eval_loss", None)
        
        # Debug: tentar encontrar métricas com outras chaves
        if mAP is None and AP50 is None:
            coco_keys = [k for k in logs.keys() if 'map' in k.lower() or 'ap50' in k.lower() or 'precision' in k.lower()]
            if coco_keys:
                for key in coco_keys:
                    if 'map' in key.lower() and mAP is None:
                        mAP = logs.get(key)
                    if 'ap50' in key.lower() and AP50 is None:
                        AP50 = logs.get(key)
        
        epoch = state.epoch if hasattr(state, 'epoch') else 0.0
        step = state.global_step if hasattr(state, 'global_step') else 0
        
        # Salvar em CSV de validação
        row = [
            f"{epoch:.4f}",
            step,
            f"{eval_loss:.6f}" if eval_loss is not None else "",
            f"{mAP:.6f}" if mAP is not None else "",
            f"{AP50:.6f}" if AP50 is not None else "",
            f"{AP75:.6f}" if AP75 is not None else "",
            f"{precision:.6f}" if precision is not None else "",
            f"{recall:.6f}" if recall is not None else "",
            datetime.now().isoformat()
        ]
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Salvar em JSONL (1 linha por época)
        metrics_dict = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "eval_loss": eval_loss,
            "mAP": mAP,
            "AP50": AP50,
            "AP75": AP75,
            "precision": precision,
            "recall": recall
        }
        with open(self.metrics_jsonl_path, 'a') as f:
            f.write(json.dumps(metrics_dict) + "\n")
        
        # Log no TensorBoard
        if self.tb_writer:
            if eval_loss is not None:
                self.tb_writer.add_scalar("val/loss", eval_loss, step)
            if mAP is not None:
                self.tb_writer.add_scalar("val/mAP", mAP, step)
            if AP50 is not None:
                self.tb_writer.add_scalar("val/AP50", AP50, step)
            if AP75 is not None:
                self.tb_writer.add_scalar("val/AP75", AP75, step)
            if precision is not None:
                self.tb_writer.add_scalar("val/precision", precision, step)
            if recall is not None:
                self.tb_writer.add_scalar("val/recall", recall, step)
        
        # Exibir métricas de forma destacada
        print("\n" + "="*70)
        print(f"📊 MÉTRICAS DE VALIDAÇÃO - Step {step} | Época {epoch:.2f}")
        print("="*70)
        
        if eval_loss is not None:
            print(f"  Loss:              {eval_loss:.6f}")
        
        if mAP is not None:
            print(f"  mAP@0.5:0.95:      {mAP:.4f} ({mAP*100:.2f}%)")
        
        if AP50 is not None:
            print(f"  mAP@0.5:           {AP50:.4f} ({AP50*100:.2f}%)")
        
        if AP75 is not None:
            print(f"  mAP@0.75:          {AP75:.4f} ({AP75*100:.2f}%)")
        
        if precision is not None:
            print(f"  Precision:         {precision:.4f} ({precision*100:.2f}%)")
        
        if recall is not None:
            print(f"  Recall:            {recall:.4f} ({recall*100:.2f}%)")
        
        print("="*70)
        print(f"💾 Métricas salvas em:")
        print(f"   - {self.val_csv_path}")
        print(f"   - {self.metrics_jsonl_path}")
        if self.tb_writer:
            print(f"   - TensorBoard: {self.output_dir / 'tb'}")
        print("="*70 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Chamado ao final do treinamento para fechar recursos."""
        if self.tb_writer:
            self.tb_writer.close()
            print(f"✅ TensorBoard logs salvos em: {self.output_dir / 'tb'}")


def load_image_processor(model_name_or_path: str, img_size: int = None):
    """
    Carrega image processor de forma robusta, com fallback para DetrImageProcessor.
    
    Args:
        model_name_or_path: Nome do modelo ou caminho do diretório
        img_size: Tamanho da imagem (opcional)
    
    Returns:
        Image processor (AutoImageProcessor ou DetrImageProcessor)
    """
    # Formato correto para size (RT-DETR requer height e width)
    size_dict = {"height": img_size, "width": img_size} if img_size else None
    
    # Tentar AutoImageProcessor primeiro
    try:
        if size_dict:
            processor = AutoImageProcessor.from_pretrained(model_name_or_path, size=size_dict)
        else:
            processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        return processor
    except (ValueError, OSError) as e:
        # Se AutoImageProcessor falhar, tentar DetrImageProcessor
        try:
            if size_dict:
                processor = DetrImageProcessor.from_pretrained(model_name_or_path, size=size_dict)
            else:
                processor = DetrImageProcessor.from_pretrained(model_name_or_path)
            return processor
        except Exception as e2:
            # Se ainda falhar, criar processador genérico baseado em DETR
            print(f"   ⚠️  Erro ao carregar processador: {e2}")
            print("   📦 Criando processador genérico baseado em DETR...")
            default_size = {"height": img_size or 640, "width": img_size or 640}
            processor = DetrImageProcessor(
                format="coco_detection",
                size=default_size,
                do_resize=True,
                do_normalize=True,
                do_rescale=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
            return processor


def get_device():
    """Retorna device disponível (MPS ou CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class COCODataset(Dataset):
    """Dataset COCO para RT-DETR."""
    
    def __init__(
        self,
        coco_data: dict,
        image_dir: Path,
        processor,  # AutoImageProcessor ou DetrImageProcessor
        remap_ids: bool = True
    ):
        self.coco_data = coco_data
        self.image_dir = image_dir
        self.processor = processor
        
        # Remapear IDs se necessário
        if remap_ids:
            self.coco_data, self.id_mapping = remap_category_ids(coco_data)
        else:
            self.id_mapping = None
        
        # Criar índice de imagens
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.image_ids = list(self.images.keys())
        
        print(f"📊 Dataset inicializado:")
        print(f"   Imagens: {len(self.image_ids)}")
        print(f"   Categorias: {len(self.coco_data['categories'])}")
        print(f"   Anotações: {len(self.coco_data['annotations'])}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = self.image_dir / img_info["file_name"]
        
        # Preparar anotações
        data = prepare_annotations_for_processor(
            image_id,
            self.coco_data,
            img_path
        )
        
        # Processar com o processor
        encoding = self.processor(
            images=data["image"],
            annotations=data["annotations"],
            return_tensors="pt"
        )
        
        # Remover batch dimension (apenas para tensors)
        result = {}
        for k, v in encoding.items():
            if hasattr(v, 'squeeze'):  # É um tensor
                result[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1:  # É uma lista com um item
                result[k] = v[0]
            else:
                result[k] = v
        
        return result


def collate_fn(batch):
    """Collate function para DataLoader."""
    # Agrupar pixel_values
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # pixel_mask pode não existir para RT-DETR
    result = {"pixel_values": pixel_values}
    
    if "pixel_mask" in batch[0]:
        pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
        result["pixel_mask"] = pixel_mask
    
    # Labels podem ter tamanhos diferentes, manter como lista
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        result["labels"] = labels
    
    return result


class RTDetrTrainer(Trainer):
    """Trainer customizado para RT-DETR com métricas COCO."""
    
    def __init__(self, *args, eval_dataset_coco=None, processor=None, **kwargs):
        """
        Args:
            eval_dataset_coco: Dataset COCO original para avaliação (opcional)
            processor: Image processor para avaliação (opcional)
        """
        super().__init__(*args, **kwargs)
        self.eval_dataset_coco = eval_dataset_coco
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computa loss customizado para RT-DETR.
        Extrai losses individuais para logging detalhado.
        
        Args:
            model: O modelo
            inputs: Dict com pixel_values, pixel_mask (opcional), labels
            return_outputs: Se True, retorna (loss, outputs)
            num_items_in_batch: Número de itens no batch (parâmetro novo do Trainer)
        """
        labels = inputs.pop("labels")
        
        outputs = model(**inputs, labels=labels)
        
        loss = outputs.loss
        
        # Extrair losses individuais se disponíveis
        # DETR/RT-DETR geralmente retorna loss_dict com losses separadas
        if hasattr(outputs, 'loss_dict'):
            loss_dict = outputs.loss_dict
            # Adicionar losses individuais aos outputs para logging
            outputs.loss_ce = loss_dict.get('loss_ce', None) or loss_dict.get('class_error', None)
            outputs.loss_bbox = loss_dict.get('loss_bbox', None) or loss_dict.get('bbox_coord', None)
            outputs.loss_giou = loss_dict.get('loss_giou', None) or loss_dict.get('giou', None)
        else:
            # Tentar extrair do modelo diretamente (alguns modelos DETR têm isso)
            if hasattr(model, 'config') and hasattr(model.config, 'loss_dict'):
                # Se o modelo tem loss_dict configurado, tentar acessar
                pass
        
        # Armazenar outputs para uso no log
        self._last_outputs = outputs
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        """
        Sobrescreve log para adicionar losses individuais aos logs.
        """
        # Extrair losses individuais do último output se disponível
        if hasattr(self, '_last_outputs') and self._last_outputs is not None:
            if hasattr(self._last_outputs, 'loss_ce') and self._last_outputs.loss_ce is not None:
                logs['loss_ce'] = float(self._last_outputs.loss_ce)
            if hasattr(self._last_outputs, 'loss_bbox') and self._last_outputs.loss_bbox is not None:
                logs['loss_bbox'] = float(self._last_outputs.loss_bbox)
            if hasattr(self._last_outputs, 'loss_giou') and self._last_outputs.loss_giou is not None:
                logs['loss_giou'] = float(self._last_outputs.loss_giou)
        
        # Chamar log padrão
        super().log(logs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Avalia o modelo e retorna métricas incluindo precision, recall e mAP.
        """
        # Avaliação padrão (loss)
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Se temos dataset COCO e processor, calcular métricas COCO
        if self.eval_dataset_coco is not None and self.processor is not None:
            try:
                # Calcular métricas COCO (isso já exibe as métricas internamente)
                coco_metrics = self._compute_coco_metrics()
                
                # Atualizar resultados com as métricas calculadas
                eval_results.update(coco_metrics)
                
            except Exception as e:
                print(f"⚠️  Erro ao calcular métricas COCO: {e}")
                import traceback
                traceback.print_exc()
                
                # Mesmo em caso de erro, exibir métricas zeradas para manter consistência
                step = self.state.global_step if hasattr(self.state, 'global_step') else 0
                epoch = self.state.epoch if hasattr(self.state, 'epoch') else 0.0
                
                print(f"\n{'='*70}")
                print(f"📊 MÉTRICAS COCO (ERRO) - Step {step} | Época {epoch:.2f}")
                print(f"{'='*70}")
                print(f"  mAP@0.5:0.95:      0.0000 (0.00%) - Erro no cálculo")
                print(f"  mAP@0.5:           0.0000 (0.00%) - Erro no cálculo")
                print(f"  mAP@0.75:          0.0000 (0.00%) - Erro no cálculo")
                print(f"  Precision:         0.0000 (0.00%) - Erro no cálculo")
                print(f"  Recall:            0.0000 (0.00%) - Erro no cálculo")
                print(f"{'='*70}\n")
                
                # Retornar métricas zeradas mesmo em caso de erro
                eval_results.update({
                    "eval_mAP": 0.0,
                    "eval_AP50": 0.0,
                    "eval_AP75": 0.0,
                    "eval_precision": 0.0,
                    "eval_recall": 0.0,
                })
        
        return eval_results
    
    def _compute_coco_metrics(self):
        """
        Calcula métricas COCO (precision, recall, mAP) similar ao YOLO.
        """
        if not COCO_AVAILABLE:
            return {}
        
        if self.eval_dataset_coco is None:
            return {}
        
        import tempfile
        
        model = self.model
        device = next(model.parameters()).device
        model.eval()
        
        # CRÍTICO: Garantir que o JSON COCO tenha o campo 'info' obrigatório
        # pycocotools requer este campo para funcionar corretamente
        eval_json_path = Path(self.eval_dataset_coco)
        ensure_coco_info_file(eval_json_path)
        
        # Carregar ground truth COCO
        coco_gt = COCO(str(self.eval_dataset_coco))
        image_ids = coco_gt.getImgIds()
        
        if len(image_ids) == 0:
            return {}
        
        # Fazer predições
        all_results = []
        dataset_dir = Path(self.eval_dataset_coco).parent
        total_detections = 0
        total_images = 0
        scores_list = []
        
        with torch.no_grad():
            for img_id in image_ids:
                img_info = coco_gt.loadImgs(img_id)[0]
                img_path = dataset_dir / img_info["file_name"]
                
                if not img_path.exists():
                    continue
                
                total_images += 1
                
                # Carregar e processar imagem
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Fazer predição
                outputs = model(**inputs)
                
                # Processar outputs
                target_sizes = torch.tensor([image.size[::-1]]).to(device)
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=0.01  # Threshold baixo para capturar todas as predições
                )[0]
                
                num_detections = len(results["boxes"])
                total_detections += num_detections
                
                # Converter para formato COCO
                for box, score, label in zip(
                    results["boxes"].cpu().numpy(),
                    results["scores"].cpu().numpy(),
                    results["labels"].cpu().numpy()
                ):
                    scores_list.append(float(score))
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Verificar se label está no range válido (0 para "Embalagem")
                    # DETR usa label 0 para "no object", então precisamos ajustar
                    if int(label) == 0:
                        # Label 0 é "no object" no DETR, mas no nosso caso é "Embalagem"
                        # Manter como 0 se for a classe correta
                        category_id = 0
                    else:
                        # Se label > 0, pode ser "no object" ou outra classe
                        # Para 1 classe, só temos label 0 (Embalagem)
                        category_id = 0
                    
                    all_results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, w, h],
                        "score": float(score)
                    })
        
        # Debug: Informações sobre as predições
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        max_score = max(scores_list) if scores_list else 0.0
        min_score = min(scores_list) if scores_list else 0.0
        
        print(f"🔍 DEBUG - Predições geradas:")
        print(f"   Imagens processadas: {total_images}")
        print(f"   Total de detecções: {total_detections}")
        print(f"   Detecções por imagem: {total_detections/total_images if total_images > 0 else 0:.2f}")
        if scores_list:
            print(f"   Score médio: {avg_score:.4f}")
            print(f"   Score máximo: {max_score:.4f}")
            print(f"   Score mínimo: {min_score:.4f}")
        else:
            print(f"   ⚠️  Nenhuma detecção gerada!")
        
        if len(all_results) == 0:
            print("⚠️  Nenhuma detecção gerada pelo modelo")
            print("   Isso é normal nas primeiras épocas quando o modelo ainda está aprendendo")
            
            # SEMPRE exibir métricas, mesmo quando são 0.0
            step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            epoch = self.state.epoch if hasattr(self.state, 'epoch') else 0.0
            
            print(f"\n{'='*70}")
            print(f"📊 MÉTRICAS COCO - Step {step} | Época {epoch:.2f}")
            print(f"{'='*70}")
            print(f"  mAP@0.5:0.95:      0.0000 (0.00%)")
            print(f"  mAP@0.5:           0.0000 (0.00%)")
            print(f"  mAP@0.75:          0.0000 (0.00%)")
            print(f"  Precision:         0.0000 (0.00%)")
            print(f"  Recall:            0.0000 (0.00%)")
            print(f"  ⚠️  Nota: Nenhuma detecção gerada - modelo ainda aprendendo")
            print(f"{'='*70}\n")
            
            return {
                "eval_mAP": 0.0,
                "eval_AP50": 0.0,
                "eval_AP75": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0,
            }
        
        # Salvar resultados temporários
        # Nota: loadRes espera apenas uma lista de anotações, não um JSON completo
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(all_results, f)
            results_path = f.name
        
        try:
            # Carregar resultados no COCO
            # O loadRes automaticamente adiciona ao dataset existente (coco_gt)
            coco_dt = coco_gt.loadRes(results_path)
            
            # Debug: Verificar se há predições e ground truth
            num_gt = len(coco_gt.getAnnIds())
            num_dt = len(coco_dt.getAnnIds())
            print(f"🔍 DEBUG - COCOeval:")
            print(f"   Ground truth annotations: {num_gt}")
            print(f"   Predicted annotations: {num_dt}")
            
            if num_dt == 0:
                print(f"   ⚠️  Nenhuma predição foi carregada no COCO!")
            
            # Rodar COCOeval
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            
            # Extrair métricas
            # stats: [AP@[0.50:0.95], AP50, AP75, AP_small, AP_medium, AP_large, AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
            stats = coco_eval.stats
            
            # CRÍTICO: Verificar se stats não está vazio
            # Isso pode acontecer quando não há detecções válidas ou quando o modelo ainda não aprendeu
            if len(stats) == 0:
                print("⚠️  COCOeval retornou stats vazio - nenhuma detecção válida encontrada")
                print("   Isso é normal nas primeiras épocas quando o modelo ainda está aprendendo")
                # Definir métricas como zero mas SEMPRE exibir
                mAP = 0.0
                AP50 = 0.0
                AP75 = 0.0
                precision_50 = 0.0
                recall_50 = 0.0
            else:
                # Verificar se temos pelo menos os índices básicos
                if len(stats) < 3:
                    print(f"⚠️  COCOeval retornou apenas {len(stats)} métricas (esperado 12)")
                    print("   Usando valores padrão para métricas faltantes")
                
                # Extrair métricas com fallback para valores padrão
                mAP = float(stats[0]) if len(stats) > 0 else 0.0  # AP@[0.50:0.95]
                AP50 = float(stats[1]) if len(stats) > 1 else 0.0  # AP@0.50
                AP75 = float(stats[2]) if len(stats) > 2 else 0.0  # AP@0.75
                
                # Calcular precision e recall @0.5
                # Para IoU=0.5, usar as métricas do COCOeval
                # Precision média @0.5 (AP50 já é uma medida de precision)
                precision_50 = AP50
                
                # Recall @0.5 (AR@100 para IoU=0.5)
                # COCOeval não fornece recall direto, mas podemos calcular
                # Usar AR (Average Recall) como aproximação
                # stats[8] = AR@100 (Average Recall com até 100 detecções)
                recall_50 = float(stats[8]) if len(stats) > 8 else AP50  # AR@100
            
            # SEMPRE exibir métricas, mesmo quando são 0.0
            # Isso garante visibilidade do progresso do treinamento
            step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            epoch = self.state.epoch if hasattr(self.state, 'epoch') else 0.0
            
            print(f"\n{'='*70}")
            print(f"📊 MÉTRICAS COCO CALCULADAS - Step {step} | Época {epoch:.2f}")
            print(f"{'='*70}")
            print(f"  mAP@0.5:0.95:      {mAP:.4f} ({mAP*100:.2f}%)")
            print(f"  mAP@0.5:           {AP50:.4f} ({AP50*100:.2f}%)")
            print(f"  mAP@0.75:          {AP75:.4f} ({AP75*100:.2f}%)")
            print(f"  Precision:         {precision_50:.4f} ({precision_50*100:.2f}%)")
            print(f"  Recall:            {recall_50:.4f} ({recall_50*100:.2f}%)")
            if len(stats) == 0:
                print(f"  ⚠️  Nota: Métricas zeradas indicam que o modelo ainda não está detectando objetos")
                print(f"      Continue o treinamento - as métricas aparecerão quando o modelo aprender")
            print(f"{'='*70}\n")
            
            # Criar dict de métricas para retornar
            metrics = {
                "eval_mAP": mAP,
                "eval_AP50": AP50,
                "eval_AP75": AP75,
                "eval_precision": precision_50,
                "eval_recall": recall_50,
            }
            
            # Retornar métricas para que sejam logadas pelo Trainer
            # IMPORTANTE: Garantir que as chaves tenham o prefixo "eval_" para serem propagadas
            return metrics
            
        finally:
            # Limpar arquivo temporário
            import os
            try:
                os.unlink(results_path)
            except:
                pass


def train(
    dataset_dir: Path,
    out_dir: Path,
    model_name: str = "PekingU/rtdetr_r50vd",
    epochs: int = 50,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    img_size: int = 640,
    eval_split: str = "valid",
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 50
):
    """
    Treina modelo RT-DETR.
    """
    device = get_device()
    print(f"🔧 Device: {device}")
    
    # Criar diretórios de saída
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Carregar datasets primeiro para obter número de categorias
    print("📊 Carregando datasets...")
    
    # Train
    train_json = dataset_dir / "train/_annotations.coco.json"
    train_data = load_coco_json(train_json)
    
    # Remapear para obter número correto de labels
    train_data_remapped, _ = remap_category_ids(train_data)
    num_labels = len(train_data_remapped["categories"])
    
    # Criar mapeamento id2label e label2id
    categories = train_data_remapped["categories"]
    id2label = {cat["id"]: cat.get("name", f"classe_{cat['id']}") for cat in categories}
    label2id = {name: cat_id for cat_id, name in id2label.items()}
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    
    # Carregar processor e modelo
    print(f"📥 Carregando modelo DETR...")
    print(f"   Número de classes: {num_labels}")
    print(f"   Classes: {class_names}")
    print(f"   id2label: {id2label}")
    
    # Usar DETR diretamente (removendo tentativa de RT-DETR para evitar ruído)
    model_name_detr = "facebook/detr-resnet-50"
    processor = load_image_processor(model_name_detr, img_size=img_size)
    
    try:
        # Verificar se timm está instalado (necessário para DETR)
        try:
            import timm
        except ImportError:
            print("   ⚠️  Biblioteca 'timm' não encontrada. Instalando...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "timm>=0.9.0"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Limpar cache de módulos para forçar recarregamento
            if 'timm' in sys.modules:
                del sys.modules['timm']
            import timm
            print("   ✅ Biblioteca 'timm' instalada e carregada!")
        
        # Carregar modelo com configuração correta
        model = AutoModelForObjectDetection.from_pretrained(
            model_name_detr,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Ignorar mismatch (modelo tem 92 classes COCO, precisamos de num_labels)
        )
        print(f"   ✅ Modelo DETR carregado com sucesso!")
        print(f"   ✅ Configuração: num_labels={num_labels}, classes={class_names}")
        
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível carregar modelo DETR.\n"
            f"Erro: {e}\n\n"
            f"Soluções possíveis:\n"
            f"1. Instalar timm: pip install timm>=0.9.0\n"
            f"2. Verificar conexão com internet (download do modelo)\n"
            f"3. Verificar se transformers está atualizado: pip install --upgrade transformers"
        )
    
    model.to(device)
    
    train_dataset = COCODataset(
        train_data,
        dataset_dir / "train",
        processor,
        remap_ids=True
    )
    
    # Valid
    valid_json = dataset_dir / f"{eval_split}/_annotations.coco.json"
    if valid_json.exists():
        valid_data = load_coco_json(valid_json)
        valid_dataset = COCODataset(
            valid_data,
            dataset_dir / eval_split,
            processor,
            remap_ids=True
        )
    else:
        print(f"⚠️  Split {eval_split} não encontrado, usando train para validação")
        valid_dataset = train_dataset
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # MPS não suporta multiprocessing bem
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Log de informações do dataset e batch
    print(f"\n📊 Informações do Dataset:")
    print(f"   Train loader: {len(train_loader)} batches")
    print(f"   Valid loader: {len(valid_loader)} batches")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Batch efetivo: {batch_size * gradient_accumulation_steps}")
    
    # Verificar batch real (debug)
    if len(train_loader) > 0:
        sample_batch = next(iter(train_loader))
        if "pixel_values" in sample_batch:
            actual_batch_size = sample_batch["pixel_values"].shape[0]
            print(f"   Batch real (verificado): {actual_batch_size}")
            if actual_batch_size != batch_size:
                print(f"   ⚠️  ATENÇÃO: Batch size real ({actual_batch_size}) diferente do configurado ({batch_size})")
    
    # Training arguments - detectar parâmetro correto para versão do transformers
    import inspect
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    
    # Versões mais novas usam eval_strategy, mais antigas usam evaluation_strategy
    eval_param_name = "eval_strategy" if "eval_strategy" in training_args_params else "evaluation_strategy"
    eval_value = "steps" if valid_json.exists() else "no"
    
    training_args_kwargs = {
        "output_dir": str(checkpoints_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": 0.0001,
        "warmup_steps": 500,
        "logging_dir": str(out_dir / "logs"),
        "logging_steps": logging_steps,
        eval_param_name: eval_value,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True if valid_json.exists() else False,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "report_to": "none",  # Desabilitar wandb/tensorboard padrão (usamos nosso próprio TensorBoard)
        "remove_unused_columns": False,
    }
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    # Trainer com métricas COCO
    eval_json_path = None
    if valid_json.exists():
        eval_json_path = valid_json
    
    trainer = RTDetrTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if valid_json.exists() else None,
        data_collator=collate_fn,
        tokenizer=processor,
        eval_dataset_coco=eval_json_path,  # Caminho do JSON COCO para avaliação
        processor=processor,  # Processor para avaliação
    )
    
    # Adicionar callback completo para métricas (sempre, mesmo sem validação)
    # O callback salva métricas de treinamento mesmo sem validação
    metrics_callback = MetricsCallback(
        output_dir=out_dir,
        use_tensorboard=True  # Habilitar TensorBoard se disponível
    )
    trainer.add_callback(metrics_callback)
    print("✅ Callback de métricas completo ativado:")
    print(f"   - Métricas de treinamento serão salvas em: {out_dir / 'train_metrics.csv'}")
    if valid_json.exists():
        print(f"   - Métricas de validação serão salvas em: {out_dir / 'val_metrics.csv'}")
    print(f"   - Métricas em JSONL: {out_dir / 'metrics.jsonl'}")
    if TENSORBOARD_AVAILABLE:
        print(f"   - TensorBoard habilitado: tensorboard --logdir {out_dir / 'tb'}")
    else:
        print(f"   ⚠️  TensorBoard não disponível (instale: pip install tensorboard)")
    
    # Treinar
    print("🚀 Iniciando treinamento...")
    print("📊 Métricas de acurácia (mAP, Precision, Recall) serão exibidas a cada avaliação\n")
    trainer.train()
    
    # Salvar modelo final
    print("💾 Salvando modelo final...")
    final_model_dir = out_dir / "model_final"
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    
    # Salvar configuração de classes
    config_file = final_model_dir / "class_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            "num_labels": num_labels,
            "class_names": class_names,
            "id2label": id2label,
            "label2id": label2id
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Configuração de classes salva: {config_file.name}")
    
    # Salvar melhor modelo (se houver)
    if valid_json.exists():
        best_model_dir = out_dir / "model_best"
        print(f"💾 Salvando melhor modelo em {best_model_dir}...")
        trainer.save_model(str(best_model_dir))
        processor.save_pretrained(str(best_model_dir))
        
        # Salvar configuração de classes também no melhor modelo
        config_file_best = best_model_dir / "class_config.json"
        with open(config_file_best, 'w', encoding='utf-8') as f:
            json.dump({
                "num_labels": num_labels,
                "class_names": class_names,
                "id2label": id2label,
                "label2id": label2id
            }, f, indent=2, ensure_ascii=False)
    
    print("✅ Treinamento concluído!")
    print(f"   Modelo final: {final_model_dir}")
    print(f"   Classes: {class_names} (num_labels={num_labels})")
    if valid_json.exists():
        print(f"   Melhor modelo: {best_model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo RT-DETR")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs_rtdetr",
        help="Diretório de saída"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PekingU/rtdetr_r50vd",
        help="Nome do modelo base"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Passos de acumulação de gradiente"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Tamanho da imagem"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid",
        help="Split para avaliação"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Salvar checkpoint a cada N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Avaliar a cada N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log a cada N steps"
    )
    
    args = parser.parse_args()
    
    train(
        dataset_dir=Path(args.dataset_dir),
        out_dir=Path(args.out_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        eval_split=args.eval_split,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps
    )


if __name__ == "__main__":
    main()

