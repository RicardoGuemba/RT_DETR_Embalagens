#!/usr/bin/env python3
"""
Script para treinar modelo RT-DETR.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

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
    Trainer
)
import numpy as np
from PIL import Image

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
    prepare_annotations_for_processor
)


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
        
        Args:
            model: O modelo
            inputs: Dict com pixel_values, pixel_mask (opcional), labels
            return_outputs: Se True, retorna (loss, outputs)
            num_items_in_batch: Número de itens no batch (parâmetro novo do Trainer)
        """
        labels = inputs.pop("labels")
        
        outputs = model(**inputs, labels=labels)
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Avalia o modelo e retorna métricas incluindo precision, recall e mAP.
        """
        # Avaliação padrão (loss)
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Se temos dataset COCO e processor, calcular métricas COCO
        if self.eval_dataset_coco is not None and self.processor is not None:
            try:
                coco_metrics = self._compute_coco_metrics()
                eval_results.update(coco_metrics)
            except Exception as e:
                print(f"⚠️  Erro ao calcular métricas COCO: {e}")
                import traceback
                traceback.print_exc()
        
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
        
        # Carregar ground truth COCO
        coco_gt = COCO(str(self.eval_dataset_coco))
        image_ids = coco_gt.getImgIds()
        
        if len(image_ids) == 0:
            return {}
        
        # Fazer predições
        all_results = []
        dataset_dir = Path(self.eval_dataset_coco).parent
        
        with torch.no_grad():
            for img_id in image_ids:
                img_info = coco_gt.loadImgs(img_id)[0]
                img_path = dataset_dir / img_info["file_name"]
                
                if not img_path.exists():
                    continue
                
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
                
                # Converter para formato COCO
                for box, score, label in zip(
                    results["boxes"].cpu().numpy(),
                    results["scores"].cpu().numpy(),
                    results["labels"].cpu().numpy()
                ):
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "score": float(score)
                    })
        
        if len(all_results) == 0:
            return {
                "eval_mAP": 0.0,
                "eval_AP50": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0
            }
        
        # Salvar resultados temporários
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(all_results, f)
            results_path = f.name
        
        try:
            # Carregar resultados no COCO
            coco_dt = coco_gt.loadRes(results_path)
            
            # Rodar COCOeval
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            
            # Extrair métricas
            # stats: [AP@[0.50:0.95], AP50, AP75, AP_small, AP_medium, AP_large, AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
            stats = coco_eval.stats
            
            mAP = float(stats[0])  # AP@[0.50:0.95]
            AP50 = float(stats[1])  # AP@0.50
            
            # Calcular precision e recall @0.5
            # Para IoU=0.5, usar as métricas do COCOeval
            # Precision média @0.5 (AP50 já é uma medida de precision)
            precision_50 = AP50
            
            # Recall @0.5 (AR@100 para IoU=0.5)
            # COCOeval não fornece recall direto, mas podemos calcular
            # Usar AR (Average Recall) como aproximação
            recall_50 = float(stats[8]) if len(stats) > 8 else AP50  # AR@100
            
            # Métricas adicionais
            AP75 = float(stats[2]) if len(stats) > 2 else 0.0
            
            metrics = {
                "eval_mAP": mAP,
                "eval_AP50": AP50,
                "eval_AP75": AP75,
                "eval_precision": precision_50,
                "eval_recall": recall_50,
            }
            
            # Log formatado similar ao YOLO
            print(f"\n{'='*60}")
            print(f"📊 MÉTRICAS DE VALIDAÇÃO (Época {self.state.epoch:.1f})")
            print(f"{'='*60}")
            print(f"  mAP@0.5:0.95: {mAP:.4f}")
            print(f"  mAP@0.5:     {AP50:.4f}")
            print(f"  mAP@0.75:    {AP75:.4f}")
            print(f"  Precision:   {precision_50:.4f}")
            print(f"  Recall:      {recall_50:.4f}")
            print(f"{'='*60}\n")
            
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
        "report_to": "none",  # Desabilitar wandb/tensorboard por padrão
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
    
    # Treinar
    print("🚀 Iniciando treinamento...")
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

