#!/usr/bin/env python3
"""
Script para avaliar modelo RT-DETR usando COCOeval.
"""

import json
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrImageProcessor
from PIL import Image

from coco_utils import load_coco_json, create_coco_results


def get_device():
    """Retorna device disponível (MPS ou CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(
    model_dir: Path,
    dataset_dir: Path,
    split: str = "valid",
    batch_size: int = 1,
    score_threshold: float = 0.01
):
    """
    Avalia modelo RT-DETR no split especificado.
    
    Args:
        model_dir: Diretório contendo o modelo
        dataset_dir: Diretório do dataset
        split: Split a avaliar (train/valid/test)
        batch_size: Tamanho do batch
        score_threshold: Threshold mínimo de score para predições
    """
    device = get_device()
    print(f"🔧 Device: {device}")
    
    # Carregar modelo
    print(f"📥 Carregando modelo de {model_dir}...")
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    
    # Carregar processor com fallback
    try:
        processor = AutoImageProcessor.from_pretrained(str(model_dir))
    except (ValueError, OSError):
        try:
            processor = DetrImageProcessor.from_pretrained(str(model_dir))
        except Exception:
            # Criar processador genérico
            processor = DetrImageProcessor(
                format="coco_detection",
                size={"shortest_edge": 640, "longest_edge": 640},
                do_resize=True,
                do_normalize=True,
                do_rescale=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
    
    # Carregar dados COCO
    json_path = dataset_dir / f"{split}/_annotations.coco.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON não encontrado: {json_path}")
    
    coco_gt = COCO(str(json_path))
    coco_data = load_coco_json(json_path)
    
    # Obter todas as imagens
    image_ids = coco_gt.getImgIds()
    print(f"📊 Avaliando {len(image_ids)} imagens do split '{split}'...")
    
    # Fazer predições
    all_results = []
    
    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="Processando imagens"):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = dataset_dir / split / img_info["file_name"]
            
            if not img_path.exists():
                print(f"⚠️  Imagem não encontrada: {img_path}")
                continue
            
            # Carregar e processar imagem
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Fazer predição
            outputs = model(**inputs)
            
            # Processar outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=score_threshold
            )[0]
            
            # Converter para formato COCO
            for box, score, label in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy()
            ):
                # Converter box de [x1, y1, x2, y2] para [x, y, width, height]
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                
                all_results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "score": float(score)
                })
    
    # Carregar resultados no formato COCO
    if len(all_results) == 0:
        print("❌ Nenhuma predição gerada!")
        return
    
    # Salvar resultados temporários
    results_path = model_dir / f"results_{split}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f)
    
    # Carregar no COCO
    coco_dt = coco_gt.loadRes(str(results_path))
    
    # Rodar COCOeval
    print("\n📈 Executando COCOeval...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extrair métricas
    metrics = {
        "mAP": coco_eval.stats[0],  # AP@[0.50:0.95]
        "AP50": coco_eval.stats[1],  # AP@0.50
        "AP75": coco_eval.stats[2],  # AP@0.75
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
    }
    
    # Calcular precision e recall @0.5
    # Usar métricas do COCOeval para IoU=0.5
    precision_50 = coco_eval.stats[1]  # AP50 já é precision média @0.5
    recall_50 = coco_eval.stats[6] if len(coco_eval.stats) > 6 else None
    
    print("\n" + "="*50)
    print("📊 MÉTRICAS FINAIS")
    print("="*50)
    print(f"mAP (AP@[0.50:0.95]): {metrics['mAP']:.4f}")
    print(f"AP50: {metrics['AP50']:.4f}")
    print(f"AP75: {metrics['AP75']:.4f}")
    print(f"AP (small): {metrics['AP_small']:.4f}")
    print(f"AP (medium): {metrics['AP_medium']:.4f}")
    print(f"AP (large): {metrics['AP_large']:.4f}")
    
    if recall_50 is not None:
        print(f"AR@0.5: {recall_50:.4f}")
    
    print("="*50)
    
    # Salvar métricas
    metrics_path = model_dir / f"metrics_{split}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Métricas salvas em: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Avaliar modelo RT-DETR")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Diretório contendo o modelo"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Split a avaliar"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.01,
        help="Threshold mínimo de score"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=Path(args.model_dir),
        dataset_dir=Path(args.dataset_dir),
        split=args.split,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold
    )


if __name__ == "__main__":
    main()

