#!/usr/bin/env python3
"""
Script para fazer inferência com modelo RT-DETR treinado.
"""

import json
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import numpy as np

from coco_utils import load_coco_json


def get_device():
    """Retorna device disponível (MPS ou CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def draw_bbox(image, bbox, label, score, color=(255, 0, 0), width=2):
    """
    Desenha bounding box na imagem.
    
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2]
        label: String com label
        score: Score da predição
        color: Cor RGB
        width: Largura da linha
    """
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    
    # Desenhar retângulo
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # Texto com label e score
    text = f"{label} {score:.2f}"
    
    # Tentar carregar fonte, usar padrão se falhar
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    # Calcular tamanho do texto
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Desenhar fundo do texto
    draw.rectangle(
        [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
        fill=color
    )
    
    # Desenhar texto
    draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)


def infer_images(
    model_dir: Path,
    input_dir: Path,
    out_dir: Path,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    dataset_dir: Path = None
):
    """
    Faz inferência em uma pasta de imagens.
    
    Args:
        model_dir: Diretório contendo o modelo
        input_dir: Diretório com imagens de entrada
        out_dir: Diretório de saída
        score_threshold: Threshold mínimo de score
        dataset_dir: Diretório do dataset (para obter nomes de categorias)
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
    
    # Carregar nomes de categorias se disponível
    category_names = {}
    if dataset_dir:
        try:
            # Tentar carregar de qualquer split disponível
            for split in ["train", "valid", "test"]:
                json_path = dataset_dir / f"{split}/_annotations.coco.json"
                if json_path.exists():
                    coco_data = load_coco_json(json_path)
                    category_names = {
                        cat["id"]: cat["name"]
                        for cat in coco_data["categories"]
                    }
                    break
        except Exception as e:
            print(f"⚠️  Não foi possível carregar nomes de categorias: {e}")
    
    # Criar diretório de saída
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar imagens
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"❌ Nenhuma imagem encontrada em {input_dir}")
        return
    
    print(f"📊 Processando {len(image_files)} imagens...")
    
    # Resultados
    all_results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processando"):
            # Carregar imagem
            image = Image.open(img_path).convert("RGB")
            original_size = image.size
            
            # Processar
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
            
            # Nota: IOU threshold é usado internamente pelo RT-DETR durante o treinamento
            # O post_process_object_detection já aplica NMS com IOU padrão
            # O parâmetro iou_threshold é mantido para compatibilidade futura
            
            # Criar cópia da imagem para desenhar
            annotated_image = image.copy()
            
            # Desenhar bboxes
            for box, score, label in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy()
            ):
                x1, y1, x2, y2 = box.tolist()
                
                # Obter nome da categoria
                label_name = category_names.get(int(label), f"Class_{int(label)}")
                
                # Desenhar
                draw_bbox(
                    annotated_image,
                    [x1, y1, x2, y2],
                    label_name,
                    float(score)
                )
                
                # Adicionar ao resultado
                all_results.append({
                    "image": img_path.name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "category_id": int(label),
                    "category_name": label_name,
                    "score": float(score)
                })
            
            # Salvar imagem anotada
            out_img_path = out_dir / f"annotated_{img_path.name}"
            annotated_image.save(out_img_path)
    
    # Salvar resultados JSON
    results_path = out_dir / "predictions.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Inferência concluída!")
    print(f"   Imagens salvas em: {out_dir}")
    print(f"   Resultados JSON: {results_path}")
    print(f"   Total de detecções: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="Fazer inferência com modelo RT-DETR")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Diretório contendo o modelo"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Diretório com imagens de entrada"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Diretório de saída"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Threshold mínimo de score"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IOU threshold (nota: RT-DETR já aplica NMS internamente)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Diretório do dataset (para obter nomes de categorias)"
    )
    
    args = parser.parse_args()
    
    infer_images(
        model_dir=Path(args.model_dir),
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None
    )


if __name__ == "__main__":
    main()

