#!/usr/bin/env python3
"""
Script para fazer inferência em vídeos com modelo DETR/RT-DETR treinado - ObjectDetection_DETR.
Processa vídeo frame por frame mantendo velocidade natural.
"""

import json
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import cv2
import uuid
import csv
from datetime import datetime

from coco_utils import load_coco_json


def get_device():
    """Retorna device disponível (MPS ou CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def draw_bbox_opencv(frame, bbox, label, score, color=(0, 255, 0), thickness=2, control_id=None, show_control_id=False):
    """
    Desenha bounding box no frame OpenCV.
    
    Args:
        frame: Frame OpenCV (numpy array BGR)
        bbox: [x1, y1, x2, y2]
        label: String com label
        score: Score da predição
        color: Cor BGR (padrão verde)
        thickness: Espessura da linha
        control_id: ID único de controle (opcional)
        show_control_id: Se True, exibe o control_id no texto
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Desenhar retângulo
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Texto com label e score
    text = f"{label} {score:.2f}"
    if show_control_id and control_id is not None:
        # Usar apenas os primeiros 8 caracteres do UUID para legibilidade
        short_id = str(control_id)[:8]
        text = f"{label} {score:.2f} [{short_id}]"
    
    # Calcular tamanho do texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Desenhar fundo do texto
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Desenhar texto
    cv2.putText(
        frame,
        text,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )


def infer_video(
    model_dir: Path,
    video_path: Path,
    out_path: Path,
    score_threshold: float = 0.3,
    dataset_dir: Path = None,
    show_preview: bool = False,
    show_control_id: bool = False
):
    """
    Faz inferência em um vídeo frame por frame.
    
    Args:
        model_dir: Diretório contendo o modelo
        video_path: Caminho do vídeo de entrada
        out_path: Caminho do vídeo de saída
        score_threshold: Threshold mínimo de score
        dataset_dir: Diretório do dataset (para obter nomes de categorias)
        show_preview: Se True, mostra preview durante processamento
        show_control_id: Se True, exibe o control_id nos frames anotados
    """
    device = get_device()
    print(f"🔧 Device: {device}")
    
    # Verificar se vídeo existe
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    
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
    
    # Abrir vídeo
    print(f"📹 Abrindo vídeo: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    
    # Obter propriedades do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Propriedades do vídeo:")
    print(f"   FPS: {fps}")
    print(f"   Resolução: {width}x{height}")
    print(f"   Total de frames: {total_frames}")
    print(f"   Duração: {total_frames/fps:.2f} segundos")
    
    # Criar writer de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    if not out_video.isOpened():
        raise RuntimeError(f"Não foi possível criar vídeo de saída: {out_path}")
    
    # Resultados
    all_results = []
    frame_count = 0
    
    # Arquivo CSV para salvar detecções por frame
    csv_path = out_path.parent / f"{out_path.stem}_detections.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    # Cabeçalhos do CSV
    csv_headers = [
        "frame",
        "timestamp_seconds",
        "control_id",
        "category_id",
        "category_name",
        "score",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "bbox_center_x",
        "bbox_center_y",
        "detection_timestamp"
    ]
    csv_writer.writerow(csv_headers)
    
    # Buffer para escrita CSV em lote (otimização para tempo real)
    csv_buffer = []
    csv_buffer_size = 100  # Escrever a cada 100 detecções
    control_id_counter = 0  # Contador sequencial (mais rápido que UUID)
    
    print(f"\n🎬 Processando vídeo frame por frame...")
    print(f"   Velocidade: {fps} FPS (velocidade natural do vídeo)")
    print(f"   CSV será salvo em: {csv_path}")
    print(f"   ⚡ Otimização: Buffer de {csv_buffer_size} detecções para escrita em lote\n")
    
    # Barra de progresso
    pbar = tqdm(total=total_frames, desc="Processando frames")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Timestamp único por frame (otimização)
            timestamp_seconds = frame_count / fps if fps > 0 else 0.0
            frame_timestamp = datetime.now().isoformat()
            
            # Converter frame BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Processar
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Fazer predição
            outputs = model(**inputs)
            
            # Processar outputs
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=score_threshold
            )[0]
            
            # Desenhar bboxes no frame
            frame_detections = []
            for box, score, label in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy()
            ):
                x1, y1, x2, y2 = box.tolist()
                
                # Gerar ID único de controle (contador sequencial - mais rápido que UUID)
                control_id_counter += 1
                control_id = f"det_{control_id_counter:08d}"
                
                # Obter nome da categoria
                label_name = category_names.get(int(label), f"Class_{int(label)}")
                
                # Desenhar no frame
                draw_bbox_opencv(
                    frame,
                    [x1, y1, x2, y2],
                    label_name,
                    float(score),
                    control_id=control_id,
                    show_control_id=show_control_id
                )
                
                # Calcular informações adicionais do bbox (manter em float para performance)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                
                # Adicionar ao resultado
                detection_data = {
                    "frame": frame_count,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "category_id": int(label),
                    "category_name": label_name,
                    "score": float(score),
                    "control_id": control_id
                }
                frame_detections.append(detection_data)
                
                # Adicionar ao buffer CSV (otimização: escrita em lote)
                csv_buffer.append([
                    frame_count,
                    f"{timestamp_seconds:.6f}",
                    control_id,
                    int(label),
                    label_name,
                    f"{float(score):.6f}",
                    f"{float(x1):.2f}",
                    f"{float(y1):.2f}",
                    f"{float(x2):.2f}",
                    f"{float(y2):.2f}",
                    f"{bbox_width:.2f}",
                    f"{bbox_height:.2f}",
                    f"{bbox_center_x:.2f}",
                    f"{bbox_center_y:.2f}",
                    frame_timestamp
                ])
                
                # Escrever buffer quando atingir tamanho limite
                if len(csv_buffer) >= csv_buffer_size:
                    csv_writer.writerows(csv_buffer)
                    csv_buffer.clear()
            
            all_results.extend(frame_detections)
            
            # Escrever frame processado
            out_video.write(frame)
            
            # Mostrar preview se solicitado
            if show_preview:
                cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Preview interrompido pelo usuário")
                    break
            
            # Atualizar progresso
            pbar.update(1)
            
            # Manter velocidade natural do vídeo (não necessário aqui pois estamos salvando)
            # Mas podemos adicionar um pequeno delay se necessário para preview
    
    pbar.close()
    cap.release()
    out_video.release()
    
    # Escrever buffer restante antes de fechar
    if csv_buffer:
        csv_writer.writerows(csv_buffer)
        csv_buffer.clear()
    
    csv_file.close()  # Fechar arquivo CSV
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # Salvar resultados JSON
    results_path = out_path.parent / f"{out_path.stem}_predictions.json"
    with open(results_path, 'w') as f:
        json.dump({
            "video": str(video_path),
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "total_frames": total_frames,
            "detections": all_results
        }, f, indent=2)
    
    print(f"\n✅ Processamento concluído!")
    print(f"   Vídeo salvo em: {out_path}")
    print(f"   Resultados JSON: {results_path}")
    print(f"   Resultados CSV: {csv_path}")
    print(f"   Total de detecções: {len(all_results)}")
    print(f"   Frames processados: {frame_count}/{total_frames}")


def main():
    parser = argparse.ArgumentParser(description="Fazer inferência em vídeo com modelo RT-DETR")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Diretório contendo o modelo"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Caminho do vídeo de entrada"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Caminho do vídeo de saída"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Threshold mínimo de score"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Diretório do dataset (para obter nomes de categorias)"
    )
    parser.add_argument(
        "--show_preview",
        action="store_true",
        help="Mostrar preview durante processamento"
    )
    parser.add_argument(
        "--show_control_id",
        action="store_true",
        help="Exibir control_id nos frames anotados"
    )
    
    args = parser.parse_args()
    
    infer_video(
        model_dir=Path(args.model_dir),
        video_path=Path(args.video_path),
        out_path=Path(args.out_path),
        score_threshold=args.score_threshold,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        show_preview=args.show_preview,
        show_control_id=args.show_control_id
    )


if __name__ == "__main__":
    main()

