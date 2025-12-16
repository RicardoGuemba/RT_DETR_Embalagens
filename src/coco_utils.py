"""
Utilitários para manipulação de datasets COCO.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from PIL import Image
from datetime import datetime


def ensure_coco_info(coco_data: Dict) -> Dict:
    """
    Garante que o JSON COCO tenha o campo 'info' obrigatório.
    pycocotools requer este campo para funcionar corretamente.
    
    Args:
        coco_data: Dict com dados COCO
        
    Returns:
        Dict com campo 'info' garantido
    """
    if "info" not in coco_data:
        coco_data["info"] = {
            "description": "Dataset COCO exportado do Roboflow",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Roboflow",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        }
    
    return coco_data


def ensure_coco_info_file(json_path: Path) -> None:
    """
    Garante que o arquivo JSON COCO tenha o campo 'info' obrigatório.
    Modifica o arquivo in-place se necessário.
    
    Args:
        json_path: Caminho do arquivo JSON COCO
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "info" not in data:
            data["info"] = {
                "description": "Dataset COCO exportado do Roboflow",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Roboflow",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Fazer backup antes de modificar
            backup_path = json_path.with_suffix('.json.backup')
            if not backup_path.exists():
                import shutil
                shutil.copy2(json_path, backup_path)
            
            # Salvar arquivo corrigido
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Campo 'info' adicionado em {json_path.name}")
    except Exception as e:
        print(f"⚠️  Erro ao garantir campo 'info' em {json_path}: {e}")


def load_coco_json(json_path: Path) -> Dict:
    """Carrega arquivo JSON COCO."""
    with open(json_path, 'r') as f:
        return json.load(f)


def remap_category_ids(coco_data: Dict) -> Tuple[Dict, Dict]:
    """
    Remapeia category_id para serem contíguos (0..N-1).
    Remove categorias não utilizadas nas anotações.
    
    Caso especial: Se há apenas 1 classe, garante que seja id=0.
    
    Returns:
        coco_data_remapped: Dados COCO com IDs remapeados
        id_mapping: Dict {old_id: new_id}
    """
    # Identificar categorias realmente usadas nas anotações
    used_category_ids = set(ann["category_id"] for ann in coco_data["annotations"])
    
    # Filtrar apenas categorias usadas e ordenar
    used_categories = [cat for cat in coco_data["categories"] if cat["id"] in used_category_ids]
    used_categories = sorted(used_categories, key=lambda x: x["id"])
    
    # Se nenhuma categoria usada, usar todas as categorias
    if not used_categories:
        used_categories = sorted(coco_data["categories"], key=lambda x: x["id"])
    
    # Criar mapeamento old_id -> new_id (0..N-1)
    id_mapping = {cat["id"]: idx for idx, cat in enumerate(used_categories)}
    
    # Criar cópia dos dados
    coco_data_remapped = {
        "images": coco_data["images"].copy(),
        "annotations": [],
        "categories": []
    }
    
    # Remapear categorias
    for idx, cat in enumerate(used_categories):
        new_cat = cat.copy()
        new_cat["id"] = idx
        coco_data_remapped["categories"].append(new_cat)
    
    # Remapear anotações
    for ann in coco_data["annotations"]:
        new_ann = ann.copy()
        new_ann["category_id"] = id_mapping[ann["category_id"]]
        coco_data_remapped["annotations"].append(new_ann)
    
    # Caso especial: Se há apenas 1 classe, garantir id=0
    if len(used_categories) == 1 and used_categories[0]["id"] != 0:
        # Se a categoria única não tem id=0, corrigir
        used_cat = used_categories[0]
        cat_name = used_cat.get("name", "embalagem")
        
        # Atualizar categoria para id=0
        coco_data_remapped["categories"] = [{
            "id": 0,
            "name": cat_name,
            "supercategory": used_cat.get("supercategory", "none")
        }]
        
        # Atualizar mapeamento
        old_id = used_cat["id"]
        id_mapping = {old_id: 0}
        
        # Garantir que todas as anotações usam id=0
        for ann in coco_data_remapped["annotations"]:
            ann["category_id"] = 0
    
    print(f"   📊 Categorias usadas: {len(used_categories)} (de {len(coco_data['categories'])} definidas)")
    
    return coco_data_remapped, id_mapping


def get_image_annotations(image_id: int, coco_data: Dict) -> List[Dict]:
    """Retorna todas as anotações de uma imagem."""
    return [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]


def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """
    Converte bbox COCO [x, y, width, height] para [x1, y1, x2, y2].
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_coco_bbox(bbox: List[float]) -> List[float]:
    """
    Converte bbox [x1, y1, x2, y2] para COCO [x, y, width, height].
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def prepare_annotations_for_processor(
    image_id: int,
    coco_data: Dict,
    image_path: Path
) -> Dict:
    """
    Prepara anotações no formato esperado pelo AutoImageProcessor (RT-DETR/DETR).
    
    O formato correto para RT-DETR é:
    {
        'image_id': int,
        'annotations': [{'bbox': [...], 'category_id': int, 'area': float, 'iscrowd': int}, ...]
    }
    
    Returns:
        Dict com 'image' (PIL Image) e 'annotations' (dict no formato COCO)
    """
    # Carregar imagem
    image = Image.open(image_path).convert("RGB")
    
    # Obter anotações da imagem
    annotations = get_image_annotations(image_id, coco_data)
    
    # Formatar anotações
    formatted_annotations = []
    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, width, height]
        formatted_annotations.append({
            "bbox": bbox,
            "category_id": ann["category_id"],
            "area": ann.get("area", bbox[2] * bbox[3]),
            "iscrowd": ann.get("iscrowd", 0)
        })
    
    # Retornar no formato esperado pelo RT-DETR processor
    return {
        "image": image,
        "annotations": {
            "image_id": image_id,
            "annotations": formatted_annotations
        }
    }


def create_coco_results(
    predictions: List[Dict],
    image_ids: List[int],
    category_ids: List[int]
) -> List[Dict]:
    """
    Cria lista de resultados no formato COCO para avaliação.
    
    Args:
        predictions: Lista de dicts com 'bbox', 'score', 'category_id'
        image_ids: Lista de image_ids correspondentes
        category_ids: Lista de category_ids correspondentes
    
    Returns:
        Lista de resultados no formato COCO
    """
    results = []
    for pred, img_id, cat_id in zip(predictions, image_ids, category_ids):
        bbox = pred["bbox"]  # [x1, y1, x2, y2]
        # Converter para [x, y, width, height]
        coco_bbox = xyxy_to_coco_bbox(bbox)
        
        results.append({
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": coco_bbox,
            "score": float(pred["score"])
        })
    
    return results

