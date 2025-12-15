#!/usr/bin/env python3
"""
Script para validar estrutura do dataset COCO.
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Validar dataset COCO")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset"
    )
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ Diretório não encontrado: {dataset_dir}")
        return
    
    print(f"🔍 Validando dataset em: {dataset_dir}\n")
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        json_file = dataset_dir / f"{split}/_annotations.coco.json"
        
        if not json_file.exists():
            print(f"⚠️  {split}: JSON não encontrado ({json_file})")
            continue
        
        print(f"📊 {split.upper()}:")
        print(f"   Arquivo: {json_file}")
        
        # Carregar JSON
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Estatísticas
        n_images = len(coco_data.get("images", []))
        n_annotations = len(coco_data.get("annotations", []))
        n_categories = len(coco_data.get("categories", []))
        
        print(f"   Imagens: {n_images}")
        print(f"   Anotações: {n_annotations}")
        print(f"   Categorias: {n_categories}")
        
        # Verificar imagens
        if n_images > 0:
            image_dir = dataset_dir / split
            missing_images = 0
            found_images = 0
            
            for img_info in coco_data["images"][:10]:  # Verificar primeiras 10
                img_path = image_dir / img_info["file_name"]
                if img_path.exists():
                    found_images += 1
                else:
                    missing_images += 1
            
            if missing_images == 0:
                print(f"   ✅ Imagens encontradas (verificadas {min(10, n_images)} primeiras)")
            else:
                print(f"   ⚠️  {missing_images} imagens não encontradas nas primeiras 10")
        
        # Estatísticas de categorias
        if n_annotations > 0:
            category_ids = [ann["category_id"] for ann in coco_data["annotations"]]
            category_counts = Counter(category_ids)
            
            print(f"   Distribuição de categorias:")
            for cat_id, count in category_counts.most_common():
                cat_name = next(
                    (c["name"] for c in coco_data["categories"] if c["id"] == cat_id),
                    f"ID_{cat_id}"
                )
                print(f"      {cat_name} (ID {cat_id}): {count} anotações")
        
        # Listar categorias
        if n_categories > 0:
            print(f"   Categorias:")
            for cat in coco_data["categories"]:
                print(f"      - {cat['name']} (ID: {cat['id']})")
        
        print()

if __name__ == "__main__":
    main()

