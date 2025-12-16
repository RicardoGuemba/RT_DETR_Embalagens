#!/usr/bin/env python3
"""
Script standalone para corrigir todos os JSONs COCO do projeto,
adicionando o campo 'info' obrigatório se não existir.

Uso:
    python scripts/fix_coco_info.py [--dataset_dir dataset]
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coco_utils import ensure_coco_info_file


def fix_all_coco_jsons(dataset_dir: Path):
    """
    Corrige todos os JSONs COCO no diretório do dataset.
    
    Args:
        dataset_dir: Diretório raiz do dataset
    """
    splits = ["train", "valid", "test"]
    fixed_count = 0
    already_ok_count = 0
    
    print("="*70)
    print("🔧 CORREÇÃO DE JSONs COCO - Adicionando campo 'info'")
    print("="*70)
    print(f"\n📁 Diretório do dataset: {dataset_dir}\n")
    
    for split in splits:
        json_file = dataset_dir / f"{split}/_annotations.coco.json"
        
        if not json_file.exists():
            print(f"⏭️  {split.upper()}: Arquivo não encontrado, pulando...")
            continue
        
        print(f"📄 Processando {split.upper()}: {json_file.name}")
        
        # Verificar se já tem campo 'info'
        import json
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "info" in data:
                print(f"   ✅ Já possui campo 'info'")
                already_ok_count += 1
            else:
                # Aplicar correção
                ensure_coco_info_file(json_file)
                fixed_count += 1
                
        except Exception as e:
            print(f"   ❌ Erro ao processar: {e}")
    
    print("\n" + "="*70)
    print("📊 RESUMO")
    print("="*70)
    print(f"   ✅ Arquivos corrigidos: {fixed_count}")
    print(f"   ✓ Arquivos já corretos: {already_ok_count}")
    print(f"   📁 Total processado: {fixed_count + already_ok_count}")
    print("="*70)
    
    if fixed_count > 0:
        print("\n💡 Backups foram criados com extensão .backup")
        print("✅ Todos os JSONs COCO agora têm o campo 'info' obrigatório!")
    else:
        print("\n✅ Todos os JSONs já estavam corretos!")


def main():
    parser = argparse.ArgumentParser(
        description="Corrige todos os JSONs COCO adicionando campo 'info'"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset (padrão: dataset)"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ Diretório não encontrado: {dataset_dir}")
        sys.exit(1)
    
    fix_all_coco_jsons(dataset_dir)


if __name__ == "__main__":
    main()

