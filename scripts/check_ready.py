#!/usr/bin/env python3
"""
Script para verificar se tudo está pronto para treinar o modelo.
"""

import sys
from pathlib import Path

def check_file(path, name):
    """Verifica se arquivo existe."""
    if Path(path).exists():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name}: {path} não encontrado")
        return False

def check_dir(path, name):
    """Verifica se diretório existe e não está vazio."""
    p = Path(path)
    if p.exists() and p.is_dir():
        files = list(p.iterdir())
        if files:
            print(f"✅ {name}: {path} ({len(files)} itens)")
            return True
        else:
            print(f"⚠️  {name}: {path} existe mas está vazio")
            return False
    else:
        print(f"❌ {name}: {path} não encontrado")
        return False

def main():
    print("🔍 Verificando se está pronto para treinar...\n")
    
    all_ok = True
    
    # Verificar estrutura básica
    print("📁 Estrutura do Projeto:")
    all_ok &= check_file("requirements.txt", "requirements.txt")
    all_ok &= check_file(".env", ".env")
    all_ok &= check_dir(".venv", "Ambiente virtual")
    print()
    
    # Verificar dataset
    print("📊 Dataset:")
    dataset_ok = check_dir("dataset", "dataset/")
    if dataset_ok:
        train_json = Path("dataset/train/_annotations.coco.json")
        if train_json.exists():
            import json
            with open(train_json) as f:
                data = json.load(f)
            print(f"   ✅ Train: {len(data.get('images', []))} imagens, {len(data.get('annotations', []))} anotações")
        
        valid_json = Path("dataset/valid/_annotations.coco.json")
        if valid_json.exists():
            import json
            with open(valid_json) as f:
                data = json.load(f)
            print(f"   ✅ Valid: {len(data.get('images', []))} imagens, {len(data.get('annotations', []))} anotações")
        else:
            print(f"   ⚠️  Valid: não encontrado (será usado train para validação)")
        
        test_json = Path("dataset/test/_annotations.coco.json")
        if test_json.exists():
            import json
            with open(test_json) as f:
                data = json.load(f)
            print(f"   ✅ Test: {len(data.get('images', []))} imagens, {len(data.get('annotations', []))} anotações")
        else:
            print(f"   ⚠️  Test: não encontrado")
    
    all_ok &= dataset_ok
    print()
    
    # Verificar dependências Python
    print("🐍 Dependências Python:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print(f"   ✅ MPS disponível: {torch.backends.mps.is_available()}")
        else:
            print(f"   ⚠️  MPS não disponível (usará CPU)")
    except ImportError:
        print(f"   ❌ PyTorch não instalado")
        all_ok = False
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print(f"   ❌ Transformers não instalado")
        all_ok = False
    
    try:
        import pycocotools
        print(f"   ✅ pycocotools instalado")
    except ImportError:
        print(f"   ❌ pycocotools não instalado")
        all_ok = False
    
    print()
    
    # Verificar scripts
    print("📜 Scripts:")
    scripts_ok = True
    scripts_ok &= check_file("scripts/download_roboflow_coco.py", "download_roboflow_coco.py")
    scripts_ok &= check_file("src/train_rtdetr.py", "train_rtdetr.py")
    scripts_ok &= check_file("src/eval_coco.py", "eval_coco.py")
    scripts_ok &= check_file("src/infer_images.py", "infer_images.py")
    all_ok &= scripts_ok
    print()
    
    # Resultado final
    print("="*60)
    if all_ok:
        print("✅ TUDO PRONTO PARA TREINAR!")
        print()
        print("Para iniciar o treinamento, execute:")
        print("  python src/train_rtdetr.py --dataset_dir dataset --out_dir runs_rtdetr --epochs 50 --img_size 640")
    else:
        print("❌ ALGUNS ITENS FALTANDO")
        print()
        print("Execute os seguintes passos:")
        if not Path(".venv").exists():
            print("  1. ./scripts/bootstrap_mac.sh")
        if not Path("dataset/train").exists():
            print("  2. python scripts/download_roboflow_coco.py")
        if not Path(".env").exists():
            print("  3. python scripts/setup_env.py")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

