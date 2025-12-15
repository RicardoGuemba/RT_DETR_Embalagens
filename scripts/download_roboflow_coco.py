#!/usr/bin/env python3
"""
Script para baixar dataset COCO do Roboflow.
Suporta três métodos: SDK Python, curl (Terminal), ou usar dataset existente.
"""

import os
import argparse
import tempfile
import shutil
import time
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def download_with_curl(download_url: str, output_dir: Path) -> Path:
    """Baixa dataset usando curl + unzip (método Terminal do Roboflow)."""
    print("📥 Usando método curl (Terminal)...")
    
    temp_zip = output_dir / "roboflow_temp.zip"
    
    try:
        # Baixar com curl
        print(f"   ⏳ Baixando ZIP de {download_url[:50]}...")
        result = subprocess.run(
            ["curl", "-L", download_url, "-o", str(temp_zip)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Extrair ZIP
        print("   ⏳ Extraindo arquivos...")
        subprocess.run(
            ["unzip", "-q", str(temp_zip), "-d", str(output_dir)],
            check=True
        )
        
        # Remover ZIP temporário
        temp_zip.unlink()
        
        # Encontrar diretório extraído (pode ter estrutura aninhada)
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(extracted_dirs) == 1:
            # Se há apenas um diretório, mover conteúdo para o nível superior
            extracted_dir = extracted_dirs[0]
            for item in extracted_dir.iterdir():
                dest = output_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(output_dir))
            extracted_dir.rmdir()
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Erro ao baixar com curl: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("curl ou unzip não encontrado. Instale: brew install curl unzip")

def download_with_sdk(api_key: str, workspace: str, project: str, version: int, output_dir: Path) -> Path:
    """Baixa dataset usando SDK Python do Roboflow."""
    print("📥 Usando método SDK Python...")
    
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("SDK do Roboflow não instalado. Execute: pip install roboflow")
    
    # Usar diretório temporário
    temp_dir = Path(tempfile.mkdtemp(prefix="roboflow_download_"))
    
    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        
        print("   ⏳ Aguardando download e extração...")
        dataset = project_obj.version(version).download("coco", location=str(temp_dir))
        
        actual_location = Path(dataset.location) if hasattr(dataset, 'location') else temp_dir
        
        # Aguardar extração
        max_wait = 45
        waited = 0
        
        while waited < max_wait:
            found_splits = []
            for split in ["train", "valid", "test"]:
                json_file = actual_location / f"{split}/_annotations.coco.json"
                if json_file.exists():
                    found_splits.append(split)
            
            if len(found_splits) >= 1:
                time.sleep(3)
                break
            
            time.sleep(1)
            waited += 1
            if waited % 5 == 0:
                print(f"   ⏳ Aguardando... ({waited}s)")
        
        # Mover para diretório final
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif not item.name.startswith('.'):
                    item.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for item in actual_location.iterdir():
            if not item.name.startswith('.'):
                dest = output_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(item), str(dest))
                else:
                    shutil.move(str(item), str(dest))
        
        return output_dir
        
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def verify_dataset(dataset_dir: Path) -> dict:
    """Verifica estrutura do dataset e retorna estatísticas."""
    splits = ["train", "valid", "test"]
    stats = {
        "splits_found": [],
        "total_images": 0,
        "total_annotations": 0,
        "categories": set(),
        "valid": True
    }
    
    for split in splits:
        json_file = dataset_dir / f"{split}/_annotations.coco.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                n_images = len(coco_data.get("images", []))
                n_annotations = len(coco_data.get("annotations", []))
                categories = coco_data.get("categories", [])
                
                stats["splits_found"].append(split)
                stats["total_images"] += n_images
                stats["total_annotations"] += n_annotations
                
                for cat in categories:
                    stats["categories"].add(cat.get("name", "unknown"))
                    
            except Exception as e:
                stats["valid"] = False
                stats["error"] = str(e)
        else:
            stats["valid"] = False
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Baixar dataset COCO do Roboflow")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório de saída para o dataset"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API Key do Roboflow (ou use .env)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace do Roboflow (ou use .env)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Projeto do Roboflow (ou use .env)"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Versão do dataset (ou use .env)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sdk", "curl", "auto"],
        default="auto",
        help="Método de download: sdk, curl, ou auto (tenta SDK primeiro, depois curl)"
    )
    parser.add_argument(
        "--download_url",
        type=str,
        default=None,
        help="URL direta para download (método Raw URL do Roboflow). Se fornecido, usa curl."
    )
    parser.add_argument(
        "--use_existing",
        type=str,
        default=None,
        help="Usar dataset já baixado neste diretório (apenas verifica e copia se necessário)"
    )
    
    args = parser.parse_args()
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Se usar dataset existente
    if args.use_existing:
        existing_dir = Path(args.use_existing)
        dataset_dir = Path(args.dataset_dir)
        
        if not existing_dir.exists():
            raise ValueError(f"Diretório não encontrado: {existing_dir}")
        
        print(f"📂 Usando dataset existente em: {existing_dir}")
        
        # Verificar se já está no local correto
        if existing_dir.resolve() == dataset_dir.resolve():
            print("   ✅ Dataset já está no local correto")
        else:
            # Copiar para o local desejado
            print(f"   📦 Copiando para: {dataset_dir}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            shutil.copytree(existing_dir, dataset_dir)
        
        # Verificar estrutura
        stats = verify_dataset(dataset_dir)
        
        print(f"\n📊 Verificando estrutura do dataset:")
        print("-" * 60)
        
        for split in ["train", "valid", "test"]:
            json_file = dataset_dir / f"{split}/_annotations.coco.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                n_images = len(coco_data.get("images", []))
                n_annotations = len(coco_data.get("annotations", []))
                n_categories = len(coco_data.get("categories", []))
                print(f"   ✅ {split.upper():6s}: {n_images:4d} imagens, {n_annotations:4d} anotações, {n_categories} categorias")
            else:
                print(f"   ❌ {split.upper():6s}: JSON não encontrado")
        
        print("-" * 60)
        print(f"   📈 TOTAL: {stats['total_images']} imagens, {stats['total_annotations']} anotações")
        
        if stats["valid"] and len(stats["splits_found"]) == 3:
            print(f"\n✅ Dataset válido com todos os splits!")
            return
        else:
            missing = [s for s in ["train", "valid", "test"] if s not in stats["splits_found"]]
            print(f"\n⚠️  ATENÇÃO: Splits faltando: {', '.join(missing)}")
            return
    
    # Obter credenciais (args têm prioridade sobre .env)
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    workspace = args.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project = args.project or os.getenv("ROBOFLOW_PROJECT")
    version = args.version or int(os.getenv("ROBOFLOW_VERSION", "2"))
    
    if not api_key:
        raise ValueError("API Key do Roboflow não encontrada. Configure ROBOFLOW_API_KEY no .env ou use --api_key")
    if not workspace:
        raise ValueError("Workspace não encontrado. Configure ROBOFLOW_WORKSPACE no .env ou use --workspace")
    if not project:
        raise ValueError("Projeto não encontrado. Configure ROBOFLOW_PROJECT no .env ou use --project")
    
    print(f"🔗 Conectando ao Roboflow...")
    print(f"   Workspace: {workspace}")
    print(f"   Projeto: {project}")
    print(f"   Versão: {version}")
    
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Se URL direta fornecida, usar curl
    if args.download_url:
        download_with_curl(args.download_url, dataset_dir)
    elif args.method == "curl":
        # Tentar construir URL (requer acesso à API)
        download_url = f"https://app.roboflow.com/ds/...?key={api_key}"
        print("⚠️  URL direta não fornecida. Use --download_url ou --method sdk")
        raise ValueError("URL de download necessária para método curl")
    elif args.method == "sdk":
        download_with_sdk(api_key, workspace, project, version, dataset_dir)
    else:  # auto
        try:
            print("🔄 Tentando método SDK primeiro...")
            download_with_sdk(api_key, workspace, project, version, dataset_dir)
        except Exception as e:
            print(f"⚠️  SDK falhou: {e}")
            print("🔄 Tentando método curl...")
            if args.download_url:
                download_with_curl(args.download_url, dataset_dir)
            else:
                raise ValueError("SDK falhou e URL de download não fornecida. Use --download_url ou --method sdk")
    
    print(f"✅ Dataset baixado e extraído com sucesso!")
    print(f"   Localização: {dataset_dir}")
    
    # Verificar estrutura
    stats = verify_dataset(dataset_dir)
    
    print(f"\n📊 Verificando estrutura do dataset:")
    print("-" * 60)
    
    for split in ["train", "valid", "test"]:
        json_file = dataset_dir / f"{split}/_annotations.coco.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                n_images = len(coco_data.get("images", []))
                n_annotations = len(coco_data.get("annotations", []))
                n_categories = len(coco_data.get("categories", []))
                print(f"   ✅ {split.upper():6s}: {n_images:4d} imagens, {n_annotations:4d} anotações, {n_categories} categorias")
            except Exception as e:
                print(f"   ⚠️  {split.upper():6s}: JSON encontrado mas erro ao ler - {e}")
        else:
            print(f"   ❌ {split.upper():6s}: JSON não encontrado")
    
    print("-" * 60)
    print(f"   📈 TOTAL: {stats['total_images']} imagens, {stats['total_annotations']} anotações")
    
    # Verificar se todos os splits estão presentes
    missing_splits = [s for s in ["train", "valid", "test"] if s not in stats["splits_found"]]
    if missing_splits:
        print(f"\n⚠️  ATENÇÃO: Splits faltando: {', '.join(missing_splits)}")
        print(f"   Certifique-se de que a versão {version} tem todos os splits configurados no Roboflow.")
    else:
        print(f"\n✅ Todos os splits (train/valid/test) estão presentes!")

if __name__ == "__main__":
    main()
