#!/usr/bin/env python3
"""
Script para baixar dataset COCO do Roboflow.
Implementação simples seguindo a documentação oficial.
"""

import os
import argparse
import shutil
import time
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def download_with_curl(download_url: str, output_dir: Path) -> Path:
    """Baixa dataset usando curl + unzip (método Terminal do Roboflow)."""
    print("📥 Usando método curl (Terminal)...")
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limpar diretório de destino antes de baixar
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif not item.name.startswith('.'):
            item.unlink()
    
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
        
        if not temp_zip.exists() or temp_zip.stat().st_size == 0:
            raise RuntimeError("Download falhou: arquivo ZIP vazio ou não encontrado")
        
        print(f"   ✅ Download concluído: {temp_zip.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Extrair ZIP
        print("   ⏳ Extraindo arquivos...")
        subprocess.run(
            ["unzip", "-q", "-o", str(temp_zip), "-d", str(output_dir)],
            check=True
        )
        
        # Remover ZIP temporário
        temp_zip.unlink()
        print("   ✅ Extração concluída")
        
        # Encontrar diretório extraído (pode ter estrutura aninhada)
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(extracted_dirs) == 1:
            # Se há apenas um diretório, mover conteúdo para o nível superior
            extracted_dir = extracted_dirs[0]
            print(f"   📦 Reorganizando estrutura (diretório aninhado: {extracted_dir.name})...")
            for item in extracted_dir.iterdir():
                dest = output_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(output_dir))
            extracted_dir.rmdir()
            print("   ✅ Estrutura reorganizada")
        
        # Aguardar um pouco para garantir que arquivos foram escritos
        time.sleep(1)
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        if temp_zip.exists():
            temp_zip.unlink()
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"Erro ao baixar com curl: {error_msg}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Ferramenta não encontrada: {e}. Instale: brew install curl unzip")
    except Exception as e:
        if temp_zip.exists():
            temp_zip.unlink()
        raise RuntimeError(f"Erro inesperado no download curl: {e}")

def download_with_sdk(api_key: str, workspace: str, project: str, version: int, output_dir: Path) -> Path:
    """
    Baixa dataset usando SDK Python do Roboflow.
    Implementação simples e direta seguindo a documentação oficial.
    """
    print("📥 Usando método SDK Python do Roboflow...")
    
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("SDK do Roboflow não instalado. Execute: pip install roboflow")
    
    # Limpar e criar diretório de saída
    if output_dir.exists():
        print(f"   🗑️  Limpando diretório existente...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Seguir exatamente a documentação do Roboflow
        print(f"   🔗 Conectando ao Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        print(f"   📂 Acessando workspace '{workspace}' e projeto '{project}'...")
        project_obj = rf.workspace(workspace).project(project)
        
        print(f"   📥 Baixando versão {version} (formato COCO)...")
        # Seguir exatamente o código do Roboflow
        version_obj = project_obj.version(version)
        dataset = version_obj.download("coco", location=str(output_dir))
        
        print(f"   ✅ Download concluído!")
        print(f"   ⏳ Aguardando extração (isso pode demorar 10-30 segundos)...")
        
        # Aguardar a extração completa - o SDK demora para extrair
        max_wait = 60
        waited = 0
        found = False
        
        while waited < max_wait:
            time.sleep(2)
            waited += 2
            
            # Verificar se já apareceram os splits
            if output_dir.exists():
                all_dirs = [d for d in output_dir.rglob("*") if d.is_dir()]
                has_splits = any("train" in d.name or "valid" in d.name or "test" in d.name for d in all_dirs)
                
                if has_splits:
                    print(f"   ✅ Arquivos detectados após {waited}s")
                    found = True
                    break
            
            if waited % 10 == 0:
                print(f"   ⏳ Ainda aguardando... ({waited}s)")
        
        if not found:
            print(f"   ⚠️  Timeout após {waited}s. Verificando o que foi baixado...")
        
        # Verificar o que foi baixado e encontrar onde estão os splits
        print(f"   🔍 Localizando arquivos do dataset...")
        
        # DEBUG: Listar tudo que foi baixado
        print(f"   📁 Conteúdo em {output_dir}:")
        if output_dir.exists():
            all_items = sorted(list(output_dir.rglob("*")))[:40]
            for item in all_items:
                rel_path = item.relative_to(output_dir)
                prefix = "📁" if item.is_dir() else "📄"
                print(f"      {prefix} {rel_path}")
        
        # O Roboflow pode criar subdiretórios. Procurar onde estão os splits
        possible_roots = [
            output_dir,
            output_dir / project,
            output_dir / f"{project}-{version}",
        ]
        
        # Adicionar todos os subdiretórios encontrados
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and item not in possible_roots:
                    possible_roots.append(item)
        
        dataset_root = None
        for root in possible_roots:
            if root.exists():
                # Verificar se tem os diretórios de splits
                has_train = (root / "train" / "_annotations.coco.json").exists()
                has_valid = (root / "valid" / "_annotations.coco.json").exists()
                has_test = (root / "test" / "_annotations.coco.json").exists()
                
                if has_train or has_valid or has_test:
                    dataset_root = root
                    print(f"   📍 Dataset encontrado em: {root.relative_to(output_dir.parent)}")
                    break
        
        if dataset_root is None:
            # Listar o que foi baixado para debug
            print(f"   ⚠️  Estrutura de splits não encontrada. Conteúdo baixado:")
            if output_dir.exists():
                for item in sorted(output_dir.rglob("*"))[:20]:
                    rel_path = item.relative_to(output_dir)
                    print(f"      {'📁' if item.is_dir() else '📄'} {rel_path}")
            raise RuntimeError(
                f"Dataset baixado mas estrutura de splits não encontrada.\n"
                f"Verifique se a versão {version} está corretamente configurada no Roboflow."
            )
        
        # Se o dataset está em um subdiretório, mover para o diretório principal
        if dataset_root != output_dir:
            print(f"   📦 Movendo arquivos para o diretório principal...")
            
            # Criar temp para evitar conflitos
            temp_root = output_dir.parent / f"temp_{output_dir.name}"
            
            # Copiar conteúdo
            for item in dataset_root.iterdir():
                if not item.name.startswith('.'):
                    dest = temp_root / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        temp_root.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)
            
            # Remover output_dir e renomear temp
            shutil.rmtree(output_dir)
            shutil.move(str(temp_root), str(output_dir))
            
            print(f"   ✅ Arquivos reorganizados")
        
        # Verificar splits finais
        splits_found = []
        for split in ["train", "valid", "test"]:
            split_dir = output_dir / split
            json_file = split_dir / "_annotations.coco.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        coco_data = json.load(f)
                    n_images = len(coco_data.get("images", []))
                    n_anns = len(coco_data.get("annotations", []))
                    print(f"   ✅ {split.upper():5s}: {n_images} imagens, {n_anns} anotações")
                    splits_found.append(split)
                except Exception as e:
                    print(f"   ⚠️  {split.upper():5s}: erro ao ler JSON - {e}")
        
        if not splits_found:
            raise RuntimeError("Nenhum split válido encontrado após download")
        
        print(f"   🎉 Dataset pronto: {len(splits_found)} split(s)")
        return output_dir
        
    except Exception as e:
        print(f"   ❌ Erro no download SDK: {e}")
        # Limpar em caso de erro
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
            except:
                pass
        raise

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
        help="Método de download: sdk, curl, ou auto"
    )
    parser.add_argument(
        "--download_url",
        type=str,
        default=None,
        help="URL direta para download (método curl)"
    )
    parser.add_argument(
        "--use_existing",
        type=str,
        default=None,
        help="Usar dataset já baixado (apenas verifica)"
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
    
    # Obter credenciais
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    workspace = args.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project = args.project or os.getenv("ROBOFLOW_PROJECT")
    version = args.version or int(os.getenv("ROBOFLOW_VERSION", "3"))
    
    if not api_key:
        raise ValueError("API Key não encontrada. Configure no .env ou use --api_key")
    if not workspace:
        raise ValueError("Workspace não encontrado. Configure no .env ou use --workspace")
    if not project:
        raise ValueError("Projeto não encontrado. Configure no .env ou use --project")
    
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
        raise ValueError("Método curl requer --download_url")
    elif args.method == "sdk":
        download_with_sdk(api_key, workspace, project, version, dataset_dir)
    else:  # auto
        try:
            print("🔄 Método automático: tentando SDK primeiro...")
            download_with_sdk(api_key, workspace, project, version, dataset_dir)
        except Exception as e:
            print(f"⚠️  SDK falhou: {e}")
            if args.download_url:
                print("🔄 Tentando método curl...")
                download_with_curl(args.download_url, dataset_dir)
            else:
                print("\n❌ SDK falhou e nenhuma URL fornecida.")
                raise ValueError(
                    "SDK falhou e URL não fornecida.\n\n"
                    "Soluções:\n"
                    "1. Verifique as credenciais no .env\n"
                    "2. Ou forneça --download_url para usar curl"
                )
    
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
                print(f"   ⚠️  {split.upper():6s}: erro - {e}")
        else:
            print(f"   ❌ {split.upper():6s}: JSON não encontrado")
    
    print("-" * 60)
    print(f"   📈 TOTAL: {stats['total_images']} imagens, {stats['total_annotations']} anotações")
    
    missing_splits = [s for s in ["train", "valid", "test"] if s not in stats["splits_found"]]
    if missing_splits:
        print(f"\n⚠️  ATENÇÃO: Splits faltando: {', '.join(missing_splits)}")
    else:
        print(f"\n✅ Todos os splits (train/valid/test) estão presentes!")

if __name__ == "__main__":
    main()
