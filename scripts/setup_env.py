#!/usr/bin/env python3
"""
Script auxiliar para configurar o arquivo .env de forma segura.
"""

import os
import shutil
from pathlib import Path
from getpass import getpass

def setup_env():
    """Configura o arquivo .env de forma interativa e segura."""
    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    print("🔐 Configuração Segura do Ambiente")
    print("="*50)
    
    # Verificar se .env já existe
    if env_file.exists():
        response = input(f"\n⚠️  O arquivo .env já existe. Sobrescrever? (s/N): ")
        if response.lower() != 's':
            print("Operação cancelada.")
            return
    
    # Copiar exemplo se não existir
    if not env_example.exists():
        print("❌ Arquivo .env.example não encontrado!")
        return
    
    # Ler exemplo
    with open(env_example, 'r') as f:
        template = f.read()
    
    print("\n📝 Configure suas credenciais:")
    print("   (Pressione Enter para usar valores padrão ou do .env existente)\n")
    
    # Valores existentes (se .env já existe)
    existing_values = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_values[key] = value
    
    # Solicitar API Key de forma segura
    api_key = getpass("🔑 ROBOFLOW_API_KEY (não será exibido): ")
    if not api_key:
        api_key = existing_values.get("ROBOFLOW_API_KEY", "")
        if api_key:
            print("   Usando valor existente.")
        else:
            print("⚠️  API Key não fornecida! Configure manualmente depois.")
            api_key = "coloque_sua_chave_aqui"
    
    # Outros valores
    workspace = input(f"📁 ROBOFLOW_WORKSPACE [{existing_values.get('ROBOFLOW_WORKSPACE', 'guemba')}]: ").strip()
    if not workspace:
        workspace = existing_values.get("ROBOFLOW_WORKSPACE", "guemba")
    
    project = input(f"📦 ROBOFLOW_PROJECT [{existing_values.get('ROBOFLOW_PROJECT', 'buddmeyer')}]: ").strip()
    if not project:
        project = existing_values.get("ROBOFLOW_PROJECT", "buddmeyer")
    
    version = input(f"🔢 ROBOFLOW_VERSION [{existing_values.get('ROBOFLOW_VERSION', '2')}]: ").strip()
    if not version:
        version = existing_values.get("ROBOFLOW_VERSION", "2")
    
    mps_fallback = input(f"⚙️  PYTORCH_ENABLE_MPS_FALLBACK [{existing_values.get('PYTORCH_ENABLE_MPS_FALLBACK', '1')}]: ").strip()
    if not mps_fallback:
        mps_fallback = existing_values.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    # Criar conteúdo do .env
    env_content = f"""# ⚠️ IMPORTANTE: Este arquivo contém credenciais sensíveis!
# NUNCA commite este arquivo no Git!
# O arquivo .env está no .gitignore

ROBOFLOW_API_KEY={api_key}
ROBOFLOW_WORKSPACE={workspace}
ROBOFLOW_PROJECT={project}
ROBOFLOW_VERSION={version}
PYTORCH_ENABLE_MPS_FALLBACK={mps_fallback}
"""
    
    # Salvar
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    # Definir permissões restritivas (apenas owner pode ler)
    os.chmod(env_file, 0o600)
    
    print(f"\n✅ Arquivo .env criado com sucesso!")
    print(f"   Localização: {env_file}")
    print(f"   Permissões: 600 (apenas você pode ler)")
    print(f"\n⚠️  Lembre-se: Este arquivo está no .gitignore e não será commitado.")


if __name__ == "__main__":
    setup_env()

