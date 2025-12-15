#!/usr/bin/env python3
"""
Script para verificar segurança antes de fazer commit.
Verifica se arquivos sensíveis não estão sendo commitados.
"""

import subprocess
import sys
from pathlib import Path

def check_git_status():
    """Verifica se há arquivos sensíveis no staging area."""
    try:
        # Verificar se estamos em um repositório git
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("⚠️  Não é um repositório git. Pulando verificação.")
            return True
    except FileNotFoundError:
        print("⚠️  Git não encontrado. Pulando verificação.")
        return True
    
    # Verificar arquivos staged
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        # Não há staging area ou erro
        return True
    
    staged_files = result.stdout.strip().split('\n')
    sensitive_files = ['.env', 'env']
    sensitive_patterns = ['api_key', 'apikey', 'secret', 'password', 'token']
    
    issues = []
    
    for file in staged_files:
        if not file:
            continue
        
        # Verificar nomes de arquivos sensíveis
        file_lower = file.lower()
        if any(sensitive in file_lower for sensitive in sensitive_files):
            issues.append(f"❌ Arquivo sensível detectado: {file}")
        
        # Verificar conteúdo (apenas para arquivos pequenos)
        try:
            if Path(file).exists() and Path(file).stat().st_size < 10000:  # < 10KB
                content = Path(file).read_text().lower()
                if 'roboflow_api_key' in content or 'api_key=' in content:
                    # Verificar se não é apenas o exemplo
                    if 'coloque_sua_chave_aqui' not in content and 'your_api_key' not in content:
                        issues.append(f"⚠️  Possível API key em: {file}")
        except Exception:
            pass
    
    if issues:
        print("\n" + "="*60)
        print("🚨 PROBLEMAS DE SEGURANÇA DETECTADOS!")
        print("="*60)
        for issue in issues:
            print(f"  {issue}")
        print("\n⚠️  NÃO faça commit de arquivos com credenciais reais!")
        print("   Certifique-se de que:")
        print("   1. O arquivo .env está no .gitignore")
        print("   2. Você está usando env.example para valores de exemplo")
        print("   3. Nenhuma API key real está no código")
        print("="*60 + "\n")
        return False
    
    print("✅ Verificação de segurança passou!")
    return True


if __name__ == "__main__":
    success = check_git_status()
    sys.exit(0 if success else 1)

