#!/bin/bash

# Script para instalar hook pré-commit do Git
# Este hook verifica segurança antes de cada commit

set -e

GIT_DIR="$(git rev-parse --git-dir 2>/dev/null || echo '.git')"
HOOK_FILE="$GIT_DIR/hooks/pre-commit"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check_security.py"

if [ ! -d "$GIT_DIR" ]; then
    echo "❌ Não é um repositório Git!"
    exit 1
fi

echo "🔧 Instalando hook pré-commit de segurança..."

# Criar diretório de hooks se não existir
mkdir -p "$GIT_DIR/hooks"

# Criar hook
cat > "$HOOK_FILE" << EOF
#!/bin/bash
# Hook pré-commit para verificação de segurança
# Instalado por scripts/install_pre_commit_hook.sh

python3 "$CHECK_SCRIPT"
EOF

# Tornar executável
chmod +x "$HOOK_FILE"

echo "✅ Hook pré-commit instalado com sucesso!"
echo "   Localização: $HOOK_FILE"
echo ""
echo "📝 O hook será executado automaticamente antes de cada commit."
echo "   Para desabilitar temporariamente: SKIP_SECURITY_CHECK=1 git commit"
echo "   Para remover: rm $HOOK_FILE"

