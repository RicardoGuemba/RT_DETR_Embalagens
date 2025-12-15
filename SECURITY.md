# 🔒 Guia de Segurança

Este documento descreve as práticas de segurança implementadas no projeto para proteger credenciais e informações sensíveis.

## Proteção de API Keys

### Arquitetura de Segurança

O projeto utiliza múltiplas camadas de proteção:

1. **Variáveis de Ambiente**: Todas as credenciais são carregadas via arquivo `.env`
2. **Gitignore**: O arquivo `.env` está no `.gitignore` e nunca será commitado
3. **Template Seguro**: `.env.example` contém apenas placeholders, nunca credenciais reais
4. **Verificação Pré-Commit**: Script automático verifica commits antes de enviar
5. **Permissões de Arquivo**: `.env` é criado com permissões 600 (apenas owner pode ler)

### Como Configurar Credenciais

#### Método 1: Setup Interativo (Recomendado)

```bash
python scripts/setup_env.py
```

Este script:
- Solicita credenciais de forma segura (API key não é exibida)
- Cria `.env` com permissões restritivas (600)
- Valida entradas básicas

#### Método 2: Manual

```bash
cp .env.example .env
# Edite .env com suas credenciais
chmod 600 .env  # Definir permissões restritivas
```

### Verificação Antes de Commits

**SEMPRE** execute antes de fazer commit:

```bash
python scripts/check_security.py
```

Este script verifica:
- ✅ Arquivos sensíveis no staging area
- ✅ Possíveis API keys expostas no código
- ✅ Arquivos `.env` sendo commitados

### Estrutura de Arquivos Segura

```
projeto/
├── .env                    # ⚠️ NUNCA commitado (no .gitignore)
├── .env.example            # ✅ Pode ser commitado (sem credenciais reais)
├── .gitignore              # ✅ Inclui .env
└── scripts/
    ├── check_security.py   # ✅ Verificação pré-commit
    └── setup_env.py        # ✅ Setup seguro
```

## Checklist de Segurança

Antes de fazer commit ou compartilhar o código:

- [ ] Execute `python scripts/check_security.py`
- [ ] Verifique `git status` - `.env` não deve aparecer
- [ ] Verifique `git diff` - nenhuma API key real deve estar visível
- [ ] Confirme que `.env.example` não contém credenciais reais
- [ ] Se usar GitHub/GitLab, verifique se há secrets configurados no repositório

## O Que Fazer Se Você Expôs uma API Key

### Ação Imediata

1. **REVOGUE A CHAVE IMEDIATAMENTE** no Roboflow:
   - Acesse https://app.roboflow.com/settings
   - Revogue a chave comprometida

2. **Gere uma nova API key** no Roboflow

3. **Remova do histórico do Git** (se necessário):
   ```bash
   # Opção 1: Usar git filter-branch (cuidado!)
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Opção 2: Usar BFG Repo-Cleaner (mais seguro)
   # https://rtyley.github.io/bfg-repo-cleaner/
   ```

4. **Configure nova chave**:
   ```bash
   python scripts/setup_env.py
   ```

5. **Force push** (se necessário, avise colaboradores):
   ```bash
   git push --force
   ```

### Prevenção Futura

- Configure um hook pré-commit do Git:
  ```bash
  # Criar hook
  cat > .git/hooks/pre-commit << 'EOF'
  #!/bin/bash
  python scripts/check_security.py || exit 1
  EOF
  chmod +x .git/hooks/pre-commit
  ```

## Boas Práticas Adicionais

### Para Desenvolvimento

- ✅ Use diferentes API keys para desenvolvimento e produção
- ✅ Rotacione API keys periodicamente
- ✅ Monitore uso da API no Roboflow
- ✅ Use variáveis de ambiente do sistema quando possível em produção

### Para Compartilhamento

- ✅ Compartilhe apenas o código, nunca o `.env`
- ✅ Use `.env.example` como referência
- ✅ Documente claramente como configurar credenciais
- ✅ Considere usar secrets managers em produção (AWS Secrets Manager, HashiCorp Vault, etc.)

## Recursos Adicionais

- [Roboflow API Documentation](https://docs.roboflow.com/api)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Git Secrets Best Practices](https://git-secret.io/)

## Suporte

Se você encontrou uma vulnerabilidade de segurança, por favor:
1. **NÃO** abra uma issue pública
2. Entre em contato diretamente com o mantenedor do projeto
3. Aguarde confirmação antes de divulgar

