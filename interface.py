#!/usr/bin/env python3
"""
Interface interativa para gerenciar treinamento, predição e download de dados.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def clear_screen():
    """Limpa a tela."""
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header():
    """Imprime cabeçalho."""
    print("="*70)
    print("🚀 RT-DETR - Interface de Gerenciamento")
    print("="*70)
    print()

def print_menu():
    """Imprime menu principal."""
    print("📋 MENU PRINCIPAL")
    print("-"*70)
    print("1. 📥 Baixar Dataset do Roboflow")
    print("2. 🔍 Verificar Status do Dataset")
    print("3. 🏋️  Treinar Modelo")
    print("4. 📊 Avaliar Modelo")
    print("5. 🔮 Fazer Predição/Inferência")
    print("6. ✅ Verificar Se Está Pronto para Treinar")
    print("7. ⚙️  Configurar Ambiente (.env)")
    print("0. 🚪 Sair")
    print("-"*70)

def download_dataset():
    """Baixa dataset do Roboflow."""
    clear_screen()
    print_header()
    print("📥 BAIXAR DATASET DO ROBOFLOW")
    print("-"*70)
    print()
    
    # Verificar se .env existe
    if not Path(".env").exists():
        print("❌ Arquivo .env não encontrado!")
        print("   Execute a opção 7 para configurar primeiro.")
        input("\nPressione Enter para voltar...")
        return
    
    # Carregar versão atual do .env
    load_dotenv()
    current_version = os.getenv("ROBOFLOW_VERSION", "2")
    
    # Perguntar versão
    print(f"Versão atual do dataset (do .env): {current_version}")
    version_input = input(f"Digite a versão do dataset [{current_version}]: ").strip()
    
    if version_input:
        try:
            version = int(version_input)
        except ValueError:
            print("❌ Versão inválida! Usando versão do .env.")
            version = int(current_version)
    else:
        version = int(current_version)
    
    print(f"\n📦 Versão selecionada: {version}")
    
    # Confirmar
    print("\n⚠️  ATENÇÃO: Isso vai sobrescrever o dataset atual!")
    resposta = input("Continuar? (s/N): ").strip().lower()
    if resposta != 's':
        print("Operação cancelada.")
        input("\nPressione Enter para voltar...")
        return
    
    print("\n📥 Baixando dataset...")
    print("-"*70)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/download_roboflow_coco.py", 
             "--dataset_dir", "dataset", "--version", str(version)],
            check=True,
            capture_output=False
        )
        print("\n✅ Download concluído!")
        
        # Executar sanity check
        print("\n🔍 Verificando dataset...")
        subprocess.run(
            [sys.executable, "scripts/sanity_coco.py", "--dataset_dir", "dataset"],
            check=False
        )
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao baixar dataset: {e}")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
    
    input("\nPressione Enter para voltar...")

def check_dataset():
    """Verifica status do dataset."""
    clear_screen()
    print_header()
    print("🔍 VERIFICAR STATUS DO DATASET")
    print("-"*70)
    print()
    
    try:
        subprocess.run(
            [sys.executable, "scripts/sanity_coco.py", "--dataset_dir", "dataset"],
            check=False
        )
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\nPressione Enter para voltar...")

def train_model():
    """Inicia treinamento do modelo."""
    clear_screen()
    print_header()
    print("🏋️  TREINAR MODELO")
    print("-"*70)
    print()
    
    # Verificar se dataset existe
    train_json = Path("dataset/train/_annotations.coco.json")
    if not train_json.exists():
        print("❌ Dataset não encontrado!")
        print("   Execute a opção 1 para baixar o dataset primeiro.")
        input("\nPressione Enter para voltar...")
        return
    
    # Parâmetros padrão
    print("Parâmetros de Treinamento:")
    print("(Pressione Enter para usar valores padrão)")
    print()
    
    epochs = input("Número de épocas [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    
    batch_size = input("Batch size [1]: ").strip()
    batch_size = int(batch_size) if batch_size else 1
    
    img_size = input("Tamanho da imagem [640]: ").strip()
    img_size = int(img_size) if img_size else 640
    
    learning_rate = input("Learning rate [1e-5]: ").strip()
    learning_rate = float(learning_rate) if learning_rate else 1e-5
    
    gradient_accum = input("Gradient accumulation steps [4]: ").strip()
    gradient_accum = int(gradient_accum) if gradient_accum else 4
    
    print()
    print("="*70)
    print("Parâmetros configurados:")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Tamanho da imagem: {img_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient accumulation: {gradient_accum}")
    print("="*70)
    print()
    
    resposta = input("Iniciar treinamento? (s/N): ").strip().lower()
    if resposta != 's':
        print("Operação cancelada.")
        input("\nPressione Enter para voltar...")
        return
    
    print("\n🚀 Iniciando treinamento...")
    print("="*70)
    print("(Isso pode levar bastante tempo. Você pode acompanhar o progresso abaixo)")
    print("="*70)
    print()
    
    try:
        cmd = [
            sys.executable, "src/train_rtdetr.py",
            "--dataset_dir", "dataset",
            "--out_dir", "runs_rtdetr",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--img_size", str(img_size),
            "--learning_rate", str(learning_rate),
            "--gradient_accumulation_steps", str(gradient_accum)
        ]
        
        subprocess.run(cmd, check=False)
        
        print("\n" + "="*70)
        print("✅ Treinamento concluído!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
    
    input("\nPressione Enter para voltar...")

def evaluate_model():
    """Avalia modelo treinado."""
    clear_screen()
    print_header()
    print("📊 AVALIAR MODELO")
    print("-"*70)
    print()
    
    # Verificar modelos disponíveis
    model_best = Path("runs_rtdetr/model_best")
    model_final = Path("runs_rtdetr/model_final")
    
    if not model_best.exists() and not model_final.exists():
        print("❌ Nenhum modelo treinado encontrado!")
        print("   Execute a opção 3 para treinar um modelo primeiro.")
        input("\nPressione Enter para voltar...")
        return
    
    # Escolher modelo
    print("Modelos disponíveis:")
    if model_best.exists():
        print("  1. model_best (melhor modelo)")
    if model_final.exists():
        print("  2. model_final (modelo final)")
    
    escolha = input("\nEscolha o modelo [1]: ").strip()
    if not escolha:
        escolha = "1"
    
    if escolha == "1" and model_best.exists():
        model_dir = "runs_rtdetr/model_best"
    elif escolha == "2" and model_final.exists():
        model_dir = "runs_rtdetr/model_final"
    else:
        print("❌ Opção inválida!")
        input("\nPressione Enter para voltar...")
        return
    
    # Escolher split
    print("\nSplit para avaliação:")
    print("  1. valid")
    print("  2. test")
    print("  3. train")
    
    split_choice = input("\nEscolha o split [1]: ").strip()
    split_map = {"1": "valid", "2": "test", "3": "train"}
    split = split_map.get(split_choice, "valid")
    
    print(f"\n📊 Avaliando modelo {model_dir} no split {split}...")
    print("="*70)
    
    try:
        cmd = [
            sys.executable, "src/eval_coco.py",
            "--model_dir", model_dir,
            "--dataset_dir", "dataset",
            "--split", split
        ]
        
        subprocess.run(cmd, check=False)
        
    except Exception as e:
        print(f"\n❌ Erro durante avaliação: {e}")
    
    input("\nPressione Enter para voltar...")

def run_inference():
    """Executa inferência/predição."""
    clear_screen()
    print_header()
    print("🔮 FAZER PREDIÇÃO/INFERÊNCIA")
    print("-"*70)
    print()
    
    # Verificar modelos disponíveis
    model_best = Path("runs_rtdetr/model_best")
    model_final = Path("runs_rtdetr/model_final")
    
    if not model_best.exists() and not model_final.exists():
        print("❌ Nenhum modelo treinado encontrado!")
        print("   Execute a opção 3 para treinar um modelo primeiro.")
        input("\nPressione Enter para voltar...")
        return
    
    # Escolher modelo
    print("Modelos disponíveis:")
    if model_best.exists():
        print("  1. model_best (melhor modelo)")
    if model_final.exists():
        print("  2. model_final (modelo final)")
    
    escolha = input("\nEscolha o modelo [1]: ").strip()
    if not escolha:
        escolha = "1"
    
    if escolha == "1" and model_best.exists():
        model_dir = "runs_rtdetr/model_best"
    elif escolha == "2" and model_final.exists():
        model_dir = "runs_rtdetr/model_final"
    else:
        print("❌ Opção inválida!")
        input("\nPressione Enter para voltar...")
        return
    
    # Diretório de entrada
    print("\nDiretório de entrada:")
    print("(Pode ser uma pasta com imagens ou um split do dataset)")
    input_dir = input("Caminho [dataset/test]: ").strip()
    if not input_dir:
        input_dir = "dataset/test"
    
    if not Path(input_dir).exists():
        print(f"❌ Diretório não encontrado: {input_dir}")
        input("\nPressione Enter para voltar...")
        return
    
    # Diretório de saída
    output_dir = input("Diretório de saída [runs_rtdetr/infer_out]: ").strip()
    if not output_dir:
        output_dir = "runs_rtdetr/infer_out"
    
    # Score threshold
    threshold = input("Score threshold [0.3]: ").strip()
    threshold = float(threshold) if threshold else 0.3
    
    print(f"\n🔮 Executando inferência...")
    print(f"   Modelo: {model_dir}")
    print(f"   Entrada: {input_dir}")
    print(f"   Saída: {output_dir}")
    print(f"   Threshold: {threshold}")
    print("="*70)
    
    try:
        cmd = [
            sys.executable, "src/infer_images.py",
            "--model_dir", model_dir,
            "--input_dir", input_dir,
            "--out_dir", output_dir,
            "--score_threshold", str(threshold),
            "--dataset_dir", "dataset"  # Para obter nomes de categorias
        ]
        
        subprocess.run(cmd, check=False)
        
        print("\n✅ Inferência concluída!")
        print(f"   Resultados salvos em: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erro durante inferência: {e}")
    
    input("\nPressione Enter para voltar...")

def check_ready():
    """Verifica se está pronto para treinar."""
    clear_screen()
    print_header()
    
    try:
        subprocess.run([sys.executable, "scripts/check_ready.py"], check=False)
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\nPressione Enter para voltar...")

def setup_env():
    """Configura ambiente."""
    clear_screen()
    print_header()
    print("⚙️  CONFIGURAR AMBIENTE (.env)")
    print("-"*70)
    print()
    
    try:
        subprocess.run([sys.executable, "scripts/setup_env.py"], check=False)
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\nPressione Enter para voltar...")

def main():
    """Loop principal da interface."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        escolha = input("\nEscolha uma opção: ").strip()
        
        if escolha == "0":
            print("\n👋 Até logo!")
            break
        elif escolha == "1":
            download_dataset()
        elif escolha == "2":
            check_dataset()
        elif escolha == "3":
            train_model()
        elif escolha == "4":
            evaluate_model()
        elif escolha == "5":
            run_inference()
        elif escolha == "6":
            check_ready()
        elif escolha == "7":
            setup_env()
        else:
            print("\n❌ Opção inválida!")
            input("Pressione Enter para continuar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interface encerrada.")
        sys.exit(0)

