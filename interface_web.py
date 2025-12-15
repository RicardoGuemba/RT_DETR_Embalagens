#!/usr/bin/env python3
"""
Interface Web usando Streamlit para gerenciar treinamento, predição e download.
"""

import streamlit as st
import subprocess
import sys
from pathlib import Path
import json

# Configuração da página
st.set_page_config(
    page_title="RT-DETR - Interface de Gerenciamento",
    page_icon="🚀",
    layout="wide"
)

def check_file_exists(path):
    """Verifica se arquivo existe."""
    return Path(path).exists()

def get_dataset_info():
    """Obtém informações do dataset."""
    info = {
        "train": {"exists": False, "images": 0, "annotations": 0},
        "valid": {"exists": False, "images": 0, "annotations": 0},
        "test": {"exists": False, "images": 0, "annotations": 0}
    }
    
    for split in ["train", "valid", "test"]:
        json_path = Path(f"dataset/{split}/_annotations.coco.json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                info[split] = {
                    "exists": True,
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                    "categories": len(data.get("categories", []))
                }
            except:
                pass
    
    return info

def main():
    st.title("🚀 RT-DETR - Interface de Gerenciamento")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Menu")
    page = st.sidebar.radio(
        "Navegação",
        ["🏠 Início", "📥 Download Dataset", "🏋️ Treinar Modelo", "📊 Avaliar Modelo", "🔮 Inferência", "⚙️ Configurações"]
    )
    
    # Página Início
    if page == "🏠 Início":
        st.header("Bem-vindo!")
        st.markdown("""
        Esta interface permite gerenciar todo o ciclo de vida do modelo RT-DETR:
        - **Download** de datasets do Roboflow
        - **Treinamento** de modelos
        - **Avaliação** de modelos treinados
        - **Inferência** em novas imagens
        """)
        
        st.markdown("---")
        st.subheader("📊 Status do Sistema")
        
        # Verificar dataset
        dataset_info = get_dataset_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Train", f"{dataset_info['train']['images']} imagens" if dataset_info['train']['exists'] else "Não encontrado")
        
        with col2:
            st.metric("Valid", f"{dataset_info['valid']['images']} imagens" if dataset_info['valid']['exists'] else "Não encontrado")
        
        with col3:
            st.metric("Test", f"{dataset_info['test']['images']} imagens" if dataset_info['test']['exists'] else "Não encontrado")
        
        # Verificar modelos
        st.markdown("---")
        st.subheader("🤖 Modelos Treinados")
        
        model_best = Path("runs_rtdetr/model_best")
        model_final = Path("runs_rtdetr/model_final")
        
        if model_best.exists():
            st.success("✅ model_best disponível")
        else:
            st.info("ℹ️ model_best não encontrado")
        
        if model_final.exists():
            st.success("✅ model_final disponível")
        else:
            st.info("ℹ️ model_final não encontrado")
    
    # Página Download
    elif page == "📥 Download Dataset":
        st.header("📥 Baixar Dataset do Roboflow")
        
        if not check_file_exists(".env"):
            st.error("❌ Arquivo .env não encontrado! Configure primeiro na página de Configurações.")
        else:
            # Carregar versão atual do .env
            import os
            from dotenv import load_dotenv
            load_dotenv()
            current_version = int(os.getenv("ROBOFLOW_VERSION", "2"))
            
            st.info("""
            ⚠️ **ATENÇÃO**: Isso vai sobrescrever o dataset atual!
            
            Certifique-se de que você já adicionou as anotações necessárias no Roboflow
            e que os splits (train/valid/test) estão configurados corretamente.
            """)
            
            # Seleção de versão
            st.subheader("Versão do Dataset")
            version = st.number_input(
                "Versão do dataset no Roboflow",
                min_value=1,
                max_value=100,
                value=current_version,
                help=f"Versão atual no .env: {current_version}"
            )
            
            st.write(f"📦 Versão selecionada: **{version}**")
            
            if st.button("📥 Baixar Dataset", type="primary"):
                with st.spinner(f"Baixando dataset versão {version} do Roboflow..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "scripts/download_roboflow_coco.py", 
                             "--dataset_dir", "dataset", "--version", str(version)],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Dataset baixado com sucesso!")
                            st.code(result.stdout)
                            
                            # Mostrar informações do dataset
                            dataset_info = get_dataset_info()
                            st.json(dataset_info)
                        else:
                            st.error(f"❌ Erro ao baixar dataset:\n{result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Erro: {e}")
    
    # Página Treinar
    elif page == "🏋️ Treinar Modelo":
        st.header("🏋️ Treinar Modelo RT-DETR")
        
        train_json = Path("dataset/train/_annotations.coco.json")
        if not train_json.exists():
            st.error("❌ Dataset não encontrado! Baixe o dataset primeiro.")
        else:
            st.sidebar.subheader("Parâmetros de Treinamento")
            
            epochs = st.sidebar.number_input("Épocas", min_value=1, max_value=1000, value=50)
            batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=16, value=1)
            img_size = st.sidebar.selectbox("Tamanho da Imagem", [640, 832, 960], index=0)
            learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, value=1e-5, format="%e")
            gradient_accum = st.sidebar.number_input("Gradient Accumulation Steps", min_value=1, max_value=32, value=4)
            
            st.info(f"""
            **Parâmetros configurados:**
            - Épocas: {epochs}
            - Batch Size: {batch_size}
            - Tamanho da Imagem: {img_size}
            - Learning Rate: {learning_rate}
            - Gradient Accumulation: {gradient_accum}
            """)
            
            if st.button("🚀 Iniciar Treinamento", type="primary"):
                st.warning("⚠️ O treinamento pode levar bastante tempo. Não feche esta página!")
                
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
                
                with st.spinner("Treinando modelo..."):
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    output_container = st.empty()
                    output_lines = []
                    
                    for line in process.stdout:
                        output_lines.append(line)
                        if len(output_lines) > 100:
                            output_lines.pop(0)
                        output_container.code("\n".join(output_lines[-50:]))
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        st.success("✅ Treinamento concluído!")
                    else:
                        st.error("❌ Erro durante treinamento. Verifique os logs acima.")
    
    # Página Avaliar
    elif page == "📊 Avaliar Modelo":
        st.header("📊 Avaliar Modelo")
        
        model_best = Path("runs_rtdetr/model_best")
        model_final = Path("runs_rtdetr/model_final")
        
        if not model_best.exists() and not model_final.exists():
            st.error("❌ Nenhum modelo treinado encontrado!")
        else:
            model_choice = st.selectbox(
                "Escolha o modelo",
                [("model_best", model_best.exists()), ("model_final", model_final.exists())],
                format_func=lambda x: f"{x[0]} {'✅' if x[1] else '❌'}"
            )
            
            split_choice = st.selectbox("Split para avaliação", ["valid", "test", "train"])
            
            if st.button("📊 Avaliar", type="primary"):
                model_dir = f"runs_rtdetr/{model_choice[0]}"
                
                with st.spinner("Avaliando modelo..."):
                    result = subprocess.run(
                        [sys.executable, "src/eval_coco.py",
                         "--model_dir", model_dir,
                         "--dataset_dir", "dataset",
                         "--split", split_choice],
                        capture_output=True,
                        text=True
                    )
                    
                    st.code(result.stdout)
                    if result.returncode != 0:
                        st.error(result.stderr)
    
    # Página Inferência
    elif page == "🔮 Inferência":
        st.header("🔮 Fazer Predição/Inferência")
        
        model_best = Path("runs_rtdetr/model_best")
        model_final = Path("runs_rtdetr/model_final")
        
        if not model_best.exists() and not model_final.exists():
            st.error("❌ Nenhum modelo treinado encontrado!")
        else:
            model_choice = st.selectbox(
                "Escolha o modelo",
                [("model_best", model_best.exists()), ("model_final", model_final.exists())],
                format_func=lambda x: f"{x[0]} {'✅' if x[1] else '❌'}"
            )
            
            input_dir = st.text_input("Diretório de entrada", "dataset/test")
            output_dir = st.text_input("Diretório de saída", "runs_rtdetr/infer_out")
            threshold = st.slider("Score Threshold", 0.0, 1.0, 0.3, 0.05)
            
            if st.button("🔮 Executar Inferência", type="primary"):
                model_dir = f"runs_rtdetr/{model_choice[0]}"
                
                if not Path(input_dir).exists():
                    st.error(f"❌ Diretório não encontrado: {input_dir}")
                else:
                    with st.spinner("Executando inferência..."):
                        result = subprocess.run(
                            [sys.executable, "src/infer_images.py",
                             "--model_dir", model_dir,
                             "--input_dir", input_dir,
                             "--out_dir", output_dir,
                             "--score_threshold", str(threshold),
                             "--dataset_dir", "dataset"],
                            capture_output=True,
                            text=True
                        )
                        
                        st.code(result.stdout)
                        if result.returncode == 0:
                            st.success(f"✅ Inferência concluída! Resultados em: {output_dir}")
                        else:
                            st.error(f"❌ Erro: {result.stderr}")
    
    # Página Configurações
    elif page == "⚙️ Configurações":
        st.header("⚙️ Configurações")
        
        st.subheader("Variáveis de Ambiente (.env)")
        
        if check_file_exists(".env"):
            st.success("✅ Arquivo .env encontrado")
            
            with open(".env") as f:
                env_content = f.read()
            
            # Não mostrar API key completa
            lines = env_content.split("\n")
            masked_lines = []
            for line in lines:
                if "ROBOFLOW_API_KEY" in line and "=" in line:
                    key, value = line.split("=", 1)
                    if len(value) > 8:
                        masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
                    else:
                        masked_value = "*" * len(value)
                    masked_lines.append(f"{key}={masked_value}")
                else:
                    masked_lines.append(line)
            
            st.code("\n".join(masked_lines))
        else:
            st.warning("⚠️ Arquivo .env não encontrado")
        
        if st.button("🔧 Configurar .env"):
            with st.spinner("Abrindo configuração..."):
                result = subprocess.run(
                    [sys.executable, "scripts/setup_env.py"],
                    capture_output=True,
                    text=True
                )
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("✅ Configuração concluída!")
                    st.experimental_rerun()

if __name__ == "__main__":
    main()

