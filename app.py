import streamlit as st
import torch
from diffusers import DiffusionPipeline  # またはGAN専用軽量パイプライン

st.title("Text-to-Image GAN / Lightweight Generation")

@st.cache_resource
def load_model():
    # Hugging Face上に重み(.pth / .safetensors)が確実に存在する軽量GAN/軽量モデル
    model_id = "praxis/galip-coco"  # または "clip-gan" 構成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 重みを自動取得してロード
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    return pipe

prompt = st.text_input("プロンプト", "a cat sitting on a chair")

if st.button("生成開始"):
    with st.spinner("重みを読み込んで生成中..."):
        pipe = load_model()
        image = pipe(prompt).images[0]
        st.image(image, caption="生成完了")
