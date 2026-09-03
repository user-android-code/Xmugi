import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="超高速画像生成App", page_icon="⚡️")
st.title("⚡️ 爆速・軽量画像生成App")

@st.cache_resource
def load_model():
    # 極小・最速クラスのモデル (segmind/tiny-sd)
    model_id = "segmind/tiny-sd"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# ステップ数を少なく指定できるようにする（CPU最速設定）
steps = st.sidebar.slider("ステップ数 (少ないほど速い)", 1, 15, 6)

prompt = st.text_input("プロンプト (英語)", "a cute cat")

if st.button("生成"):
    if prompt:
        with st.spinner("生成中..."):
            image = pipe(
                prompt=prompt, 
                num_inference_steps=steps
            ).images[0]
            st.image(image, caption=prompt, use_column_width=True)
