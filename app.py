import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("軽量・高画質 AI画像生成 (256x256)")

@st.cache_resource
def load_pipeline():
    # Small-SDを1GBメモリ枠に安全にロード
    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/small-sd",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return pipe

prompt = st.text_input("プロンプト (英語):", "a detailed portrait of a fantasy anime girl, masterpiece, highly detailed")

if st.button("生成"):
    with st.spinner("生成中..."):
        pipe = load_pipeline()
        # 256x256固定でメモリ消費を最少に抑える
        image = pipe(prompt, height=256, width=256, num_inference_steps=20).images[0]
        st.image(image, caption="生成結果 (256x256)")
