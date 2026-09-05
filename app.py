import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("軽量画像生成アプリ (1GBメモリ対応)")

@st.cache_resource
def load_pipeline():
    # 1GB制限に対応した超軽量モデル（Hugging Face上に実在）
    model_id = "segmind/tiny-sd"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 低メモリモードで読み込み
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    return pipe

prompt = st.text_input("プロンプト", "a cat sitting on a chair")

if st.button("生成開始"):
    with st.spinner("モデルを準備中...（初回のみダウンロード）"):
        try:
            pipe = load_pipeline()
            with st.spinner("画像生成中..."):
                # メモリ消費を抑えるためステップ数を少なめに設定
                image = pipe(prompt, num_inference_steps=15).images[0]
                st.image(image, caption="生成完了", use_container_width=False)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
