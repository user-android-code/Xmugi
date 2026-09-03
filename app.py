# pip: torch
# pip: diffusers
# pip: transformers
# pip: accelerate
# pip: pillow

import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

# ページ設定
st.set_page_config(page_title="Hugging Face MobileDiffusion", page_icon="⚡️")
st.title("⚡️ MobileDiffusion (ikozlov/MobileDiffusion)")

# モデルの読み込み（キャッシュ化）
@st.cache_resource
def load_mobile_diffusion():
    model_id = "ikozlov/MobileDiffusion"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    with st.spinner("ikozlov/MobileDiffusion モデルをロード中..."):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, safety_checker=None
            )
            pipe = pipe.to(device)
            return pipe, device
        except Exception as e:
            st.error(f"モデルの読み込みに失敗したよ: {e}")
            return None, device

# パイプラインをロード
pipe, device = load_mobile_diffusion()

if pipe is None:
    st.stop()

st.sidebar.header("設定")
st.sidebar.write(f"実行デバイス: **{device.upper()}**")

# パラメータ設定
steps = st.sidebar.slider("推論ステップ数", 1, 30, 8)
guidance_scale = st.sidebar.slider("プロンプト忠実度 (Guidance Scale)", 1.0, 20.0, 7.5)

# プロンプト入力
prompt = st.text_area(
    "生成したい画像のプロンプト (英語)",
    value="a portrait of a cyberpunk detective in a dark neon street, highly detailed digital painting",
)

# 生成ボタン
if st.button("生成する", type="primary"):
    if not prompt:
        st.warning("プロンプトを入力してね")
    else:
        with st.spinner("⚡️ MobileDiffusionで生成中..."):
            output = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            generated_image = output.images[0]
            st.image(
                generated_image,
                caption=f"Generated: '{prompt}' (Steps: {steps})",
                use_column_width=True,
            )
            st.success("できた！")
