import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# ページ設定
st.set_page_config(page_title="MobileDiffusion", page_icon="⚡️")
st.title("⚡️ MobileDiffusion (ikozlov/MobileDiffusion)")

@st.cache_resource
def load_mobile_diffusion():
    model_id = "ikozlov/MobileDiffusion"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    with st.spinner("ikozlov/MobileDiffusion モデルをロード中..."):
        try:
            # UNet部分の読み込みでエラーになるのを防ぐため、
            # 標準的なSD1.5のコンポーネントをベースにリポジトリの重みを適用
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
            pipe = pipe.to(device)
            return pipe, device
        except Exception as e:
            st.error(f"モデルの読み込みエラー: {e}")
            return None, device

pipe, device = load_mobile_diffusion()

if pipe is None:
    st.stop()

st.sidebar.header("設定")
st.sidebar.write(f"実行デバイス: **{device.upper()}**")

steps = st.sidebar.slider("推論ステップ数", 1, 30, 8)
guidance_scale = st.sidebar.slider("プロンプト忠実度", 1.0, 20.0, 7.5)

prompt = st.text_area(
    "プロンプト (英語)",
    value="a cute cat sitting in a sunny garden, digital art",
)

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
            st.image(generated_image, caption=f"Prompt: {prompt}", use_column_width=True)
            st.success("できた！")
