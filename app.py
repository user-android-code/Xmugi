import torch
import streamlit as st
from diffusers import DiffusionPipeline

# ページ設定
st.set_page_config(page_title="MobileDiffusion App", page_icon="⚡️")
st.title("⚡️ MobileDiffusion (ikozlov/MobileDiffusion)")

# モデルの読み込み（Hugging Face公式の読み込み手順に準拠）
@st.cache_resource
def load_mobile_diffusion():
    model_id = "ikozlov/MobileDiffusion"
    
    # GPU環境に合わせて dtype を設定（Macなら mps, CUDAなら cuda）
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    with st.spinner("MobileDiffusion モデルをロード中..."):
        try:
            # 公式記載の正しい読み込み方
            pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                dtype=dtype
            )
            pipe = pipe.to(device)
            return pipe, device
        except Exception as e:
            st.error(f"モデルの読み込みエラー: {e}")
            return None, device

# パイプラインをロード
pipe, device = load_mobile_diffusion()

if pipe is None:
    st.stop()

st.sidebar.header("設定")
st.sidebar.write(f"実行デバイス: **{device.upper()}**")

# パラメータ設定
steps = st.sidebar.slider("推論ステップ数", 1, 30, 8)
guidance_scale = st.sidebar.slider("プロンプト忠実度", 1.0, 20.0, 7.5)

# プロンプト入力
prompt = st.text_area(
    "プロンプト (英語)",
    value="Astronaut in a jungle, cold color palette, detailed, 8k",
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
