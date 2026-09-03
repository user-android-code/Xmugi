import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

# ページ設定
st.set_page_config(page_title="超軽量・画像生成AI", page_icon="🎨")
st.title("🎨 超軽量・画像生成アプリ")
st.write("プロンプトを入力して、画像を作ってみよう！")

# 超軽量モデルの読み込み
@st.cache_resource
def load_lightweight_model():
    # 超小型のStable Diffusionモデル (サイズがめちゃくちゃ小さい)
    model_id = "lambdalabs/miniSD-diffusers"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    with st.spinner("軽量モデルを準備中..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)
        return pipe, device

pipe, device = load_lightweight_model()

st.sidebar.header("設定")
st.sidebar.write(f"実行デバイス: **{device.upper()}**")

# 設定スライダー
steps = st.sidebar.slider("生成ステップ数", 10, 30, 15)

# プロンプト入力画面
prompt = st.text_input(
    "どんな画像を作る？（英語がおすすめ）",
    value="a tiny cute robot drinking coffee, digital art",
)

# 生成ボタン
if st.button("画像を生成する！", type="primary"):
    if not prompt:
        st.warning("プロンプトを入力してね！")
    else:
        with st.spinner("画像を生成中...⚡️"):
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
            ).images[0]

            st.image(image, caption=prompt, use_column_width=True)
            st.success("できたよ！")
