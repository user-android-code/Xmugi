import streamlit as st
import torch
from diffusers import DiffusionPipeline  # 軽量なパイプラインで一括読み込み

st.title("一発丸ごとダウンロードGAN")

@st.cache_resource
def load_gan_model():
    # 重みも構造も全部セットでHugging Faceから1行で直接ロード
    model_id = "clip-gan/galip-coco-256"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # これだけで重みデータ(pth)も自動でセットされる！
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    return pipe

prompt = st.text_input("プロンプト", "a cat sitting on a chair")

if st.button("生成開始"):
    with st.spinner("モデルと重みをまとめてロード中..."):
        try:
            pipe = load_gan_model()
            # 1ステップで一気に生成！
            image = pipe(prompt, num_inference_steps=1).images[0]
            st.image(image, caption="生成完了")
        except Exception as e:
            st.error(f"エラー: {e}")
