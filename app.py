import streamlit as st
import torch
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    save_as_images,
    one_hot_from_names,
)
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

st.title("自由プロンプト対応 GAN画像生成")

@st.cache_resource
def load_models():
    # CLIP（文章と画像を関連付けるモデル）とBigGANをロード
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    biggan_model = BigGAN.from_pretrained('biggan-deep-256')
    return clip_model, clip_processor, biggan_model

with st.spinner("モデルを準備中..."):
    clip_model, clip_processor, biggan_model = load_models()

# ユーザー入力（自由な日本語・英語テキスト）
prompt_text = st.text_input("どんな画像を作りたい？（自由に入力OK）", "赤いスポーツカー")

if st.button("画像を生成"):
    with st.spinner("プロンプトを解析して画像を生成中..."):
        try:
            # 1. BigGANが理解できる代表的なカテゴリ候補
            candidate_classes = [
                "sports car", "golden retriever", "cat", "castle", "beach",
                "pizza", "flower", "mountain", "airplane", "robot",
                "coffee", "forest", "guitar", "space", "house"
            ]

            # 2. CLIPを使って、入力プロンプトに一番近いカテゴリを自動判定
            inputs = clip_processor(
                text=[f"a photo of {c}" for c in candidate_classes],
                images=None,
                return_tensors="pt",
                padding=True
            )
            
            # 入力文と候補の類似度を計算
            text_inputs = clip_processor(text=[prompt_text], return_tensors="pt", padding=True)
            text_embeds = clip_model.get_text_features(**text_inputs)
            cand_inputs = clip_processor(text=[f"a photo of {c}" for c in candidate_classes], return_tensors="pt", padding=True)
            cand_embeds = clip_model.get_text_features(**cand_inputs)

            # 一番近いカテゴリを選択
            similarity = torch.matmul(text_embeds, cand_embeds.T)
            best_idx = similarity.argmax().item()
            matched_class = candidate_classes[best_idx]

            st.info(f"解析結果: プロンプトから「**{matched_class}**」の要素を検出したぞ！")

            # 3. 検出したカテゴリでGAN画像を生成
            class_vector = one_hot_from_names([matched_class], batch_size=1)
            noise_vector = truncated_noise_sample(batch_size=1, truncation=0.4)
            
            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector)

            with torch.no_grad():
                output = biggan_model(noise_vector, class_vector, 0.4)

            # 4. 画像保存と表示
            file_prefix = "output_gen"
            save_as_images(output, origin_class=[file_prefix])
            
            img_path = f"{file_prefix}_0.png"
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption=f"生成結果 (概念: {matched_class})", use_column_width=True)
                st.success("生成完了！")

        except Exception as e:
            st.error(f"エラーが発生したぞ: {e}")
