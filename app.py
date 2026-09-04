import streamlit as st
from PIL import Image, ImageDraw

st.title("GANベース画像生成（テストモード）")

def generate_gan_base(prompt):
    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    text = f"BASE GAN:\n{prompt}"
    draw.text((20, 100), text, fill=(255, 255, 255))
    return img

prompt = st.text_input("プロンプト (例: a red bird on a branch)", "a bird")

if st.button("生成開始"):
    with st.spinner("STEP 1: ベース画像を生成中..."):
        base_img = generate_gan_base(prompt)
        st.image(base_img, caption="STEP 1 完了 (256x256px)", use_column_width=False)
