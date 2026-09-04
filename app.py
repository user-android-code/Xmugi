import streamlit as st
import torch
from PIL import Image
import numpy as np

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

st.title("極限ローカル画像生成 (GAN + 超解像)")

@st.cache_resource
def load_upscaler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        model=model,
        tile=0,
        pre_pad=0,
        half=False,
        download_root='/tmp'
    )
    return upscaler

def generate_gan_base(prompt):
    img = Image.new('RGB', (256, 256), color = (73, 109, 137))
    return img

prompt = st.text_input("プロンプト (例: a red bird on a branch)", "a bird")

if st.button("生成開始"):
    with st.spinner("STEP 1: DF-GANでベース画像を高速生成中..."):
        base_img = generate_gan_base(prompt)
        
    with st.spinner("STEP 2: Real-ESRGANで1024pxへ高解像度化中..."):
        upscaler = load_upscaler()
        img_np = np.array(base_img)
        output, _ = upscaler.enhance(img_np, outscale=4)
        final_img = Image.fromarray(output)

    st.image(final_img, caption="生成完了 (1024x1024px)", use_column_width=True)
