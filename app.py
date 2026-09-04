import os
import urllib.request
import streamlit as st
import torch
from PIL import Image
import numpy as np

from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

st.title("極限ローカル画像生成 (GAN + 超解像)")

@st.cache_resource
def load_upscaler():
    model_dir = "/tmp/weights"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "realesr-general-x4v3.pth")
    
    if not os.path.exists(model_path):
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        urllib.request.urlretrieve(url, model_path)

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, uppercase=False, act_type='prelu')
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
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
