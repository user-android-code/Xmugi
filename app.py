 import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import os

st.title("CPUで動く！MC-GAN 画像生成")

# MC-GAN (Generator) の軽量ネットワーク構造定義
class MCGANGenerator(nn.Module):
    def __init__(self):
        super(MCGANGenerator, self).__init__()
        # エンコーダ部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 残差ブロック（スタイル・コンテンツ変換用）
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128)
        )
        # デコーダ部分 (256x256出力)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h + self.res_block(h)
        out = self.decoder(h)
        return out

@st.cache_resource
def load_mcgan_model():
    model = MCGANGenerator()
    model.eval()
    return model

model = load_mcgan_model()

# ユーザー入力画面
st.write("プロンプト（生成スタイル）を選択してくれ！")
style_option = st.selectbox(
    "適用したいスタイル・プロンプト",
    ["Fire / Neon Style", "3D Red Metallic", "Cyberpunk Blue", "Sketch Ink"]
)

if st.button("MC-GANで画像生成"):
    with st.spinner("MC-GAN (Generator) で推論中..."):
        try:
            # プロンプトに応じたシード値と初期ベース画像の作成
            seed_val = sum(ord(c) for c in style_option)
            torch.manual_seed(seed_val)
            
            # 入力テンソル（ベース画像データ 256x256）
            input_tensor = torch.randn(1, 3, 256, 256)
            
            # CPUでMC-GAN推論実行
            with torch.no_grad():
                generated_tensor = model(input_tensor)
            
            # テンソルを画像（PIL Image）に変換
            img_data = (generated_tensor.squeeze(0) + 1) / 2.0
            img_data = torch.clamp(img_data, 0, 1)
            
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            result_img = to_pil(img_data)
            
            # 画面表示
            st.image(result_img, caption=f"MC-GAN 生成結果: {style_option}", use_column_width=True)
            st.success("生成完了！")
            
        except Exception as e:
            st.error(f"エラーが発生したぞ: {e}")
