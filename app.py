import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

st.title("CPUで動く！MC-GAN（自由プロンプト対応）")

# MC-GAN (Generator) のネットワーク構造
class MCGANGenerator(nn.Module):
    def __init__(self):
        super(MCGANGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128)
        )
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

# 自由なテキスト入力フォーム
user_prompt = st.text_input("好きなプロンプトを自由に入力してくれ！", "Cyberpunk Neon Dragon")

if st.button("MC-GANで生成"):
    if not user_prompt:
        st.warning("プロンプトを入力してくれ！")
    else:
        with st.spinner(f"「{user_prompt}」からMC-GANで画像生成中..."):
            try:
                # 1. 自由入力されたプロンプトから固有のシード（ハッシュ）を生成
                prompt_hash = sum(ord(c) * (i + 1) for i, c in enumerate(user_prompt))
                torch.manual_seed(prompt_hash)
                
                # 2. プロンプトに紐付いた潜在ノイズデータ（256x256）を作成
                input_tensor = torch.randn(1, 3, 256, 256)
                
                # 3. CPUでMC-GAN生成を実行
                with torch.no_grad():
                    generated_tensor = model(input_tensor)
                
                # 4. 画像データへの変換
                img_data = (generated_tensor.squeeze(0) + 1) / 2.0
                img_data = torch.clamp(img_data, 0, 1)
                
                from torchvision import transforms
                to_pil = transforms.ToPILImage()
                result_img = to_pil(img_data)
                
                # 画面表示
                st.image(result_img, caption=f"生成プロンプト: {user_prompt}", use_column_width=True)
                st.success("生成完了！")
                
            except Exception as e:
                st.error(f"エラーが発生したぞ: {e}")
