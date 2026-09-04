import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.fc_gamma = nn.Linear(cond_dim, num_features)
        self.fc_beta = nn.Linear(cond_dim, num_features)

    def forward(self, x, c):
        gamma = self.fc_gamma(c).unsqueeze(2).unsqueeze(3)
        beta = self.fc_beta(c).unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta

class DFBBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.aff1 = Affine(cond_dim, in_ch)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.aff2 = Affine(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, c):
        h = self.conv1(self.relu(self.aff1(x, c)))
        h = self.conv2(self.relu(self.aff2(h, c)))
        return h + self.shortcut(x)

class DFGANGenerator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=384, ngf=64):
        super().__init__()
        self.fc = nn.Linear(noise_dim + cond_dim, ngf * 8 * 4 * 4)
        self.ngf = ngf

        self.block0 = DFBBlock(ngf * 8, ngf * 8, cond_dim)
        self.block1 = DFBBlock(ngf * 8, ngf * 4, cond_dim)
        self.block2 = DFBBlock(ngf * 4, ngf * 2, cond_dim)
        self.block3 = DFBBlock(ngf * 2, ngf * 1, cond_dim)
        self.block4 = DFBBlock(ngf * 1, ngf // 2, cond_dim)
        self.block5 = DFBBlock(ngf // 2, ngf // 4, cond_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(ngf // 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf // 4, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, cond):
        inp = torch.cat([z, cond], dim=1)
        out = self.fc(inp).view(-1, self.ngf * 8, 4, 4)

        out = self.upsample(self.block0(out, cond))
        out = self.upsample(self.block1(out, cond))
        out = self.upsample(self.block2(out, cond))
        out = self.upsample(self.block3(out, cond))
        out = self.upsample(self.block4(out, cond))
        out = self.block5(out, cond)

        img = self.to_rgb(out)
        return img

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    generator = DFGANGenerator(noise_dim=100, cond_dim=384).to(device)
    generator.eval()
    return device, text_encoder, generator

st.title("DF-GANベース画像生成")

prompt = st.text_input("プロンプト (例: a red bird on a branch)", "a bird")

if st.button("生成開始"):
    with st.spinner("モデルを読み込み中..."):
        device, text_encoder, generator = load_pipeline()
    
    with st.spinner("STEP 1: ベース画像を生成中..."):
        with torch.no_grad():
            text_embedding = text_encoder.encode([prompt], convert_to_tensor=True).to(device)
            noise = torch.randn(1, 100, device=device)
            fake_tensor = generator(noise, text_embedding)

            fake_tensor = (fake_tensor.squeeze(0).clamp(-1, 1) + 1) / 2.0
            img_np = (fake_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            base_img = Image.fromarray(img_np)

        st.image(base_img, caption="STEP 1 完了 (256x256px)", use_container_width=False)
