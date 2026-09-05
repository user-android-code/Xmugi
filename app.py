import os
import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# DF-GAN Generator ネットワーク構造
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# パイプライン・自動重みロード
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    generator = DFGANGenerator(noise_dim=100, cond_dim=384).to(device)
    
    # torchvisionの公式学習済み重みをCPUで自動取得
    with st.spinner("公式ベース重みを自動取得中..."):
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pretrained_dict = base_model.state_dict()
        
        # DF-GANの各層にサイズが合う重みを移植
        gen_dict = generator.state_dict()
        for (k_g, v_g), (k_p, v_p) in zip(gen_dict.items(), pretrained_dict.items()):
            if v_g.shape == v_p.shape:
                gen_dict[k_g] = v_p
        
        generator.load_state_dict(gen_dict)
        st.success("重みの自動適用完了！")
            
    generator.eval()
    return device, text_encoder, generator

# ---------------------------------------------------------
# UI・推論実行部
# ---------------------------------------------------------
st.title("DF-GANベース画像生成 (COCO擬似重み適用)")

prompt = st.text_input("プロンプト (例: a person standing in a room)", "a cat")

if st.button("生成開始"):
    with st.spinner("モデルを準備中..."):
        device, text_encoder, generator = load_pipeline()
    
    with st.spinner("画像生成中..."):
        with torch.no_grad():
            text_embedding = text_encoder.encode([prompt], convert_to_tensor=True).to(device)
            noise = torch.randn(1, 100, device=device)
            fake_tensor = generator(noise, text_embedding)

            fake_tensor = (fake_tensor.squeeze(0).clamp(-1, 1) + 1) / 2.0
            img_np = (fake_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            base_img = Image.fromarray(img_np)

        st.image(base_img, caption="生成完了 (256x256px)", use_container_width=False)
