import sys
import subprocess

# 1. 必要なライブラリを自動インストールする関数
def install_packages():
    packages = ["torch", "torchvision", "pytorch-pretrained-biggan", "Pillow"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"'{package}' が見つからないから自動インストール中...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 自動インストール実行
install_packages()

# ライブラリの読み込み
import torch
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    save_as_images,
    one_hot_from_names,
)

def generate_image(prompt_text):
    print(f"\nプロンプト: '{prompt_text}' で画像を生成中...")
    
    # 256x256用の軽量モデル（約300MB）をロード
    model = BigGAN.from_pretrained('biggan-deep-256')

    try:
        # 単語からカテゴリベクトルを作成
        class_vector = one_hot_from_names([prompt_text], batch_size=1)
    except Exception:
        print(f"エラー: '{prompt_text}' は認識できなかったぞ。")
        print("一般的な英単語（例: 'sports car', 'golden retriever', 'pizza'）で試してくれ！")
        return

    # ノイズベクトルの作成と変換
    noise_vector = truncated_noise_sample(batch_size=1, truncation=0.4)
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # CPUで推論（画像生成）
    with torch.no_grad():
        output = model(noise_vector, class_vector, 0.4)

    # 画像の保存
    file_name = f"output_{prompt_text.replace(' ', '_')}"
    save_as_images(output, origin_class=[file_name])
    print(f"生成完了！ '{file_name}_0.png' として保存されたぞ！")

if __name__ == "__main__":
    # 好きな英語の単語をここに入れて実行！
    my_prompt = "sports car"
    generate_image(my_prompt)
