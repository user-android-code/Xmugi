import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

st.title("1GB CPU Text-to-Image (GAN)")

torch.set_num_threads(1)

@st.cache_resource
def load_hf_gan_model():
    model_id = "sail-sg/AttnGAN-birds"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

try:
    with st.spinner("Loading model..."):
        tokenizer, model = load_hf_gan_model()
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Error: {e}")

prompt = st.text_input("Prompt", "a small yellow bird")

if st.button("Generate"):
    with st.spinner("Processing..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_tensor = model.generate(**inputs)
            
        img_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        st.image(img, caption=prompt, width=256)
