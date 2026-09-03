import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("512x512 Lightweight Image Generator")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-tiny", 
        torch_dtype=torch.float32,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline()

prompt = st.text_input("Prompt (English)", "a cute dog")

if st.button("Generate"):
    with st.spinner("Generating..."):
        image = pipe(
            prompt, 
            height=512, 
            width=512, 
            num_inference_steps=15
        ).images[0]
        
        st.image(image, caption="Result (512x512)")
