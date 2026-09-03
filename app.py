import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("256x256 Lightweight Image Generator")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-tiny", 
        torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline()

prompt = st.text_input("Prompt (English)", "a cute dog")

if st.button("Generate"):
    with st.spinner("Generating..."):
        image = pipe(
            prompt, 
            height=256, 
            width=256, 
            num_inference_steps=6
        ).images[0]
        
        st.image(image, caption="Result (256x256)")
