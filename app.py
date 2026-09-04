import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("BK-SDM-Tiny")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-tiny", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    pipe.enable_sequential_cpu_offload()
    return pipe

try:
    pipe = load_pipeline()
    prompt = st.text_input("Prompt (English)", "a photo of a cute cat")

    if st.button("Generate"):
        with st.spinner("Processing..."):
            image = pipe(
                prompt,
                height=256,
                width=256,
                num_inference_steps=1
            ).images[0]
            
            st.image(image, caption="Result")

except Exception as e:
    st.error(f"Error: {e}")
