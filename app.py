import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("Segmind Tiny-SD (256x256)")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", 
        torch_dtype=torch.float32,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline()

prompt = st.text_input("Prompt (English)", "a photo of a cute cat")

if st.button("Generate"):
    with st.spinner("Generating..."):
        image = pipe(
            prompt, 
            negative_prompt="ugly, deformed, disfigured, bad anatomy",
            height=256, 
            width=256, 
            num_inference_steps=15
        ).images[0]
        
        st.image(image, caption="Result (256x256)")
