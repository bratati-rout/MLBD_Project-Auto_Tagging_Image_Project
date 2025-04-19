import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration
)

# Page Configuration
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>Image Caption Generator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Choose a model and generate image captions effortlessly!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Model Selector
model_choice = st.selectbox("Choose a model", ["ViT-GPT2", "BLIP"], index=0)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Load models with caching
@st.cache_resource(show_spinner="Loading ViT-GPT2 model...")
def load_vit_model():
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return processor, tokenizer, model

@st.cache_resource(show_spinner="Loading BLIP model...")
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Caption generation functions
def generate_vit_caption(image: Image.Image):
    processor, tokenizer, model = load_vit_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        max_length=64,
        num_beams=5,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def generate_blip_caption(image: Image.Image):
    processor, model = load_blip_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Main Logic
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption... Please wait."):
            try:
                if model_choice == "ViT-GPT2":
                    caption = generate_vit_caption(image)
                else:
                    caption = generate_blip_caption(image)

                st.success("Caption Generated!")
                st.markdown(f"<h3 style='text-align: center;'>{caption}</h3>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating caption: {e}")
else:
    st.info("Please upload an image to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        Developed by <strong>Bratati Rout [M24DE3032]</strong>, 
        <strong>Sharone Verma [M24DE3072]</strong>, and 
        <strong>Yash Engendala [M24DE3086]</strong>  
    </div>
    """,
    unsafe_allow_html=True
)
