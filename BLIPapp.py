import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Page Configuration
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and Description
st.markdown("""
# Image Caption Generator using BLIP  
Generate natural language descriptions for any image using the BLIP model!
""")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(" Upload an image", type=["png", "jpg", "jpeg"])

@st.cache_resource(show_spinner=" Loading BLIP model...")
def load_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f" Failed to load model: {e}")
        return None, None

def generate_caption(image: Image.Image):
    processor, model = load_model()

    if processor is None or model is None:
        return " Model loading failed."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Display image and caption output
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_container_width=True)

    if st.button(" Generate Caption"):
        with st.spinner(" Generating caption... please wait"):
            try:
                caption = generate_caption(image)
                st.success(" Caption Generated!")
                st.markdown(f"###  **{caption}**")
            except Exception as e:
                st.error(f" Error: {e}")
else:
    st.info(" Please upload an image to get started.")

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
