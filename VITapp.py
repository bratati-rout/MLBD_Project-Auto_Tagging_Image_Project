
import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Title & Styling
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>Image Caption Generator using ViT-GPT2</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Generate natural language descriptions for any image using the ViT-GPT2 model!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return processor, tokenizer, model

def generate_caption(image: Image.Image):
    processor, tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare input
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate caption with improved settings
    output_ids = model.generate(
        pixel_values,
        max_length=64,
        num_beams=5,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )

    # Decode output
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# File uploader section
uploaded_file = st.file_uploader(" Upload an image", type=["png", "jpg", "jpeg"])

# If file uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption... Please wait."):
            caption = generate_caption(image)
            st.success(" Caption Generated!")
            st.markdown(f"<h3 style='text-align: center;'>{caption}</h3>", unsafe_allow_html=True)
else:
    st.info("Please upload an image to get started.")

# Developer Credits
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
