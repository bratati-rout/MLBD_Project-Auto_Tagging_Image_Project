import streamlit as st
import torch
from PIL import Image
from model import CaptioningModel
from vocab import Vocabulary
import pickle
import os

# Fix torch.classes path issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]

# === Configuration ===
MODEL_PATH = "E:/MLBD Project/models/caption_model.pth"
VOCAB_FILE = "E:/MLBD Project/data/vocab.pkl"
MAX_LENGTH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Resources ===
@st.cache_resource
def load_resources():
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    
    model = CaptioningModel(
        embedding_dim=1536, 
        hidden_dim=512,
        vocab_size=len(vocab)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, vocab

model, vocab = load_resources()

# === Caption Generation ===
def generate_caption(image_embed):
    """Generate caption for a single image embedding."""
    caption = []
    with torch.no_grad():
        # Prepare initial inputs
        image_embed = image_embed.float().unsqueeze(0)  # Add batch dimension [1, 1536]
        start_token = torch.tensor([[vocab.word2idx['<start>']]], 
                                  dtype=torch.long, 
                                  device=DEVICE)
        
        # First forward pass with image and start token
        outputs, hidden = model(img_embeds=image_embed, captions=start_token)
        _, pred = outputs.max(-1)
        word_idx = pred.item()
        
        if word_idx != vocab.word2idx['<end>']:
            caption.append(vocab.idx2word[word_idx])
        
        # Subsequent steps
        x = pred
        for _ in range(MAX_LENGTH - 1):
            outputs, hidden = model(captions=x, hidden=hidden)
            _, pred = outputs.max(-1)
            word_idx = pred.item()
            
            if word_idx == vocab.word2idx['<end>']:
                break
                
            caption.append(vocab.idx2word[word_idx])
            x = pred
    
    return ' '.join(caption)

# === Streamlit UI ===
st.title("Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Generate Caption"):
        # Temporary: Random features (replace with actual EfficientNet extraction)
        image_embed = torch.randn(1536).to(DEVICE).float()
        
        # Generate and display caption
        caption = generate_caption(image_embed)
        st.success(f"**Generated Caption:** {caption}")
