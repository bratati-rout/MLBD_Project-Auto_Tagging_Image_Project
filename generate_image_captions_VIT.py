import os
from PIL import Image
from tqdm import tqdm
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Configurations
FOLDER_PATH = "E:/MLBD Project/data/Images"  # Path to images
OUTPUT_FILE = "E:/MLBD Project/data/vit_captions.txt"  # Output file

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model, processor, tokenizer
print("Loading ViT-GPT2 model...")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
print("Model loaded successfully.")

# Set model generation parameters
model.config.max_length = 20
model.config.num_beams = 4
model.config.early_stopping = True

# Function to generate caption from an image
def generate_caption(image: Image.Image):
    try:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(
            pixel_values,
            max_length=20,
            num_beams=4,
            early_stopping=True
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        return f"Error: {str(e)}"


# Process folder
captions = {}
image_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

print(f"Found {len(image_files)} images. Generating captions...")

for img_file in tqdm(image_files):
    img_path = os.path.join(FOLDER_PATH, img_file)
    try:
        image = Image.open(img_path).convert("RGB")
        caption = generate_caption(image)
    except Exception as e:
        caption = f"Error: {str(e)}"
    captions[img_file] = caption

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for filename, caption in captions.items():
        f.write(f"{filename}: {caption}\n")

print(f"Captions saved to {OUTPUT_FILE}")
