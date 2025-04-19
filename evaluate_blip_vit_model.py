import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm
import evaluate
import pandas as pd
import random

# Set CPU device (since your PyTorch doesn't support CUDA)
device = torch.device("cpu")

# Load BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

# Load ViT-GPT2 model
vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vit_model.eval()

# Load image paths and reference captions
image_dir = "E:/MLBD Project/data/Images"
captions_path = "E:/MLBD Project/data/captions.txt"

# Read captions from CSV file and create a dictionary
reference_dict = {}
with open(captions_path, "r") as f:
    for line in f:
        # Skip header or empty lines
        if line.strip() == "" or line.startswith("image,caption"):
            continue
        img_name, caption = line.strip().split(',', 1)  # Split by comma, limit to 1 split
        if img_name in reference_dict:
            reference_dict[img_name].append(caption)
        else:
            reference_dict[img_name] = [caption]

# Use a subset of images for faster evaluation
sample_keys = random.sample(list(reference_dict.keys()), 300)  # Adjust number as needed

# Caption generation functions
def generate_blip_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

def generate_vit_caption(image: Image.Image):
    pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = vit_model.generate(
        pixel_values,
        max_length=64,
        num_beams=5,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )
    caption = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# Evaluation loop
vit_preds, blip_preds, all_refs = [], [], []

for img_name in tqdm(sample_keys, desc="Evaluating on subset"):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        continue
    image = Image.open(img_path).convert("RGB")

    vit_caption = generate_vit_caption(image)
    blip_caption = generate_blip_caption(image)

    vit_preds.append(vit_caption)
    blip_preds.append(blip_caption)
    all_refs.append(reference_dict[img_name])

# Evaluate with BLEU
bleu = evaluate.load("bleu")

vit_bleu_score = bleu.compute(predictions=vit_preds, references=all_refs)
blip_bleu_score = bleu.compute(predictions=blip_preds, references=all_refs)

print("\nEvaluation on Subset of 300 Images")
print("===================================")
print(f"ViT-GPT2 BLEU Score: {vit_bleu_score}")
print(f"BLIP BLEU Score: {blip_bleu_score}")
