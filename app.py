from flask import Flask, render_template, request, jsonify
import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load MultiLabelBinarizer
print("Loading tag binarizer...")
try:
    with open("E:/MLBD Project/models/mlb_mlp.pkl", "rb") as f:
        mlb = pickle.load(f)
    print(f"Loaded binarizer with {len(mlb.classes_)} tag classes")
except Exception as e:
    print(f"Warning: Could not load binarizer: {e}")
    tags_df = pd.read_csv("E:/MLBD Project/data/predicted_tags.csv")
    all_tags = []
    for tags_str in tags_df['tags'].dropna():
        if isinstance(tags_str, str):
            all_tags.extend([t.strip() for t in tags_str.split(',')])
    unique_tags = list(set(all_tags))
    mlb = unique_tags
    print(f"Created fallback tag list with {len(unique_tags)} tags")

# Load MLP model
print("Loading trained MLP model...")
try:
    input_dim = 1536
    hidden_dim = 512
    output_dim = len(mlb.classes_) if hasattr(mlb, 'classes_') else len(mlb)
    
    mlp_model = MLP(input_dim, hidden_dim, output_dim)
    mlp_model.load_state_dict(torch.load("E:/MLBD Project/models/mlp_model.pt", map_location='cpu'))
    mlp_model.eval()
    print("MLP model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load MLP model: {e}")
    mlp_model = None

# Load EfficientNet model for feature extraction
print("Loading EfficientNet model...")
try:
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = torch.nn.Identity()
    model.eval()
    print("EfficientNet model loaded successfully")
except Exception as e:
    print(f"Error loading EfficientNet: {e}")
    model = None

# Define image transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            if model is not None:
                features = model.extract_features(image_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze().numpy()
                print(f"Feature shape: {features.shape}")

                if mlp_model is not None:
                    predicted_tags = predict_tags(features)
                else:
                    predicted_tags = predict_tags_fallback(features)
            else:
                predicted_tags = ["model", "loading", "failed"]

        return jsonify({
            'filename': file.filename,
            'filepath': '/' + filepath.replace('\\', '/'),
            'tags': predicted_tags
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({
            'error': str(e),
            'filename': file.filename,
            'tags': []
        })


def predict_tags(features):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = mlp_model(x).squeeze().numpy()
    threshold = 0.2  # Lowered threshold
    tags = [mlb.classes_[i] for i in np.where(out > threshold)[0]]
    if not tags:
        top_indices = out.argsort()[-5:][::-1]
        tags = [mlb.classes_[i] for i in top_indices]
    return tags


def predict_tags_fallback(features):
    """Fallback prediction using heuristics"""
    common_tags = ["mountain", "snow", "sky", "man", "water", "dog", "beach", 
                   "grass", "woman", "jumping", "running", "black", "white", "blue"]
    result_tags = []

    blue_sum = np.sum(features[100:200])
    if blue_sum > 0.5:
        result_tags.append("blue")
        result_tags.append("sky")

    green_sum = np.sum(features[300:400])
    if green_sum > 0.4:
        result_tags.append("grass")

    if np.linalg.norm(features) > 20:
        result_tags.append("mountain")

    if len(result_tags) < 3:
        import random
        remaining = 3 - len(result_tags)
        result_tags.extend(random.sample([t for t in common_tags if t not in result_tags], remaining))

    return result_tags

if __name__ == '__main__':
    app.run(debug=True)
