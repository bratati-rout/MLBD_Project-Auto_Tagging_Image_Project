import torch
import pickle
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

print("Loading resources for tag prediction...")

# Load Enhanced Graph
graph_path = "E:/MLBD Project/data/enhanced_image_graph.pkl"
with open(graph_path, "rb") as f:
    G = pickle.load(f)
print(f"Loaded enhanced graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Load trained MultiLabelBinarizer
with open("E:/MLBD Project/models/multilabel_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)
print(f"Loaded MultiLabelBinarizer with {len(mlb.classes_)} classes")

# Get list of images
image_list = list(G.nodes())

# Load node features
try:
    X = np.array([G.nodes[img]['visual_feature'] for img in image_list], dtype=np.float32)
except KeyError:
    X = np.array([G.nodes[img]['feature'] for img in image_list], dtype=np.float32)
print(f"Feature matrix shape: {X.shape}")

# Prepare edge_index
node_id_map = {img: idx for idx, img in enumerate(image_list)}
edges = [(node_id_map[a], node_id_map[b]) for a, b in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

# Create Data object
data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index)

# Define GCN model (must match training architecture exactly)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.4):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=False)

        # Second GCN layer
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=False)

        # Output layer
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# Load model checkpoint
checkpoint_path = "E:/MLBD Project/models/gnn_model.pkl"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model with correct architecture
model = GCN(
    in_channels=X.shape[1],
    hidden_channels=256,
    out_channels=len(mlb.classes_),
    dropout=0.4
)

# Load state dict and metadata
model.load_state_dict(checkpoint['model_state_dict'])
tag_vocab = checkpoint['tag_vocab']
confidence_thresholds = checkpoint['threshold_map']
tag_categories = checkpoint['tag_categories']

model.eval()
print("Successfully loaded trained model with:")
print(f"- {len(tag_vocab)} tags")
print(f"- Threshold map: {confidence_thresholds}")
print(f"- Tag categories: {list(tag_categories.keys())}")

# Prediction function with dynamic thresholding
def predict_tags(pred_vector, tag_vocab):
    sorted_indices = np.argsort(pred_vector)[::-1]
    tags = []
    confidences = []

    for i in sorted_indices:
        tag = tag_vocab[i]
        conf = pred_vector[i]

        # Determine category
        category = 'default'
        for cat, tags_in_cat in tag_categories.items():
            if tag in tags_in_cat:
                category = cat
                break

        threshold = confidence_thresholds.get(category, confidence_thresholds['default'])

        if conf >= threshold:
            tags.append(tag)
            confidences.append(conf)

            # Stop if confidence drops significantly
            if len(confidences) > 1 and conf < 0.6 * confidences[0]:
                break

    # Post-processing: remove standalone numerical terms
    standalone_numerical = {'one', 'two', 'three', 'four', 'five'}
    filtered_tags = [t for t in tags if t not in standalone_numerical or len(tags) > 1]

    return filtered_tags[:15]  # Max 15 tags

# Generate predictions
print("\nGenerating predictions...")
predicted_tags = []
with torch.no_grad():
    predictions = model(data)
    predictions = predictions.numpy()

for idx, pred_vector in enumerate(predictions):
    tags = predict_tags(pred_vector, tag_vocab)
    predicted_tags.append({
        "image": image_list[idx],
        "tags": ", ".join(tags),
        "tag_count": len(tags)
    })

# Save predictions
df = pd.DataFrame(predicted_tags)
output_path = "E:/MLBD Project/data/predicted_tags.csv"
df.to_csv(output_path, index=False)

# Display statistics
tag_counts = df['tag_count'].value_counts().sort_index()
print("\nTag distribution:")
for count, num in tag_counts.items():
    print(f"  {count} tags: {num} images ({num/len(df)*100:.1f}%)")

print(f"\nPredictions saved to: {output_path}")
