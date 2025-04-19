import pickle
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

import torch
print(torch.__version__)


# === Config ===
FEATURES_FILE = "E:/MLBD Project/data/efficientnet_image_features.pkl"
GRAPH_SAVE_PATH = "E:/MLBD Project/data/image_graph.pt"
TOP_K = 5  # Number of nearest neighbors for each node

# === Step 1: Load image features ===
with open(FEATURES_FILE, "rb") as f:
    image_features = pickle.load(f)

image_names = list(image_features.keys())
features = np.array([image_features[name] for name in image_names])  # Shape: (num_images, feature_dim)
num_nodes = features.shape[0]

print(f"Loaded {num_nodes} image feature vectors.")

# === Step 2: Build similarity matrix ===
print("Computing cosine similarity...")
similarity_matrix = cosine_similarity(features)  # Shape: (num_images, num_images)

# === Step 3: Create edge_index using top-k neighbors (excluding self) ===
edge_index = []

for i in range(num_nodes):
    # Get top-k similar indices (excluding self)
    sim_scores = similarity_matrix[i]
    top_k_indices = np.argsort(sim_scores)[-TOP_K-1:-1]  # Last TOP_K excluding self

    for j in top_k_indices:
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
x = torch.tensor(features, dtype=torch.float)

# === Step 4: Create graph data ===
data = Data(x=x, edge_index=edge_index)
data.image_names = image_names  # Optional: for mapping

# === Step 5: Save graph ===
torch.save(data, GRAPH_SAVE_PATH)
print(f"Graph saved to {GRAPH_SAVE_PATH} with {data.num_nodes} nodes and {data.num_edges} edges.")
