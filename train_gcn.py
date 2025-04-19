import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import torch.serialization

# === Config ===
GRAPH_FILE = "E:/MLBD Project/data/image_graph.pt"
EPOCHS = 20
LR = 0.01
DEVICE = torch.device("cpu")

# === Safe Load Graph Data ===
# Load without weights_only to bypass PyTorch 2.6 restriction
try:
    data = torch.load(GRAPH_FILE, map_location=DEVICE)
except Exception as e:
    print("First load failed due to safety restrictions. Trying with manual trust...")
    obj = torch.load(GRAPH_FILE, map_location=DEVICE, weights_only=False)  # Unsafe load
    torch.serialization.add_safe_globals([type(obj)])  # Trust the type
    data = obj

# === GCN Model ===
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

input_dim = data.num_node_features       # 1536
hidden_dim = 256
output_dim = data.num_node_features      # Also 1536, to match target
model = GCN(input_dim, hidden_dim, output_dim).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
model.train()
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)  # Self-supervised: reconstruct input features
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}")

# === Save GCN Embeddings ===
# Convert output tensor to numpy
final_embeddings = out.detach().cpu().numpy()

# Assume image filenames (nodes) are stored in data.image_ids, or manually create it
# For now, simulate: data.image_ids = ['img1.jpg', 'img2.jpg', ...] (ensure it's in order)

# If you have a list of image ids (must be same length as num nodes)
image_ids_path = "E:/MLBD Project/data/image_ids.txt"
with open(image_ids_path, "r") as f:
    image_ids = [line.strip() for line in f]

assert len(image_ids) == final_embeddings.shape[0], "Mismatch in image ID and embedding count!"

# Create dict: {filename: embedding}
embedding_dict = {img_id: torch.tensor(embed) for img_id, embed in zip(image_ids, final_embeddings)}

# Save dictionary
torch.save(embedding_dict, "E:/MLBD Project/data/gcn_embeddings.pt")
print("GCN training complete. Embeddings dictionary saved to gcn_embeddings.pt")

# === Verify Saved Embeddings ===
# Load the embeddings to verify
loaded_embeddings = torch.load("E:/MLBD Project/data/gcn_embeddings.pt")

# Print the first 5 embeddings to check
for img_id, embed in list(loaded_embeddings.items())[:5]:
    print(f"Image ID: {img_id}, Embedding: {embed[:5]}")  # Print first 5 elements of the embedding
