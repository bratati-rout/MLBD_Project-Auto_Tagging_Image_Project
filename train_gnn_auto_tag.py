'''import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from nltk.corpus import stopwords
import os

# Ensure directory exists
os.makedirs("E:/MLBD Project/models", exist_ok=True)

# Load Image Graph (updated to use enhanced graph)
with open("E:/MLBD Project/data/enhanced_image_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load Clean Captions 
captions_df = pd.read_csv("E:/MLBD Project/data/clean_captions.csv")

# Create Labels from Captions with stop word filtering
print("Creating tag labels from captions...")
try:
    stop_words = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

all_words = " ".join(captions_df["caption"].tolist()).split()
filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
most_common_tags = [word for word, _ in Counter(filtered_words).most_common(100)]

print(f"Top 10 tags: {most_common_tags[:10]}")

# Assign multi-labels per image (based on tag presence in caption)
tag_dict = {}
for row in captions_df.itertuples():
    tags = [word for word in row.caption.split() if word in most_common_tags]
    tag_dict[row.image] = list(set(tags))

# Use MultiLabelBinarizer to create one-hot encoded labels
mlb = MultiLabelBinarizer()
image_list = list(G.nodes())
y = mlb.fit_transform([tag_dict.get(img, []) for img in image_list])

# Save the binarizer for later use
with open("E:/MLBD Project/models/multilabel_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

print(f"Created label matrix with shape {y.shape}")

# Prepare Graph Tensors 
# Node features: use visual features from node attributes
try:
    X = np.array([G.nodes[img]['visual_feature'] for img in image_list], dtype=np.float32)
except KeyError:
    # Fallback to 'feature' if 'visual_feature' doesn't exist
    X = np.array([G.nodes[img]['feature'] for img in image_list], dtype=np.float32)

print(f"Feature matrix shape: {X.shape}")

# Map image names to integer indices
node_id_map = {img: idx for idx, img in enumerate(image_list)}
edges = [(node_id_map[a], node_id_map[b]) for a, b in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

print(f"Edge index shape: {edge_index.shape}")

# Create PyTorch Geometric data
data = Data(x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float32))

# Split into train/validation/test
indices = list(range(data.num_nodes))
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Create boolean masks
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

# GCN Model with added batch normalization
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer with batch normalization
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer with batch normalization
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer with sigmoid activation for multi-label
        x = self.conv3(x, edge_index)
        x = torch.sigmoid(x)
        return x

# Train GCN 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GCN(input_dim=X.shape[1], hidden_dim=256, output_dim=y.shape[1], dropout=0.3)
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training with early stopping
best_val_loss = float('inf')
patience = 10
counter = 0
epochs = 200

print("\nTraining GCN for auto-tagging...")
for epoch in range(1, epochs + 1):
    # Training step
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.binary_cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = F.binary_cross_entropy(out[data.val_mask], data.y[data.val_mask])
        
    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Save best model
        torch.save(model.state_dict(), "E:/MLBD Project/models/gnn_model_best.pkl")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model for evaluation
model.load_state_dict(torch.load("E:/MLBD Project/models/gnn_model_best.pkl"))

# Evaluation with multiple metrics
print("\nEvaluating model on test set...")
model.eval()
with torch.no_grad():
    out = model(data)
    preds = out[data.test_mask]
    actual = data.y[data.test_mask]
    
    # Convert to binary predictions
    pred_labels = (preds > 0.5).int().cpu().numpy()
    actual_labels = actual.int().cpu().numpy()
    
    # Calculate metrics
    accuracy = (pred_labels == actual_labels).mean()
    precision = precision_score(actual_labels, pred_labels, average='samples', zero_division=0)
    recall = recall_score(actual_labels, pred_labels, average='samples', zero_division=0)
    f1 = f1_score(actual_labels, pred_labels, average='samples', zero_division=0)
    
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Save model and metadata for future use
print("\nSaving model and metadata...")
torch.save(model.state_dict(), "E:/MLBD Project/models/gnn_model.pkl")

# Save image to index mapping for prediction
with open("E:/MLBD Project/models/image_to_index.pkl", "wb") as f:
    pickle.dump(node_id_map, f)

print("Training complete! Model saved to models/gnn_model.pkl")
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from utils import load_features_and_graph, create_label_matrix, get_class_weights
from collections import defaultdict

TAG_CATEGORIES = {
    'common': ['a', 'in', 'the', 'is', 'on', 'and', 'with', 'of'],
    'descriptive': ['white', 'black', 'red', 'blue', 'green', 'pink'],
    'activity': ['running', 'jumping', 'playing', 'standing', 'sitting'],
    'specific': ['dog', 'man', 'woman', 'water', 'beach', 'mountain']
}

THRESHOLD_MAP = {
    'common': 0.6,    # High threshold for stopwords
    'descriptive': 0.5,
    'activity': 0.55,
    'specific': 0.4,
    'default': 0.5
}




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=4):  # Increased from previous
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

class GCNTagger(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(GCNTagger, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc(x)
        return x

def get_tag_threshold(tag):
    for category, tags in TAG_CATEGORIES.items():
        if tag in tags:
            return THRESHOLD_MAP[category]
    return THRESHOLD_MAP['default']

def filter_tags(pred_tags, probs, tag_vocab):
    """Enforce strict limits on tag counts"""
    # Keep top 10 most confident predictions
    sorted_indices = np.argsort(probs)[::-1]
    filtered = [tag_vocab[i] for i in sorted_indices[:10]]
    return filtered



def train_model(model, data, train_idx, val_idx, y, loss_fn, optimizer, scheduler=None, max_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = loss_fn(val_out[val_idx], y[val_idx]).item()

        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model)
    return model

def analyze_tag_distribution(y_true, tag_vocab):
    tag_counts = y_true.sum(axis=0)
    print("\nTag Frequency Distribution in Test Set:")
    for idx in np.argsort(-tag_counts):
        print(f"{tag_vocab[idx]}: {tag_counts[idx]} samples")

def evaluate_model(model, features, edge_index, labels, mask, tag_vocab):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        probs = torch.sigmoid(logits)
        probs = probs[mask]
        labels = labels[mask]

        # Apply category-specific thresholds
        preds = torch.zeros_like(probs)
        for tag_idx, tag in enumerate(tag_vocab):
            threshold = get_tag_threshold(tag)
            preds[:, tag_idx] = (probs[:, tag_idx] > threshold).float()

        # Convert to final tags with post-processing
        y_pred = []
        for i in range(preds.shape[0]):
            image_preds = preds[i].cpu().numpy()
            raw_tags = [tag_vocab[idx] for idx in np.where(image_preds > 0)[0]]
            filtered_tags = filter_tags(raw_tags, probs[i].cpu().numpy(), tag_vocab)
            y_pred.append([1 if tag in filtered_tags else 0 for tag in tag_vocab])

        y_pred = np.array(y_pred)
        y_true = labels.cpu().numpy()

        # Calculate metrics
        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
        avg_tags = np.mean(np.sum(y_pred, axis=1))

        # Per-class metrics
        class_metrics = defaultdict(dict)
        for idx, tag in enumerate(tag_vocab):
            class_metrics[tag]['precision'] = precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
            class_metrics[tag]['recall'] = recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
            class_metrics[tag]['f1'] = f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)

        # Print performance analysis
        analyze_tag_distribution(y_true, tag_vocab)
        
        print("\nTop 10 worst performing tags:")
        sorted_tags = sorted(class_metrics.items(), key=lambda x: x[1]['f1'])
        for tag, metrics in sorted_tags[:10]:
            print(f"  {tag}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")

        print("\nEvaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Avg Tags/Image: {avg_tags:.2f}")

        # Tag distribution
        tag_counts = np.sum(y_pred, axis=1)
        count_dict = {i: np.sum(tag_counts == i) for i in range(0, 11)}
        for k in sorted(count_dict.keys()):
            v = count_dict[k]
            print(f"  {k} tags: {v} images ({v/len(y_pred)*100:.1f}%)")

def main():
    print("Creating tag labels from captions...")
    label_matrix, tag_vocab = create_label_matrix('data/clean_captions.csv', top_k=100)
    print("Created label matrix with shape", label_matrix.shape)

    print("Loading image features and graph...")
    features, edge_index = load_features_and_graph(
        'data/efficientnet_image_features.pkl',
        'data/enhanced_image_graph.pkl')

    print("Feature matrix shape:", features.shape)
    print("Edge index shape:", edge_index.shape)

    # Standardization
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)
    y = label_matrix
    data = Data(x=x, edge_index=edge_index)

    # Train/val/test split
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    data = data.to(device)
    y = y.to(device)

    # Model initialization
    model = GCNTagger(
        in_channels=x.shape[1],
        hidden_channels=256,
        out_channels=y.shape[1],
        dropout=0.4
    ).to(device)

    # Loss and optimizer
    loss_fn = FocalLoss(alpha=0.5, gamma=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training
    print("\nTraining GCN for auto-tagging...")
    model = train_model(model, data, train_idx, val_idx, y, loss_fn, optimizer, scheduler)
    
    # Evaluation
    print("\nEvaluating model on test set...")
    evaluate_model(model, x.to(device), edge_index.to(device), y, test_idx, tag_vocab)

    # Save model
    print("\nSaving model and metadata...")
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag_vocab': tag_vocab,
        'threshold_map': THRESHOLD_MAP,
        'tag_categories': TAG_CATEGORIES
    }, 'models/gnn_model.pkl')
    print("Training complete! Model saved to models/gnn_model.pkl")

if __name__ == '__main__':
    main()
