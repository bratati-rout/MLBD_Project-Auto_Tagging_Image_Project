import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from collections import Counter
import networkx as nx
from torch_geometric.utils import from_networkx

def load_features_and_graph(feature_file, edge_index_file):
    # Load image features
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)
    features = np.array(list(features_dict.values()))
    features = torch.tensor(features, dtype=torch.float)

    # Load and clean NetworkX graph
    with open(edge_index_file, 'rb') as f:
        G = pickle.load(f)

    # Remove edge attributes
    G_clean = nx.Graph()
    G_clean.add_nodes_from(G.nodes)
    G_clean.add_edges_from(G.edges)  # Only structure, no attributes

    # Convert to PyG format
    pyg_graph = from_networkx(G_clean)
    edge_index = pyg_graph.edge_index

    return features, edge_index


def create_label_matrix(captions_file, top_k=100):
    df = pd.read_csv(captions_file)
    df['tags'] = df['caption'].str.lower().str.split()
    
    all_tags = [tag for tags in df['tags'] for tag in tags]
    top_tags = [t for t, _ in Counter(all_tags).most_common(top_k)]
    
    df['filtered_tags'] = df['tags'].apply(lambda tags: [t for t in tags if t in top_tags])

    mlb = MultiLabelBinarizer(classes=top_tags)
    label_matrix = mlb.fit_transform(df['filtered_tags'])

    return torch.tensor(label_matrix, dtype=torch.float), top_tags

def split_data(num_samples, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(indices, train_size=train_frac, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_frac / (val_frac + test_frac), random_state=42)
    return train_idx, val_idx, test_idx

def get_class_weights(labels):
    # Calculate positive samples per class
    pos_counts = labels.sum(dim=0)
    # Calculate negative samples per class
    neg_counts = labels.shape[0] - pos_counts
    
    # Effective number of samples formula (from "Class-Balanced Loss")
    beta = 0.9  # Hyperparameter
    effective_num = 1.0 - torch.pow(beta, pos_counts)
    weights = (1.0 - beta) / (effective_num + 1e-5)
    
    # Normalize weights
    weights = weights / weights.mean()
    return weights
