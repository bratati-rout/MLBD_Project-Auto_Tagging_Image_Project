'''import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import pandas as pd
from tqdm import tqdm

# Define paths
EFFICIENTNET_FEATURES_FILE = "E:/MLBD Project/data/efficientnet_image_features.pkl"
TFIDF_EMBEDDINGS_FILE = "E:/MLBD Project/data/tfidf_embeddings.pkl"
CAPTIONS_FILE = "E:/MLBD Project/data/clean_captions.csv"
OUTPUT_GRAPH = "E:/MLBD Project/data/enhanced_image_graph.pkl"

# Load EfficientNet features
print("Loading EfficientNet features...")
with open(EFFICIENTNET_FEATURES_FILE, "rb") as f:
    efficientnet_features = pickle.load(f)

# Load captions for semantic information
print("Loading captions...")
captions_df = pd.read_csv(CAPTIONS_FILE)

# Load TF-IDF embeddings for semantic similarity
try:
    print("Loading TF-IDF embeddings...")
    with open(TFIDF_EMBEDDINGS_FILE, "rb") as f:
        tfidf_data = pickle.load(f)
    
    # Extract TF-IDF vectors and corresponding images
    tfidf_matrix = tfidf_data
    tfidf_images = captions_df['image'].tolist()
    has_tfidf = True
    print(f"Loaded TF-IDF embeddings with shape: {tfidf_matrix.shape}")
except:
    print("TF-IDF embeddings not found or not loadable, using only visual features.")
    has_tfidf = False

# Prepare data for graph construction
image_names = list(efficientnet_features.keys())
visual_features = np.array([efficientnet_features[img] for img in image_names])

print(f"Constructing graph with {len(image_names)} images...")

# Initialize graph
G = nx.Graph()

# Add nodes with features
print("Adding nodes to the graph...")
for idx, name in enumerate(image_names):
    # Get caption if available
    caption = captions_df[captions_df['image'] == name]['caption'].values[0] if name in captions_df['image'].values else ""
    
    # Add node with attributes
    G.add_node(
        name, 
        visual_feature=efficientnet_features[name],
        caption=caption
    )

# UPDATED: Lower similarity thresholds
VISUAL_SIMILARITY_THRESHOLD = 0.45  # Lowered from 0.75
SEMANTIC_SIMILARITY_THRESHOLD = 0.15  # Lowered from 0.3
HYBRID_SIMILARITY_THRESHOLD = 0.35  # Lowered from 0.6

# ADDED: K-nearest neighbors parameter
K_NEIGHBORS = 5  # Ensure each image connects to at least 5 similar images

# Process in batches to avoid memory issues with large matrices
BATCH_SIZE = 1000

print("Computing similarities and adding edges...")
num_edges = 0

# First pass: Add edges based on thresholds
for i in range(0, len(image_names), BATCH_SIZE):
    batch_images = image_names[i:i+BATCH_SIZE]
    batch_features = visual_features[i:i+BATCH_SIZE]
    
    # Compute similarity for this batch against all images
    batch_sim = cosine_similarity(batch_features, visual_features)
    
    # Add edges for similar pairs
    for b_idx, img1 in enumerate(batch_images):
        global_idx = i + b_idx
        
        # Find potential neighbors based on visual similarity
        potential_neighbors = [(image_names[j], batch_sim[b_idx, j]) 
                               for j in range(len(image_names)) 
                               if global_idx != j and batch_sim[b_idx, j] >= VISUAL_SIMILARITY_THRESHOLD]
        
        for img2, visual_sim in potential_neighbors:
            # Skip if edge already exists
            if G.has_edge(img1, img2):
                continue
                
            # Calculate hybrid similarity if TF-IDF is available
            if has_tfidf and img1 in tfidf_images and img2 in tfidf_images:
                idx1 = tfidf_images.index(img1)
                idx2 = tfidf_images.index(img2)
                semantic_sim = cosine_similarity(
                    tfidf_matrix[idx1].reshape(1, -1),
                    tfidf_matrix[idx2].reshape(1, -1)
                )[0][0]
                
                # Combine visual and semantic similarity (weighted average)
                hybrid_sim = 0.7 * visual_sim + 0.3 * semantic_sim
                
                # Only add edge if hybrid similarity is high enough
                if hybrid_sim >= HYBRID_SIMILARITY_THRESHOLD:
                    G.add_edge(
                        img1, img2, 
                        weight=hybrid_sim,
                        visual_similarity=visual_sim,
                        semantic_similarity=semantic_sim
                    )
                    num_edges += 1
            else:
                # If TF-IDF not available, use only visual similarity
                G.add_edge(
                    img1, img2, 
                    weight=visual_sim,
                    visual_similarity=visual_sim,
                    semantic_similarity=0.0
                )
                num_edges += 1
    
    print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(image_names)-1)//BATCH_SIZE + 1}, edges so far: {num_edges}")

# ADDED: Second pass - ensure K-nearest neighbors for each node
print(f"Ensuring each node has at least {K_NEIGHBORS} connections...")
isolated_nodes = 0
added_knn_edges = 0

for i, img1 in enumerate(image_names):
    # Check current number of neighbors
    current_neighbors = list(G.neighbors(img1))
    if len(current_neighbors) >= K_NEIGHBORS:
        continue
    
    # Need to add more neighbors
    neighbors_needed = K_NEIGHBORS - len(current_neighbors)
    
    # Get visual similarities for all other images
    similarities = []
    for j, img2 in enumerate(image_names):
        if img1 != img2 and img2 not in current_neighbors:
            sim = cosine_similarity(
                visual_features[i].reshape(1, -1),
                visual_features[j].reshape(1, -1)
            )[0][0]
            similarities.append((img2, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Add edges to top most similar images
    for img2, sim in similarities[:neighbors_needed]:
        G.add_edge(
            img1, img2,
            weight=sim,
            visual_similarity=sim,
            semantic_similarity=0.0,
            knn_added=True
        )
        added_knn_edges += 1
    
    # Check if this node is still isolated (had no neighbors and couldn't add any)
    if len(list(G.neighbors(img1))) == 0:
        isolated_nodes += 1
        
    # Print progress every 500 nodes
    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{len(image_names)} nodes, added {added_knn_edges} KNN edges")

print(f"K-nearest neighbors pass complete. Added {added_knn_edges} additional edges.")
print(f"Remaining isolated nodes: {isolated_nodes}")

print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Add graph-level metadata
G.graph['creation_date'] = 'April 14, 2025'
G.graph['feature_type'] = 'EfficientNet-B3'
G.graph['visual_threshold'] = VISUAL_SIMILARITY_THRESHOLD
G.graph['hybrid_threshold'] = HYBRID_SIMILARITY_THRESHOLD
G.graph['k_neighbors'] = K_NEIGHBORS

# Save the enhanced graph
print(f"Saving graph to {OUTPUT_GRAPH}...")
with open(OUTPUT_GRAPH, "wb") as f:
    pickle.dump(G, f)

print("Graph construction complete!")

# Print graph statistics
avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
print(f"Average node degree: {avg_degree:.2f}")
print(f"Graph density: {nx.density(G):.6f}")

# Analyze connected components
components = list(nx.connected_components(G))
print(f"Number of connected components: {len(components)}")
print(f"Largest component size: {len(max(components, key=len))}")

# Print component size distribution
component_sizes = [len(c) for c in components]
print("Component size distribution:")
print(f"  Components with size 1 (isolated): {component_sizes.count(1)}")
print(f"  Components with size 2-10: {sum(1 for s in component_sizes if 2 <= s <= 10)}")
print(f"  Components with size 11-100: {sum(1 for s in component_sizes if 11 <= s <= 100)}")
print(f"  Components with size >100: {sum(1 for s in component_sizes if s > 100)}")
'''
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import pandas as pd
from tqdm import tqdm

# Define paths
EFFICIENTNET_FEATURES_FILE = "E:/MLBD Project/data/efficientnet_image_features.pkl"
TFIDF_EMBEDDINGS_FILE = "E:/MLBD Project/data/tfidf_embeddings.pkl"
CAPTIONS_FILE = "E:/MLBD Project/data/clean_captions.csv"
OUTPUT_GRAPH = "E:/MLBD Project/data/enhanced_image_graph.pkl"

def load_features():
    print("Loading EfficientNet features...")
    with open(EFFICIENTNET_FEATURES_FILE, "rb") as f:
        efficientnet_features = pickle.load(f)

    print("Loading captions...")
    captions_df = pd.read_csv(CAPTIONS_FILE)
    image_to_caption = dict(zip(captions_df['image'], captions_df['caption']))

    try:
        print("Loading TF-IDF embeddings...")
        with open(TFIDF_EMBEDDINGS_FILE, "rb") as f:
            tfidf_matrix = pickle.load(f)
        tfidf_images = captions_df['image'].tolist()
        tfidf_index_map = {img: idx for idx, img in enumerate(tfidf_images)}
        print(f"Loaded TF-IDF embeddings with shape: {tfidf_matrix.shape}")
        return efficientnet_features, image_to_caption, tfidf_matrix, tfidf_index_map
    except Exception:
        print("TF-IDF embeddings not found, continuing with only visual features.")
        return efficientnet_features, image_to_caption, None, None

def compute_batch_similarity(query_features, reference_features):
    return cosine_similarity(query_features, reference_features)

def add_nodes(G, image_names, features_dict, image_to_caption):
    for name in image_names:
        G.add_node(name,
                   visual_feature=features_dict[name],
                   caption=image_to_caption.get(name, ""))

def add_similarity_edges(G, image_names, visual_features, tfidf_matrix, tfidf_index_map,
                         VISUAL_SIM_THRESH=0.45, HYBRID_SIM_THRESH=0.35, has_tfidf=True, BATCH_SIZE=1000):
    num_edges = 0
    N = len(image_names)

    for i in tqdm(range(0, N, BATCH_SIZE), desc="Adding similarity edges"):
        batch_imgs = image_names[i:i+BATCH_SIZE]
        batch_feats = visual_features[i:i+BATCH_SIZE]
        sim_matrix = compute_batch_similarity(batch_feats, visual_features)

        for b_idx, img1 in enumerate(batch_imgs):
            sims = sim_matrix[b_idx]
            for j, sim in enumerate(sims):
                img2 = image_names[j]
                if img1 == img2 or sim < VISUAL_SIM_THRESH or G.has_edge(img1, img2):
                    continue

                semantic_sim = 0.0
                hybrid_sim = sim
                if has_tfidf and img1 in tfidf_index_map and img2 in tfidf_index_map:
                    idx1, idx2 = tfidf_index_map[img1], tfidf_index_map[img2]
                    semantic_sim = cosine_similarity(tfidf_matrix[idx1].reshape(1, -1),
                                                     tfidf_matrix[idx2].reshape(1, -1))[0][0]
                    hybrid_sim = 0.7 * sim + 0.3 * semantic_sim

                if hybrid_sim >= HYBRID_SIM_THRESH:
                    G.add_edge(img1, img2,
                               weight=hybrid_sim,
                               visual_similarity=sim,
                               semantic_similarity=semantic_sim)
                    num_edges += 1
    return num_edges

def ensure_knn(G, image_names, visual_features, K=5):
    added_knn = 0
    isolated = 0
    visual_matrix = visual_features

    print("Ensuring KNN connectivity...")
    for i, img1 in enumerate(tqdm(image_names, desc="KNN pass")):
        current_neighbors = list(G.neighbors(img1))
        if len(current_neighbors) >= K:
            continue

        sims = cosine_similarity(visual_matrix[i].reshape(1, -1), visual_matrix)[0]
        sims[i] = -1  # Exclude self

        top_indices = sims.argsort()[-(K + len(current_neighbors)):][::-1]
        added = 0
        for j in top_indices:
            img2 = image_names[j]
            if img2 != img1 and img2 not in current_neighbors:
                G.add_edge(img1, img2,
                           weight=sims[j],
                           visual_similarity=sims[j],
                           semantic_similarity=0.0,
                           knn_added=True)
                added_knn += 1
                added += 1
            if added >= K - len(current_neighbors):
                break

        if len(list(G.neighbors(img1))) == 0:
            isolated += 1

    return isolated, added_knn

def construct_graph():
    efficientnet_features, image_to_caption, tfidf_matrix, tfidf_index_map = load_features()
    image_names = list(efficientnet_features.keys())
    visual_features = np.array([efficientnet_features[img] for img in image_names])

    print(f"Constructing graph with {len(image_names)} images...")
    G = nx.Graph()

    add_nodes(G, image_names, efficientnet_features, image_to_caption)

    VISUAL_SIMILARITY_THRESHOLD = 0.45
    HYBRID_SIMILARITY_THRESHOLD = 0.35
    K_NEIGHBORS = 5

    num_edges = add_similarity_edges(G, image_names, visual_features,
                                     tfidf_matrix, tfidf_index_map,
                                     VISUAL_SIMILARITY_THRESHOLD,
                                     HYBRID_SIMILARITY_THRESHOLD,
                                     tfidf_matrix is not None)

    isolated_nodes, added_knn = ensure_knn(G, image_names, visual_features, K_NEIGHBORS)

    print(f"Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Isolated nodes: {isolated_nodes}")
    print(f"Added KNN edges: {added_knn}")

    G.graph.update({
        'creation_date': 'April 16, 2025',
        'feature_type': 'EfficientNet-B3',
        'visual_threshold': VISUAL_SIMILARITY_THRESHOLD,
        'hybrid_threshold': HYBRID_SIMILARITY_THRESHOLD,
        'k_neighbors': K_NEIGHBORS
    })

    print(f"Saving graph to {OUTPUT_GRAPH}...")
    with open(OUTPUT_GRAPH, "wb") as f:
        pickle.dump(G, f)
    print("Graph construction complete!")

if __name__ == "__main__":
    construct_graph()
