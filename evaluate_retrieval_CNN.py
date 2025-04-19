import pandas as pd
import numpy as np
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity

# File paths (your setup)
features_path = "data/efficientnet_image_features.pkl"
captions_path = "data/clean_captions.csv"
num_test_images = 30  # Number of images to evaluate

def load_image_features(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_captions(path):
    df = pd.read_csv(path)
    captions_dict = {}
    for _, row in df.iterrows():
        img = row['image']
        caps = row['caption'].split(' a ')
        caps = ['a ' + c.strip() if not c.strip().startswith('a ') else c.strip() for c in caps]
        captions_dict[img] = caps
    return captions_dict

def evaluate_retrieval(image_features, captions_dict, test_images):
    image_list = list(image_features.keys())
    caption_vectors = []
    caption_to_image = []

    for img in image_list:
        for caption in captions_dict[img]:
            caption_vectors.append(image_features[img])
            caption_to_image.append(img)

    caption_vectors = np.array(caption_vectors)

    precision_scores = []
    recall_scores = []
    map_scores = []

    for test_img in test_images:
        test_feat = image_features[test_img].reshape(1, -1)
        sims = cosine_similarity(test_feat, caption_vectors).flatten()
        top_k_indices = np.argsort(sims)[::-1][:10]
        retrieved_images = [caption_to_image[idx] for idx in top_k_indices]

        # ground truth is all captions of test_img
        true_matches = sum([1 for img in retrieved_images if img == test_img])
        precision = true_matches / 10.0
        recall = true_matches / len(captions_dict[test_img])

        # MAP@10 calculation
        ap = 0.0
        hits = 0
        for i, img in enumerate(retrieved_images):
            if img == test_img:
                hits += 1
                ap += hits / (i + 1)
        map10 = ap / min(len(captions_dict[test_img]), 10)

        print(f"Image: {test_img}")
        print(f"  Precision@10: {precision:.4f}")
        print(f"  Recall@10: {recall:.4f}")
        print(f"  MAP@10: {map10:.4f}")
        print()

        precision_scores.append(precision)
        recall_scores.append(recall)
        map_scores.append(map10)

    print(f"Average Precision@10: {np.mean(precision_scores):.4f}")
    print(f"Average Recall@10: {np.mean(recall_scores):.4f}")
    print(f"Average MAP@10: {np.mean(map_scores):.4f}")

if __name__ == "__main__":
    print("Loading EfficientNet image features...")
    image_features = load_image_features(features_path)
    print(f"Loaded {len(image_features)} image features")

    print("Loading image captions...")
    captions_dict = load_captions(captions_path)
    print(f"Loaded {len(captions_dict)} image captions")

    print(f"\nEvaluating retrieval performance on {num_test_images} test images...")
    available_images = list(set(image_features.keys()) & set(captions_dict.keys()))
    test_images = random.sample(available_images, num_test_images)

    evaluate_retrieval(image_features, captions_dict, test_images)
