import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import os
from nltk.corpus import stopwords
import nltk
import re


try:
    stopwords.words('english')
except:
    nltk.download('stopwords')


os.makedirs("E:/MLBD Project/evaluation", exist_ok=True)

print("Loading data for evaluation...")

# Load predicted tags
predicted_df = pd.read_csv("E:/MLBD Project/data/predicted_tags.csv")

# Load captions for ground truth comparison
captions_df = pd.read_csv("E:/MLBD Project/data/clean_captions.csv")

# Configure visualization style
plt.style.use('fivethirtyeight')
sns.set_palette("muted")

#  BASIC TAG STATISTICS 

print("\n TAG DISTRIBUTION STATISTICS")

# Calculate basic statistics
total_images = len(predicted_df)
images_with_tags = len(predicted_df[predicted_df['tags'].str.len() > 0])
predicted_df['tag_count'] = predicted_df['tags'].str.split(',').str.len()
images_without_tags = total_images - images_with_tags

print(f"Total images evaluated: {total_images}")
print(f"Images with tags: {images_with_tags} ({images_with_tags/total_images*100:.1f}%)")
print(f"Images without tags: {images_without_tags} ({images_without_tags/total_images*100:.1f}%)")

# Calculate tag distribution
tag_distribution = predicted_df['tag_count'].value_counts().sort_index()
avg_tags_per_image = predicted_df['tag_count'].mean()

print(f"Average tags per image: {avg_tags_per_image:.2f}")
print("\nTag count distribution:")
for count, images in tag_distribution.items():
    print(f"  {count} tags: {images} images ({images/total_images*100:.1f}%)")

# Plot tag distribution
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='tag_count', data=predicted_df, palette='viridis')
plt.title('Number of Tags per Image', fontsize=16)
plt.xlabel('Number of Tags', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)

# Add percentage labels
total = len(predicted_df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("E:/MLBD Project/evaluation/tag_distribution.png")
print(f"Saved tag distribution plot to evaluation/tag_distribution.png")

#TAG FREQUENCY ANALYSIS

print("\n TAG FREQUENCY ANALYSIS")

# Extract all individual tags
all_tags = []
for tags in predicted_df['tags'].dropna():
    if isinstance(tags, str):
        all_tags.extend([t.strip() for t in tags.split(',')])

# Count tag frequencies
tag_counts = pd.Series(all_tags).value_counts()
total_tag_occurrences = len(all_tags)

print(f"Total tag occurrences: {total_tag_occurrences}")
print(f"Unique tags used: {len(tag_counts)}")
print("\nTop 20 most common tags:")
for tag, count in tag_counts.head(20).items():
    print(f"  {tag}: {count} occurrences ({count/total_tag_occurrences*100:.1f}%)")

# Plot tag frequencies
plt.figure(figsize=(14, 8))
tag_counts.head(20).plot(kind='bar', color='teal')
plt.title('Top 20 Most Frequent Tags', fontsize=16)
plt.xlabel('Tag', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("E:/MLBD Project/evaluation/tag_frequency.png")
print(f"Saved tag frequency plot to evaluation/tag_frequency.png")

# Save tag statistics
tag_stats = pd.DataFrame({
    'tag': tag_counts.index,
    'count': tag_counts.values,
    'percentage': tag_counts.values / total_tag_occurrences * 100
})
tag_stats.to_csv("E:/MLBD Project/evaluation/tag_statistics.csv", index=False)
print(f"Saved tag statistics to evaluation/tag_statistics.csv")

# COMPARATIVE EVALUATION WITH GROUND TRUTH 

print("\n EVALUATION AGAINST GROUND TRUTH ")

# Function to extract keywords from captions (proxy for ground truth tags)
def extract_keywords_from_caption(caption):
    if not isinstance(caption, str):
        return []
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase, remove punctuation, and split into words
    words = re.sub(r'[^\w\s]', ' ', caption.lower()).split()
    
    # Remove stopwords and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return list(set(keywords))  # Remove duplicates

# Helper function to compute precision, recall, F1 between two tag sets
def compute_metrics(ground_truth_tags, predicted_tags):
    if not ground_truth_tags or not predicted_tags:
        return 0, 0, 0
    
    true_positives = len(set(ground_truth_tags) & set(predicted_tags))
    precision = true_positives / len(predicted_tags) if predicted_tags else 0
    recall = true_positives / len(ground_truth_tags) if ground_truth_tags else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Prepare evaluation dataframe
eval_data = []

# Merge predicted tags with captions
merged_df = pd.merge(predicted_df, captions_df, on='image', how='inner')

print(f"Found {len(merged_df)} images with both predicted tags and captions")

# Calculate metrics for each image
precisions = []
recalls = []
f1_scores = []

for _, row in merged_df.iterrows():
    # Get predicted tags
    pred_tags = [tag.strip() for tag in row['tags'].split(',')] if isinstance(row['tags'], str) else []
    
    # Extract ground truth tags from caption
    gt_tags = extract_keywords_from_caption(row['caption'])
    
    # Compute metrics
    precision, recall, f1 = compute_metrics(gt_tags, pred_tags)
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    # Store evaluation data
    eval_data.append({
        'image': row['image'],
        'predicted_tags': row['tags'],
        'caption': row['caption'],
        'ground_truth_tags': ', '.join(gt_tags),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Create evaluation dataframe
eval_df = pd.DataFrame(eval_data)

# Calculate overall metrics
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)

print(f"\nOverall Evaluation Metrics:")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Plot metrics distribution
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.hist(precisions, bins=20, alpha=0.7, color='blue')
plt.axvline(avg_precision, color='red', linestyle='dashed', linewidth=2)
plt.title(f'Precision Distribution\nAvg: {avg_precision:.4f}')
plt.xlabel('Precision')
plt.ylabel('Number of Images')

plt.subplot(1, 3, 2)
plt.hist(recalls, bins=20, alpha=0.7, color='green')
plt.axvline(avg_recall, color='red', linestyle='dashed', linewidth=2)
plt.title(f'Recall Distribution\nAvg: {avg_recall:.4f}')
plt.xlabel('Recall')
plt.ylabel('Number of Images')

plt.subplot(1, 3, 3)
plt.hist(f1_scores, bins=20, alpha=0.7, color='purple')
plt.axvline(avg_f1, color='red', linestyle='dashed', linewidth=2)
plt.title(f'F1 Score Distribution\nAvg: {avg_f1:.4f}')
plt.xlabel('F1 Score')
plt.ylabel('Number of Images')

plt.tight_layout()
plt.savefig("E:/MLBD Project/evaluation/metrics_distribution.png")
print(f"Saved metrics distribution plot to evaluation/metrics_distribution.png")

# Save evaluation results
eval_df.to_csv("E:/MLBD Project/evaluation/tag_evaluation.csv", index=False)
print(f"Saved detailed evaluation results to evaluation/tag_evaluation.csv")

#  QUALITY ASSESSMENT 

print("\n=== PART 4: TAG QUALITY ASSESSMENT ===")

# Calculate tag relevance score (how many predicted tags appear in captions)
relevant_tag_percentages = []

for _, row in eval_df.iterrows():
    pred_tags = [tag.strip() for tag in row['predicted_tags'].split(',')] if isinstance(row['predicted_tags'], str) else []
    gt_tags = row['ground_truth_tags'].split(', ') if row['ground_truth_tags'] else []
    
    if pred_tags:
        relevant_tag_count = len(set(pred_tags) & set(gt_tags))
        relevant_percentage = relevant_tag_count / len(pred_tags) * 100
        relevant_tag_percentages.append(relevant_percentage)

avg_relevance = np.mean(relevant_tag_percentages) if relevant_tag_percentages else 0

print(f"Average Tag Relevance: {avg_relevance:.2f}%")

# Define quality buckets based on F1 score
eval_df['quality_bucket'] = pd.cut(
    eval_df['f1_score'], 
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
)

# Count images in each quality bucket
quality_distribution = eval_df['quality_bucket'].value_counts().sort_index()
print("\nTag Quality Distribution:")
for quality, count in quality_distribution.items():
    percentage = count / len(eval_df) * 100
    print(f"  {quality}: {count} images ({percentage:.1f}%)")

# Plot quality distribution
plt.figure(figsize=(10, 6))
quality_distribution.plot(kind='bar', color='orange')
plt.title('Tag Quality Distribution', fontsize=16)
plt.xlabel('Quality Category', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.tight_layout()
plt.savefig("E:/MLBD Project/evaluation/quality_distribution.png")
print(f"Saved quality distribution plot to evaluation/quality_distribution.png")

# Sample evaluation for demonstration
print("\nSample Evaluation (10 random images):")
sample_eval = eval_df.sample(10)
for _, row in sample_eval.iterrows():
    print(f"\nImage: {row['image']}")
    print(f"Caption: {row['caption']}")
    print(f"Ground Truth Tags: {row['ground_truth_tags']}")
    print(f"Predicted Tags: {row['predicted_tags']}")
    print(f"Metrics: Precision = {row['precision']:.2f}, Recall = {row['recall']:.2f}, F1 = {row['f1_score']:.2f}")
    print(f"Quality: {row['quality_bucket']}")

print("\nEvaluation complete! Results saved to evaluation/ directory.")
