import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import textwrap
import numpy as np
import random

# Load predictions
PREDICTED_TAGS_FILE = "E:/MLBD Project/data/predicted_tags.csv"
IMAGES_DIR = "E:/MLBD Project/data/Images"  # Corrected path with capital I

# Load the CSV file
df = pd.read_csv(PREDICTED_TAGS_FILE)
print(f"Loaded {len(df)} images with their predicted tags")
print("CSV columns:", df.columns.tolist())

# Configuration
num_images = 15       # Show more images for better evaluation
cols = 3              # Larger images in 3 columns
rows = (num_images + cols - 1) // cols

# Select images with different numbers of tags to evaluate diverse cases
tag_counts = df['tags'].str.count(',').fillna(-1) + 1  # Count tags (0 if empty)
selected_indices = []

# Try to include examples with different tag counts (0 tags to 10 tags)
for count in range(11):
    candidates = df[tag_counts == count].index.tolist()
    if candidates:
        selected_indices.append(random.choice(candidates))

# Fill remaining slots with random images
remaining = num_images - len(selected_indices)
if remaining > 0:
    other_indices = df.index.difference(selected_indices).tolist()
    if other_indices:
        selected_indices.extend(np.random.choice(other_indices, 
                                              min(remaining, len(other_indices)), 
                                              replace=False))

# Limit to desired number and get selected rows
selected_indices = selected_indices[:num_images]
selected_df = df.loc[selected_indices].reset_index(drop=True)

# Create figure with appropriate size
plt.figure(figsize=(20, 6*rows))

for i, (_, row) in enumerate(selected_df.iterrows()):
    # Get full image path
    img_path = os.path.join(IMAGES_DIR, row["image"])
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    # Create subplot
    ax = plt.subplot(rows, cols, i + 1)
    
    # Open and display image
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    
    # Image filename as title
    ax.set_title(row["image"], fontsize=10)
    
    # Format and display tags
    if pd.isna(row["tags"]) or row["tags"] == "":
        tag_text = "No tags predicted"
        color = 'red'
    else:
        tag_text = row["tags"]
        color = 'green'
    
    wrapped_tags = "\n".join(textwrap.wrap(tag_text, width=40))
    
    # Display tags with background box for better readability
    ax.text(0.5, -0.05, wrapped_tags,
            transform=ax.transAxes,
            fontsize=9,
            ha='center',
            va='top',
            color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
                     boxstyle='round,pad=0.5'))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Add more space between rows for tags



plt.show()
print("Visualization complete!")
