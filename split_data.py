import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to your existing files
captions_file = 'E:/MLBD Project/data/captions.txt'
image_ids_file = 'E:/MLBD Project/data/image_ids.txt'

# Output paths
train_captions_file = 'E:/MLBD Project/data/train_captions.txt'
test_captions_file = 'E:/MLBD Project/data/test_captions.txt'
train_image_ids_file = 'E:/MLBD Project/data/train_image_ids.txt'
test_image_ids_file = 'E:/MLBD Project/data/test_image_ids.txt'

# Load data
captions_df = pd.read_csv(captions_file)
with open(image_ids_file, 'r') as f:
    image_ids = [line.strip() for line in f]

# Split image IDs (e.g., 90% train, 10% test)
train_ids, test_ids = train_test_split(image_ids, test_size=0.1, random_state=42)

# Filter captions for train and test
train_captions_df = captions_df[captions_df['image'].isin(train_ids)]
test_captions_df = captions_df[captions_df['image'].isin(test_ids)]

# Save splits
train_captions_df.to_csv(train_captions_file, index=False)
test_captions_df.to_csv(test_captions_file, index=False)

with open(train_image_ids_file, 'w') as f:
    for img_id in train_ids:
        f.write(img_id + '\n')
with open(test_image_ids_file, 'w') as f:
    for img_id in test_ids:
        f.write(img_id + '\n')

print("Train/test splits created:")
print(f"- {train_captions_file}")
print(f"- {test_captions_file}")
print(f"- {train_image_ids_file}")
print(f"- {test_image_ids_file}")
