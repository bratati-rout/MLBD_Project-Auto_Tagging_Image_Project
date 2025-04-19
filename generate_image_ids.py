import pandas as pd

# Path to captions file
captions_file = "E:/MLBD Project/data/captions.txt"
output_file = "E:/MLBD Project/data/image_ids.txt"

# Read captions
df = pd.read_csv(captions_file)

# Extract unique image filenames in order
unique_image_ids = df['image'].drop_duplicates().tolist()

# Save to text file
with open(output_file, "w") as f:
    for img_id in unique_image_ids:
        f.write(f"{img_id}\n")

print(f"Saved {len(unique_image_ids)} image IDs to {output_file}")
