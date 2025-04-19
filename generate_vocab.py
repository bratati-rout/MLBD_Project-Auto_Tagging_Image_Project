from vocab import Vocabulary
import pandas as pd
import pickle

CAPTIONS_FILE = "E:/MLBD Project/data/captions.txt"

# Initialize vocabulary
vocab = Vocabulary()

# === Add Special Tokens First ===
vocab.add_word('<unk>')  # Essential for handling unknown words
vocab.add_word('<pad>')  # If used in your CaptionDataset
vocab.add_word('<start>')
vocab.add_word('<end>')

# Load captions
df = pd.read_csv(CAPTIONS_FILE)

# Add words to vocabulary
for _, row in df.iterrows():
    caption = row['caption']
    tokens = caption.lower().split()
    for token in tokens:
        vocab.add_word(token)

# Save vocabulary to file
with open('E:/MLBD Project/data/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
