import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import pickle
from vocab import Vocabulary
import warnings

class CaptionDataset(Dataset):
    def __init__(self, captions_file, image_ids_file, embeddings_file, vocab_file, max_length=20):
        super(CaptionDataset, self).__init__()
        self.max_length = max_length

        # Load and validate data
        self.df = pd.read_csv(captions_file)
        self._validate_columns()
        
        # Load image IDs and filter those with captions
        with open(image_ids_file, 'r') as f:
            all_image_ids = {line.strip() for line in f}
        
        self.image_caption_map = self._build_image_caption_map(all_image_ids)
        self.image_ids = self._filter_valid_images(all_image_ids)
        
        # Load and validate embeddings
        self.embeddings = torch.load(embeddings_file)
        self._validate_embeddings()
        
        # Load and validate vocabulary
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        self._validate_vocabulary()

    def _validate_columns(self):
        if not {'image', 'caption'}.issubset(self.df.columns):
            raise ValueError("CSV must contain 'image' and 'caption' columns")

    def _build_image_caption_map(self, valid_ids):
        image_map = {}
        for _, row in self.df.iterrows():
            img_id = row['image']
            if img_id in valid_ids:
                image_map.setdefault(img_id, []).append(row['caption'])
        return image_map

    def _filter_valid_images(self, all_ids):
        valid_ids = [img_id for img_id in all_ids if img_id in self.image_caption_map]
        missing_count = len(all_ids) - len(valid_ids)
        if missing_count > 0:
            warnings.warn(f"Removed {missing_count} images without captions")
        return valid_ids

    def _validate_embeddings(self):
        missing = [img_id for img_id in self.image_ids if img_id not in self.embeddings]
        if missing:
            raise KeyError(f"Missing embeddings for {len(missing)} images")

    def _validate_vocabulary(self):
        required_tokens = ['<unk>', '<start>', '<end>', '<pad>']
        for token in required_tokens:
            if token not in self.vocab.word2idx:
                raise ValueError(f"Vocabulary missing required token: {token}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        embedding = self.embeddings[img_id]
        caption = random.choice(self.image_caption_map[img_id])
        
        # Process caption
        tokens = ['<start>'] + caption.lower().split() + ['<end>']
        tokens = self._process_tokens(tokens)
        
        return (
            embedding,
            torch.tensor([self.vocab.word2idx.get(t, self.vocab.word2idx['<unk>']) for t in tokens], 
                        dtype=torch.long),
            torch.tensor(min(len(tokens), self.max_length), dtype=torch.long)
        )

    def _process_tokens(self, tokens):
        """Truncate or pad tokens to max_length"""
        if len(tokens) > self.max_length:
            return tokens[:self.max_length-1] + ['<end>']
        return tokens + ['<pad>'] * (self.max_length - len(tokens))

    def collate_fn(self, batch):
        embeddings, captions, lengths = zip(*batch)
        return (
            torch.stack(embeddings),
            torch.stack(captions),
            torch.stack(lengths)
        )


# ========== Usage Example ==========
if __name__ == "__main__":
    # Example configuration - update paths as needed
    dataset = CaptionDataset(
        captions_file="E:/MLBD Project/data/captions.txt",
        image_ids_file="E:/MLBD Project/data/image_ids.txt",
        embeddings_file="E:/MLBD Project/data/gcn_embeddings.pt",
        vocab_file="E:/MLBD Project/data/vocab.pkl",
        max_length=20
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # Test one batch
    print("\n=== Testing DataLoader ===")
    for batch_idx, (embeddings, captions, lengths) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Lengths: {lengths.tolist()}")
        
        # Decode first caption in batch
        sample_caption = ' '.join([dataset.vocab.idx2word[idx.item()] for idx in captions[0]])
        print(f"\nSample caption:\n{sample_caption}")
        
        break  # Only show first batch
