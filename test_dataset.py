# test_dataset.py (updated)
from caption_dataset import CaptionDataset
from torch.utils.data import DataLoader

def test_data_pipeline():
    dataset = CaptionDataset(
        captions_file="E:/MLBD Project/data/captions.txt",
        image_ids_file="E:/MLBD Project/data/image_ids.txt",
        embeddings_file="E:/MLBD Project/data/gcn_embeddings.pt",
        vocab_file="E:/MLBD Project/data/vocab.pkl",
        max_length=20
    )
    
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    
    # Extended testing
    for batch_idx, (embeddings, captions, lengths) in enumerate(dataloader):
        assert embeddings.shape == (4, 1536), "Embedding dimension mismatch"
        assert captions.shape == (4, 20), "Caption sequence length mismatch"
        print(f"Batch {batch_idx} validated ")
        if batch_idx == 2:  # Test 3 batches
            break

if __name__ == "__main__":
    test_data_pipeline()
