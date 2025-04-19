import torch
from torch.utils.data import DataLoader
from caption_dataset import CaptionDataset
import pickle
from model import CaptioningModel
import numpy as np

# === Config ===
CAPTIONS_FILE = "E:/MLBD Project/data/captions.txt"
IMAGE_IDS_FILE = "E:/MLBD Project/data/image_ids.txt"
EMBEDDINGS_FILE = "E:/MLBD Project/data/gcn_embeddings.pt"
VOCAB_FILE = "E:/MLBD Project/data/vocab.pkl"
MODEL_SAVE_PATH = "E:/MLBD Project/models/caption_model.pth"
MAX_LENGTH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
EMBEDDING_DIM = 1536
HIDDEN_DIM = 512
DROPOUT = 0.2
EPOCHS = 30
LR = 0.001
GRAD_CLIP = 5.0
TEACHER_FORCING_RATIO = 0.75

def main():
    # === Dataset & Loader ===
    dataset = CaptionDataset(
        CAPTIONS_FILE, 
        IMAGE_IDS_FILE,
        EMBEDDINGS_FILE,
        VOCAB_FILE,
        max_length=MAX_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # === Model Setup ===
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)

    model = CaptioningModel(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=len(vocab),
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # === Training Loop ===
    best_loss = np.inf
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (img_embeds, captions, lengths) in enumerate(dataloader):
            # Sort sequences
            lengths = lengths.cpu()
            sorted_len, sorted_idx = lengths.sort(0, descending=True)
            
            img_embeds = img_embeds[sorted_idx].to(DEVICE)
            captions = captions[sorted_idx].to(DEVICE)
            sorted_len = sorted_len.to(DEVICE)
            
            # Prepare inputs/targets
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            valid_len = sorted_len - 1

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                # Forward pass with teacher forcing
                outputs, _ = model(img_embeds, inputs, valid_len)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with loss: {best_loss:.4f}")

    print("Training complete!")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
