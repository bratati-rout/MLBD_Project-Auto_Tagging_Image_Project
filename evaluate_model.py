import torch
import pickle
from model import CaptioningModel
from caption_dataset import CaptionDataset
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

# === Configuration ===
TEST_CAPTIONS_FILE = "E:/MLBD Project/data/test_captions.txt"
TEST_IMAGE_IDS_FILE = "E:/MLBD Project/data/test_image_ids.txt"
EMBEDDINGS_FILE = "E:/MLBD Project/data/gcn_embeddings.pt"
VOCAB_FILE = "E:/MLBD Project/data/vocab.pkl"
MODEL_PATH = "E:/MLBD Project/models/caption_model.pth"
BEAM_SIZE = 5
MAX_LENGTH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Resources ===
with open(VOCAB_FILE, 'rb') as f:
    vocab = pickle.load(f)

model = CaptioningModel(
    embedding_dim=1536,
    hidden_dim=512,
    vocab_size=len(vocab)
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Beam Search Decoder ===
def beam_search(image_embed, beam_size=BEAM_SIZE, length_penalty=0.7):
    """Improved beam search with length normalization"""
    image_embed = image_embed.to(DEVICE).float()
    beam = [([vocab.word2idx['<start>']], 0.0, None)]
    for _ in range(MAX_LENGTH):
        candidates = []
        for seq, score, hidden in beam:
            if seq[-1] == vocab.word2idx['<end>']:
                candidates.append((seq, score, hidden))
                continue
            seq_tensor = torch.tensor([seq], device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                if hidden is None:
                    outputs, new_hidden = model(img_embeds=image_embed.unsqueeze(0), captions=seq_tensor)
                else:
                    outputs, new_hidden = model(captions=seq_tensor[:, -1:], hidden=hidden)
            log_probs = torch.log_softmax(outputs[:, -1, :], dim=-1)
            topk_probs, topk_indices = log_probs.topk(beam_size)
            for i in range(beam_size):
                new_seq = seq + [topk_indices[0, i].item()]
                new_score = score + topk_probs[0, i].item()
                candidates.append((new_seq, new_score, new_hidden))
        candidates = sorted(candidates, key=lambda x: x[1] / (len(x[0])**length_penalty), reverse=True)[:beam_size]
        if all(c[0][-1] == vocab.word2idx['<end>'] for c in candidates):
            break
        beam = candidates
    best_seq = [idx for idx in beam[0][0] if idx not in {vocab.word2idx['<start>'], vocab.word2idx['<end>']}]
    return [vocab.idx2word[idx] for idx in best_seq]

# === Evaluation ===
def evaluate():
    test_dataset = CaptionDataset(
        TEST_CAPTIONS_FILE,
        TEST_IMAGE_IDS_FILE,
        EMBEDDINGS_FILE,
        VOCAB_FILE,
        max_length=MAX_LENGTH
    )
    # Verify test-train split
    with open("E:/MLBD Project/data/train_image_ids.txt", "r") as f:
        train_ids = set(line.strip() for line in f)
    test_ids = set(test_dataset.image_ids)
    assert len(test_ids & train_ids) == 0, "Test images leaked into training set!"
    references = []
    hypotheses = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for i, (img_embed, _, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
        pred_tokens = beam_search(img_embed.squeeze())
        img_id = test_dataset.image_ids[i]
        refs = [c.lower().split() for c in test_dataset.image_caption_map[img_id]]
        references.append(refs)
        hypotheses.append(pred_tokens)
    # Use smoothing to avoid BLEU warnings
    chencherry = SmoothingFunction()
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=chencherry.method4)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method4)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method4)
    print("\n=== Evaluation Results ===")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

def print_sample_outputs(n=5):
    test_dataset = CaptionDataset(
        TEST_CAPTIONS_FILE,
        TEST_IMAGE_IDS_FILE,
        EMBEDDINGS_FILE,
        VOCAB_FILE,
        max_length=MAX_LENGTH
    )
    print("\nSample Generated Captions vs. Ground Truth:\n")
    for i in range(n):
        image_embed, _, _ = test_dataset[i]
        generated_caption = beam_search(image_embed)
        img_id = test_dataset.image_ids[i]
        ground_truths = test_dataset.image_caption_map[img_id]
        print(f"\nImage ID: {img_id}")
        print("Generated Caption:", ' '.join(generated_caption))
        print("Ground Truth Captions:")
        for ref in ground_truths:
            print("  -", ref)

def analyze_failures(n=5):
    test_dataset = CaptionDataset(
        TEST_CAPTIONS_FILE,
        TEST_IMAGE_IDS_FILE,
        EMBEDDINGS_FILE,
        VOCAB_FILE,
        max_length=MAX_LENGTH
    )
    scores = []
    for i in range(len(test_dataset)):
        img_embed, _, _ = test_dataset[i]
        pred_tokens = beam_search(img_embed)
        img_id = test_dataset.image_ids[i]
        refs = [c.lower().split() for c in test_dataset.image_caption_map[img_id]]
        chencherry = SmoothingFunction()
        score = corpus_bleu([refs], [pred_tokens], smoothing_function=chencherry.method4)
        scores.append((score, img_id, pred_tokens, refs))
    scores.sort(key=lambda x: x[0])
    print("\n=== Worst Performing Samples ===")
    for score, img_id, pred, refs in scores[:n]:
        print(f"\nImage ID: {img_id}")
        print(f"BLEU-4: {score:.4f}")
        print("Predicted:", ' '.join(pred))
        print("References:")
        for r in refs:
            print(f"  - {' '.join(r)}")

if __name__ == "__main__":
    evaluate()
    print_sample_outputs()
    analyze_failures()
