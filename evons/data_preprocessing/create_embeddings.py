import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path

ROOT = Path().cwd() #file to be run from project root
DATA_DIR = ROOT / 'evons' / 'data'
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV_PATH = DATA_DIR / "evons.csv"
TITLE_OUT_PATH = EMB_DIR / "title_embeddings.pt"
DESC_OUT_PATH = EMB_DIR / "desc_embeddings.pt"

MODEL_NAME = 'FacebookAI/roberta-base'
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEmbedder(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        output = self.model(**inputs)
        return output.last_hidden_state[:, 0, :]  # [CLS] token, shape [B, 768]


def compute_embeddings(texts, embedder, batch_size=BATCH_SIZE):
    all_embeddings = []
    embedder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = [str(t) for t in texts[i:i + batch_size]]
            emb = embedder(batch).detach().cpu()
            all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def main():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV_PATH}")

    df = pd.read_csv(DATA_CSV_PATH)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')

    print(f"Loaded {len(df)} rows from {DATA_CSV_PATH}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embedder = TextEmbedder(model, tokenizer)

    print("Computing title embeddings...")
    title_embeddings = compute_embeddings(df['title'].tolist(), embedder)
    torch.save(title_embeddings, TITLE_OUT_PATH)
    print(f"Saved title embeddings: {title_embeddings.shape} -> {TITLE_OUT_PATH}")

    print("Computing description embeddings...")
    desc_embeddings = compute_embeddings(df['description'].tolist(), embedder)
    torch.save(desc_embeddings, DESC_OUT_PATH)
    print(f"Saved description embeddings: {desc_embeddings.shape} -> {DESC_OUT_PATH}")


if __name__ == "__main__":
    main()

