import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from mistralai import Mistral
from pathlib import Path

import json

from tqdm.notebook import tqdm

api_key = 'YOUR-KEY-HERE'
model = "mistral-embed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Step 2: Dataloader Dataset =====
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def generate_embedding(input_jsonl_path, output_pt_path):
    # ===== Step 0: Load data =====
    with open(input_jsonl_path, 'r') as f:
        nested_data = [json.loads(line) for line in f]

    # ===== Step 1: Flatten the data and track structure =====
    flat_texts = []
    lengths = []
    for sequence in nested_data:
        lengths.append(len(sequence))
        flat_texts.extend([item['text'] for item in sequence])

    # ===== Step 3: Dataloader with batch size =====
    batch_size = 256  # puoi modificarlo in base alla memoria disponibile
    dataset = TextDataset(flat_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ===== Step 4: Create client and embed =====
    client = Mistral(api_key=api_key)

    all_embeddings = []
    for batch in tqdm(dataloader):
        embeddings_batch_response = client.embeddings.create(
            model=model,
            inputs=batch
        )
        embeddings = [x.embedding for x in embeddings_batch_response.data]
        all_embeddings.extend(embeddings)

    all_embeddings_tensor = torch.stack([torch.tensor(x) for x in all_embeddings], dim=0).cpu()

    # ===== Step 5: Rebuilding original structure =====
    mistral_nested_embeddings = []
    index = 0
    for length in lengths:
        mistral_nested_embeddings.append(list(all_embeddings_tensor[index:index+length]))
        index += length

    torch.save(mistral_nested_embeddings, output_pt_path)

# ===== Use the function on both files =====

ROOT = Path().cwd()
DATA_DIR = ROOT / "FakeNewsNet" / "data"
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

generate_embedding(f"{DATA_DIR}/ordered_real_propagation_paths.jsonl",
                   f"{EMB_DIR}/ordered_real_propagation_paths_emb_mistral.pt")

generate_embedding(f"{DATA_DIR}/ordered_fake_propagation_paths.jsonl",
                   f"{EMB_DIR}/ordered_fake_propagation_paths_emb_mistral.pt")