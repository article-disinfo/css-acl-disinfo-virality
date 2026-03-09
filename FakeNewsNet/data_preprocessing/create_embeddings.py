from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from pathlib import Path


class TextDataset(Dataset):
    """Simple dataset holding raw texts; tokenization happens in collate_fn."""

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self) -> int:  
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def collate_fn(self, batch):
        return self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )


def generate_embeddings(
    input_jsonl_path: str,
    output_pt_path: str,
    model_name: str = "FacebookAI/roberta-base",
    batch_size: int = 32,
) -> None:
    """Generate and persist nested embeddings.

    Args:
        input_jsonl_path: Path to JSONL with one JSON array (sequence) per line.
        output_pt_path: Destination .pt file to save nested list of tensors.
        model_name: HF model identifier.
        batch_size: Batch size for inference.
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load nested sequences
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        nested_data = [json.loads(line) for line in f]

    # Flatten texts
    flat_texts = []
    lengths = []

    for sequence in nested_data:
        lengths.append(len(sequence))
        flat_texts.extend([item['text'] for item in sequence])

    total_texts = sum(lengths)
    assert total_texts == len(flat_texts), "Length accounting mismatch"  # sanity

    # Prepare tokenizer/model once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = TextDataset(flat_texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    hidden_states = []
    with torch.no_grad():
        for batch in tqdm(dataloader):

            batch = {k: v.to(device) for k, v in batch.items()} #move to device

            outputs = model(**batch)

            token_embeddings = outputs.last_hidden_state  # (B, T, H)

            emb = token_embeddings[:, 0, :]  # (B, H)

            hidden_states.append(emb.cpu())

    all_embeddings = torch.cat(hidden_states, dim=0)

    assert all_embeddings.size(0) == total_texts, "Mismatch in embedding count"  # sanity

    # Reconstruct nested structure (list of list of tensors)
    nested_embeddings = []
    idx = 0
    for length in lengths:
        nested_embeddings.append(list(all_embeddings[idx : idx + length]))
        idx += length

    torch.save(nested_embeddings, output_pt_path)

    print(
        f"Saved {len(nested_embeddings)} sequences / {total_texts} texts -> {output_pt_path} (shape per embedding {all_embeddings.shape[1]})"
    )


def main():
    DATA_DIR = Path().cwd() / 'FakeNewsNet' / 'data'
    EMB_DIR = DATA_DIR / 'embeddings'
    EMB_DIR.mkdir(exist_ok=True)  
    # Adjust these relative paths if directory layout changes
    generate_embeddings(
        input_jsonl_path=f"{DATA_DIR}/ordered_fake_propagation_paths.jsonl",
        output_pt_path=f"{EMB_DIR}/ordered_fake_propagation_paths_emb.pt",
    )
    generate_embeddings(
        input_jsonl_path=f"{DATA_DIR}/ordered_real_propagation_paths.jsonl",
        output_pt_path=f"{EMB_DIR}/ordered_real_propagation_paths_emb.pt",
    )


if __name__ == "__main__":
    main()
