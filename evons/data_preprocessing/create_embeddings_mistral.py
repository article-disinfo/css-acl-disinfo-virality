import os
import torch
import pandas as pd
from mistralai import Mistral
from tqdm import tqdm
from pathlib import Path

# ========= Configuration =========
API_KEY = 'YOUR-KEY-HERE'
MODEL_NAME = 'mistral-embed'
BATCH_SIZE = 256

# Relative paths (script expected to be run from project root)
ROOT = Path().cwd()
DATA_DIR = ROOT / 'evons' / 'data'
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV_PATH = DATA_DIR / "evons.csv"  # evons/data/evons.csv
TITLE_OUT_PATH = EMB_DIR / "title_embeddings_mistral.pt"
DESC_OUT_PATH = EMB_DIR / "desc_embeddings_mistral.pt"

# ========= Helper =========
def compute_embeddings(client, texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = [str(t) for t in texts[i:i+batch_size]]
        if not batch_texts:
            continue
        resp = client.embeddings.create(model=MODEL_NAME, inputs=batch_texts)
        batch_embs = [x.embedding for x in resp.data]
        embeddings.extend(batch_embs)
    return embeddings


def main():
    if API_KEY == 'YOUR-KEY-HERE':
        raise ValueError('Edit API_KEY.')

    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f'Missing data file at {DATA_CSV_PATH}. Place evons.csv there.')

    df = pd.read_csv(DATA_CSV_PATH)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')

    titles = df['title'].tolist()
    descs = df['description'].tolist()

    client = Mistral(api_key=API_KEY)

    print('Computing title embeddings...')
    title_embeddings = compute_embeddings(client, titles)
    torch.save(title_embeddings, TITLE_OUT_PATH)

    print('Computing description embeddings...')
    desc_embeddings = compute_embeddings(client, descs)
    torch.save(desc_embeddings, DESC_OUT_PATH)

    print('Done.')


if __name__ == '__main__':
    main()
