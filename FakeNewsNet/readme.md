## FakeNewsNet

This folder contains FakeNewsNet-specific preprocessing guidance and scripts (Politifact split).

FakeNewsNet is used in this project for two tasks:

1. Disinformation detection
2. Virality prediction

Preprocessing combines per-tweet text embeddings with lightweight metadata.

Training is run with code in:

- `training/fakenewsnet_disinformation_detection.py`
- `training/fakenewsnet_virality_prediction.py`

## Recommended Pipeline

Use this workflow from repository root (`paper_disinfo/`):

1. Place anonymized propagation files in `FakeNewsNet/data/`:
  - `anonymized_fake_propagation_paths.jsonl`
  - `anonymized_real_propagation_paths.jsonl`
2. Set your API token:
  - `export TWITTER_BEARER_TOKEN="..."`
3. Rehydrate tweet and user fields:
  - `python FakeNewsNet/data_preprocessing/rebuild_dataset.py`
4. Confirm generated ordered files:
  - `FakeNewsNet/data/ordered_fake_propagation_paths.jsonl`
  - `FakeNewsNet/data/ordered_real_propagation_paths.jsonl`
5. Generate embeddings from ordered JSONL files:
  - BERT/Roberta: `python FakeNewsNet/data_preprocessing/create_embeddings.py`
  - Mistral: `python FakeNewsNet/data_preprocessing/create_embeddings_mistral.py`
  - Embeddings will be saved in `FakeNewsNet/data/embeddings/` folder

## Folder Structure

```
data/                      # Raw / intermediate data and precomputed embeddings
data_preprocessing/        # Scripts to build standardized sequence datasets
  rebuild_dataset.py       # Rehydrates anonymized IDs into tweet/user fields via Twitter/X API
  path_creation.py         # Creates ordered propagation JSONL directly from raw FakeNewsNet export
  create_embeddings.py     # Generates BERT text embeddings tensors
  create_embeddings_mistral.py # Generates Mistral embeddings through the Mistral API
```

## Alternative / Legacy Input Paths

- If you already have full raw FakeNewsNet JSON tweets locally, run `path_creation.py` to directly produce ordered JSONL files.

## Data Representation

Each propagation is represented as a short sequence of tweets.

Per tweet features include:

- Text embedding (BERT, with optional Mistral variants)
- Scalar metadata (verification flag, follower/following counts, favorites, elapsed time, and similar variables)

Instructions on how to download data are in the [data](./data) folder.

## Training

These training scripts evaluate classical and neural baselines (for example logistic regression, random forest, optional XGBoost, and MLP variants) with BERT and Mistral embeddings.