# EVONS Data

All EVONS-related scripts expect input files in `evons/data/`.

## Required Files

1. Dataset CSV (`evons.csv`)
2. Precomputed embedding tensors (`*.pt`) for title and description fields

## Download Steps

1. Get the EVONS dataset by following the official curator instructions:
	- https://github.com/krstovski/evons
2. Download precomputed embeddings from:
	- https://drive.google.com/drive/folders/1X27WjPEKzAcC5jXai8cuI8FEmWkjJb6l?usp=sharing
3. Place the dataset CSV (`evons.csv`) in `evons/data/` and the embedding `.pt` files in `evons/data/embeddings/`.

## Optional: Regenerate Embeddings

If you do not want to use precomputed files, generate embeddings with scripts in:

- `evons/data_preprocessing/create_embeddings.py` (BERT)
- `evons/data_preprocessing/create_embeddings_mistral.py` (Mistral)

Run these commands from repository root (`paper_disinfo/`) so relative paths resolve correctly.

## Quick Check

Before training, verify that:

- `evons/data/` contains `evons.csv`
- `evons/data/embeddings/` contains `title_embeddings.pt` and `desc_embeddings.pt` (or Mistral equivalents)