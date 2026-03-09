# EVONS

This folder contains EVONS-specific data notes and preprocessing scripts.

EVONS is used in this project for two tasks:

1. Disinformation detection
2. Virality prediction

## What Is In This Folder

```text
evons/
    data/                    # Dataset file
        embeddings/          # Precomputed embedding tensors (.pt)
    data_preprocessing/      # Scripts to generate embeddings
        create_embeddings.py
        create_embeddings_mistral.py
```

## Setup Order

1. Read `evons/data/readme.md` and place required files in `evons/data/`.
2. If precomputed embeddings are available, keep them as-is.
3. Only if needed, regenerate embeddings with scripts in `evons/data_preprocessing/`.

Run commands from repository root (`paper_disinfo/`).

## Data Representation

Each EVONS record includes:

- News title
- News description/caption
- Source information
- Engagement fields used in virality settings
- Label fields used in supervised training

Embedding scripts transform title and description text into tensors used by training scripts.

## Where Training Happens

Training is not run from this folder. Use scripts in `training/`:

- `training/evons_disinformation_detection.py`
- `training/evons_virality_prediction.py`

Those scripts evaluate classical and neural baselines (for example logistic regression, random forest, optional XGBoost, and MLP variants) using BERT and Mistral embeddings.
