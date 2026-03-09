# From Veracity to Diffusion

Code and reproducibility material for the paper:

_From Veracity to Diffusion: Addressing Operational Challenges in Moving From Fake-News Detection to Information Disorders_

## What This Repository Contains

This project compares two prediction targets on two datasets:

1. Disinformation detection (veracity-oriented task)
2. Virality prediction (diffusion-oriented task)

Datasets:

1. EVONS
2. FakeNewsNet (Politifact)

Each dataset-task pair has one training script in `training/`, for a total of four experiment scripts.

## Repository Structure

- `training/`: experiment scripts and generated outputs
- `tools/`: scripts for statistical evaluation and paper table generation
- `evons/`: EVONS data notes and preprocessing scripts
- `FakeNewsNet/`: FakeNewsNet data notes and preprocessing scripts
- `report/`: publication-oriented artifacts (combined CV file, stats, LaTeX tables)

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## Data Setup

Follow dataset-specific instructions before running experiments:

1. EVONS: `evons/readme.md` and `evons/data/readme.md`
2. FakeNewsNet: `FakeNewsNet/readme.md` and `FakeNewsNet/data/readme.md`

Important:

- Precomputed embeddings are expected in dataset data folders.
- Regenerating embeddings is optional and can be time-consuming.
- For FakeNewsNet rehydration, set `TWITTER_BEARER_TOKEN`.

## Run Experiments

Run from repository root:

```bash
python training/evons_disinformation_detection.py
python training/evons_virality_prediction.py
python training/fakenewsnet_disinformation_detection.py
python training/fakenewsnet_virality_prediction.py
```

Outputs are written to `training/outputs/`.

Per script, expected files are:

1. `*_merged_cv_results.csv`
2. `*_merged_summary.csv`
3. `*_publication_summary.md`

See `training/results_readme.md` for details.

## Build statistical evaluations

After all experiment outputs exist, run:

```bash
python tools/create_file_for_stat_eval.py
python tools/statistical_evaluation.py --input report/combined_cv.csv --output report/stats_current.csv
```

This produces:

- `report/combined_cv.csv`
- `report/stats_current.csv`

## Notes For Reproducibility

- Scripts use fixed CV settings (`n_splits=10`, `test_size=0.2`, `random_state=42`) where applicable.
- Weights & Biases should stay disabled for non-interactive runs.
- Some scripts conditionally use XGBoost if available.

## Abstract

Much misinformation research has focused on fake-news detection, framed as predicting veracity labels attached to articles or claims. Social-science work, however, has repeatedly shown that information manipulation also depends on amplification dynamics. This shift has direct implications for how prediction tasks are operationalized.

This repository compares fake-news detection and virality prediction on EVONS and FakeNewsNet with an evaluation-first perspective. We study how benchmark behavior changes when moving from veracity targets to diffusion targets. Results show that fake-news detection is relatively stable once strong text embeddings are available, while virality prediction is more sensitive to operational choices such as threshold definition and early observation windows.