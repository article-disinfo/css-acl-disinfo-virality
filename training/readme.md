# Training Outputs

Each script in `training/` writes result files to `training/outputs/`.

For every dataset-task experiment, three files are produced:

1. `*_merged_cv_results.csv`
2. `*_merged_summary.csv`
3. `*_publication_summary.md`

## File Meanings

1. `*_merged_cv_results.csv`
	- One row per fold and model
	- Detailed cross-validation metrics
	- Expected columns: `fold`, `model`, `accuracy`, `f1`, `precision`, `recall`, `auc`
2. `*_merged_summary.csv`
	- Mean metrics aggregated by model
	- Same metric columns as CV file
	- Typically sorted by `f1` in descending order
3. `*_publication_summary.md`
	- Human-readable summary used for paper drafting

## Current Experiments In This Repository

Expected output groups:

1. `evons_disinformation_*`
2. `evons_virality_*`
3. `fakenewsnet_disinformation_*`
4. `fakenewsnet_virality_*`

## How These Files Are Used Later

1. `tools/create_file_for_stat_eval.py` merges CV files into `report/combined_cv.csv`.
2. `tools/statistical_evaluation.py` creates `report/stats_current.csv`.
3. `tools/generate_paper_tables.py` generates LaTeX tables in `report/`.
