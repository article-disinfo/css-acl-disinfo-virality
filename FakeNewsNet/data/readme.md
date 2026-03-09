# FakeNewsNet Data

All FakeNewsNet scripts expect data files in `FakeNewsNet/data/`.

Because of size limits and Twitter/X privacy rules, files are not stored in this repository.

## Source Links

1. Original raw dataset project (optional for this workflow):
   - https://github.com/KaiDMML/FakeNewsNet
2. Anonymized propagation files used by this repository:
   - https://drive.proton.me/urls/6ZDQPJY178#XMn79ZYqEcxt

## Required Workflow

Run commands from repository root (`paper_disinfo/`).

1. Download and place these files in `FakeNewsNet/data/`:
   - `anonymized_fake_propagation_paths.jsonl`
   - `anonymized_real_propagation_paths.jsonl`
2. Set your Twitter/X bearer token:
   - `export TWITTER_BEARER_TOKEN="..."`
3. Rehydrate records:
   - `python FakeNewsNet/data_preprocessing/rebuild_dataset.py`
4. Confirm generated ordered files:
   - `ordered_fake_propagation_paths.jsonl`
   - `ordered_real_propagation_paths.jsonl`
5. Generate embeddings:
   - BERT: `python FakeNewsNet/data_preprocessing/create_embeddings.py`
   - Mistral: `python FakeNewsNet/data_preprocessing/create_embeddings_mistral.py`

## Optional Legacy Route

If you have the full raw FakeNewsNet export, `FakeNewsNet/data_preprocessing/path_creation.py` can build ordered JSONL files directly.

## Notes

- This project uses the Politifact split.
- Rehydration requires valid Twitter/X API credentials.
- API rate limits can slow or interrupt retrieval.