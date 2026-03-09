# # FakeNewsNet — Disinformation detection (merged standalone)
# 
# Notebook autonome pour la tâche de détection de désinformation, sans exécuter d'autres notebooks.
# 
# Contenu:
# - Chargement des JSONL (`ordered_fake_propagation_paths.jsonl`, `ordered_real_propagation_paths.jsonl`)
# - Chargement des embeddings BERT et Mistral pré-calculés
# - Construction des labels fake(1)/real(0)
# - Évaluation cross-validation (MLP + baselines classiques)
# 

# %%

from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path().cwd()
DATA_DIR = ROOT / 'FakeNewsNet' / 'data'

required = [
    DATA_DIR / 'ordered_fake_propagation_paths.jsonl',
    DATA_DIR / 'ordered_real_propagation_paths.jsonl',
    DATA_DIR / 'embeddings' / 'ordered_fake_propagation_paths_emb.pt',
    DATA_DIR / 'embeddings' / 'ordered_real_propagation_paths_emb.pt',
    DATA_DIR / 'embeddings' / 'ordered_fake_propagation_paths_emb_mistral.pt',
    DATA_DIR / 'embeddings' / 'ordered_real_propagation_paths_emb_mistral.pt',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Fichiers manquants\n' + '\n'.join(missing))

print('All required files found.')
print('XGBoost available:', HAS_XGB)


# %%

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def to_tensor_embeddings(x):
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        # Per ogni sequenza, fai media dei suoi embedding
        pooled = []
        for seq in x:
            seq_tensor = torch.stack([torch.as_tensor(e) if not isinstance(e, torch.Tensor) else e for e in seq]).float()
            pooled.append(seq_tensor.mean(dim=0))  # media lungo la sequenza
        t = torch.stack(pooled)
    elif isinstance(x, torch.Tensor):
        t = x.float()
    elif isinstance(x, list):
        t = torch.stack([torch.as_tensor(e) if not isinstance(e, torch.Tensor) else e for e in x]).float()
    else:
        t = torch.as_tensor(x).float()

    if t.ndim != 2:
        raise ValueError(f'Unexpected embedding shape: {tuple(t.shape)}')
    return t


def load_features(mode='bert'):
    fake_json = read_jsonl(DATA_DIR / 'ordered_fake_propagation_paths.jsonl')
    real_json = read_jsonl(DATA_DIR / 'ordered_real_propagation_paths.jsonl')

    if mode == 'bert':
        emb_fake = torch.load(DATA_DIR / 'embeddings' / 'ordered_fake_propagation_paths_emb.pt', map_location='cpu')
        emb_real = torch.load(DATA_DIR / 'embeddings' / 'ordered_real_propagation_paths_emb.pt', map_location='cpu')
    elif mode == 'mistral':
        emb_fake = torch.load(DATA_DIR / 'embeddings' / 'ordered_fake_propagation_paths_emb_mistral.pt', map_location='cpu')
        emb_real = torch.load(DATA_DIR / 'embeddings' / 'ordered_real_propagation_paths_emb_mistral.pt', map_location='cpu')
    else:
        raise ValueError('mode must be bert or mistral')

    X_fake = to_tensor_embeddings(emb_fake)
    X_real = to_tensor_embeddings(emb_real)

    n_fake = min(len(fake_json), len(X_fake))
    n_real = min(len(real_json), len(X_real))

    X = torch.cat([X_fake[:n_fake], X_real[:n_real]], dim=0)
    y = np.concatenate([np.ones(n_fake, dtype=int), np.zeros(n_real, dtype=int)])
    return X, y


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    out['auc'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return out


# %%

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def run_mlp_cv(X, y, n_splits=10, epochs=12, batch_size=64, lr=1e-3):
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    for fold, (tr, va) in enumerate(skf.split(X.numpy(), y), 1):
        model = MLP(in_dim=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        dl_tr = DataLoader(FeatureDataset(X[tr], y[tr]), batch_size=batch_size, shuffle=True)
        dl_va = DataLoader(FeatureDataset(X[va], y[va]), batch_size=batch_size, shuffle=False)

        model.train()
        for _ in range(epochs):
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in dl_va:
                probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        probs = np.concatenate(probs)

        m = compute_metrics(y[va], probs)
        m['fold'] = fold
        rows.append(m)

    return pd.DataFrame(rows)


def run_baselines_cv(X, y, n_splits=10):
    Xn = X.numpy()
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=1000, n_jobs=-1),
        'rf': RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
        )

    rows = []
    for name, model in models.items():
        for fold, (tr, va) in enumerate(skf.split(Xn, y), 1):
            model.fit(Xn[tr], y[tr])
            prob = model.predict_proba(Xn[va])[:, 1] if hasattr(model, 'predict_proba') else 1/(1+np.exp(-model.decision_function(Xn[va])))
            m = compute_metrics(y[va], prob)
            m['model'] = name
            m['fold'] = fold
            rows.append(m)

    return pd.DataFrame(rows)


# %%

results = []
for mode in ['bert', 'mistral']:
    print(f'\n=== Running disinformation ({mode}) ===')
    X, y = load_features(mode=mode)
    print('Shape:', tuple(X.shape), '| Positive rate:', round(y.mean(), 4))

    mlp_df = run_mlp_cv(X, y)
    mlp_df['model'] = f'mlp_{mode}'
    results.append(mlp_df)

    base_df = run_baselines_cv(X, y)
    base_df['model'] = base_df['model'] + f'_{mode}'
    results.append(base_df)

all_results = pd.concat(results, ignore_index=True)
summary = all_results.groupby('model')[['accuracy', 'f1', 'precision', 'recall', 'auc']].mean().sort_values('f1', ascending=False)
summary


# %%

OUT_DIR = ROOT / 'training' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
all_results.to_csv(OUT_DIR / 'fakenewsnet_disinformation_merged_cv_results.csv', index=False)
summary.to_csv(OUT_DIR / 'fakenewsnet_disinformation_merged_summary.csv')
print('Saved results to', OUT_DIR)


# %%
PUB_SUMMARY_FILENAME = 'fakenewsnet_disinformation_publication_summary.md'

# %%
# Build a publication-ready markdown synthesis from summary dataframe
summary_for_pub = summary.copy()
summary_for_pub = summary_for_pub.reset_index()
summary_for_pub[['accuracy', 'f1', 'precision', 'recall', 'auc']] = summary_for_pub[['accuracy', 'f1', 'precision', 'recall', 'auc']].round(4)

top_model = summary_for_pub.iloc[0]
md_lines = [
    '# Synthèse des résultats',
    '',
    f"- Nombre de modèles comparés: **{len(summary_for_pub)}**",
    f"- Meilleur modèle (selon F1): **{top_model['model']}**",
    f"- F1 du meilleur modèle: **{top_model['f1']:.4f}**",
    '',
    '## Tableau comparatif (moyenne CV)',
    '',
    summary_for_pub.to_markdown(index=False),
    '',
    '## Top 3 (F1)',
    '',
    summary_for_pub.head(3).to_markdown(index=False),
]

pub_md_path = OUT_DIR / PUB_SUMMARY_FILENAME
pub_md_path.write_text('\n'.join(md_lines), encoding='utf-8')
print('Saved publication summary to', pub_md_path)

# %%



