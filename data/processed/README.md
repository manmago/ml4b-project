# Processed Data

This folder holds the **output of the Phase 3 data-preparation pipeline**
(`notebooks/03_data_preparation.ipynb` / `src/ml4b/data/`).

Everything here is in `.gitignore` **except `feature_names.txt`**, which is
committed (it is the single source of truth for the model's feature order — the
app needs it without the dataset). `.gitkeep` keeps the directory on a fresh clone.

## Contents

| File | Tracked? | Purpose |
|------|:--------:|---------|
| `feature_names.txt` | ✅ in git | Newline-separated names of the **39 device-invariant** feature columns, in order (DECISIONS.md; see `docs/data/data_dictionary.md`). |
| `train_features.csv` / `val_features.csv` / `test_features.csv` | ❌ ignored | Optional local feature dumps from exploration — regenerated from the dataset, never committed. |

The current pipeline windows each recording into **200-sample (2 s @ 100 Hz)**
windows with 50 % overlap and extracts **39 device-invariant features**
(`src/ml4b/data/features_invariant.py`). The model is evaluated with
**leave-one-set-out** cross-validation grouped by recording — not a subject split.
Full column reference: `docs/data/data_dictionary.md`.

## Reproducing the model
The shipped model is rebuilt deterministically from the committed artifacts:

1. Place the Kaggle dataset under `data/raw/kaggle_gym_imu/` (bootstrap) and/or
   commit recordings under `data/Testdaten/<Exercise>/` (continual learning).
2. Run `make train` (Kaggle-only baseline) or `make update`
   (Kaggle + Testdaten — the canonical rebuild). See `docs/DECISIONS.md` §8.

## Naming convention
Lowercase snake_case. Do not commit any file from this folder except
`feature_names.txt`.
