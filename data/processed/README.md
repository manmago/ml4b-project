# Processed Data

This folder holds the **output of the Phase 3 data preparation pipeline** (`notebooks/03_data_preparation.ipynb`).

It is in `.gitignore` — nothing here is ever committed. The `.gitkeep` file is the only tracked entry so the directory exists on a fresh clone.

## Expected Contents (after running notebook 03)

| File | Purpose | Approx. size |
|------|---------|--------------|
| `train_features.csv` | ~70% of subjects, feature matrix + labels | a few MB |
| `val_features.csv`   | ~10% of subjects, feature matrix + labels | <1 MB |
| `test_features.csv`  | ~20% of subjects, feature matrix + labels | a few MB |
| `feature_names.txt`  | Newline-separated names of the 47 numeric feature columns — single source of truth for downstream notebooks | <1 KB |

Every CSV row corresponds to one 2 s sensor window (100 samples at 50 Hz) with the columns:

* `subject_id`, `exercise_name`, `window_id` — identifiers / label
* 42 per-axis statistical features (7 stats × 6 axes: ax, ay, az, gx, gy, gz)
* 3 magnitude features (`accel_magnitude_mean`, `accel_magnitude_std`, `gyro_magnitude_mean`)
* 2 frequency-domain features (`dominant_frequency`, `spectral_energy`)

## Reproducing These Files
1. Place the RecoFit `.mat` file at `data/raw/recofit/exercise_data.50.0000_singleonly.mat` (see `data/raw/recofit/README.md`).
2. Open and run `notebooks/03_data_preparation.ipynb` top-to-bottom.
3. The notebook ends with a sanity-check cell that asserts no subject overlap and no NaN/inf values.

## Naming Convention
Lowercase snake_case, `<split>_features.csv` for split feature matrices. Do not commit any file from this folder.
