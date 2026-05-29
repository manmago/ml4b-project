"""One-shot training script for ML4B gym exercise recognition.

Trains the full pipeline (load -> window -> features -> split -> train)
and saves best_model.joblib, random_forest.joblib and feature_names.txt.

Run with:
    uv run python scripts/train_model.py

This script is the alternative to running notebooks 03 + 04 manually.
Use it whenever the Jupyter kernel is unavailable, or to reproduce the
trained model from scratch (requires the RecoFit .mat dataset).

If processed feature CSVs already exist in data/processed/, the expensive
load + window + feature-extraction steps are skipped and the script jumps
straight to model training — so it also works without the raw dataset.
"""

import time

import joblib
import pandas as pd

from ml4b.data.features import extract_features
from ml4b.data.loader import load_recofit_raw
from ml4b.data.splitting import subject_based_split, undersample_majority_class
from ml4b.data.windowing import apply_sliding_window
from ml4b.models.evaluate import evaluate_model
from ml4b.models.train import train_random_forest
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR, find_project_root

# Resolve all paths from the project root — never hardcode absolute paths.
PROJECT_ROOT = find_project_root()
MAT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "recofit" / "exercise_data.50.0000_singleonly.mat"
)

# Identifier columns carried through the feature frame that are NOT model inputs.
ID_COLUMNS = ["subject_id", "exercise_name", "window_id"]


def main() -> None:
    """Run the end-to-end training pipeline and persist the model artifacts."""
    print("=" * 60)
    print("ML4B Training Script")
    print("=" * 60)

    # Reuse already-processed features when present so the script runs even
    # without the 2.5 GB raw dataset (and is much faster on repeat runs).
    if (DATA_PROCESSED / "train_features.csv").exists():
        print("Processed data found — skipping to model training")
        train_df = pd.read_csv(DATA_PROCESSED / "train_features.csv")
        val_df = pd.read_csv(DATA_PROCESSED / "val_features.csv")
        feature_names = (
            (DATA_PROCESSED / "feature_names.txt").read_text().strip().split("\n")
        )
    else:
        print("Step 1/5: Loading raw data...")
        raw_df = load_recofit_raw(MAT_FILE)
        print(f"  Raw shape: {raw_df.shape}")

        print("Step 2/5: Applying sliding window...")
        window_df = apply_sliding_window(raw_df, window_size=100, overlap=0.5)
        print(f"  Windows: {len(window_df)}")

        print("Step 3/5: Extracting features...")
        feature_df = extract_features(window_df)
        feature_names = [c for c in feature_df.columns if c not in ID_COLUMNS]
        print(f"  Features: {len(feature_names)}")

        print("Step 4/5: Splitting data (subject-based, seed=42)...")
        train_df, val_df, test_df = subject_based_split(
            feature_df, test_size=0.2, val_size=0.1, random_state=42
        )
        # Cap the dominant 'rest' class on the TRAIN split only — see ADR-008.
        train_df = undersample_majority_class(
            train_df, majority_class="rest", multiplier=2.0, random_state=42
        )

        # Persist processed splits so future runs (and the app) skip recompute.
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(DATA_PROCESSED / "train_features.csv", index=False)
        val_df.to_csv(DATA_PROCESSED / "val_features.csv", index=False)
        test_df.to_csv(DATA_PROCESSED / "test_features.csv", index=False)
        (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Step 5/5: Training Random Forest...")
    X_train = train_df[feature_names].values
    y_train = train_df["exercise_name"].values
    X_val = val_df[feature_names].values
    y_val = val_df["exercise_name"].values

    t0 = time.time()
    model = train_random_forest(X_train, y_train, random_state=42)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # Validation-set sanity check — primary metric is macro F1 (ADR-008).
    results = evaluate_model(model, X_val, y_val, "Random Forest", sorted(set(y_val)))
    print(f"  Val Macro F1: {results['macro_f1']:.4f}")
    print(f"  Val Accuracy: {results['accuracy']:.4f}")

    # Save under both names: best_model.joblib (what the app loads) and
    # random_forest.joblib (algorithm-specific archive copy).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    print(f"Model saved to {MODELS_DIR}/best_model.joblib")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
