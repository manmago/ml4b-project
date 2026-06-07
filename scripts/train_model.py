"""Train the final 3-class Apple-Watch exercise-recognition model.

Trains the production model on the Kaggle Gym Workout IMU dataset (Apple Watch,
100 Hz, single subject — DECISIONS.md) for three classes: ``bicep_curl``, ``row`` and
``tricep_extension``. The pipeline is:

    load (3-class Kaggle) -> sliding window (200 @ 100 Hz, 50% overlap)
        -> augment (rotation + time-warp + mirror + jitter, DECISIONS.md)
        -> invariant features (DECISIONS.md)
        -> Random Forest (class_weight='balanced', seed 42)

Evaluation uses **leave-one-set-out** cross-validation (each Kaggle file is one
set / group), so windows from the same set never appear in both train and test —
the only honest estimate available for a single-subject dataset. True
leave-one-*subject*-out is impossible here (one subject); this limitation is
documented in DECISIONS.md. Augmented copies of a held-out set are also excluded from
its training folds, so there is no leakage through augmentation.

Outputs (committed so the app runs with no dataset — DECISIONS.md):
    models/saved/best_model.joblib
    models/saved/random_forest.joblib
    data/processed/feature_names.txt
    reports/figures/confusion_matrix_*.png
    reports/leave_one_set_out_results.json

Run with:
    uv run python scripts/train_model.py
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from ml4b.data.activity_gate import (
    ACCEL_MAG_STD_THRESHOLD,
    GYRO_MAG_MEAN_THRESHOLD,
    window_energy,
)
from ml4b.data.augmentation import augment_windows
from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.features_invariant import extract_invariant_features, feature_columns
from ml4b.data.kaggle_loader import TARGET_CLASSES, load_kaggle_3class
from ml4b.data.windowing import apply_sliding_window
from ml4b.models.evaluate import evaluate_model
from ml4b.models.train import train_random_forest
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

# Number of augmented copies per original window (→ 6× total). See DECISIONS.md.
N_AUGMENT = 5
# Fixed class order for reproducible reports and confusion-matrix axes.
CLASS_NAMES = sorted(TARGET_CLASSES)


def _report_gate_calibration(window_df: pd.DataFrame) -> dict[str, float]:
    """Sanity-check that the activity-gate thresholds clear real exercise.

    Computes the accel-magnitude-std and gyro-magnitude-mean of every original
    exercise window and reports low percentiles, so we can confirm the gate
    thresholds (DECISIONS.md) sit safely below genuine exercise energy.

    Args:
        window_df: Original (non-augmented) windows.

    Returns:
        Dict of percentile statistics and the fraction of windows the gate
        would (correctly) keep as active.
    """
    accel_stds: list[float] = []
    gyro_means: list[float] = []
    for _, row in window_df.iterrows():
        a_std, g_mean = window_energy(
            row["raw_ax"],
            row["raw_ay"],
            row["raw_az"],
            row["raw_gx"],
            row["raw_gy"],
            row["raw_gz"],
        )
        accel_stds.append(a_std)
        gyro_means.append(g_mean)
    accel_arr = np.array(accel_stds)
    gyro_arr = np.array(gyro_means)
    # An exercise window is kept active if it clears EITHER threshold.
    kept = np.mean(
        (accel_arr > ACCEL_MAG_STD_THRESHOLD) | (gyro_arr > GYRO_MAG_MEAN_THRESHOLD)
    )
    return {
        "accel_std_p01": float(np.percentile(accel_arr, 1)),
        "accel_std_p05": float(np.percentile(accel_arr, 5)),
        "accel_std_median": float(np.median(accel_arr)),
        "gyro_mean_p01": float(np.percentile(gyro_arr, 1)),
        "gyro_mean_p05": float(np.percentile(gyro_arr, 5)),
        "gyro_mean_median": float(np.median(gyro_arr)),
        "fraction_kept_active": float(kept),
    }


def main() -> None:
    """Run the end-to-end 3-class training pipeline and persist artifacts."""
    print("=" * 64)
    print("ML4B Training — 3-class Apple-Watch model (Kaggle anchor)")
    print("=" * 64)

    print("Step 1/6: Loading 3-class Kaggle data...")
    raw_df = load_kaggle_3class()
    n_sets = raw_df["recording_id"].nunique()
    print(f"  Rows: {len(raw_df):,} | sets: {n_sets} | classes: {CLASS_NAMES}")
    print("  Sets per class:")
    print(
        raw_df.groupby("exercise_name")["recording_id"]
        .nunique()
        .to_string(header=False)
    )

    print(f"Step 2/6: Sliding window (size={WINDOW_SIZE} @100Hz, overlap={OVERLAP})...")
    window_df = apply_sliding_window(
        raw_df, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    n_orig = len(window_df)
    print(f"  Original windows: {n_orig:,}")

    # Activity-gate calibration check (does not alter training labels).
    gate_stats = _report_gate_calibration(window_df)
    print(
        f"  Activity gate: {gate_stats['fraction_kept_active'] * 100:.1f}% of "
        "exercise windows clear the rest thresholds (want ~100%)."
    )

    print(f"Step 3/6: Augmenting (n_augment={N_AUGMENT}, seed=42)...")
    combined = augment_windows(window_df, n_augment=N_AUGMENT, random_state=42)
    print(f"  Total windows after augmentation: {len(combined):,}")

    print("Step 4/6: Extracting invariant features...")
    t0 = time.time()
    feats = extract_invariant_features(combined)
    feature_names = feature_columns(feats)
    # Mark which rows are augmented: augment_windows keeps originals first.
    is_augmented = np.arange(len(feats)) >= n_orig
    print(f"  Features: {len(feature_names)} | extracted in {time.time() - t0:.1f}s")

    X = feats[feature_names].to_numpy()
    y = feats["exercise_name"].to_numpy()
    groups = feats["recording_id"].to_numpy()

    print("Step 5/6: Leave-one-set-out cross-validation (honest metric)...")
    # For each held-out set, train on all OTHER sets (incl. their augmented
    # copies) and test ONLY on the held-out set's original windows. This is the
    # leak-free, single-subject-honest estimate (DECISIONS.md).
    logo = LeaveOneGroupOut()
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    t0 = time.time()
    for train_idx, test_idx in logo.split(X, y, groups):
        # Restrict the test set to ORIGINAL windows of the held-out set.
        test_idx = test_idx[~is_augmented[test_idx]]
        if len(test_idx) == 0:
            continue
        fold_model = train_random_forest(X[train_idx], y[train_idx], random_state=42)
        y_pred_all.extend(fold_model.predict(X[test_idx]).tolist())
        y_true_all.extend(y[test_idx].tolist())
    print(f"  {logo.get_n_splits(groups=groups)} folds in {time.time() - t0:.1f}s")

    # Aggregate the held-out predictions into honest metrics + confusion matrix.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results = evaluate_model(
        _ConstPredictor(y_pred_all),
        np.zeros((len(y_true_all), 1)),  # X unused by the const predictor
        np.array(y_true_all),
        "Leave-One-Set-Out CV",
        CLASS_NAMES,
        save_dir=REPORTS_DIR,
    )
    print(f"  CV Macro F1 : {cv_results['macro_f1']:.4f}")
    print(f"  CV Accuracy : {cv_results['accuracy']:.4f}")
    print("  Per-class F1:")
    for cls, f1 in cv_results["per_class_f1"].items():
        print(f"    {cls:<18} {f1:.4f}")

    print("Step 6/6: Training final model on all data + saving...")
    # The shipped model is trained on ALL sets and ALL augmented copies so it
    # uses every available example; the honest performance estimate is the CV
    # number above, not this model's training fit.
    final_model = train_random_forest(X, y, random_state=42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 keeps the model identical but shrinks the on-disk size ~5x so it
    # stays well under GitHub's 100 MB per-file limit (the model is committed so
    # the app runs without the dataset — DECISIONS.md).
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib", compress=3)
    joblib.dump(final_model, MODELS_DIR / "random_forest.joblib", compress=3)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))

    # Persist the honest results for the docs and the Model Performance page.
    results_payload = {
        "classes": CLASS_NAMES,
        "n_sets": int(n_sets),
        "n_original_windows": int(n_orig),
        "n_features": len(feature_names),
        "n_augment": N_AUGMENT,
        "evaluation": "leave-one-set-out (single subject; no LOSO possible)",
        "cv_macro_f1": round(float(cv_results["macro_f1"]), 4),
        "cv_accuracy": round(float(cv_results["accuracy"]), 4),
        "cv_per_class_f1": {
            k: round(v, 4) for k, v in cv_results["per_class_f1"].items()
        },
        "confusion_matrix": cv_results["confusion_matrix"].tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "activity_gate_calibration": gate_stats,
        "activity_gate_thresholds": {
            "accel_mag_std": ACCEL_MAG_STD_THRESHOLD,
            "gyro_mag_mean": GYRO_MAG_MEAN_THRESHOLD,
        },
    }
    (REPORTS_DIR.parent / "leave_one_set_out_results.json").write_text(
        json.dumps(results_payload, indent=2)
    )
    # Also write a committed copy next to the model so the Streamlit Model
    # Performance page can show the real, honest metrics after a fresh clone
    # (reports/ is gitignored; models/saved/model_metrics.json is not — DECISIONS.md).
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(results_payload, indent=2)
    )

    print(f"  Saved model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"  Saved features -> {DATA_PROCESSED / 'feature_names.txt'}")
    print("=" * 64)
    print("TRAINING COMPLETE")
    print("=" * 64)


class _ConstPredictor:
    """Minimal predictor that replays pre-computed labels for evaluate_model.

    The leave-one-set-out loop already produced per-window predictions; this thin
    adapter lets us reuse :func:`ml4b.models.evaluate.evaluate_model` (which
    calls ``model.predict(X)``) to compute the aggregate metrics and confusion
    matrix without retraining.
    """

    def __init__(self, predictions: list[str]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the stored predictions, ignoring ``X``."""
        return self._predictions


if __name__ == "__main__":
    main()
