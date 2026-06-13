"""Train the 3-class Apple-Watch model on the Kaggle anchor (bootstrap).

Trains the Kaggle-only model — the project's **Model 1 (baseline)** — on the
Kaggle Gym Workout IMU dataset (Apple Watch, 100 Hz, single subject —
DECISIONS.md) for three classes: ``bicep_curl``, ``row`` and ``tricep_extension``.
The pipeline (shared with the app and the rebuild via ``ml4b.models.pipeline``) is:

    load (3-class Kaggle) -> sliding window (200 @ 100 Hz, 50% overlap)
        -> augment 6x (rotation + time-warp + mirror + jitter, DECISIONS.md)
        -> invariant features (DECISIONS.md)
        -> Random Forest (class_weight='balanced', seed 42) + novelty detector

Evaluation uses **leave-one-set-out** cross-validation (each Kaggle file is one
set / group), so windows from the same set never appear in both train and test —
the only honest estimate for a single-subject dataset. Augmented copies of a
held-out set are excluded from its training folds, so there is no leakage.

Because there is no Testdaten here, this script writes the SAME Kaggle-only model
to BOTH the baseline slot and the current slot, so the app's two-model comparison
always has both models present. ``make update``
(``scripts/rebuild_from_testdaten.py``) then retrains the current slot on
Kaggle + Testdaten, at which point Model 2 diverges from the frozen baseline and
the comparison shows the effect of our own data.

Outputs (committed so the app runs with no dataset — DECISIONS.md):
    models/saved/best_model.joblib            (Model 2 slot — Kaggle-only here)
    models/saved/random_forest.joblib
    models/saved/novelty_detector.joblib
    models/saved/model_metrics.json
    models/saved/baseline_model.joblib        (Model 1 — Kaggle-only)
    models/saved/baseline_novelty_detector.joblib
    models/saved/baseline_metrics.json
    data/processed/feature_names.txt

Run with:
    uv run python scripts/train_model.py
"""

from __future__ import annotations

import json

import joblib

from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.kaggle_loader import TARGET_CLASSES, load_kaggle_3class
from ml4b.data.windowing import apply_sliding_window
from ml4b.models.pipeline import (
    N_AUGMENT,
    build_metrics_payload,
    build_training_matrix,
    fit_model_and_novelty,
    gate_calibration,
    leave_one_set_out_cv,
)
from ml4b.utils.config import (
    BASELINE_METRICS_FILE,
    BASELINE_MODEL_FILE,
    BASELINE_NOVELTY_FILE,
    BEST_MODEL_FILE,
    DATA_PROCESSED,
    METRICS_FILE,
    MODELS_DIR,
    NOVELTY_FILE,
    REPORTS_DIR,
)

# Fixed class order for reproducible reports and confusion-matrix axes.
CLASS_NAMES = sorted(TARGET_CLASSES)


def main() -> None:
    """Run the end-to-end Kaggle-only training pipeline and persist artifacts."""
    print("=" * 64)
    print("ML4B Training — 3-class Apple-Watch model (Kaggle anchor / baseline)")
    print("=" * 64)

    print("Step 1/5: Loading 3-class Kaggle data...")
    raw_df = load_kaggle_3class()
    n_sets = raw_df["recording_id"].nunique()
    print(f"  Rows: {len(raw_df):,} | sets: {n_sets} | classes: {CLASS_NAMES}")

    print(f"Step 2/5: Sliding window (size={WINDOW_SIZE} @100Hz, overlap={OVERLAP})...")
    window_df = apply_sliding_window(
        raw_df, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    n_orig = len(window_df)
    print(f"  Original windows: {n_orig:,}")

    gate_stats = gate_calibration(window_df)
    print(
        f"  Activity gate: {gate_stats['fraction_kept_active'] * 100:.1f}% of "
        "exercise windows clear the rest thresholds (want ~100%)."
    )

    print(f"Step 3/5: Augment (n_augment={N_AUGMENT}, seed=42) + invariant features...")
    tm = build_training_matrix(window_df, n_augment=N_AUGMENT)
    print(f"  Features: {len(tm.feature_names)} | windows after aug: {len(tm.X):,}")

    print("Step 4/5: Leave-one-set-out cross-validation (honest metric)...")
    cv_results = leave_one_set_out_cv(tm, CLASS_NAMES, REPORTS_DIR)
    print(f"  CV Macro F1 : {cv_results['macro_f1']:.4f}")
    print(f"  CV Accuracy : {cv_results['accuracy']:.4f}")
    for cls, f1 in cv_results["per_class_f1"].items():
        print(f"    {cls:<18} {f1:.4f}")

    print("Step 5/5: Training final model + novelty detector + saving...")
    model, detector = fit_model_and_novelty(tm)
    payload = build_metrics_payload(
        cv_results,
        class_names=CLASS_NAMES,
        n_sets=n_sets,
        n_orig=n_orig,
        n_features=len(tm.feature_names),
        n_augment=N_AUGMENT,
        gate_stats=gate_stats,
        evaluation="leave-one-set-out (single subject; no LOSO possible)",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 keeps the model identical but shrinks the on-disk size ~5x so it
    # stays well under GitHub's 100 MB per-file limit (the model is committed).
    # No Testdaten here -> write the SAME Kaggle-only model to BOTH slots so the
    # app's two-model comparison always has both models; `make update` then
    # retrains the current slot on Kaggle + Testdaten.
    for model_file in (BEST_MODEL_FILE, BASELINE_MODEL_FILE, "random_forest.joblib"):
        joblib.dump(model, MODELS_DIR / model_file, compress=3)
    for novelty_file in (NOVELTY_FILE, BASELINE_NOVELTY_FILE):
        joblib.dump(detector, MODELS_DIR / novelty_file, compress=3)
    for metrics_file in (METRICS_FILE, BASELINE_METRICS_FILE):
        (MODELS_DIR / metrics_file).write_text(json.dumps(payload, indent=2))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(tm.feature_names))
    # Keep the legacy reports/ copy for the docs (reports/ is gitignored).
    (REPORTS_DIR.parent / "leave_one_set_out_results.json").write_text(
        json.dumps(payload, indent=2)
    )

    print(f"  Saved models -> {MODELS_DIR}")
    print("=" * 64)
    print("TRAINING COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
