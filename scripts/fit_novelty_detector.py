"""Fit the open-set novelty detector for the 3-class Apple-Watch model.

The shipped Random Forest is closed-set: it labels every active window as one of
its three exercises, even when the user performs an exercise it was never trained
on. This script fits a :class:`ml4b.data.novelty.NoveltyDetector` that flags such
out-of-distribution windows as ``unknown`` (DECISIONS.md).

It regenerates the SAME device-invariant features the model is trained on, from
the SAME Kaggle 3-class source (``data/raw/kaggle_gym_imu``), but WITHOUT
augmentation — the detector must be calibrated on the real motion distribution,
not synthetic copies. The feature column order is read from
``data/processed/feature_names.txt`` so it matches the model exactly.

Output (committed so the app runs with no dataset — same rationale as DECISIONS.md):
    models/saved/novelty_detector.joblib

Run with:
    uv run python scripts/fit_novelty_detector.py
"""

from __future__ import annotations

import joblib

from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.features_invariant import extract_invariant_features
from ml4b.data.kaggle_loader import load_kaggle_3class
from ml4b.data.novelty import NoveltyDetector
from ml4b.data.windowing import apply_sliding_window
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR


def main() -> None:
    """Fit the novelty detector on the invariant Kaggle features and save it."""
    print("=" * 64)
    print("ML4B — Fitting open-set novelty detector (DECISIONS.md)")
    print("=" * 64)

    # Read the model's feature order so the detector lives in the SAME space the
    # classifier sees at inference time.
    feature_names = (
        (DATA_PROCESSED / "feature_names.txt").read_text().strip().split("\n")
    )
    print(f"Loaded {len(feature_names)} feature names from feature_names.txt")

    print("Step 1/3: Loading 3-class Kaggle data...")
    raw_df = load_kaggle_3class()
    print(f"  Rows: {len(raw_df):,} | sets: {raw_df['recording_id'].nunique()}")

    print(f"Step 2/3: Windowing (size={WINDOW_SIZE} @100Hz, overlap={OVERLAP})...")
    # No augmentation: calibrate the detector on the genuine motion distribution.
    window_df = apply_sliding_window(
        raw_df, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    feats = extract_invariant_features(window_df)
    X = feats[feature_names].to_numpy()
    y = feats["exercise_name"].to_numpy()
    print(f"  Windows: {len(feats):,}")

    print("Step 3/3: Fitting NoveltyDetector + calibrating thresholds...")
    detector = NoveltyDetector().fit(X, y, feature_names)
    for cls in detector.classes_:
        print(f"  threshold[{cls:<18}] = {detector.thresholds_[cls]:.3f}")

    # Sanity check: nearly all of the training windows should be "known" (the
    # calibration percentile sets this; ~99% is expected, never far below).
    known_frac = float(detector.is_known(X).mean())
    print(f"  Fraction of training windows kept as known: {known_frac:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "novelty_detector.joblib"
    joblib.dump(detector, out_path, compress=3)
    print(f"  Saved detector -> {out_path}")
    print("=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()
