"""Build processed feature CSVs from the MM-Fit dataset.

This script replaces the RecoFit-based data preparation (see ADR-013). It loads
the wrist-worn smartwatch streams from MM-Fit, runs the **same** windowing and
feature-extraction pipeline used everywhere else in the project, and writes the
standard processed artifacts that ``scripts/train_model.py`` and the notebooks
consume::

    data/processed/train_features.csv
    data/processed/val_features.csv
    data/processed/test_features.csv
    data/processed/feature_names.txt

Splits follow MM-Fit's official workout-id partition (train/val/test), which is
a session/subject-level split — no workout appears in more than one split, so
there is no data leakage.

Run with:
    uv run python scripts/build_mmfit_dataset.py

Requires the unzipped dataset at ``data/raw/mm-fit/`` (download:
https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip).
"""

import argparse
import time

from ml4b.data.augmentation import augment_windows_with_rotation
from ml4b.data.features import extract_features
from ml4b.data.mmfit_loader import (
    TEST_W_IDS,
    TRAIN_W_IDS,
    VAL_W_IDS,
    load_mmfit_split,
)
from ml4b.data.splitting import undersample_majority_class
from ml4b.data.windowing import apply_sliding_window
from ml4b.utils.config import DATA_PROCESSED, DATA_RAW

# Identifier columns carried through the feature frame that are NOT model inputs.
ID_COLUMNS = ["subject_id", "exercise_name", "window_id"]

MMFIT_ROOT = DATA_RAW / "mm-fit"

# Map split name -> workout ids. Ordered so 'train' is built first (its column
# order defines feature_names.txt).
SPLITS = {
    "train": TRAIN_W_IDS,
    "val": VAL_W_IDS,
    "test": TEST_W_IDS,
}


def _build_split(name: str, workout_ids: list[str], n_rotations: int = 0):
    """Load -> window -> (augment) -> extract features for one split.

    Args:
        name: Split name ("train" / "val" / "test"), used only for logging.
        workout_ids: MM-Fit workout ids belonging to this split.
        n_rotations: Rotation-augmentation copies per window. Should be > 0 only
            for the train split; val/test are never augmented (honest metrics).

    Returns:
        Feature DataFrame (one row per window) for the split.
    """
    print(f"[{name}] loading {len(workout_ids)} workouts: {workout_ids}")
    raw = load_mmfit_split(MMFIT_ROOT, workout_ids)
    print(f"[{name}]   raw samples: {len(raw):,}")

    windows = apply_sliding_window(raw, window_size=100, overlap=0.5)
    print(f"[{name}]   windows: {len(windows):,}")

    if n_rotations > 0:
        windows = augment_windows_with_rotation(
            windows, n_rotations=n_rotations, random_state=42
        )
        print(f"[{name}]   windows after {n_rotations}x rotation aug: {len(windows):,}")

    feats = extract_features(windows)
    print(f"[{name}]   feature rows: {len(feats):,}")
    return feats


def main() -> None:
    """Build and persist train/val/test feature CSVs from MM-Fit."""
    parser = argparse.ArgumentParser(description="Build MM-Fit feature CSVs.")
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="Rotation-augmentation copies per TRAIN window (0 = off). "
        "Adds cross-device orientation robustness — see ADR-014.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MM-Fit dataset builder")
    print(f"  rotation augmentation: {args.augment} copies/window (train only)")
    print("=" * 60)
    if not MMFIT_ROOT.exists():
        raise FileNotFoundError(
            f"MM-Fit not found at {MMFIT_ROOT}. Download and unzip mm-fit.zip first "
            "(see scripts/build_mmfit_dataset.py docstring)."
        )

    t0 = time.time()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train_df = _build_split("train", TRAIN_W_IDS, n_rotations=args.augment)
    feature_names = [c for c in train_df.columns if c not in ID_COLUMNS]

    # Cap the dominant 'rest' class on the TRAIN split only — see ADR-008.
    # Rest is ~80% of windows in MM-Fit. multiplier=1.5 (down from 2.0) keeps
    # rest close to the largest exercise class, which reduces the model's
    # tendency to over-predict 'rest' on real Apple Watch data — see ADR-015.
    before = len(train_df)
    train_df = undersample_majority_class(
        train_df, majority_class="rest", multiplier=1.5, random_state=42
    )
    print(f"[train]   undersampled rest: {before:,} -> {len(train_df):,} rows")

    val_df = _build_split("val", VAL_W_IDS)
    test_df = _build_split("test", TEST_W_IDS)

    train_df.to_csv(DATA_PROCESSED / "train_features.csv", index=False)
    val_df.to_csv(DATA_PROCESSED / "val_features.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test_features.csv", index=False)
    (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))

    print("-" * 60)
    print(f"Features: {len(feature_names)}")
    print(f"Class distribution (train):\n{train_df['exercise_name'].value_counts()}")
    print(f"Wrote train/val/test CSVs to {DATA_PROCESSED}")
    print(f"Done in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
