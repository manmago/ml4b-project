"""Retrain the model from base data + user corrections (ADR-027).

This is the "continual learning" step. The Random Forest the project ships has no
incremental ``partial_fit``, and updating a model from a handful of corrections in
isolation would cause catastrophic forgetting. The robust, standard
human-in-the-loop pattern is therefore a **feedback-augmented retrain**: rebuild
the model from the full base training set *plus* the accumulated corrections,
using the identical windowing → augmentation → invariant-feature → Random Forest
pipeline as initial training (so nothing about the model contract changes except
that it has now also seen the user's own examples — and any new exercise labels
they introduced).

Because corrections are few relative to the thousands of base windows, they are
(a) repeated ``feedback_repeat`` times and (b) augmented like every other window,
so they carry real weight without us hand-tuning sample weights.

Retraining needs the base Kaggle dataset (``data/raw/kaggle_gym_imu/``). On a
fresh handover clone that is absent; :func:`base_available` reports this so the
caller can disable the action while still *collecting* feedback for later.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from ml4b.data.augmentation import augment_windows
from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.features_invariant import extract_invariant_features, feature_columns
from ml4b.data.kaggle_loader import load_kaggle_3class
from ml4b.data.windowing import apply_sliding_window
from ml4b.feedback import store
from ml4b.models.train import train_random_forest
from ml4b.utils.config import DATA_PROCESSED, DATA_RAW, MODELS_DIR

# Default number of augmented copies per window (→6×), matching initial training.
N_AUGMENT = 5
# Default times each correction is repeated before augmentation, to give a small
# number of corrections meaningful weight against the large base set.
FEEDBACK_REPEAT = 3

KAGGLE_DIR = DATA_RAW / "kaggle_gym_imu"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
# Pristine copy of the originally-shipped model, made before the first retrain so
# the base model can always be restored.
BASE_BACKUP_FILE = MODELS_DIR / "best_model_base.joblib"
MANIFEST_FILE = MODELS_DIR / "model_manifest.json"
FEATURE_NAMES_FILE = DATA_PROCESSED / "feature_names.txt"


def base_available() -> bool:
    """Return True if the base Kaggle dataset is present for retraining."""
    return KAGGLE_DIR.is_dir() and any(KAGGLE_DIR.glob("*.csv"))


def _base_window_df() -> pd.DataFrame:
    """Load and window the base 3-class Kaggle data (same as initial training)."""
    raw_df = load_kaggle_3class()
    return apply_sliding_window(
        raw_df, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )


def build_training_windows(feedback_repeat: int = FEEDBACK_REPEAT) -> pd.DataFrame:
    """Combine base windows with repeated feedback windows (pre-augmentation).

    Args:
        feedback_repeat: How many times to repeat each stored correction.

    Returns:
        A windowed DataFrame (base first, then the repeated feedback windows)
        ready for augmentation and feature extraction.
    """
    base = _base_window_df()
    fb_windows = store.to_window_df(store.load())
    if fb_windows.empty or feedback_repeat <= 0:
        return base
    repeated = pd.concat([fb_windows] * feedback_repeat, ignore_index=True)
    return pd.concat([base, repeated], ignore_index=True)


def retrain_model(
    feedback_repeat: int = FEEDBACK_REPEAT,
    n_augment: int = N_AUGMENT,
    random_state: int = 42,
    backup: bool = True,
) -> dict[str, Any]:
    """Retrain and persist the model on base data + accumulated corrections.

    Args:
        feedback_repeat: Times to repeat each correction before augmentation.
        n_augment: Augmented copies per window (→ ``n_augment + 1`` total).
        random_state: Seed for augmentation and the Random Forest.
        backup: If True, preserve the current model as ``best_model_base.joblib``
            the first time a retrain runs (so the base model is recoverable).

    Returns:
        The manifest dict describing the retrain (also written next to the model).

    Raises:
        RuntimeError: If the base dataset is unavailable, or there is no feedback
            to learn from.
    """
    if not base_available():
        raise RuntimeError(
            "Base dataset not found at data/raw/kaggle_gym_imu/. Retraining needs "
            "it so the model does not forget its base classes. Your corrections "
            "are safely stored and can be used once the dataset is present."
        )
    feedback = store.load()
    if feedback.empty:
        raise RuntimeError("No corrections recorded yet — nothing to learn from.")

    # Preserve the originally-shipped model once, before the first overwrite.
    if backup and MODEL_FILE.exists() and not BASE_BACKUP_FILE.exists():
        shutil.copy2(MODEL_FILE, BASE_BACKUP_FILE)

    base = _base_window_df()
    n_base = len(base)
    combined = build_training_windows(feedback_repeat=feedback_repeat)

    # Same augmentation + invariant features as initial training.
    augmented = augment_windows(
        combined, n_augment=n_augment, random_state=random_state
    )
    feats = extract_invariant_features(augmented)
    feature_names = feature_columns(feats)
    X = feats[feature_names].to_numpy()
    y = feats["exercise_name"].to_numpy()

    model = train_random_forest(X, y, random_state=random_state)

    # Persist the new model + feature list (compress like scripts/train_model.py).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FEATURE_NAMES_FILE.write_text("\n".join(feature_names))

    fb_stats = store.stats()
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "base + user feedback (ADR-027 continual learning)",
        "base_windows": int(n_base),
        "feedback_corrections": int(fb_stats["total"]),
        "feedback_repeat": int(feedback_repeat),
        "n_augment": int(n_augment),
        "classes": sorted(set(map(str, y))),
        "feedback_per_label": fb_stats["per_label"],
        "base_model_backup": BASE_BACKUP_FILE.name
        if BASE_BACKUP_FILE.exists()
        else None,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_base_model() -> bool:
    """Restore the originally-shipped model from its backup, if one exists.

    Returns:
        True if the base model was restored, False if no backup is present.
    """
    if not BASE_BACKUP_FILE.exists():
        return False
    shutil.copy2(BASE_BACKUP_FILE, MODEL_FILE)
    MANIFEST_FILE.unlink(missing_ok=True)
    return True
