"""Shared training-pipeline orchestration for the 3-class model.

Single source of truth for the core "augment -> invariant features -> Random
Forest + novelty detector" sequence, plus leave-one-set-out cross-validation,
the activity-gate calibration report, and the metrics payload. Both training
entry points import from here so the two committed models stay byte-for-byte
comparable:

  * **Model 1 (baseline)** — Kaggle anchor only.
  * **Model 2 (current)**  — Kaggle anchor + committed Testdaten.

Keeping this in one module enforces the project's single-pipeline rule: there is
exactly ONE place where windows become a trained model, so a fair "what did our
own uploaded training data change?" comparison is guaranteed — both models are
produced by identical preprocessing, augmentation and hyper-parameters; the only
difference is the data they see (see ``docs/DECISIONS.md`` and
``docs/project/continual_training.md``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from ml4b.data.activity_gate import (
    ACCEL_MAG_STD_THRESHOLD,
    GYRO_MAG_MEAN_THRESHOLD,
    window_energy,
)
from ml4b.data.augmentation import augment_windows
from ml4b.data.features_invariant import extract_invariant_features, feature_columns
from ml4b.data.novelty import DEFAULT_PERCENTILE, NoveltyDetector
from ml4b.models.evaluate import evaluate_model
from ml4b.models.train import train_random_forest

# Number of augmented copies per original window (-> 6x total). See DECISIONS.md.
# The SAME value is applied to the Kaggle anchor and to the uploaded Testdaten:
# both go through the identical 6x rotation+time-warp+mirror+jitter augmentation.
N_AUGMENT = 5


@dataclass
class TrainingMatrix:
    """Feature matrix + metadata for one set of training windows.

    Attributes:
        X: Feature matrix ``(n_windows_after_aug, n_features)`` — originals first,
            then their augmented copies.
        y: Class label per row.
        groups: ``recording_id`` per row (the leave-one-set-out group key).
        is_augmented: Boolean mask, ``True`` for augmented copies. Originals keep
            the leading ``n_orig`` positions, so held-out CV folds can exclude the
            augmented copies of the test set (no leakage).
        feature_names: Ordered invariant-feature column names.
        n_orig: Number of original (non-augmented) windows.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    is_augmented: np.ndarray
    feature_names: list[str]
    n_orig: int


def build_training_matrix(
    window_df: pd.DataFrame,
    *,
    n_augment: int = N_AUGMENT,
    random_state: int = 42,
) -> TrainingMatrix:
    """Augment windows and extract invariant features into a training matrix.

    This is the front half of training shared by both models. ``augment_windows``
    keeps the original windows first and appends the augmented copies, so the
    ``is_augmented`` mask is a simple position threshold.

    Args:
        window_df: Per-window raw-signal frame from ``apply_sliding_window``.
        n_augment: Augmented copies per original window (-> ``n_augment+1``x).
        random_state: Seed for reproducible augmentation.

    Returns:
        A :class:`TrainingMatrix`.
    """
    n_orig = len(window_df)
    augmented = augment_windows(
        window_df, n_augment=n_augment, random_state=random_state
    )
    feats = extract_invariant_features(augmented)
    feature_names = feature_columns(feats)
    is_augmented = np.arange(len(feats)) >= n_orig
    return TrainingMatrix(
        X=feats[feature_names].to_numpy(),
        y=feats["exercise_name"].to_numpy(),
        groups=feats["recording_id"].to_numpy(),
        is_augmented=is_augmented,
        feature_names=feature_names,
        n_orig=n_orig,
    )


def fit_model_and_novelty(
    tm: TrainingMatrix,
    *,
    novelty_percentile: float = DEFAULT_PERCENTILE,
    random_state: int = 42,
) -> tuple[Any, NoveltyDetector]:
    """Train the final Random Forest and refit the novelty detector.

    The Random Forest uses every window (originals + augmented copies); the
    novelty detector is fit on the ORIGINAL windows only so our own exercises
    count as "known" and are never rejected as out-of-distribution.

    Args:
        tm: Training matrix from :func:`build_training_matrix`.
        novelty_percentile: Per-class calibration percentile for open-set
            rejection (lower rejects foreign motion more aggressively).
        random_state: Seed for the Random Forest.

    Returns:
        ``(model, novelty_detector)``, both fitted.
    """
    model = train_random_forest(tm.X, tm.y, random_state=random_state)
    X_orig, y_orig = tm.X[~tm.is_augmented], tm.y[~tm.is_augmented]
    detector = NoveltyDetector(percentile=novelty_percentile).fit(
        X_orig, y_orig, tm.feature_names
    )
    return model, detector


class _ConstPredictor:
    """Replays pre-computed CV predictions so we can reuse ``evaluate_model``.

    The leave-one-set-out loop already produced per-window predictions; this thin
    adapter lets us reuse :func:`ml4b.models.evaluate.evaluate_model` (which calls
    ``model.predict(X)``) to compute aggregate metrics + confusion matrix without
    retraining.
    """

    def __init__(self, predictions: list[str]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the stored predictions, ignoring ``X``."""
        return self._predictions


def leave_one_set_out_cv(
    tm: TrainingMatrix,
    class_names: list[str],
    save_dir: Path,
    *,
    label: str = "Leave-One-Set-Out CV",
    random_state: int = 42,
    progress: bool = False,
) -> dict:
    """Honest leave-one-set-out CV: train on all other sets, test on held-out
    originals only.

    Augmented copies of the held-out set are excluded from its test fold, so
    there is no leakage through augmentation or window overlap — the only honest
    estimate available for a single-subject anchor (DECISIONS.md).

    Args:
        tm: Training matrix from :func:`build_training_matrix`.
        class_names: Fixed class order for reports / confusion-matrix axes.
        save_dir: Directory the confusion-matrix figure is written to.
        label: Title used in the saved figure / report.
        random_state: Seed for the per-fold Random Forest.
        progress: When True, print fold progress (used by the long rebuild).

    Returns:
        The ``evaluate_model`` results dict (macro_f1, accuracy, per_class_f1,
        confusion_matrix, ...).
    """
    logo = LeaveOneGroupOut()
    y_true: list[str] = []
    y_pred: list[str] = []
    t0 = time.time()
    n_splits = logo.get_n_splits(groups=tm.groups)
    for i, (train_idx, test_idx) in enumerate(
        logo.split(tm.X, tm.y, tm.groups), start=1
    ):
        test_idx = test_idx[~tm.is_augmented[test_idx]]
        if len(test_idx) == 0:
            continue
        fold = train_random_forest(
            tm.X[train_idx], tm.y[train_idx], random_state=random_state
        )
        y_pred.extend(fold.predict(tm.X[test_idx]).tolist())
        y_true.extend(tm.y[test_idx].tolist())
        if progress and (i % 25 == 0 or i == n_splits):
            print(f"    fold {i}/{n_splits} ({time.time() - t0:.0f}s)")
    save_dir.mkdir(parents=True, exist_ok=True)
    return evaluate_model(
        _ConstPredictor(y_pred),
        np.zeros((len(y_true), 1)),
        np.array(y_true),
        label,
        class_names,
        save_dir=save_dir,
    )


def gate_calibration(window_df: pd.DataFrame) -> dict[str, float]:
    """Report how comfortably real exercise windows clear the rest thresholds.

    Computes the accel-magnitude-std and gyro-magnitude-mean of every original
    exercise window so we can confirm the activity-gate thresholds (DECISIONS.md)
    sit safely below genuine exercise energy. Used by both training scripts to
    keep the activity-gate calibration block in ``model_metrics.json`` current.

    Args:
        window_df: Original (non-augmented) windows.

    Returns:
        Dict of percentile statistics plus the fraction of windows the gate would
        (correctly) keep as active.
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


def build_metrics_payload(
    cv_results: dict,
    *,
    class_names: list[str],
    n_sets: int,
    n_orig: int,
    n_features: int,
    n_augment: int,
    gate_stats: dict[str, float],
    evaluation: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the committed ``*_metrics.json`` payload shown by the app.

    Args:
        cv_results: Output of :func:`leave_one_set_out_cv`.
        class_names: Fixed class order.
        n_sets: Number of training sets (groups).
        n_orig: Number of original (non-augmented) windows.
        n_features: Number of invariant features.
        n_augment: Augmented copies per original window.
        gate_stats: Output of :func:`gate_calibration`.
        evaluation: Human-readable description of the evaluation protocol.
        extra: Optional extra keys merged in (e.g. Testdaten counts for Model 2).

    Returns:
        A JSON-serialisable metrics dict.
    """
    payload: dict[str, Any] = {
        "classes": class_names,
        "n_sets": int(n_sets),
        "n_original_windows": int(n_orig),
        "n_features": int(n_features),
        "n_augment": int(n_augment),
        "evaluation": evaluation,
        "cv_macro_f1": round(float(cv_results["macro_f1"]), 4),
        "cv_accuracy": round(float(cv_results["accuracy"]), 4),
        "cv_per_class_f1": {
            k: round(v, 4) for k, v in cv_results["per_class_f1"].items()
        },
        "confusion_matrix": cv_results["confusion_matrix"].tolist(),
        "confusion_matrix_labels": class_names,
        "activity_gate_calibration": gate_stats,
        "activity_gate_thresholds": {
            "accel_mag_std": ACCEL_MAG_STD_THRESHOLD,
            "gyro_mag_mean": GYRO_MAG_MEAN_THRESHOLD,
        },
    }
    if extra:
        payload.update(extra)
    return payload
