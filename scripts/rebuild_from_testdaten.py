"""Rebuild the model, novelty detector and metrics from committed Testdaten.

The single, deterministic "continual learning" entry point for the project
(``make update``). It is a **full retrain**, not incremental "further training":
the Random Forest has no ``partial_fit``, so every update rebuilds the model from
scratch on the *current* data (see ``docs/project/continual_training.md`` for the
retraining-vs-further-training explanation).

It rebuilds from:

    Kaggle 3-class anchor  (data/raw/kaggle_gym_imu — Apple Watch SE, 100 Hz,
                            the SAME CoreMotion device domain as our recordings)
  + every labelled set committed under  Testdaten/<Exercise>/

so the shipped model is a pure function of committed data: anyone who runs this on
the same commit gets the same artifacts. That is exactly what keeps the whole team
on ONE model state instead of N divergent per-laptop models.

Folder -> training label (matched by folder-name PREFIX, so filename typos such as
``rowsr_...`` inside ``Rows/`` do not matter — only the folder decides the label):

    Biceps_Curls*        -> bicep_curl
    Rows*                -> row
    Triceps_Extensions*  -> tricep_extension

Two folders are deliberately NOT training classes (see continual_training.md):

    Rest*       Rest is detected by an energy threshold, not a learned class
                (a learned rest class transfers badly across people/devices —
                ``activity_gate.py``). Rest recordings are used only to VALIDATE
                that the gate actually gates them out.
    Uncertain*  Recordings of OTHER exercises (not our three). Used only to
                VALIDATE open-set rejection: the novelty detector should flag them
                ``unknown``. Training an "everything-else" class generalises badly
                (it can only memorise the few foreign exercises recorded); the
                novelty detector rejects unseen foreign motion too.

Pipeline — byte-for-byte the same windowing/augmentation/feature/RF code as
``scripts/train_model.py`` (the project's single-pipeline rule):

    load Kaggle + window  ─┐
    each Testdaten set:     │ concat -> augment (6x) -> invariant features
      load -> resample      │   -> leave-one-set-out CV (honest metric)
      -> window -> gate ────┘   -> final Random Forest on ALL sets
                                -> refit NoveltyDetector on base+Testdaten (no aug)
                                -> write model + novelty + model_metrics.json

Roll back any rebuild with git (the model is committed):
    git checkout HEAD~1 -- models/saved/

Run:
    uv run python scripts/rebuild_from_testdaten.py          # full rebuild + CV
    uv run python scripts/rebuild_from_testdaten.py --no-cv  # fast: skip CV (metrics not refreshed)

Outputs (all committed so the app runs with no dataset):
    models/saved/best_model.joblib
    models/saved/random_forest.joblib
    models/saved/novelty_detector.joblib
    models/saved/model_metrics.json
    data/processed/feature_names.txt
"""

from __future__ import annotations

import argparse
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
from ml4b.data.apple_watch_loader import predict_from_sensor_logger
from ml4b.data.augmentation import augment_windows
from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.features_invariant import extract_invariant_features, feature_columns
from ml4b.data.kaggle_loader import TARGET_CLASSES, load_kaggle_3class
from ml4b.data.novelty import NoveltyDetector
from ml4b.data.testdaten import (
    REST_PREFIXES,
    UNCERTAIN_PREFIXES,
    iter_category,
    load_exercise_windows,
    recording_name,
)
from ml4b.data.windowing import apply_sliding_window
from ml4b.models.evaluate import evaluate_model
from ml4b.models.train import train_random_forest
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

# Number of augmented copies per original window (-> 6x total). Same as training.
N_AUGMENT = 5
# Fixed class order for reproducible reports and confusion-matrix axes.
CLASS_NAMES = sorted(TARGET_CLASSES)


def _gate_calibration(window_df: pd.DataFrame) -> dict[str, float]:
    """Report how comfortably real exercise windows clear the rest thresholds.

    Mirrors ``scripts/train_model.py`` so ``model_metrics.json`` keeps the
    activity-gate calibration block the Model Performance page shows.
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


class _ConstPredictor:
    """Replays pre-computed CV predictions so we can reuse ``evaluate_model``."""

    def __init__(self, predictions: list[str]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the stored predictions, ignoring ``X``."""
        return self._predictions


def _leave_one_set_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    is_augmented: np.ndarray,
) -> dict:
    """Honest leave-one-set-out CV: train on all other sets, test on held-out
    originals only (augmented copies of the held-out set are excluded), so there
    is no leakage through augmentation or window overlap.
    """
    logo = LeaveOneGroupOut()
    y_true: list[str] = []
    y_pred: list[str] = []
    t0 = time.time()
    n_splits = logo.get_n_splits(groups=groups)
    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        test_idx = test_idx[~is_augmented[test_idx]]
        if len(test_idx) == 0:
            continue
        fold = train_random_forest(X[train_idx], y[train_idx], random_state=42)
        y_pred.extend(fold.predict(X[test_idx]).tolist())
        y_true.extend(y[test_idx].tolist())
        if i % 25 == 0 or i == n_splits:
            print(f"    fold {i}/{n_splits} ({time.time() - t0:.0f}s)")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return evaluate_model(
        _ConstPredictor(y_pred),
        np.zeros((len(y_true), 1)),
        np.array(y_true),
        "Leave-One-Set-Out CV (base + Testdaten)",
        CLASS_NAMES,
        save_dir=REPORTS_DIR,
    )


def _validate_open_set(
    model, detector: NoveltyDetector, feature_names: list[str]
) -> None:
    """Validate the Uncertain (foreign-exercise) and Rest folders.

    Runs the **real app pipeline** (``predict_from_sensor_logger``: gate → novelty
    → confidence threshold), so the numbers reflect what a user would actually see.
    For each Uncertain recording (a foreign exercise) we want the app to REFUSE a
    confident call — i.e. label the active windows ``unknown`` (novelty) or
    ``uncertain`` (low confidence) rather than one of the three exercises. For each
    Rest recording we want the windows gated out as ``rest``. Neither folder is a
    training class.
    """
    rest_recs = list(iter_category(REST_PREFIXES))
    unc_recs = list(iter_category(UNCERTAIN_PREFIXES))
    if not rest_recs and not unc_recs:
        return
    print("\nValidation — real app output (not used for training):")

    def _labels(rec):
        results = predict_from_sensor_logger(
            rec, model, feature_names, novelty_detector=detector
        )
        return results["predicted_class"].astype(str)

    for _folder, rec in rest_recs:
        try:
            labels = _labels(rec)
        except ValueError as exc:
            print(f"  ! skip {recording_name(rec)}: {exc}")
            continue
        # Want a HIGH rest fraction: a rest recording should be gated out.
        frac_rest = float((labels == "rest").mean())
        print(
            f"  [Rest]      {recording_name(rec)}: {frac_rest:5.1%} windows rest (want high)"
        )

    for _folder, rec in unc_recs:
        try:
            labels = _labels(rec)
        except ValueError as exc:
            print(f"  ! skip {recording_name(rec)}: {exc}")
            continue
        # Among ACTIVE (non-rest) windows, how many did the app refuse to call a
        # confident exercise (unknown OR uncertain)?
        active = labels[labels != "rest"]
        if active.empty:
            print(f"  [Uncertain] {recording_name(rec)}: no active windows")
            continue
        refused = active.isin(["unknown", "uncertain"]).mean()
        guessed = active[~active.isin(["unknown", "uncertain"])]
        top = ", ".join(
            f"{k}:{v:.0%}"
            for k, v in guessed.value_counts(normalize=True).head(2).items()
        )
        print(
            f"  [Uncertain] {recording_name(rec)}: {refused:5.1%} refused as "
            f"unknown/uncertain (want high) | else confidently [{top or '—'}]"
        )


def main() -> None:
    """Run the end-to-end rebuild from Kaggle + Testdaten and persist artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip leave-one-set-out CV (faster; model_metrics.json NOT refreshed).",
    )
    parser.add_argument(
        "--novelty-percentile",
        type=float,
        default=NoveltyDetector().percentile,
        help=(
            "Per-class calibration percentile for open-set rejection (default 99). "
            "LOWER it (e.g. 95) to reject foreign exercises more aggressively at the "
            "cost of occasionally flagging a genuine exercise as unknown."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ML4B — Rebuild model + novelty + metrics from Kaggle + Testdaten")
    print("=" * 70)

    print("Step 1/6: Loading Kaggle 3-class anchor...")
    base_raw = load_kaggle_3class()
    base_win = apply_sliding_window(
        base_raw, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    print(
        f"  Kaggle: {base_win['recording_id'].nunique()} sets, {len(base_win):,} windows"
    )

    print("Step 2/6: Loading committed Testdaten recordings...")
    td_win, td_counts = load_exercise_windows()
    if td_win.empty:
        print("  No Testdaten exercise recordings found — rebuilding on Kaggle only.")
        combined = base_win
    else:
        n_td_sets = td_win["recording_id"].nunique()
        print(f"  Testdaten: {n_td_sets} sets, {len(td_win):,} active windows")
        for label in CLASS_NAMES:
            print(f"    {label:<18} {td_counts.get(label, 0)} recording(s)")
        # Align columns and stack base + Testdaten into one training frame.
        combined = pd.concat([base_win, td_win[base_win.columns]], ignore_index=True)

    n_sets = combined["recording_id"].nunique()
    n_orig = len(combined)
    print(f"  Combined: {n_sets} sets, {n_orig:,} original windows")

    gate_stats = _gate_calibration(combined)
    print(
        f"  Activity gate: {gate_stats['fraction_kept_active'] * 100:.1f}% of "
        "exercise windows clear the rest thresholds (want ~100%)."
    )

    print(f"Step 3/6: Augmenting (n_augment={N_AUGMENT}, seed=42)...")
    augmented = augment_windows(combined, n_augment=N_AUGMENT, random_state=42)
    print("Step 4/6: Extracting invariant features...")
    feats = extract_invariant_features(augmented)
    feature_names = feature_columns(feats)
    is_augmented = np.arange(len(feats)) >= n_orig
    X = feats[feature_names].to_numpy()
    y = feats["exercise_name"].to_numpy()
    groups = feats["recording_id"].to_numpy()
    print(
        f"  {len(feature_names)} features | {len(feats):,} windows after augmentation"
    )

    cv_results: dict | None = None
    if args.no_cv:
        print("Step 5/6: Skipping CV (--no-cv); model_metrics.json left unchanged.")
    else:
        print(f"Step 5/6: Leave-one-set-out CV over {n_sets} sets (honest metric)...")
        cv_results = _leave_one_set_out(X, y, groups, is_augmented)
        print(f"  CV Macro F1 : {cv_results['macro_f1']:.4f}")
        print(f"  CV Accuracy : {cv_results['accuracy']:.4f}")
        for cls, f1 in cv_results["per_class_f1"].items():
            print(f"    {cls:<18} {f1:.4f}")

    print("Step 6/6: Training final model + novelty detector + saving...")
    # Final model: every set, every augmented copy (the honest estimate is the CV
    # above, not this model's training fit).
    final_model = train_random_forest(X, y, random_state=42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib", compress=3)
    joblib.dump(final_model, MODELS_DIR / "random_forest.joblib", compress=3)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))

    # Refit novelty on the SAME distribution the model learned (base + Testdaten),
    # NON-augmented, so our own exercises count as "known" and are not rejected.
    X_orig, y_orig = X[~is_augmented], y[~is_augmented]
    detector = NoveltyDetector(percentile=args.novelty_percentile).fit(
        X_orig, y_orig, feature_names
    )
    known_frac = float(detector.is_known(X_orig).mean())
    joblib.dump(detector, MODELS_DIR / "novelty_detector.joblib", compress=3)
    print(f"  Novelty detector refit | {known_frac:.1%} of training windows kept known")

    if cv_results is not None:
        payload = {
            "classes": CLASS_NAMES,
            "n_sets": int(n_sets),
            "n_testdaten_sets": int(
                0 if td_win.empty else td_win["recording_id"].nunique()
            ),
            "testdaten_sets_per_class": td_counts,
            "n_original_windows": int(n_orig),
            "n_features": len(feature_names),
            "n_augment": N_AUGMENT,
            "data_sources": "Kaggle 3-class anchor + committed Testdaten",
            "evaluation": "leave-one-set-out (base + Testdaten; single subject base)",
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
        (MODELS_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2))
        (REPORTS_DIR.parent / "leave_one_set_out_results.json").write_text(
            json.dumps(payload, indent=2)
        )
        print("  Wrote model_metrics.json (Model Performance page is now current).")

    _validate_open_set(final_model, detector, feature_names)

    print("=" * 70)
    print("REBUILD COMPLETE — commit models/saved/ + data/processed/feature_names.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
