"""Rebuild BOTH shipped models, the novelty detectors and metrics from data.

The single, deterministic "continual learning" entry point for the project
(``make update``). It is a **full retrain**, not incremental "further training":
the Random Forest has no ``partial_fit``, so every update rebuilds from scratch on
the *current* data (see ``docs/project/continual_training.md``).

It produces the project's **two-model comparison** in one run, so the app can show
the effect of our own uploaded training data side by side (DECISIONS.md):

  * **Model 1 (baseline)** — Kaggle 3-class anchor ONLY.
        -> baseline_model.joblib / baseline_novelty_detector.joblib / baseline_metrics.json
  * **Model 2 (current)**  — Kaggle anchor + every labelled set under Testdaten/.
        -> best_model.joblib / random_forest.joblib / novelty_detector.joblib / model_metrics.json

Both models go through byte-for-byte the SAME windowing/augmentation (6x)/feature/RF
code (``ml4b.models.pipeline`` — the project's single-pipeline rule); the ONLY
difference is the data they see, so any change in their predictions is purely the
effect of the Testdaten. Augmentation is applied to the Kaggle anchor and to the
uploaded Testdaten alike (both are concatenated *before* augmenting).

Folder -> training label (matched by folder-name PREFIX, so filename typos such as
``rowsr_...`` inside ``Rows/`` do not matter — only the folder decides the label):

    Biceps_Curls*        -> bicep_curl
    Rows*                -> row
    Triceps_Extensions*  -> tricep_extension

Two folders are deliberately NOT training classes (see continual_training.md):

    Rest*       Rest is detected by an energy threshold, not a learned class
                (``activity_gate.py``). Rest recordings only VALIDATE the gate.
    Uncertain*  Recordings of OTHER exercises. Used only to VALIDATE open-set
                rejection: the novelty detector should flag them ``unknown``.

Roll back any rebuild with git (the models are committed):
    git checkout HEAD~1 -- models/saved/

Run:
    uv run python scripts/rebuild_from_testdaten.py          # full rebuild + CV
    uv run python scripts/rebuild_from_testdaten.py --no-cv  # fast: skip CV (metrics not refreshed)

Outputs (all committed so the app runs with no dataset):
    models/saved/best_model.joblib
    models/saved/random_forest.joblib
    models/saved/novelty_detector.joblib
    models/saved/model_metrics.json
    models/saved/baseline_model.joblib
    models/saved/baseline_novelty_detector.joblib
    models/saved/baseline_metrics.json
    data/processed/feature_names.txt
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from ml4b.data.apple_watch_loader import predict_from_sensor_logger
from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
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
from ml4b.models.pipeline import (
    N_AUGMENT,
    TrainingMatrix,
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


def _validate_open_set(
    model, detector: NoveltyDetector, feature_names: list[str]
) -> None:
    """Validate the Uncertain (foreign-exercise) and Rest folders.

    Runs the **real app pipeline** (``predict_from_sensor_logger``: gate → novelty
    → confidence threshold) on the *current* (Model 2) model, so the numbers
    reflect what a user would actually see. For each Uncertain recording (a foreign
    exercise) we want the app to REFUSE a confident call — label the active windows
    ``unknown`` (novelty) or ``uncertain`` (low confidence). For each Rest recording
    we want the windows gated out as ``rest``. Neither folder is a training class.
    """
    rest_recs = list(iter_category(REST_PREFIXES))
    unc_recs = list(iter_category(UNCERTAIN_PREFIXES))
    if not rest_recs and not unc_recs:
        return
    print("\nValidation — real app output, Model 2 (not used for training):")

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


def _train_and_save(
    name: str,
    window_df: pd.DataFrame,
    *,
    model_file: str,
    novelty_file: str,
    metrics_file: str | None,
    evaluation: str,
    novelty_percentile: float,
    run_cv: bool,
    metrics_extra: dict | None = None,
    also_save_rf: bool = False,
) -> tuple[object, NoveltyDetector, list[str]]:
    """Train one model + novelty detector from windows and persist the artifacts.

    Shared by both Model 1 (baseline) and Model 2 (current) so the only thing that
    differs between them is ``window_df`` (the data). Returns the fitted model,
    detector and feature names for downstream validation.

    Args:
        name: Human-readable model name for the log.
        window_df: Original (non-augmented) training windows.
        model_file: Filename for the classifier under ``MODELS_DIR``.
        novelty_file: Filename for the novelty detector under ``MODELS_DIR``.
        metrics_file: Filename for the metrics JSON, or ``None`` to skip writing.
        evaluation: Human-readable evaluation-protocol description for the JSON.
        novelty_percentile: Per-class calibration percentile for the detector.
        run_cv: Whether to run leave-one-set-out CV and (re)write the metrics JSON.
        metrics_extra: Extra keys merged into the metrics payload (e.g. Testdaten counts).
        also_save_rf: When True, also save a ``random_forest.joblib`` copy (legacy
            name kept for Model 2 so existing references resolve).

    Returns:
        ``(model, detector, feature_names)``.
    """
    n_sets = window_df["recording_id"].nunique()
    n_orig = len(window_df)
    print(f"\n[{name}] {n_sets} sets, {n_orig:,} original windows")

    gate_stats = gate_calibration(window_df)
    print(
        f"  Activity gate: {gate_stats['fraction_kept_active'] * 100:.1f}% of "
        "exercise windows clear the rest thresholds (want ~100%)."
    )

    print(f"  Augmenting (n_augment={N_AUGMENT}, seed=42) + invariant features...")
    tm: TrainingMatrix = build_training_matrix(window_df, n_augment=N_AUGMENT)
    print(
        f"  {len(tm.feature_names)} features | {len(tm.X):,} windows after augmentation"
    )

    cv_results: dict | None = None
    if run_cv:
        print(f"  Leave-one-set-out CV over {n_sets} sets (honest metric)...")
        cv_results = leave_one_set_out_cv(
            tm, CLASS_NAMES, REPORTS_DIR, label=f"LOSO CV — {name}", progress=True
        )
        print(f"  CV Macro F1 : {cv_results['macro_f1']:.4f}")
        print(f"  CV Accuracy : {cv_results['accuracy']:.4f}")
        for cls, f1 in cv_results["per_class_f1"].items():
            print(f"    {cls:<18} {f1:.4f}")
    else:
        print("  Skipping CV (--no-cv); metrics JSON left unchanged.")

    model, detector = fit_model_and_novelty(tm, novelty_percentile=novelty_percentile)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / model_file, compress=3)
    if also_save_rf:
        joblib.dump(model, MODELS_DIR / "random_forest.joblib", compress=3)
    joblib.dump(detector, MODELS_DIR / novelty_file, compress=3)
    known_frac = float(detector.is_known(tm.X[~tm.is_augmented]).mean())
    print(
        f"  Saved {model_file} + {novelty_file} | "
        f"{known_frac:.1%} of training windows kept known"
    )

    if metrics_file and cv_results is not None:
        payload = build_metrics_payload(
            cv_results,
            class_names=CLASS_NAMES,
            n_sets=n_sets,
            n_orig=n_orig,
            n_features=len(tm.feature_names),
            n_augment=N_AUGMENT,
            gate_stats=gate_stats,
            evaluation=evaluation,
            extra=metrics_extra,
        )
        (MODELS_DIR / metrics_file).write_text(json.dumps(payload, indent=2))
        print(f"  Wrote {metrics_file}.")

    return model, detector, tm.feature_names


def main() -> None:
    """Rebuild both models (baseline + current) from Kaggle + Testdaten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip leave-one-set-out CV (faster; metrics JSONs NOT refreshed).",
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
    run_cv = not args.no_cv

    print("=" * 70)
    print("ML4B — Rebuild BOTH models (baseline + current) from Kaggle + Testdaten")
    print("=" * 70)

    print("Loading Kaggle 3-class anchor...")
    base_raw = load_kaggle_3class()
    base_win = apply_sliding_window(
        base_raw, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    print(
        f"  Kaggle: {base_win['recording_id'].nunique()} sets, {len(base_win):,} windows"
    )

    print("Loading committed Testdaten recordings...")
    td_win, td_counts = load_exercise_windows()
    if td_win.empty:
        print("  No Testdaten exercise recordings found.")
        combined = base_win
    else:
        n_td_sets = td_win["recording_id"].nunique()
        print(f"  Testdaten: {n_td_sets} sets, {len(td_win):,} active windows")
        for label in CLASS_NAMES:
            print(f"    {label:<18} {td_counts.get(label, 0)} recording(s)")
        # Align columns and stack base + Testdaten into one training frame.
        combined = pd.concat([base_win, td_win[base_win.columns]], ignore_index=True)

    # --- Model 1 (baseline): Kaggle anchor ONLY -------------------------------
    print("\n" + "-" * 70)
    print("MODEL 1 (baseline) — Kaggle anchor only")
    print("-" * 70)
    _train_and_save(
        "Model 1 (baseline, Kaggle only)",
        base_win,
        model_file=BASELINE_MODEL_FILE,
        novelty_file=BASELINE_NOVELTY_FILE,
        metrics_file=BASELINE_METRICS_FILE,
        evaluation="leave-one-set-out (Kaggle anchor only; single subject)",
        novelty_percentile=args.novelty_percentile,
        run_cv=run_cv,
    )

    # --- Model 2 (current): Kaggle anchor + Testdaten -------------------------
    print("\n" + "-" * 70)
    print("MODEL 2 (current) — Kaggle anchor + Testdaten")
    print("-" * 70)
    td_extra = {
        "n_testdaten_sets": int(
            0 if td_win.empty else td_win["recording_id"].nunique()
        ),
        "testdaten_sets_per_class": td_counts,
        "data_sources": "Kaggle 3-class anchor + committed Testdaten",
    }
    model2, detector2, feature_names = _train_and_save(
        "Model 2 (current, Kaggle + Testdaten)",
        combined,
        model_file=BEST_MODEL_FILE,
        novelty_file=NOVELTY_FILE,
        metrics_file=METRICS_FILE,
        evaluation="leave-one-set-out (base + Testdaten; single subject base)",
        novelty_percentile=args.novelty_percentile,
        run_cv=run_cv,
        metrics_extra=td_extra,
        also_save_rf=True,
    )

    # Feature names are identical for both models (the invariant feature set is
    # fixed); one committed copy serves the app for both.
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))

    _validate_open_set(model2, detector2, feature_names)

    print("\n" + "=" * 70)
    print("REBUILD COMPLETE — commit models/saved/ + data/processed/feature_names.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
