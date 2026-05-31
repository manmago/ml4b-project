"""Run the 3-class model on the real Apple Watch sanity-check samples.

Predicts on every ``WristMotion.csv`` in ``data/raw/apple_watch/test_samples/``
and prints, per file, the detected sampling rate, the per-window label
distribution, the dominant *exercise* (excluding the gated ``rest`` and the
``uncertain`` abstention), and the mean confidence of classified windows.

The filename hints at the true label. Files whose true label is **not** one of
the three trained classes (e.g. ``push_up``, which is absent from the Kaggle
dataset — ADR-016) are reported as *out of scope*: the script shows what the
model outputs without calling it right or wrong.

This is a read-only sanity check — it does NOT tune anything. Results are
recorded in ``docs/project/apple_watch_validation_results.md``.

Run with: uv run python scripts/test_apple_watch_prediction.py
"""

from __future__ import annotations

import joblib

from ml4b.data.apple_watch_loader import predict_from_sensor_logger
from ml4b.data.kaggle_loader import TARGET_CLASSES
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR, find_project_root

PROJECT_ROOT = find_project_root()
TEST_SAMPLES_DIR = PROJECT_ROOT / "data" / "raw" / "apple_watch" / "test_samples"

# The three classes the model can output; rest/uncertain are produced outside it.
KNOWN_CLASSES = set(TARGET_CLASSES)


def _true_label_from_name(filename: str) -> str:
    """Infer the intended exercise from a sample filename.

    Args:
        filename: e.g. ``"bicep_curl_sample_1.csv"`` or ``"push_up_sample.csv"``.

    Returns:
        The leading exercise token, e.g. ``"bicep_curl"`` or ``"push_up"``.
    """
    stem = filename.replace(".csv", "")
    # Strip a trailing "_sample"/"_session" suffix, leaving the exercise name.
    for marker in ("_sample", "_session"):
        if marker in stem:
            stem = stem.split(marker)[0]
    return stem


def main() -> None:
    """Predict on every sanity-check sample and print an honest summary."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    feature_names = (
        (DATA_PROCESSED / "feature_names.txt").read_text().strip().split("\n")
    )

    samples = sorted(TEST_SAMPLES_DIR.glob("*.csv"))
    if not samples:
        print(f"No sample CSVs found in {TEST_SAMPLES_DIR}.")
        return

    print("=" * 70)
    print("Apple Watch sanity check — 3-class model (read-only, no tuning)")
    print("=" * 70)

    for csv_file in samples:
        true_label = _true_label_from_name(csv_file.name)
        in_scope = true_label in KNOWN_CLASSES

        results = predict_from_sensor_logger(csv_file, model, feature_names)
        counts = results["predicted_class"].value_counts().to_dict()
        # Dominant exercise excludes the non-model outputs (rest, uncertain).
        exercises = results[~results["predicted_class"].isin({"rest", "uncertain"})]
        top = (
            exercises["predicted_class"].mode().iloc[0]
            if not exercises.empty
            else "none"
        )
        avg_conf = (
            float(results["confidence"].mean())
            if results["confidence"].notna().any()
            else float("nan")
        )

        print(f"\n{csv_file.name}")
        scope_note = "" if in_scope else "  (OUT OF SCOPE)"
        print(f"  true label        : {true_label}{scope_note}")
        print(f"  detected rate     : {results.attrs.get('detected_hz')} Hz")
        print(f"  windows           : {len(results)}")
        print(f"  distribution      : {counts}")
        print(f"  dominant exercise : {top}")
        print(f"  avg confidence    : {avg_conf:.3f}")
        if in_scope:
            verdict = "CORRECT" if top == true_label else "MISCLASSIFIED"
            print(f"  verdict           : {verdict}")
        else:
            print("  verdict           : n/a — class not in the model's 3-class scope")

    print("\n" + "=" * 70)
    print("Done. Full write-up: docs/project/apple_watch_validation_results.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
