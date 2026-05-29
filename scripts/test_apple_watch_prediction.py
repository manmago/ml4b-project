"""Test script to validate apple_watch_loader.py fixes.

Tests the prediction pipeline on real Apple Watch WristMotion.csv samples in
data/raw/apple_watch/test_samples/. The filename indicates the true label, so
the script checks whether the most common prediction matches.

If a sample is misclassified, the script automatically prints a diagnostic:
the first normalized rows and a feature-distribution comparison against the
training data, flagging the features that are most out of distribution.

Run with: uv run python scripts/test_apple_watch_prediction.py
"""

import joblib
import pandas as pd

from ml4b.data.apple_watch_loader import (
    detect_sampling_rate,
    load_sensor_logger_csv,
    predict_from_sensor_logger,
)
from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR, find_project_root

PROJECT_ROOT = find_project_root()
TEST_SAMPLES_DIR = PROJECT_ROOT / "data" / "raw" / "apple_watch" / "test_samples"

# The six classes the model can output (anything else cannot be predicted).
KNOWN_CLASSES = {
    "bicep_curl",
    "lateral_raise",
    "rest",
    "shoulder_press",
    "squat",
    "tricep_extension",
}

# Load model and feature names once.
model = joblib.load(MODELS_DIR / "best_model.joblib")
feature_names = (DATA_PROCESSED / "feature_names.txt").read_text().strip().split("\n")


def _diagnose(csv_file) -> None:
    """Print raw normalized rows and a feature-distribution comparison.

    Helps decide whether a misprediction is a loader bug or a fundamental
    sensor-calibration difference between the RecoFit device and Apple Watch.

    Args:
        csv_file: Path to the WristMotion.csv sample to diagnose.
    """
    print("  --- DIAGNOSTIC ---")
    raw_df = load_sensor_logger_csv(csv_file)

    # 1) Sanity-check the normalized signal columns.
    print("  First 3 normalized rows [ax, ay, az, gx, gy, gz]:")
    print(raw_df[["ax", "ay", "az", "gx", "gy", "gz"]].head(3).to_string(index=False))
    sig = raw_df[["ax", "ay", "az", "gx", "gy", "gz"]]
    print(
        f"  NaNs: {int(sig.isna().sum().sum())}, all-zero cols: "
        f"{[c for c in sig.columns if (sig[c] == 0).all()]}"
    )

    # 2) Compare this sample's feature distribution to the training data.
    train_df = pd.read_csv(DATA_PROCESSED / "train_features.csv")
    sample_results = predict_from_sensor_logger(csv_file, model, feature_names)
    _ = sample_results  # predictions already printed by caller

    # Recompute the sample's feature matrix the same way the pipeline does.
    from ml4b.data.features import extract_features
    from ml4b.data.windowing import apply_sliding_window

    rdf = load_sensor_logger_csv(csv_file)
    hz = detect_sampling_rate(rdf)
    if abs(hz - 50) >= 5:
        from ml4b.data.apple_watch_loader import resample_to_target_hz

        rdf = resample_to_target_hz(rdf, source_hz=hz, target_hz=50.0)
    rdf["subject_id"] = 0
    rdf["exercise_name"] = "unknown"
    rdf["recording_id"] = 0
    feats = extract_features(apply_sliding_window(rdf, window_size=100, overlap=0.5))

    # z-distance of the sample's mean feature value from the training mean.
    rows = []
    for f in feature_names:
        tr_mean, tr_std = train_df[f].mean(), train_df[f].std()
        aw_mean = feats[f].mean() if f in feats else 0.0
        z = abs(aw_mean - tr_mean) / tr_std if tr_std > 0 else 0.0
        rows.append((f, tr_mean, tr_std, aw_mean, z))
    rows.sort(key=lambda r: r[4], reverse=True)

    print("  Top 8 most out-of-distribution features (train vs Apple Watch):")
    print(
        f"  {'feature':<24}{'train_mean':>12}{'train_std':>11}{'aw_mean':>11}{'z':>7}"
    )
    for f, tm, ts, am, z in rows[:8]:
        flag = "  <-- far" if z > 3 else ""
        print(f"  {f:<24}{tm:>12.3f}{ts:>11.3f}{am:>11.3f}{z:>7.1f}{flag}")


def main() -> None:
    """Run the prediction test over every sample and report correctness."""
    print("=" * 60)
    print("Apple Watch Prediction Test")
    print("=" * 60)

    any_wrong = False
    for csv_file in sorted(TEST_SAMPLES_DIR.glob("*.csv")):
        print(f"\nFile: {csv_file.name}")
        print(f"True label (from filename): {csv_file.stem}")

        raw_df = load_sensor_logger_csv(csv_file)
        hz = detect_sampling_rate(raw_df)
        print(f"Detected sampling rate: {hz} Hz")
        print(f"Duration: {raw_df['timestamp'].max():.1f} seconds")
        print(f"Samples: {len(raw_df)}")

        results = predict_from_sensor_logger(
            csv_file=csv_file, model=model, feature_names=feature_names
        )
        print(
            f"Resampled to 50 Hz: {results.attrs.get('n_samples_after_resample')} "
            f"samples -> {len(results)} windows"
        )
        print("Prediction distribution:")
        dist = results["predicted_class"].value_counts()
        for cls, count in dist.items():
            pct = count / len(results) * 100
            marker = " <- CORRECT" if cls in csv_file.stem.lower() else ""
            print(f"  {cls:<20} {count:3d} windows ({pct:.1f}%){marker}")

        most_common = results["predicted_class"].mode()[0]
        true_is_known = any(c in csv_file.stem.lower() for c in KNOWN_CLASSES)
        correct = most_common in csv_file.stem.lower()
        if not true_is_known:
            print(
                f"Most common prediction: {most_common} "
                f"(true class '{csv_file.stem}' is NOT one of the 6 trained "
                "classes — cannot be correct by design)"
            )
        else:
            print(
                f"Most common prediction: {most_common} {'PASS' if correct else 'FAIL'}"
            )
            if not correct:
                any_wrong = True
                _diagnose(csv_file)

    print("\n" + "=" * 60)
    print("Test complete" + ("" if not any_wrong else " — see diagnostics above"))
    print("=" * 60)


if __name__ == "__main__":
    main()
