"""Apple Watch data loader for ML4B gym exercise recognition.

Loads and preprocesses CSV files exported from the Sensor Logger app
to match the format expected by the trained model pipeline.
Used in Phase 5 generalization testing and in the Streamlit app.

Pipeline for a single Sensor Logger CSV file:
    load_sensor_logger_csv() → sliding window → extract_features() → model.predict()

The feature extraction reuses the same functions as training (ml4b.data.features)
so that training and inference remain identical — see the architectural rule in CLAUDE.md.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Expected raw signal columns after renaming — must match the names used in
# ml4b.data.features._AXES so that feature extraction works without changes.
_SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

# Sensor Logger iOS app exports accelerometer data in m/s². The RecoFit
# dataset uses g (9.81 m/s² per g). We convert here so the model sees the
# same unit distribution it was trained on.
_G_TO_MS2 = 9.80665  # standard gravity in m/s²


def load_sensor_logger_csv(csv_file: Path) -> pd.DataFrame:
    """Load a Sensor Logger CSV export from Apple Watch.

    Reads the raw CSV, renames columns to match the RecoFit format
    (ax, ay, az, gx, gy, gz, timestamp), and returns a clean DataFrame
    ready for the windowing and feature extraction pipeline.

    Sensor Logger may export column names in several formats depending on
    the app version and export settings. This function tries the most common
    variants in priority order and raises a clear error if none match.

    Args:
        csv_file: Path to the Sensor Logger CSV export.

    Returns:
        DataFrame with columns: [timestamp, ax, ay, az, gx, gy, gz]
        with the same format as the RecoFit raw DataFrame.
        Accelerometer values are in g (not m/s²).

    Raises:
        FileNotFoundError: If csv_file does not exist.
        ValueError: If the CSV does not contain recognisable sensor columns.
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"Sensor Logger CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()

    # Try to map whatever column names Sensor Logger used to the standard names.
    # Priority order: most common export formats first.
    rename_map = _detect_column_map(df)
    df = df.rename(columns=rename_map)

    # Keep only the sensor columns and a timestamp — drop any extras
    keep = ["timestamp"] + _SENSOR_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing} in {csv_file.name}. "
            f"Available columns: {list(df.columns)}. "
            "Check docs/project/apple_watch_data_collection_guide.md for expected format."
        )

    df = df[keep].dropna()

    # Convert accelerometer axes from m/s² (Sensor Logger default) to g
    # (RecoFit training format). Gyroscope is already in dps (degrees/s).
    for col in ["ax", "ay", "az"]:
        df[col] = df[col] / _G_TO_MS2

    return df.reset_index(drop=True)


def _detect_column_map(df: pd.DataFrame) -> dict[str, str]:
    """Infer the column rename map for common Sensor Logger export formats.

    Args:
        df: Raw DataFrame with lowercased column names.

    Returns:
        Dict mapping existing column names to standard names.
    """
    cols = set(df.columns)

    # Format A: combined motion file with explicit sensor prefix
    # e.g. accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    if {"accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}.issubset(cols):
        candidate = {
            "accel_x": "ax",
            "accel_y": "ay",
            "accel_z": "az",
            "gyro_x": "gx",
            "gyro_y": "gy",
            "gyro_z": "gz",
        }
        # Timestamp alias
        for ts_name in ("time", "seconds_elapsed", "timestamp"):
            if ts_name in cols:
                candidate[ts_name] = "timestamp"
                break
        return candidate

    # Format B: x/y/z columns with a time column (accelerometer-only file)
    # Sensor Logger exports each sensor type separately in this format.
    if {"x", "y", "z"}.issubset(cols):
        candidate = {"x": "ax", "y": "ay", "z": "az"}
        for ts_name in ("time", "seconds_elapsed", "timestamp"):
            if ts_name in cols:
                candidate[ts_name] = "timestamp"
                break
        # Gyroscope not in this file — caller should merge or use a combined export
        return candidate

    # Format C: already uses the standard column names
    if {"ax", "ay", "az"}.issubset(cols):
        candidate = {}
        for ts_name in ("time", "seconds_elapsed"):
            if ts_name in cols:
                candidate[ts_name] = "timestamp"
                break
        return candidate

    return {}


def predict_from_sensor_logger(
    csv_file: Path,
    model: Any,
    feature_names: list[str],
    window_size: int = 100,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """Run full prediction pipeline on a Sensor Logger CSV file.

    Loads CSV → applies sliding window → extracts features →
    predicts exercise class for each window.
    This is the function called by the Streamlit app.

    The sliding window and feature extraction steps are identical to
    the training pipeline (same window_size, overlap, and feature functions)
    to avoid train-serve skew.

    Args:
        csv_file: Path to Sensor Logger CSV export.
        model: Trained sklearn-compatible classifier (best_model.joblib).
        feature_names: Ordered list of feature names from feature_names.txt.
            Must match the columns the model was trained on.
        window_size: Samples per window. Default 100 = 2 s at 50 Hz.
        overlap: Overlap fraction between windows. Default 0.5 = 50%.

    Returns:
        DataFrame with columns: [window_id, predicted_class, confidence]
        where confidence is the maximum predicted probability (0–1).
        One row per window in the input CSV.

    Raises:
        ValueError: If the CSV has fewer rows than one window.
    """
    from ml4b.data.features import extract_features

    raw_df = load_sensor_logger_csv(csv_file)

    if len(raw_df) < window_size:
        raise ValueError(
            f"CSV has only {len(raw_df)} rows — need at least {window_size} "
            f"for one window (window_size={window_size})."
        )

    # Build a synthetic windowed DataFrame in the format expected by extract_features().
    # extract_features() reads raw_ax, raw_ay, ... list columns plus metadata fields.
    # We use dummy metadata (subject_id=0, exercise_name='unknown') since the
    # true label is not known at inference time.
    step = max(1, int(round(window_size * (1 - overlap))))
    windowed_rows = []

    for window_id, start in enumerate(range(0, len(raw_df) - window_size + 1, step)):
        segment = raw_df.iloc[start : start + window_size]
        row: dict[str, Any] = {
            "subject_id": 0,
            "exercise_name": "unknown",  # placeholder — not used in feature extraction
            "window_id": window_id,
        }
        # Store raw signal lists for each axis — mirrors the format of apply_sliding_window()
        for axis in _SENSOR_COLS:
            row[f"raw_{axis}"] = segment[axis].tolist()
        windowed_rows.append(row)

    window_df = pd.DataFrame(windowed_rows)

    # Reuse the same extract_features() function used during training — guarantees
    # identical feature computation. Drop the metadata columns before predicting.
    feature_df = extract_features(window_df)
    X = feature_df[feature_names].values

    predicted_classes = model.predict(X)

    # predict_proba returns shape (n_windows, n_classes); confidence = max probability
    probabilities = model.predict_proba(X)
    confidence = np.max(probabilities, axis=1)

    return pd.DataFrame(
        {
            "window_id": range(len(predicted_classes)),
            "predicted_class": predicted_classes,
            "confidence": confidence.round(4),
        }
    )
