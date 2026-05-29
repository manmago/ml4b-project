"""Apple Watch / Sensor Logger data loader for ML4B gym exercise recognition.

Handles CSV files exported from the Sensor Logger iOS app (free, App Store).
Sensor Logger exports a folder (or ZIP) containing multiple CSV files. This
module reads ``WristMotion.csv`` (accelerometer + gyroscope from the wrist),
normalizes its column names, and runs the *identical* preprocessing pipeline
used in training (sliding window -> feature extraction -> model prediction).

This guarantees the architectural rule in CLAUDE.md: training and inference
share the exact same preprocessing code (``ml4b.data.windowing`` and
``ml4b.data.features``), so predictions are consistent.

Supported Sensor Logger export formats (auto-detected):
  Format A: time, seconds_elapsed, x, y, z, roll, pitch, yaw
  Format B: timestamp, ax, ay, az, gx, gy, gz
  Format C: seconds_elapsed, x, y, z, roll, pitch, yaw
  Format D: time, accelerationX/Y/Z, rotationRateX/Y/Z (DeviceMotion export)

All formats are normalized to the internal schema:
  [timestamp, ax, ay, az, gx, gy, gz]
"""

import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from ml4b.data.features import extract_features
from ml4b.data.windowing import apply_sliding_window

# Sampling rate of the wrist sensor, in Hz. Sensor Logger Wrist Motion records
# at 50 Hz, matching the RecoFit training data — see ADR-006.
SAMPLING_RATE_HZ = 50

# Internal schema every loader output must conform to. These names match
# ml4b.data.features._AXES so feature extraction works without changes.
INTERNAL_COLUMNS = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]

# Known Sensor Logger column-name mappings -> internal format.
# Add a new dict here if Sensor Logger changes its export format.
# Keys are matched case-insensitively against the CSV's columns.
WRIST_MOTION_COLUMN_MAPPINGS: list[dict[str, str]] = [
    # Format A: Sensor Logger default WristMotion.csv (attitude as roll/pitch/yaw).
    {
        "time": "timestamp",
        "x": "ax",
        "y": "ay",
        "z": "az",
        "roll": "gx",
        "pitch": "gy",
        "yaw": "gz",
    },
    # Format B: pre-normalized export (ax/ay/az + gx/gy/gz).
    {
        "timestamp": "timestamp",
        "ax": "ax",
        "ay": "ay",
        "az": "az",
        "gx": "gx",
        "gy": "gy",
        "gz": "gz",
    },
    # Format C: seconds_elapsed used as the timestamp.
    {
        "seconds_elapsed": "timestamp",
        "x": "ax",
        "y": "ay",
        "z": "az",
        "roll": "gx",
        "pitch": "gy",
        "yaw": "gz",
    },
    # Format D: DeviceMotion-style export with explicit accel + rotation rate.
    {
        "time": "timestamp",
        "accelerationx": "ax",
        "accelerationy": "ay",
        "accelerationz": "az",
        "rotationratex": "gx",
        "rotationratey": "gy",
        "rotationratez": "gz",
    },
]


def detect_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect the Sensor Logger column format and normalize it.

    Tries each known column mapping in order and uses the first whose source
    keys are all present in the input (case-insensitive match).

    Args:
        df: Raw DataFrame loaded from a Sensor Logger CSV.

    Returns:
        DataFrame with exactly the internal columns
        ``[timestamp, ax, ay, az, gx, gy, gz]``.

    Raises:
        ValueError: If no known column format matches the input columns.
    """
    # Lower-cased view of the input columns for case-insensitive matching.
    cols_lower = [c.lower() for c in df.columns]

    for mapping in WRIST_MOTION_COLUMN_MAPPINGS:
        # A mapping applies only if every one of its source keys is present.
        if all(key.lower() in cols_lower for key in mapping):
            # Build the actual (original-case) column rename map.
            col_map: dict[str, str] = {}
            for src, dst in mapping.items():
                # Find the original column whose lower-case form equals src.
                actual = df.columns[cols_lower.index(src.lower())]
                col_map[actual] = dst
            renamed = df.rename(columns=col_map)
            return renamed[INTERNAL_COLUMNS]

    # No mapping matched — raise a clear, actionable error listing the columns.
    raise ValueError(
        "Could not detect Sensor Logger column format.\n"
        f"Columns found: {list(df.columns)}\n"
        f"Expected one of: {[list(m.keys()) for m in WRIST_MOTION_COLUMN_MAPPINGS]}\n"
        "See docs/project/apple_watch_data_collection_guide.md for the "
        "supported formats."
    )


def load_sensor_logger_csv(csv_file: Path) -> pd.DataFrame:
    """Load and normalize a ``WristMotion.csv`` from Sensor Logger.

    Args:
        csv_file: Path to ``WristMotion.csv`` exported from Sensor Logger.

    Returns:
        DataFrame with columns ``[timestamp, ax, ay, az, gx, gy, gz]``,
        sorted by timestamp with rows containing NaNs dropped.
    """
    df = pd.read_csv(csv_file)
    df = detect_and_normalize_columns(df)
    # Drop incomplete rows and order by time so windows are contiguous.
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def load_sensor_logger_zip(zip_file: Path) -> pd.DataFrame:
    """Load and normalize ``WristMotion.csv`` from a Sensor Logger ZIP export.

    Sensor Logger can share the whole recording as a ZIP of CSV files. This
    extracts the wrist-motion file (case-insensitive, possibly nested in a
    subfolder) and normalizes it like :func:`load_sensor_logger_csv`.

    Args:
        zip_file: Path to the ZIP file containing Sensor Logger CSV exports.

    Returns:
        DataFrame with columns ``[timestamp, ax, ay, az, gx, gy, gz]``,
        sorted by timestamp with rows containing NaNs dropped.

    Raises:
        FileNotFoundError: If no ``WristMotion.csv`` is found in the ZIP.
    """
    with zipfile.ZipFile(zip_file, "r") as zf:
        # Match WristMotion.csv anywhere in the archive, ignoring case/folders.
        wrist_files = [
            name
            for name in zf.namelist()
            if "wristmotion" in name.lower() and name.lower().endswith(".csv")
        ]
        if not wrist_files:
            raise FileNotFoundError(
                f"WristMotion.csv not found in ZIP. Files found: {zf.namelist()}"
            )
        # Read the first match through a text wrapper so pandas can parse it.
        with zf.open(wrist_files[0]) as handle:
            df = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8"))

    df = detect_and_normalize_columns(df)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def predict_from_sensor_logger(
    csv_file: Path,
    model: Any,
    feature_names: list[str],
    window_size: int = 100,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """Run the full prediction pipeline on a Sensor Logger export.

    Pipeline: load (CSV or ZIP) -> normalize columns -> sliding window ->
    feature extraction -> model prediction -> confidence scores. This is the
    main entry point used by the Streamlit app, and it uses the same
    preprocessing functions as training so predictions stay consistent.

    Args:
        csv_file: Path to ``WristMotion.csv`` or a ZIP from Sensor Logger.
        model: Trained sklearn-compatible classifier (best_model.joblib).
        feature_names: Ordered feature names from feature_names.txt (47 features).
        window_size: Samples per window. Default 100 = 2 s at 50 Hz.
        overlap: Overlap fraction between windows. Default 0.5 = 50%.

    Returns:
        DataFrame with one row per window and columns
        ``[window_id, predicted_class, confidence, time_start_seconds]``.

    Raises:
        ValueError: If the recording is too short to form a single window.
    """
    # Load + normalize depending on the file type the user uploaded.
    suffix = Path(csv_file).suffix.lower()
    if suffix == ".zip":
        raw_df = load_sensor_logger_zip(csv_file)
    else:
        raw_df = load_sensor_logger_csv(csv_file)

    # The windowing function groups by these columns; for a single unlabeled
    # recording we inject constant dummy values so one contiguous group forms.
    raw_df["subject_id"] = 0
    raw_df["exercise_name"] = "unknown"
    raw_df["recording_id"] = 0

    # Apply the SAME sliding window as training (2 s windows, 50% overlap).
    window_df = apply_sliding_window(raw_df, window_size=window_size, overlap=overlap)
    if window_df.empty:
        raise ValueError(
            f"Recording too short: need at least {window_size} samples "
            f"(~{window_size / SAMPLING_RATE_HZ:.0f} s at {SAMPLING_RATE_HZ} Hz) "
            "to form one window."
        )

    # Extract the SAME 47 features as training.
    feature_df = extract_features(window_df)

    # Align feature columns to the exact training order. Any feature missing
    # for a short/odd signal is filled with 0.0 so the matrix shape is valid.
    X = feature_df.reindex(columns=feature_names, fill_value=0.0).values

    # Predict the class and take the max class probability as confidence.
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)

    # Convert each window index to an approximate start time in seconds.
    step_seconds = window_size * (1 - overlap) / SAMPLING_RATE_HZ
    results = pd.DataFrame(
        {
            "window_id": range(len(predictions)),
            "predicted_class": predictions,
            "confidence": confidence,
            "time_start_seconds": [i * step_seconds for i in range(len(predictions))],
        }
    )
    return results
