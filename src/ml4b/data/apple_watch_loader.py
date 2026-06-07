"""Apple Watch / Sensor Logger inference pipeline for the 3-class model.

Handles CSV (or ZIP) files exported from the Sensor Logger iOS app and runs the
*identical* preprocessing used in training, satisfying the project's rule that
training and inference share one pipeline. The shared building blocks are:

  * :mod:`ml4b.data.canonical`          — units, sampling rate, window size
  * :mod:`ml4b.data.windowing`          — sliding window
  * :mod:`ml4b.data.activity_gate`      — energy-threshold rest detection
  * :mod:`ml4b.data.features_invariant` — device-invariant features

Both the Kaggle training data and Sensor Logger uploads are Apple CoreMotion
``DeviceMotion`` streams, so the canonicalization is the same on both sides:
total acceleration (userAcceleration + gravity) in **g** and rotation rate in
**rad/s** — no cross-device unit conversion (the device-domain match that
motivated DECISIONS.md).

Pipeline (see DECISIONS.md):
    load (CSV/ZIP) -> normalize columns -> resample to 100 Hz
        -> sliding window (200, 50% overlap)
        -> activity gate (rest is NOT a model class)
        -> invariant features
        -> novelty gate (unknown is NOT a model class; optional)
        -> predict (3 classes)
        -> confidence threshold -> "uncertain"

The model never predicts ``rest``, ``unknown`` or ``uncertain``; those are
produced by the activity gate, the novelty detector and the confidence threshold
respectively, so the three trained classes only compete on genuine, confident,
in-distribution movement.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml4b.data.activity_gate import REST_LABEL, gate_window_df
from ml4b.data.canonical import (
    CONFIDENCE_THRESHOLD,
    OVERLAP,
    TARGET_HZ,
    WINDOW_SIZE,
    resample_uniform,
)
from ml4b.data.features_invariant import extract_invariant_features
from ml4b.data.windowing import apply_sliding_window

# Label used when the model's top probability is below CONFIDENCE_THRESHOLD.
UNCERTAIN_LABEL = "uncertain"

# Label used for active windows the (optional) novelty detector rejects as
# out-of-distribution — an exercise the closed-set model was never trained on
# (DECISIONS.md). Distinct from UNCERTAIN_LABEL: novelty is "not one of our classes",
# uncertainty is "one of our classes, but the model is not confident".
NOVEL_LABEL = "unknown"

# Internal schema every loader output must conform to.
INTERNAL_COLUMNS = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]

# Known Sensor Logger column-name mappings -> internal format. Keys are matched
# case-insensitively. The PRIMARY format is the confirmed real Apple Watch
# WristMotion export (CoreMotion DeviceMotion).
WRIST_MOTION_COLUMN_MAPPINGS: list[dict[str, str]] = [
    # Format PRIMARY: accelerationX/Y/Z = user acceleration in g,
    # rotationRateX/Y/Z = gyroscope in rad/s, seconds_elapsed = timestamp.
    {
        "seconds_elapsed": "timestamp",
        "accelerationX": "ax",
        "accelerationY": "ay",
        "accelerationZ": "az",
        "rotationRateX": "gx",
        "rotationRateY": "gy",
        "rotationRateZ": "gz",
    },
    # Format A: default WristMotion.csv (attitude as roll/pitch/yaw).
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

    Tries each known mapping in order and uses the first whose source keys are
    all present (case-insensitive). When the source is CoreMotion user
    acceleration AND a gravity vector is present, total acceleration in g is
    reconstructed (userAccel + gravity) to match the training canonicalization.

    Args:
        df: Raw DataFrame loaded from a Sensor Logger CSV.

    Returns:
        DataFrame with exactly ``[timestamp, ax, ay, az, gx, gy, gz]`` where
        acceleration is total acceleration in g and gyro is rad/s.

    Raises:
        ValueError: If no known column format matches the input columns.
    """
    cols_lower = [c.lower() for c in df.columns]

    for mapping in WRIST_MOTION_COLUMN_MAPPINGS:
        # A mapping applies only if every one of its source keys is present.
        if all(key.lower() in cols_lower for key in mapping):
            col_map: dict[str, str] = {}
            for src, dst in mapping.items():
                actual = df.columns[cols_lower.index(src.lower())]
                col_map[actual] = dst
            renamed = df.rename(columns=col_map)
            result = renamed[INTERNAL_COLUMNS].copy()

            # Reconstruct TOTAL acceleration in g (userAccel + gravity) when the
            # mapping read CoreMotion user acceleration and a gravity vector is
            # available — this matches the Kaggle training canonicalization
            # exactly (total accel in g, no m/s² conversion). See DECISIONS.md.
            mapped_from_acceleration = any(
                key.lower().startswith("acceleration") for key in mapping
            )
            grav = {c.lower(): c for c in df.columns}
            has_gravity = all(f"gravity{axis}" in grav for axis in ("x", "y", "z"))
            if mapped_from_acceleration and has_gravity:
                result["ax"] = result["ax"].to_numpy() + df[grav["gravityx"]].to_numpy()
                result["ay"] = result["ay"].to_numpy() + df[grav["gravityy"]].to_numpy()
                result["az"] = result["az"].to_numpy() + df[grav["gravityz"]].to_numpy()

            # Gyroscope (rotationRate) is already rad/s — matches training.
            return result

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
        DataFrame ``[timestamp, ax, ay, az, gx, gy, gz]``, sorted by timestamp
        with NaN rows dropped.
    """
    df = pd.read_csv(csv_file)
    df = detect_and_normalize_columns(df)
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def load_sensor_logger_zip(zip_file: Path) -> pd.DataFrame:
    """Load and normalize ``WristMotion.csv`` from a Sensor Logger ZIP export.

    Args:
        zip_file: Path to the ZIP containing Sensor Logger CSV exports.

    Returns:
        DataFrame ``[timestamp, ax, ay, az, gx, gy, gz]``, sorted by timestamp
        with NaN rows dropped.

    Raises:
        FileNotFoundError: If no ``WristMotion.csv`` is found in the ZIP.
    """
    with zipfile.ZipFile(zip_file, "r") as zf:
        wrist_files = [
            name
            for name in zf.namelist()
            if "wristmotion" in name.lower() and name.lower().endswith(".csv")
        ]
        if not wrist_files:
            raise FileNotFoundError(
                f"WristMotion.csv not found in ZIP. Files found: {zf.namelist()}"
            )
        with zf.open(wrist_files[0]) as handle:
            df = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8"))

    df = detect_and_normalize_columns(df)
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def detect_sampling_rate(df: pd.DataFrame, timestamp_col: str = "timestamp") -> float:
    """Detect the sampling rate from median timestamp differences.

    Args:
        df: DataFrame with a seconds-valued timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        Estimated sampling rate in Hz; falls back to the target rate if the
        timestamps are degenerate.
    """
    diffs = df[timestamp_col].diff().dropna()
    median_diff = diffs.median()
    if not median_diff or median_diff <= 0:
        return float(TARGET_HZ)
    return round(1.0 / median_diff)


def load_and_window_recording(
    csv_file: Path,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> tuple[pd.DataFrame, float, int]:
    """Load a Sensor Logger export and fold it into raw signal windows.

    Shared front half of the prediction pipeline: load (CSV/ZIP) -> normalize
    columns -> resample to 100 Hz -> sliding window. Defined once and reused by
    both :func:`predict_from_sensor_logger` and the labelled-recording importer
    (``scripts/add_labelled_recording.py``), so windowing never diverges between
    prediction and feedback ingestion (the project's single-pipeline rule).

    Args:
        csv_file: Path to ``WristMotion.csv`` or a Sensor Logger ZIP.
        window_size: Samples per window. Default 200 = 2 s at 100 Hz.
        overlap: Overlap fraction. Default 0.5.

    Returns:
        ``(window_df, detected_hz, n_resampled)`` — the per-window raw-channel
        frame (row position == ``window_id``), the recording's detected sampling
        rate, and the sample count after resampling to 100 Hz.

    Raises:
        ValueError: If the recording is too short to form a single window.
    """
    suffix = Path(csv_file).suffix.lower()
    raw_df = (
        load_sensor_logger_zip(csv_file)
        if suffix == ".zip"
        else load_sensor_logger_csv(csv_file)
    )

    # Detect the real rate (for reporting) and resample onto a uniform 100 Hz
    # grid so a 200-sample window is exactly 2 s regardless of source rate.
    detected_hz = detect_sampling_rate(raw_df)
    raw_df = resample_uniform(raw_df, target_hz=TARGET_HZ)

    # The windower groups by these columns; inject constants so one contiguous
    # group forms for a single unlabeled recording.
    raw_df["subject_id"] = 0
    raw_df["exercise_name"] = "unknown"
    raw_df["recording_id"] = 0

    window_df = apply_sliding_window(
        raw_df, window_size=window_size, overlap=overlap, sampling_rate=TARGET_HZ
    ).reset_index(drop=True)
    if window_df.empty:
        raise ValueError(
            f"Recording too short: need at least {window_size} samples "
            f"(~{window_size / TARGET_HZ:.0f} s at {TARGET_HZ} Hz) to form one window."
        )
    return window_df, detected_hz, len(raw_df)


def predict_from_sensor_logger(
    csv_file: Path,
    model: Any,
    feature_names: list[str],
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
    novelty_detector: Any | None = None,
    return_windows: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full 3-class prediction pipeline on a Sensor Logger export.

    Pipeline: load (CSV/ZIP) -> normalize -> resample to 100 Hz -> sliding
    window -> activity gate (rest) -> invariant features -> novelty gate
    (unknown) -> model predict -> confidence threshold (uncertain). Uses the same
    preprocessing modules as training so predictions are consistent.

    Args:
        csv_file: Path to ``WristMotion.csv`` or a Sensor Logger ZIP.
        model: Trained classifier (``best_model.joblib``).
        feature_names: Ordered feature names from ``feature_names.txt``.
        window_size: Samples per window. Default 200 = 2 s at 100 Hz.
        overlap: Overlap fraction. Default 0.5.
        novelty_detector: Optional fitted
            :class:`ml4b.data.novelty.NoveltyDetector`. When provided, active
            windows it rejects as out-of-distribution are labelled ``unknown``
            and never reach the model. When ``None``, behaviour is unchanged.
        return_windows: When True, also return the raw windowed DataFrame so the
            caller can capture user corrections for continual learning (DECISIONS.md §8).
            Its row order matches ``results`` (row position == ``window_id``).

    Returns:
        DataFrame with one row per window and columns
        ``[window_id, predicted_class, confidence, time_start_seconds]``. The
        ``predicted_class`` is one of the three exercises, ``rest`` (gated),
        ``unknown`` (novelty-rejected), or ``uncertain`` (low confidence).
        ``DataFrame.attrs`` carries ``detected_hz`` and
        ``n_samples_after_resample``. If ``return_windows`` is True, returns a
        ``(results, window_df)`` tuple instead, where ``window_df`` holds the
        per-window raw signal channels (``raw_a*`` / ``raw_g*``).

    Raises:
        ValueError: If the recording is too short to form a single window.
    """
    # Shared load -> normalize -> resample -> window front half of the pipeline.
    window_df, detected_hz, n_resampled = load_and_window_recording(
        csv_file, window_size=window_size, overlap=overlap
    )

    # Energy-threshold activity gate: rest windows never reach the model.
    active_mask = gate_window_df(window_df).to_numpy()

    # Invariant features for every window (cheap; keeps indexing aligned).
    feature_df = extract_invariant_features(window_df)
    X = feature_df.reindex(columns=feature_names, fill_value=0.0).to_numpy()

    n_windows = len(window_df)
    # Defaults: every window is rest until proven active; confidence NaN.
    predicted = np.array([REST_LABEL] * n_windows, dtype=object)
    confidence = np.full(n_windows, np.nan)

    if active_mask.any():
        active_idx = np.where(active_mask)[0]

        # Novelty gate (optional): reject out-of-distribution windows as
        # "unknown" so an unseen exercise is not forced into one of the three
        # trained classes. Only the windows the detector accepts as known reach
        # the model. See DECISIONS.md.
        if novelty_detector is not None:
            known = novelty_detector.is_known(X[active_idx])
            predicted[active_idx[~known]] = NOVEL_LABEL  # confidence stays NaN
            model_idx = active_idx[known]
        else:
            model_idx = active_idx

        if len(model_idx) > 0:
            probs = model.predict_proba(X[model_idx])
            top_conf = probs.max(axis=1)
            top_pred = model.classes_[probs.argmax(axis=1)]
            # Below the confidence threshold we abstain with "uncertain".
            labels = np.where(
                top_conf >= CONFIDENCE_THRESHOLD, top_pred, UNCERTAIN_LABEL
            )
            predicted[model_idx] = labels
            confidence[model_idx] = top_conf

    # Approximate start time of each window in seconds.
    step_seconds = window_size * (1 - overlap) / TARGET_HZ
    results = pd.DataFrame(
        {
            "window_id": range(n_windows),
            "predicted_class": predicted,
            "confidence": confidence,
            "time_start_seconds": [i * step_seconds for i in range(n_windows)],
        }
    )
    results.attrs["detected_hz"] = detected_hz
    results.attrs["n_samples_after_resample"] = n_resampled
    if return_windows:
        return results, window_df
    return results
