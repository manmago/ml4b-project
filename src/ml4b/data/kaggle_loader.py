"""Loader for the Kaggle 'Gym Workout IMU Dataset' (Apple Watch, 100 Hz).

This is the training-data source for the final 3-class model (see DECISIONS.md). The
dataset is 164 single-set CSV files recorded on an Apple Watch SE worn on the
left wrist, sampled at 100 Hz. Each file is one set; the filename encodes the
exercise (``DDMMYY_ABBREV_Wweight_Sset_Rreps-timestamp.csv``) and the in-file
``activity`` column carries the same abbreviation.

Role in the project: this module turns the raw Kaggle files into the same
long-format DataFrame the rest of the pipeline expects
(``[subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz]``),
selecting only the files that map to our three target classes and canonicalizing
the channels via :mod:`ml4b.data.canonical` so they are byte-for-byte compatible
with the Sensor Logger inference path.

Why three classes (see DECISIONS.md): they maximise per-class coverage AND are
biomechanically distinct across different movement axes — elbow flexion
(curl), elbow extension overhead (triceps), and horizontal pull (row).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml4b.data.canonical import (
    LAG_SECONDS,
    reconstruct_total_acceleration,
)
from ml4b.utils.config import DATA_RAW

# Directory holding the raw Kaggle CSV files (resolved from the project root).
KAGGLE_DIR: Path = DATA_RAW / "kaggle_gym_imu"

# Maps each raw Kaggle exercise abbreviation to one of our three target classes.
# Only abbreviations listed here are loaded; every other exercise in the dataset
# (presses, raises, pulldowns, wrist curls, ...) is ignored. See DECISIONS.md for the
# decode of every abbreviation and the rationale for these groupings.
ABBREV_TO_CLASS: dict[str, str] = {
    # bicep_curl — elbow flexion (24 sets total)
    "AIDBC": "bicep_curl",  # Alternating Incline Dumbbell Bicep Curl
    "IDBC": "bicep_curl",  # Incline Dumbbell Bicep Curl
    "PREC": "bicep_curl",  # Machine Preacher (Bicep) Curl
    # tricep_extension — elbow extension overhead (30 sets total)
    "CGOCTE": "tricep_extension",  # Close-Grip Overhead Cable Triceps Extension
    "MTE": "tricep_extension",  # Machine Triceps Extension
    "SAOCTE": "tricep_extension",  # Single-Arm Overhead Cable Triceps Extension
    "SAODTE": "tricep_extension",  # Single-Arm Overhead Dumbbell Triceps Extension
    # row — horizontal pull (21 sets total)
    "CGCR": "row",  # Close-Grip Cable Row
    "NGCR": "row",  # Neutral-Grip Cable Row
    "MGTBR": "row",  # Machine/Mid-Grip T-Bar Row
}

# The three target class labels, in a fixed order for reproducible reports.
TARGET_CLASSES: list[str] = ["bicep_curl", "row", "tricep_extension"]

# Raw Kaggle CoreMotion column names (prefixed with ``wristMotion_``).
_USER_ACCEL_COLS = (
    "wristMotion_accelerationX",
    "wristMotion_accelerationY",
    "wristMotion_accelerationZ",
)
_GRAVITY_COLS = (
    "wristMotion_gravityX",
    "wristMotion_gravityY",
    "wristMotion_gravityZ",
)
_GYRO_COLS = (
    "wristMotion_rotationRateX",
    "wristMotion_rotationRateY",
    "wristMotion_rotationRateZ",
)


def load_kaggle_file(csv_path: Path) -> pd.DataFrame:
    """Load and canonicalize a single Kaggle set file.

    Drops the leading CoreMotion warm-up lag (the first ``LAG_SECONDS`` of every
    file are NaN), reconstructs total acceleration in g, and returns the six
    canonical channels plus a seconds timestamp.

    Args:
        csv_path: Path to one Kaggle ``*.csv`` set file.

    Returns:
        DataFrame with columns ``[timestamp, ax, ay, az, gx, gy, gz]`` at the
        native 100 Hz rate, with the warm-up lag removed and NaN rows dropped.
    """
    raw = pd.read_csv(csv_path)

    # The first ~0.24 s of every file is sensor warm-up lag (all-NaN rows).
    # Drop both explicit NaNs and anything before LAG_SECONDS to be safe.
    raw = raw[raw["secondsElapsed"] >= LAG_SECONDS]
    raw = raw.dropna(subset=list(_USER_ACCEL_COLS + _GYRO_COLS))

    # Reconstruct total acceleration (user accel + gravity) in g.
    accel = reconstruct_total_acceleration(raw, _USER_ACCEL_COLS, _GRAVITY_COLS)

    # Assemble the canonical frame; rotation rate (rad/s) passes through.
    out = pd.DataFrame(
        {
            "timestamp": raw["secondsElapsed"].to_numpy(dtype=float),
            "ax": accel["ax"].to_numpy(),
            "ay": accel["ay"].to_numpy(),
            "az": accel["az"].to_numpy(),
            "gx": raw[_GYRO_COLS[0]].to_numpy(dtype=float),
            "gy": raw[_GYRO_COLS[1]].to_numpy(dtype=float),
            "gz": raw[_GYRO_COLS[2]].to_numpy(dtype=float),
        }
    )
    return out.dropna().sort_values("timestamp").reset_index(drop=True)


def load_kaggle_3class(kaggle_dir: Path = KAGGLE_DIR) -> pd.DataFrame:
    """Load every Kaggle file belonging to the three target classes.

    Each file becomes one ``recording_id`` (its filename stem) so the sliding
    window never crosses a set boundary and so the trainer can group by set for
    leave-one-set-out evaluation (see DECISIONS.md). The single-subject dataset is
    given a constant ``subject_id`` of 0.

    Args:
        kaggle_dir: Directory containing the raw Kaggle CSV files. Defaults to
            ``data/raw/kaggle_gym_imu``.

    Returns:
        Long-format DataFrame with columns
        ``[subject_id, exercise_name, recording_id, timestamp, ax, ay, az,
        gx, gy, gz]`` containing only the three target classes.

    Raises:
        FileNotFoundError: If the Kaggle directory does not exist.
    """
    if not kaggle_dir.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at {kaggle_dir}. "
            "Download it from "
            "https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset "
            "and unzip into data/raw/kaggle_gym_imu/."
        )

    frames: list[pd.DataFrame] = []
    # Sorted for deterministic ordering across machines.
    for csv_path in sorted(kaggle_dir.glob("*.csv")):
        # The abbreviation is the second underscore-delimited token of the name.
        abbrev = csv_path.name.split("_")[1]
        target_class = ABBREV_TO_CLASS.get(abbrev)
        if target_class is None:
            # Not one of the three target exercises — skip this file.
            continue

        df = load_kaggle_file(csv_path)
        if df.empty:
            continue
        # Identify each set by its filename stem so windows stay within one set.
        df["subject_id"] = 0
        df["exercise_name"] = target_class
        df["recording_id"] = csv_path.stem
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No target-class files found in {kaggle_dir}. "
            f"Expected abbreviations: {sorted(ABBREV_TO_CLASS)}."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Reorder so identifier columns lead, matching the rest of the pipeline.
    return combined[
        [
            "subject_id",
            "exercise_name",
            "recording_id",
            "timestamp",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]
    ]
