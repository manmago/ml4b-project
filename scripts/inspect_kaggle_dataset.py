"""Inspect the Kaggle 'Gym Workout IMU Dataset' (Apple Watch) under data/raw/kaggle_gym_imu/.

Role in the project: read-only data understanding (CRISP-DM phase 2). This script does NOT
modify the model, training pipeline, or any data. It reports the structure, columns, units,
exercise label vocabulary, per-exercise set counts, subject information, and the sampling
rate so we can decide whether the dataset is a suitable Apple-Watch training anchor for our
3 target exercises (bicep curl, triceps extension, push-up).

Run:
    uv run python scripts/inspect_kaggle_dataset.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Dataset location (relative to the repo root; resolved via this file's location so the
# script works regardless of the current working directory).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "raw" / "kaggle_gym_imu"

# Apple Watch CoreMotion units inferred from the channel names. CoreMotion 'CMDeviceMotion'
# reports rotationRate in rad/s, gravity and userAcceleration in units of g (gravitational
# acceleration), and attitude quaternion as a unitless unit-quaternion. secondsElapsed is a
# relative time axis in seconds. These are documented here so a new team member understands
# the raw units BEFORE any unit alignment to Sensor Logger is applied downstream.
COLUMN_UNITS: dict[str, str] = {
    "secondsElapsed": "seconds (relative time axis)",
    "wristMotion_rotationRateX": "rad/s (gyroscope, CoreMotion rotationRate)",
    "wristMotion_rotationRateY": "rad/s (gyroscope, CoreMotion rotationRate)",
    "wristMotion_rotationRateZ": "rad/s (gyroscope, CoreMotion rotationRate)",
    "wristMotion_gravityX": "g (gravity vector component)",
    "wristMotion_gravityY": "g (gravity vector component)",
    "wristMotion_gravityZ": "g (gravity vector component)",
    "wristMotion_accelerationX": "g (userAcceleration, gravity removed)",
    "wristMotion_accelerationY": "g (userAcceleration, gravity removed)",
    "wristMotion_accelerationZ": "g (userAcceleration, gravity removed)",
    "wristMotion_quaternionW": "unitless (attitude unit-quaternion)",
    "wristMotion_quaternionX": "unitless (attitude unit-quaternion)",
    "wristMotion_quaternionY": "unitless (attitude unit-quaternion)",
    "wristMotion_quaternionZ": "unitless (attitude unit-quaternion)",
    "weight": "kg or lb (weight used for the set; from filename Wxx)",
    "set": "integer (set index within session)",
    "reps": "integer (repetitions in the set)",
    "activity": "string (exercise abbreviation label)",
    "activityEncoded": "integer (numeric label encoding of 'activity')",
}

# Filename pattern: DDMMYY_ACTIVITY_Wweight_Sset_Rreps-YYYY-MM-DD_HH-MM-SS.csv
# We extract the leading date token (proxy for a recording session) and the activity token.
FILENAME_RE = re.compile(r"^(?P<date>\d{6})_(?P<activity>[A-Za-z0-9]+)_")


def section(title: str) -> None:
    """Print a visually separated section header."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    """Run the full read-only inspection and print a structured report."""
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset directory not found: {DATASET_DIR}")

    csv_files = sorted(DATASET_DIR.glob("*.csv"))

    # --- 1. Files and folder structure -----------------------------------------------
    section("1. FILES AND FOLDER STRUCTURE")
    subdirs = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    print(f"Dataset directory : {DATASET_DIR}")
    print(
        f"Sub-folders       : {len(subdirs)} ({[p.name for p in subdirs] or 'none — flat layout'})"
    )
    print(f"CSV file count    : {len(csv_files)}")
    print("Example filenames :")
    for p in csv_files[:5]:
        print(f"  - {p.name}")

    # --- 2. Column names and units ----------------------------------------------------
    section("2. COLUMN NAMES AND UNITS")
    example_df = pd.read_csv(csv_files[0])
    print(f"Columns in example file ({csv_files[0].name}): {len(example_df.columns)}")
    for col in example_df.columns:
        unit = COLUMN_UNITS.get(col, "UNKNOWN — verify manually")
        print(f"  {col:<32} -> {unit}")

    # Confirm the schema is consistent across all files (cheap header-only check).
    inconsistent = [
        p.name
        for p in csv_files
        if list(pd.read_csv(p, nrows=0).columns) != list(example_df.columns)
    ]
    print(
        "\nSchema consistency: "
        + (
            "ALL files share the same columns"
            if not inconsistent
            else f"DIFFERENT columns in {inconsistent}"
        )
    )

    # --- 3 & 4 & 5. Activities, subjects, sets per activity ---------------------------
    # We derive metadata from filenames AND from the in-file 'activity' column, and report
    # any disagreement. The date token acts as a session proxy since there is no athlete ID.
    activities_from_col: Counter[str] = Counter()
    activities_from_name: Counter[str] = Counter()
    session_dates: set[str] = set()
    subject_columns_seen: set[str] = set()

    # Column names that would indicate a subject/athlete identifier, if present anywhere.
    SUBJECT_HINTS = ("subject", "athlete", "participant", "user", "person", "id")

    for p in csv_files:
        m = FILENAME_RE.match(p.name)
        if m:
            activities_from_name[m.group("activity")] += 1
            session_dates.add(m.group("date"))
        # Read only the label column cheaply to count by the authoritative in-file label.
        df_head = pd.read_csv(p, nrows=1)
        for c in df_head.columns:
            if any(h in c.lower() for h in SUBJECT_HINTS) and c not in ("reps",):
                subject_columns_seen.add(c)
        if "activity" in df_head.columns and pd.notna(df_head["activity"].iloc[0]):
            activities_from_col[str(df_head["activity"].iloc[0])] += 1

    section("3. UNIQUE EXERCISE LABEL ABBREVIATIONS")
    all_labels = sorted(set(activities_from_col) | set(activities_from_name))
    print(f"Distinct exercise abbreviations: {len(all_labels)}")
    print(f"  {all_labels}")

    section("4. SUBJECTS / ATHLETES")
    print(
        f"Subject/athlete identifier column found: {sorted(subject_columns_seen) or 'NONE'}"
    )
    print(f"Distinct recording-session dates (proxy): {len(session_dates)}")
    print(
        "Note: no athlete/subject column exists. The dataset documentation describes a single\n"
        "collector on one Apple Watch SE (left wrist), so treat this as effectively 1 subject."
    )

    section("5. SETS PER EXERCISE ABBREVIATION (one CSV == one set)")
    print(f"{'abbrev':<10}{'sets (by filename)':<22}{'sets (by in-file label)':<24}")
    for label in all_labels:
        print(
            f"{label:<10}{activities_from_name.get(label, 0):<22}{activities_from_col.get(label, 0):<24}"
        )
    print(
        f"\nTOTAL sets: {sum(activities_from_name.values())} (filename) / {sum(activities_from_col.values())} (in-file)"
    )

    # --- 6. Sampling rate from timestamps --------------------------------------------
    section("6. SAMPLING RATE (computed from secondsElapsed)")
    # Use the example file; secondsElapsed is a monotonically increasing seconds axis.
    dt = example_df["secondsElapsed"].diff().dropna()
    median_dt = float(dt.median())
    print(f"Example file        : {csv_files[0].name}")
    print(f"Rows                : {len(example_df)}")
    print(f"Median delta-t      : {median_dt:.5f} s")
    print(
        f"Implied sample rate : {1.0 / median_dt:.2f} Hz"
        if median_dt > 0
        else "Implied rate: n/a"
    )
    # Cross-check a few files so the rate is not a fluke of one recording.
    rates = []
    for p in csv_files[:10]:
        d = pd.read_csv(p, usecols=["secondsElapsed"])["secondsElapsed"].diff().dropna()
        if (d.median() or 0) > 0:
            rates.append(1.0 / d.median())
    print(
        f"Rate across 10 files: min={min(rates):.1f} Hz, max={max(rates):.1f} Hz, mean={sum(rates) / len(rates):.1f} Hz"
    )

    # --- 7. First 5 rows of an example file ------------------------------------------
    section("7. FIRST 5 ROWS OF AN EXAMPLE FILE")
    print(f"File: {csv_files[0].name}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(example_df.head(5).to_string())
    # Also show first 5 rows that actually contain sensor values, since the documented
    # ~1.5s sensor lag means the leading rows are often empty.
    valued = example_df.dropna(subset=["wristMotion_accelerationX"]).head(5)
    section("7b. FIRST 5 ROWS WITH NON-EMPTY SENSOR VALUES (leading lag dropped)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(valued.to_string())


if __name__ == "__main__":
    main()
