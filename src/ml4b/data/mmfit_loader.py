"""Loader for the MM-Fit dataset (wrist-worn smartwatch gym exercises).

MM-Fit (Strömbäck, Huang & Radu, UbiComp/ISWC 2020) records full-body gym
workouts with **two smartwatches worn on the wrists** (Mobvoi TicWatch Pro),
plus phones, earbuds and RGB-D video. We use only the **smartwatch
accelerometer + gyroscope** streams because they match the deployment device
— an Apple Watch worn on the wrist.

Why MM-Fit replaced RecoFit (see DECISIONS.md):
    RecoFit's inertial sensor was an armband on the **forearm**; the Apple Watch
    sits on the **wrist**. That placement gap (the wrist adds supination /
    pronation the forearm sensor never sees) is the root cause of the domain
    shift that made RecoFit-trained models fail on real Apple Watch recordings,
    even after unit alignment. MM-Fit is wrist-worn, so the placement matches.

Dataset layout (one folder per workout ``w00`` … ``w20``)::

    mm-fit/w01/
        w01_sw_l_acc.npy   left-wrist accelerometer
        w01_sw_l_gyr.npy   left-wrist gyroscope
        w01_sw_r_acc.npy   right-wrist accelerometer
        w01_sw_r_gyr.npy   right-wrist gyroscope
        w01_labels.csv     exercise segments
        ... (phone / earbud / pose modalities we ignore)

Each sensor ``.npy`` is a 2-D array with columns
``[frame, timestamp_ms, x, y, z]`` sampled at ~100 Hz. ``frame`` is the index
into the (30 Hz) pose stream and is what the labels reference.

``w{ID}_labels.csv`` rows are ``[start_frame, end_frame, repetitions, activity]``.

This loader produces the **same long-format schema as**
:func:`ml4b.data.loader.load_recofit_raw` so the rest of the pipeline
(:func:`ml4b.data.windowing.apply_sliding_window` →
:func:`ml4b.data.features.extract_features`) is reused unchanged::

    [subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz]

Key processing decisions (DECISIONS.md):
    * **Both wrists** are loaded as independent recording streams. This doubles
      the data and — because the left/right wrist axes are mirror images —
      makes the model agnostic to which wrist the Apple Watch is worn on.
    * **100 Hz → 50 Hz decimation** keeps ``window_size=100`` equal to a 2 s
      window and the feature definitions identical to the original pipeline.
    * **Class mapping** keeps the 6 original classes and adds ``push_up``
      (MM-Fit ``pushups``); MM-Fit exercises with no ML4B equivalent
      (lunges, sit-ups, dumbbell rows, jumping jacks) are dropped.

Units: MM-Fit's TicWatch (Wear OS / Android) reports accelerometer in
**m/s² including gravity** and gyroscope in **rad/s**. These raw units are
passed through unchanged; the Apple Watch loader is aligned to *these* units
(see ``apple_watch_loader.py`` and DECISIONS.md), not to RecoFit's.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

# Internal long-format schema shared with load_recofit_raw — the windowing and
# feature stages depend on exactly these column names.
LONG_FORMAT_COLUMNS = [
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

# Target rate after decimation — matches the RecoFit training rate and the
# window_size=100 (= 2 s) assumption baked into the pipeline.
TARGET_HZ = 50.0

# Map MM-Fit activity strings to the ML4B class scheme. Activities not present
# here (lunges, situps, dumbbell_rows, jumping_jacks) are dropped because they
# have no equivalent among the deployed classes. ``non_activity`` becomes the
# ``rest`` class. ``pushups`` is included as ``push_up`` (a 7th class) since
# MM-Fit provides it and it is a common exercise users will perform.
MMFIT_TO_ML4B: dict[str, str] = {
    "squats": "squat",
    "bicep_curls": "bicep_curl",
    "tricep_extensions": "tricep_extension",
    "dumbbell_shoulder_press": "shoulder_press",
    "lateral_shoulder_raises": "lateral_raise",
    "pushups": "push_up",
    "non_activity": "rest",
}

# Official MM-Fit workout-ID splits (from the reference repo train script).
# Splitting by workout = splitting by session/subject, matching our
# subject-based methodology (no leakage of a workout across splits).
TRAIN_W_IDS = ["01", "02", "03", "04", "06", "07", "08", "16", "17", "18"]
VAL_W_IDS = ["14", "15", "19"]
TEST_W_IDS = ["09", "10", "11"]
# The "unseen" test set defined by MM-Fit — kept for completeness.
UNSEEN_TEST_W_IDS = ["00", "05", "12", "13", "20"]


def load_mmfit_labels(label_path: Path) -> list[tuple[int, int, str]]:
    """Read an MM-Fit ``w{ID}_labels.csv`` file.

    Args:
        label_path: Path to a ``*_labels.csv`` file with rows
            ``[start_frame, end_frame, repetitions, activity]``.

    Returns:
        List of ``(start_frame, end_frame, activity)`` tuples. The repetition
        count is intentionally dropped — this project recognizes exercises, it
        does not count reps.
    """
    labels: list[tuple[int, int, str]] = []
    with open(label_path, newline="") as fh:
        for row in csv.reader(fh):
            # Defensive: skip blank or malformed lines rather than crashing on
            # a stray trailing newline.
            if len(row) < 4:
                continue
            labels.append((int(row[0]), int(row[1]), row[3].strip()))
    return labels


def _load_sensor_npy(path: Path) -> pd.DataFrame:
    """Load one MM-Fit sensor ``.npy`` into a [frame, ts_ms, c0, c1, c2] frame.

    Args:
        path: Path to a sensor modality file (``*_acc.npy`` or ``*_gyr.npy``).

    Returns:
        DataFrame with columns ``[frame, ts_ms, c0, c1, c2]`` sorted by
        timestamp, where c0/c1/c2 are the three sensor axes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the array does not have the expected 5 columns.
    """
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            f"{path.name}: expected a 2-D array with 5 columns "
            f"[frame, ts_ms, x, y, z], got shape {arr.shape}"
        )
    df = pd.DataFrame(arr, columns=["frame", "ts_ms", "c0", "c1", "c2"])
    # merge_asof below requires the key column to be sorted ascending.
    return df.sort_values("ts_ms").reset_index(drop=True)


def _label_for_frames(
    frames: np.ndarray, labels: list[tuple[int, int, str]]
) -> np.ndarray:
    """Assign a raw MM-Fit activity string to every sample by its pose frame.

    Samples whose frame falls inside a labelled ``[start, end]`` interval get
    that activity; everything else is ``non_activity`` (rest).

    Args:
        frames: 1-D array of per-sample pose-frame indices (sensor column 0).
        labels: ``(start_frame, end_frame, activity)`` tuples for the workout.

    Returns:
        1-D object array of activity strings, same length as ``frames``.
    """
    # Start everyone as rest, then paint each labelled interval over the top.
    out = np.full(frames.shape, "non_activity", dtype=object)
    for start, end, activity in labels:
        mask = (frames >= start) & (frames <= end)
        out[mask] = activity
    return out


def _detect_hz(ts_ms: np.ndarray) -> float:
    """Estimate sampling rate (Hz) from the median timestamp delta in ms.

    Args:
        ts_ms: Monotonically increasing timestamps in milliseconds.

    Returns:
        Estimated samples per second; falls back to ``TARGET_HZ`` when the
        timestamps are too short or non-increasing to measure.
    """
    if len(ts_ms) < 2:
        return TARGET_HZ
    median_dt = float(np.median(np.diff(ts_ms)))
    if median_dt <= 0:
        return TARGET_HZ
    return 1000.0 / median_dt


def load_mmfit_workout(
    workout_dir: Path,
    wrists: tuple[str, ...] = ("l", "r"),
    class_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load one MM-Fit workout folder into the project's long-format schema.

    For each requested wrist the accelerometer and gyroscope streams are paired
    by nearest timestamp, labelled per sample, decimated to 50 Hz, mapped to the
    ML4B class scheme, and split into contiguous single-label segments (each a
    separate ``recording_id`` so windows never straddle a label boundary).

    Args:
        workout_dir: Path to a ``w{ID}`` folder.
        wrists: Which wrists to load — ``("l", "r")`` by default (both).
        class_map: MM-Fit-activity → ML4B-class mapping. Defaults to
            :data:`MMFIT_TO_ML4B`. Activities absent from the map are dropped.

    Returns:
        Long-format DataFrame with :data:`LONG_FORMAT_COLUMNS`. ``subject_id``
        is the workout id (e.g. ``"w01"``); ``recording_id`` encodes the wrist
        and segment so the windower treats each as an independent recording.
        Empty if the workout has no usable labelled data.

    Raises:
        FileNotFoundError: If the label file is missing.
    """
    class_map = class_map if class_map is not None else MMFIT_TO_ML4B
    wid = workout_dir.name  # e.g. "w01"

    label_path = workout_dir / f"{wid}_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = load_mmfit_labels(label_path)

    frames: list[pd.DataFrame] = []
    for wrist in wrists:
        acc_path = workout_dir / f"{wid}_sw_{wrist}_acc.npy"
        gyr_path = workout_dir / f"{wid}_sw_{wrist}_gyr.npy"
        # A workout may be missing one wrist's data — skip it gracefully.
        if not acc_path.exists() or not gyr_path.exists():
            continue

        acc = _load_sensor_npy(acc_path).rename(
            columns={"c0": "ax", "c1": "ay", "c2": "az"}
        )
        gyr = _load_sensor_npy(gyr_path).rename(
            columns={"c0": "gx", "c1": "gy", "c2": "gz"}
        )

        # Pair each accelerometer sample with the nearest-in-time gyroscope
        # sample. acc/gyr are recorded independently so their timestamps do not
        # line up exactly; nearest-timestamp matching keeps both channels in
        # physical sync without resampling artefacts.
        merged = pd.merge_asof(
            acc[["frame", "ts_ms", "ax", "ay", "az"]],
            gyr[["ts_ms", "gx", "gy", "gz"]],
            on="ts_ms",
            direction="nearest",
        )
        # Drop any rows where no gyro match existed (edges) to avoid NaNs.
        merged = merged.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)
        if merged.empty:
            continue

        # Label every sample by its pose frame, then map to the ML4B scheme.
        merged["activity"] = _label_for_frames(merged["frame"].to_numpy(), labels)
        merged["exercise_name"] = merged["activity"].map(class_map)
        # Drop samples whose activity has no ML4B equivalent (NaN after map).
        merged = merged.dropna(subset=["exercise_name"]).reset_index(drop=True)
        if merged.empty:
            continue

        # Decimate ~100 Hz → 50 Hz by keeping every Nth sample, so a
        # window_size=100 window spans 2 s exactly as in training.
        source_hz = _detect_hz(merged["ts_ms"].to_numpy())
        step = max(1, int(round(source_hz / TARGET_HZ)))
        merged = merged.iloc[::step].reset_index(drop=True)

        # Split into contiguous runs of the same class. A new segment starts
        # wherever the label changes (including across dropped gaps). Each
        # segment is one recording_id so the sliding window stays within a
        # single exercise bout.
        seg_id = (merged["exercise_name"] != merged["exercise_name"].shift()).cumsum()
        merged["recording_id"] = [f"{wid}_{wrist}_{s}" for s in seg_id]
        merged["subject_id"] = wid
        # Re-base timestamp to seconds for downstream readability/consistency.
        merged["timestamp"] = merged["ts_ms"].to_numpy() / 1000.0

        frames.append(merged[LONG_FORMAT_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=LONG_FORMAT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_mmfit_split(
    mmfit_root: Path,
    workout_ids: list[str],
    wrists: tuple[str, ...] = ("l", "r"),
    class_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate several MM-Fit workouts into one long-format frame.

    Args:
        mmfit_root: Path to the unzipped ``mm-fit/`` directory.
        workout_ids: Two-digit workout ids to load, e.g. ``["01", "02"]``.
        wrists: Which wrists to load (default both).
        class_map: MM-Fit → ML4B class mapping (default :data:`MMFIT_TO_ML4B`).

    Returns:
        Concatenated long-format DataFrame for all requested workouts. Workouts
        whose folder is missing are skipped with a printed warning.
    """
    frames: list[pd.DataFrame] = []
    for wid in workout_ids:
        workout_dir = mmfit_root / f"w{wid}"
        if not workout_dir.exists():
            print(f"  [warn] workout folder missing, skipping: {workout_dir}")
            continue
        frames.append(
            load_mmfit_workout(workout_dir, wrists=wrists, class_map=class_map)
        )
    if not frames:
        return pd.DataFrame(columns=LONG_FORMAT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
