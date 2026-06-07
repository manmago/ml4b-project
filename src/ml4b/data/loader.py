"""RecoFit .mat file loader for the ML4B exercise recognition pipeline.

This module is the first stage of the data preparation pipeline. It reads the
RecoFit single-activity MATLAB file (``exercise_data.50.0000_singleonly.mat``)
produced by Microsoft Research (Morris et al., CHI 2014), iterates over the
(n_subjects × n_exercises) cell matrix it contains, and flattens every
recording into a single long-format pandas DataFrame.

The loader also applies ``EXERCISE_MAPPING``: only the RecoFit class labels
listed there are kept and they are renamed to our 6 standardized target
classes (bicep_curl, shoulder_press, squat, tricep_extension, lateral_raise,
rest). The selection rationale is documented in DECISIONS.md.

Downstream modules in this subpackage consume the DataFrame produced by
:func:`load_recofit_raw` — see ``windowing.py`` next.

Typical usage::

    from ml4b.data.loader import load_recofit_raw
    from ml4b.utils.config import DATA_RAW

    df = load_recofit_raw(DATA_RAW / "recofit" / "exercise_data.50.0000_singleonly.mat")
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# Mapping from RecoFit's raw exercise label strings to our 6 standardized
# target classes. Multiple RecoFit variants of the same movement are merged
# (e.g. "Bicep Curl" and "Two-arm Dumbbell Curl" both map to "bicep_curl").
# Any RecoFit label NOT present here is filtered out during loading.
# Selection rationale: DECISIONS.md (data-driven, >30 participants per class).
EXERCISE_MAPPING: dict[str, str] = {
    # Bicep Curl variants → bicep_curl
    "Two-arm Dumbbell Curl (both arms, not alternating)": "bicep_curl",
    "Bicep Curl": "bicep_curl",
    # Shoulder Press variants → shoulder_press
    "Shoulder Press (dumbbell)": "shoulder_press",
    "Squat Rack Shoulder Press": "shoulder_press",
    # Squat variants → squat
    "Squat (arms in front of body, parallel to ground)": "squat",
    "Squat": "squat",
    "Dumbbell Squat (hands at side)": "squat",
    # Tricep Extension variants → tricep_extension
    "Overhead Triceps Extension": "tricep_extension",
    "Triceps extension (lying down)": "tricep_extension",
    # Lateral Raise → lateral_raise
    "Lateral Raise": "lateral_raise",
    # Rest / no-activity variants → rest
    "Non-Exercise": "rest",
    "Device on Table": "rest",
    "Rest": "rest",
}


def _is_empty_cell(cell: object) -> bool:
    """Check whether a (subject, exercise) cell holds no recordings.

    RecoFit stores missing slots as a zero-length numpy array rather than
    ``None``, so a plain ``is None`` check is not sufficient.

    Args:
        cell: One element of the ``subject_data`` matrix.

    Returns:
        True if the cell is an empty ndarray, False otherwise.
    """
    return isinstance(cell, np.ndarray) and cell.size == 0


def _iter_recordings(cell: object) -> list:
    """Normalize a RecoFit cell into a list of recording structs.

    A cell can contain either a single ``mat_struct`` (one recording for that
    subject+exercise combination) or an ndarray of ``mat_struct`` objects
    (multiple separate recordings). This helper returns a uniform list so the
    caller can always iterate without special-casing the shape.

    Args:
        cell: One element of the ``subject_data`` matrix.

    Returns:
        List of mat_struct objects; empty list if the cell is empty.
    """
    if _is_empty_cell(cell):
        return []
    # ndarray → already an iterable of recordings; single struct → wrap in list
    return list(cell) if isinstance(cell, np.ndarray) else [cell]


def load_recofit_raw(mat_file: Path) -> pd.DataFrame:
    """Load raw RecoFit .mat file and convert to a flat pandas DataFrame.

    Reads the single-activity .mat file, iterates over all subjects and
    exercise classes, and returns one row per time sample with columns:
    subject_id, exercise_name, timestamp, ax, ay, az, gx, gy, gz

    Args:
        mat_file: Path to the exercise_data.50.0000_singleonly.mat file.

    Returns:
        DataFrame with columns:
        [subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz]
    """
    # Fail loudly and early if the file is missing — the .mat file is 2.5 GB
    # and not in git, so a missing-file error is the most common setup issue.
    if not mat_file.exists():
        raise FileNotFoundError(
            f"RecoFit .mat file not found at {mat_file}. "
            "Place the file under data/raw/recofit/ — see data/raw/recofit/README.md."
        )

    # simplify_cells=True converts MATLAB cell arrays and structs into plain
    # Python lists/objects, which makes attribute access much cleaner.
    mat = scipy.io.loadmat(str(mat_file), simplify_cells=True)

    # The label strings live in exerciseConstants.activities, indexed by the
    # column position in subject_data.
    activities: list[str] = list(mat["exerciseConstants"]["activities"])
    subject_data: np.ndarray = mat["subject_data"]
    n_subjects, n_exercises = subject_data.shape

    # Collect per-recording DataFrames in a list and concat once at the end —
    # repeated pd.concat in a loop is O(n²) and very slow for ~4700 recordings.
    chunks: list[pd.DataFrame] = []

    for subj_idx in range(n_subjects):
        for exer_idx in range(n_exercises):
            raw_label = activities[exer_idx]

            # Skip exercise classes that are not in our 6 target classes.
            # This filters the data early and keeps memory usage modest.
            if raw_label not in EXERCISE_MAPPING:
                continue

            target_label = EXERCISE_MAPPING[raw_label]
            cell = subject_data[subj_idx, exer_idx]

            # One subject+exercise slot can contain multiple recordings (reps
            # captured in separate sessions). recording_id distinguishes them
            # so windowing later does not mix samples across sessions.
            for rec_id, recording in enumerate(_iter_recordings(cell)):
                try:
                    # mat_struct fields are accessed as attributes; columns are
                    # [t, x, y, z] for both accel (g) and gyro (dps).
                    accel = recording.data.accelDataMatrix
                    gyro = recording.data.gyroDataMatrix
                except AttributeError:
                    # Some cells (e.g. "Note", "Invalid") lack the sensor
                    # matrices entirely — skip them rather than crash.
                    continue

                # Accel and gyro are sampled on the same 50 Hz clock, but the
                # row counts can occasionally differ by one sample due to the
                # original capture device — align to the shorter length.
                n = min(accel.shape[0], gyro.shape[0])
                if n == 0:
                    continue

                chunk = pd.DataFrame(
                    {
                        "subject_id": subj_idx,
                        "exercise_name": target_label,
                        "recording_id": rec_id,
                        # Use the accelerometer timestamp as the canonical time;
                        # gyro timestamps match to within a sample.
                        "timestamp": accel[:n, 0],
                        "ax": accel[:n, 1],
                        "ay": accel[:n, 2],
                        "az": accel[:n, 3],
                        "gx": gyro[:n, 1],
                        "gy": gyro[:n, 2],
                        "gz": gyro[:n, 3],
                    }
                )
                chunks.append(chunk)

    if not chunks:
        # Empty result usually means the .mat file is from a different
        # RecoFit release with different label strings — raise loudly so the
        # caller can update EXERCISE_MAPPING instead of debugging silently.
        raise RuntimeError(
            "No recordings matched EXERCISE_MAPPING. "
            "Check the activity label strings in the .mat file against EXERCISE_MAPPING."
        )

    # Concatenate once at the end and reset the index so row positions are
    # contiguous — downstream code can rely on .iloc-style indexing.
    df = pd.concat(chunks, ignore_index=True)

    # Enforce a stable column order so downstream modules can rely on it.
    return df[
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
