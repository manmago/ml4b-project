"""Sensor-orientation data augmentation for cross-device robustness.

The model is trained on MM-Fit (Mobvoi TicWatch on the wrist) but deployed on an
Apple Watch. Even with units aligned, the *resting orientation* of the watch on
the wrist — and therefore how gravity is distributed across the x/y/z axes —
differs between devices and between users. This shows up as a domain shift in the
per-axis features (e.g. ``az_mean``, ``ay_mean``) and is the main reason similar
arm exercises (bicep curl vs tricep extension) get confused on real recordings.

**Rotation augmentation** (a.k.a. domain randomization) addresses this without
collecting any device-specific data: each training window's 3-axis accelerometer
and gyroscope vectors are rotated by random 3-D rotations. Because the device's
accelerometer and gyroscope share one physical frame, the *same* rotation is
applied to both. The classifier then sees every exercise in many orientations and
learns decision boundaries that do not depend on the exact watch mounting — see
ADR-014.

Only the TRAIN split is augmented; validation and test stay untouched so metrics
remain honest.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Raw signal columns produced by apply_sliding_window (lists per window).
_ACCEL_COLS = ["raw_ax", "raw_ay", "raw_az"]
_GYRO_COLS = ["raw_gx", "raw_gy", "raw_gz"]


def _rotate_triplet(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rot_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate three same-length axis signals by a 3x3 rotation matrix.

    Args:
        x: Samples for the first axis.
        y: Samples for the second axis.
        z: Samples for the third axis.
        rot_matrix: A 3x3 rotation matrix (from scipy Rotation).

    Returns:
        Tuple ``(x', y', z')`` of the rotated axis signals.
    """
    # Stack into an (N, 3) array, rotate every sample vector at once, unpack.
    vecs = np.column_stack([x, y, z])
    rotated = vecs @ rot_matrix.T
    return rotated[:, 0], rotated[:, 1], rotated[:, 2]


def augment_windows_with_rotation(
    window_df: pd.DataFrame,
    n_rotations: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Augment windowed sensor data with random 3-D rotations.

    For every input window, ``n_rotations`` extra copies are created, each with a
    random rotation applied identically to the accelerometer and gyroscope
    vectors. The original (unrotated) windows are kept as well.

    Args:
        window_df: Windowed DataFrame from
            :func:`ml4b.data.windowing.apply_sliding_window`, with the
            ``raw_a*`` / ``raw_g*`` list columns plus ``subject_id``,
            ``exercise_name`` and ``window_id``.
        n_rotations: Number of rotated copies to add per window. ``0`` returns
            the input unchanged. Default 3.
        random_state: Seed for the random rotation generator (reproducibility).

    Returns:
        A new DataFrame containing the original windows followed by the rotated
        copies, with ``window_id`` renumbered to stay unique.
    """
    # Nothing to do — return the input untouched so callers can toggle cleanly.
    if n_rotations <= 0 or window_df.empty:
        return window_df.reset_index(drop=True)

    # One reproducible bundle of random rotations, reused across all windows so
    # the whole augmented set is generated from a single seed.
    rotations = Rotation.random(n_rotations, random_state=random_state).as_matrix()

    augmented_rows: list[dict] = []
    for _, row in window_df.iterrows():
        for rot_matrix in rotations:
            ax2, ay2, az2 = _rotate_triplet(
                np.asarray(row["raw_ax"], dtype=float),
                np.asarray(row["raw_ay"], dtype=float),
                np.asarray(row["raw_az"], dtype=float),
                rot_matrix,
            )
            gx2, gy2, gz2 = _rotate_triplet(
                np.asarray(row["raw_gx"], dtype=float),
                np.asarray(row["raw_gy"], dtype=float),
                np.asarray(row["raw_gz"], dtype=float),
                rot_matrix,
            )
            augmented_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "exercise_name": row["exercise_name"],
                    # window_id is renumbered below; placeholder for now.
                    "window_id": -1,
                    "raw_ax": ax2.tolist(),
                    "raw_ay": ay2.tolist(),
                    "raw_az": az2.tolist(),
                    "raw_gx": gx2.tolist(),
                    "raw_gy": gy2.tolist(),
                    "raw_gz": gz2.tolist(),
                }
            )

    augmented_df = pd.DataFrame(augmented_rows, columns=window_df.columns)
    combined = pd.concat([window_df, augmented_df], ignore_index=True)
    # Renumber window_id so every (original + augmented) window is unique.
    combined["window_id"] = range(len(combined))
    return combined
