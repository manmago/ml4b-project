"""Sensor data augmentation — a synthetic substitute for subject diversity.

The final model is trained on a **single-subject** Apple-Watch dataset (the
Kaggle Gym Workout IMU dataset; ADR-016) yet must generalise to *other* people
uploading from *their* Apple Watch. We cannot collect more subjects, so instead
we synthesise the variability that more subjects would have provided (ADR-019).
Four physically-motivated augmentations are composed per copy:

* **Random 3-D rotation** — simulates different watch orientations / mounting and
  handedness of arm posture. Accelerometer and gyroscope share one physical
  frame, so the *same* rotation is applied to both. Norm-preserving.
* **Time-warp** — resamples the window at a random tempo factor to simulate
  faster/slower repetition speeds between people.
* **Axis mirror** — reflects the frame to approximate wearing the watch on the
  *other* wrist (the Kaggle data is left-wrist only).
* **Jitter** — adds small Gaussian noise to model sensor and body-size variation.

Each original window spawns several augmented copies (default 5, i.e. 6× total),
multiplying the effective number of "subjects". Only TRAIN windows are
augmented; held-out sets stay pristine so leave-one-set-out metrics stay honest.

``augment_windows`` is the orchestrator used by training. The older
``augment_windows_with_rotation`` is retained for the rotation-only unit tests.
"""

from __future__ import annotations

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


def _time_warp(signal: np.ndarray, factor: float) -> np.ndarray:
    """Resample a window to a different tempo while keeping its sample count.

    A ``factor`` > 1 plays the motion faster (compresses time), < 1 slower. The
    output keeps the same length so downstream feature extraction is unchanged.

    Args:
        signal: 1-D window signal.
        factor: Tempo multiplier; positions are read at ``i * factor``.

    Returns:
        The time-warped signal, same length as the input.
    """
    n = signal.size
    src_idx = np.arange(n)
    # Read the signal at warped positions, clipped to stay in range.
    warped_pos = np.clip(np.arange(n) * factor, 0, n - 1)
    return np.interp(warped_pos, src_idx, signal)


def _mirror_axes(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reflect the sensor frame to approximate the opposite wrist.

    A reflection across the plane normal to x flips the x acceleration
    component; angular velocity is a pseudovector, so under the same reflection
    its y and z components flip while x is preserved. This is an approximation of
    switching wrists (the Kaggle data is left-wrist only).

    Args:
        arrays: Dict with keys ``ax, ay, az, gx, gy, gz`` of 1-D signals.

    Returns:
        A new dict with the mirrored signals.
    """
    return {
        "ax": -arrays["ax"],
        "ay": arrays["ay"],
        "az": arrays["az"],
        "gx": arrays["gx"],
        "gy": -arrays["gy"],
        "gz": -arrays["gz"],
    }


def augment_windows(
    window_df: pd.DataFrame,
    n_augment: int = 5,
    random_state: int = 42,
    jitter_scale: float = 0.05,
    warp_range: tuple[float, float] = (0.8, 1.2),
    mirror_prob: float = 0.5,
) -> pd.DataFrame:
    """Augment windows with composed rotation, time-warp, mirror and jitter.

    For each input window, ``n_augment`` extra copies are appended (the original
    is always kept). Every copy applies, in order: a random 3-D rotation (shared
    by accel + gyro), a random time-warp, a random axis mirror (with probability
    ``mirror_prob``), and per-axis Gaussian jitter. All identifier columns
    present in ``window_df`` (including ``recording_id``) are carried through so
    augmented copies remain grouped with their source set.

    Args:
        window_df: Windowed DataFrame from
            :func:`ml4b.data.windowing.apply_sliding_window`.
        n_augment: Number of augmented copies per window. ``0`` returns the input
            unchanged. Default 5 (→ 6× total).
        random_state: Seed for reproducible augmentation.
        jitter_scale: Gaussian noise std as a fraction of each axis's own std.
        warp_range: Inclusive range the tempo factor is drawn from.
        mirror_prob: Probability a copy is mirrored to the opposite wrist.

    Returns:
        DataFrame with the originals followed by the augmented copies, with
        ``window_id`` renumbered to stay unique.
    """
    # No-op path keeps callers simple when augmentation is toggled off.
    if n_augment <= 0 or window_df.empty:
        return window_df.reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    # Identifier columns to copy verbatim onto each augmented row.
    id_cols = [c for c in window_df.columns if not c.startswith("raw_")]

    augmented_rows: list[dict] = []
    for _, row in window_df.iterrows():
        # Base arrays for this window.
        base = {
            ax: np.asarray(row[f"raw_{ax}"], dtype=float)
            for ax in ("ax", "ay", "az", "gx", "gy", "gz")
        }
        for _copy in range(n_augment):
            # 1) Random rotation, applied identically to accel and gyro frames.
            rot = Rotation.random(random_state=rng.integers(0, 2**31 - 1)).as_matrix()
            ax2, ay2, az2 = _rotate_triplet(base["ax"], base["ay"], base["az"], rot)
            gx2, gy2, gz2 = _rotate_triplet(base["gx"], base["gy"], base["gz"], rot)
            arrs = {"ax": ax2, "ay": ay2, "az": az2, "gx": gx2, "gy": gy2, "gz": gz2}

            # 2) Time-warp every channel by the same random tempo factor.
            factor = float(rng.uniform(*warp_range))
            arrs = {k: _time_warp(v, factor) for k, v in arrs.items()}

            # 3) Mirror to the opposite wrist with the configured probability.
            if rng.random() < mirror_prob:
                arrs = _mirror_axes(arrs)

            # 4) Per-axis Gaussian jitter scaled to each axis's variability.
            for k, v in arrs.items():
                sigma = jitter_scale * (np.std(v) + 1e-8)
                arrs[k] = v + rng.normal(0.0, sigma, size=v.shape)

            new_row = {col: row[col] for col in id_cols}
            new_row["window_id"] = -1  # renumbered after concatenation
            new_row["raw_ax"] = arrs["ax"].tolist()
            new_row["raw_ay"] = arrs["ay"].tolist()
            new_row["raw_az"] = arrs["az"].tolist()
            new_row["raw_gx"] = arrs["gx"].tolist()
            new_row["raw_gy"] = arrs["gy"].tolist()
            new_row["raw_gz"] = arrs["gz"].tolist()
            augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows, columns=window_df.columns)
    combined = pd.concat([window_df, augmented_df], ignore_index=True)
    # Renumber window_id so every (original + augmented) window is unique.
    combined["window_id"] = range(len(combined))
    return combined
