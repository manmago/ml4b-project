"""Unit tests for ml4b.data.augmentation (rotation augmentation).

Verifies that rotation augmentation preserves vector magnitudes (a rotation is
norm-preserving), multiplies the dataset size as expected, keeps the originals,
respects the n_rotations=0 no-op, and is deterministic for a fixed seed.
"""

import numpy as np
import pandas as pd

from ml4b.data.augmentation import augment_windows, augment_windows_with_rotation

WIN = 100


def _windows(n: int = 5) -> pd.DataFrame:
    """Build a small windowed DataFrame matching apply_sliding_window output."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        rows.append(
            {
                "subject_id": "w01",
                "exercise_name": "bicep_curl",
                "window_id": i,
                "raw_ax": rng.standard_normal(WIN).tolist(),
                "raw_ay": rng.standard_normal(WIN).tolist(),
                "raw_az": rng.standard_normal(WIN).tolist(),
                "raw_gx": rng.standard_normal(WIN).tolist(),
                "raw_gy": rng.standard_normal(WIN).tolist(),
                "raw_gz": rng.standard_normal(WIN).tolist(),
            }
        )
    return pd.DataFrame(rows)


def test_no_op_when_zero_rotations() -> None:
    """n_rotations=0 returns the input unchanged."""
    df = _windows()
    out = augment_windows_with_rotation(df, n_rotations=0)
    assert len(out) == len(df)


def test_size_multiplies() -> None:
    """Each window gains n_rotations copies on top of the original."""
    df = _windows(5)
    out = augment_windows_with_rotation(df, n_rotations=3)
    assert len(out) == 5 * (1 + 3)
    # window_id must be unique after renumbering.
    assert out["window_id"].nunique() == len(out)


def test_rotation_preserves_magnitude() -> None:
    """Per-sample accel magnitude is invariant under rotation."""
    df = _windows(1)
    out = augment_windows_with_rotation(df, n_rotations=2)
    orig = np.array([df.loc[0, c] for c in ("raw_ax", "raw_ay", "raw_az")])
    orig_mag = np.sqrt((orig**2).sum(axis=0))
    for _, row in out.iloc[1:].iterrows():  # skip the original (row 0)
        rot = np.array([row[c] for c in ("raw_ax", "raw_ay", "raw_az")])
        rot_mag = np.sqrt((rot**2).sum(axis=0))
        assert np.allclose(orig_mag, rot_mag, atol=1e-9)


def test_deterministic_for_seed() -> None:
    """Same seed produces identical augmented data."""
    df = _windows(3)
    a = augment_windows_with_rotation(df, n_rotations=2, random_state=42)
    b = augment_windows_with_rotation(df, n_rotations=2, random_state=42)
    assert np.allclose(a.iloc[-1]["raw_ax"], b.iloc[-1]["raw_ax"])


def _windows_with_recording_id(n: int = 3) -> pd.DataFrame:
    """Windows that also carry a recording_id, like the Kaggle pipeline."""
    df = _windows(n)
    df.insert(2, "recording_id", [f"set_{i}" for i in range(n)])
    return df


def test_augment_windows_size_and_noop() -> None:
    """augment_windows adds n_augment copies per window; 0 is a no-op."""
    df = _windows_with_recording_id(3)
    assert len(augment_windows(df, n_augment=0)) == 3
    out = augment_windows(df, n_augment=5, random_state=42)
    assert len(out) == 3 * (1 + 5)
    assert out["window_id"].nunique() == len(out)


def test_augment_windows_preserves_recording_id() -> None:
    """recording_id (the leave-one-set-out group key) is carried onto copies."""
    df = _windows_with_recording_id(3)
    out = augment_windows(df, n_augment=4, random_state=42)
    # No augmented row may have a missing/NaN recording_id.
    assert out["recording_id"].notna().all()
    # Each original set id still appears among the augmented copies.
    assert set(out["recording_id"]) == {"set_0", "set_1", "set_2"}


def test_augment_windows_keeps_window_length() -> None:
    """Time-warp keeps each augmented window the same length as the original."""
    df = _windows_with_recording_id(1)
    out = augment_windows(df, n_augment=3, random_state=7)
    assert all(len(row) == WIN for row in out["raw_ax"])


def test_augment_windows_deterministic() -> None:
    """A fixed seed yields identical augmented output."""
    df = _windows_with_recording_id(2)
    a = augment_windows(df, n_augment=3, random_state=42)
    b = augment_windows(df, n_augment=3, random_state=42)
    assert np.allclose(a.iloc[-1]["raw_ax"], b.iloc[-1]["raw_ax"])
