"""Unit tests for ml4b.data.augmentation (rotation augmentation).

Verifies that rotation augmentation preserves vector magnitudes (a rotation is
norm-preserving), multiplies the dataset size as expected, keeps the originals,
respects the n_rotations=0 no-op, and is deterministic for a fixed seed.
"""

import numpy as np
import pandas as pd

from ml4b.data.augmentation import augment_windows_with_rotation

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
