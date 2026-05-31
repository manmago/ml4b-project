"""Unit tests for ml4b.data.features_invariant.

Verifies the feature count, that identifier columns (including recording_id) are
carried through, that magnitude features are rotation-invariant (the core design
property), and that empty input yields an empty frame.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ml4b.data.features_invariant import extract_invariant_features, feature_columns

WIN = 200


def _windows(n: int = 4) -> pd.DataFrame:
    """Build a small windowed DataFrame matching apply_sliding_window output."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "subject_id": 0,
                "exercise_name": "row",
                "recording_id": f"set_{i}",
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


def test_empty_input_returns_empty() -> None:
    """Empty window frame yields an empty feature frame."""
    assert extract_invariant_features(pd.DataFrame()).empty


def test_carries_identifier_columns() -> None:
    """Identifier columns, including recording_id, are preserved for grouping."""
    feats = extract_invariant_features(_windows())
    for col in ("subject_id", "exercise_name", "recording_id", "window_id"):
        assert col in feats.columns
    # recording_id values survive unchanged (used for leave-one-set-out CV).
    assert sorted(feats["recording_id"]) == ["set_0", "set_1", "set_2", "set_3"]


def test_feature_count_and_no_ids_in_features() -> None:
    """39 numeric features; identifier columns are excluded from feature_columns."""
    feats = extract_invariant_features(_windows())
    fcols = feature_columns(feats)
    assert len(fcols) == 39
    assert "recording_id" not in fcols and "exercise_name" not in fcols


def test_magnitude_features_rotation_invariant() -> None:
    """Rotating a window leaves the magnitude-based features unchanged."""
    base = _windows(1)
    rot = Rotation.random(random_state=3).as_matrix()

    # Rotate the accelerometer and gyroscope triplets by the same rotation.
    rotated = base.copy()
    for axes in (("raw_ax", "raw_ay", "raw_az"), ("raw_gx", "raw_gy", "raw_gz")):
        vecs = np.column_stack([base.loc[0, c] for c in axes]) @ rot.T
        for j, c in enumerate(axes):
            rotated.at[0, c] = vecs[:, j].tolist()

    f_base = extract_invariant_features(base)
    f_rot = extract_invariant_features(rotated)
    # Every magnitude feature must be invariant to the rotation.
    mag_cols = [
        c for c in feature_columns(f_base) if c.startswith(("accel_mag", "gyro_mag"))
    ]
    for c in mag_cols:
        assert np.isclose(f_base.loc[0, c], f_rot.loc[0, c], atol=1e-8), c
