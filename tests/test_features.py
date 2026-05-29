"""Unit tests for ml4b.data.features.

Verifies that feature extraction produces the expected fixed-size feature
vector (47 features + identifier columns), is deterministic, and handles the
empty-input edge case gracefully.
"""

import numpy as np
import pandas as pd

from ml4b.data.features import extract_features

# The pipeline contract: 47 engineered features per window.
EXPECTED_FEATURE_COUNT = 47
ID_COLUMNS = {"subject_id", "exercise_name", "window_id"}


def _make_window_df(n_windows: int = 3, window_size: int = 100) -> pd.DataFrame:
    """Build a windowed DataFrame matching apply_sliding_window() output.

    Args:
        n_windows: Number of window rows to create.
        window_size: Number of samples per window (per axis).

    Returns:
        DataFrame with subject_id, exercise_name, window_id and raw_* list columns.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_windows):
        rows.append(
            {
                "subject_id": 1,
                "exercise_name": "bicep_curl",
                "window_id": i,
                "raw_ax": rng.standard_normal(window_size).tolist(),
                "raw_ay": rng.standard_normal(window_size).tolist(),
                "raw_az": rng.standard_normal(window_size).tolist(),
                "raw_gx": rng.standard_normal(window_size).tolist(),
                "raw_gy": rng.standard_normal(window_size).tolist(),
                "raw_gz": rng.standard_normal(window_size).tolist(),
            }
        )
    return pd.DataFrame(rows)


def test_extract_features_produces_47_features() -> None:
    """extract_features returns exactly 47 feature columns plus the 3 id columns."""
    window_df = _make_window_df()
    feats = extract_features(window_df)

    feature_cols = [c for c in feats.columns if c not in ID_COLUMNS]
    assert len(feature_cols) == EXPECTED_FEATURE_COUNT
    assert ID_COLUMNS.issubset(set(feats.columns))


def test_extract_features_row_count_matches_windows() -> None:
    """One feature row is produced per input window."""
    window_df = _make_window_df(n_windows=5)
    feats = extract_features(window_df)
    assert len(feats) == 5


def test_extract_features_all_finite() -> None:
    """Engineered features contain no NaN or inf values."""
    feats = extract_features(_make_window_df())
    numeric = feats.drop(columns=list(ID_COLUMNS))
    assert np.isfinite(numeric.to_numpy()).all()


def test_extract_features_deterministic() -> None:
    """Same input yields identical features (no hidden randomness)."""
    window_df = _make_window_df()
    a = extract_features(window_df)
    b = extract_features(window_df)
    pd.testing.assert_frame_equal(a, b)


def test_extract_features_empty_input() -> None:
    """Empty input returns an empty DataFrame instead of raising."""
    empty = pd.DataFrame(
        columns=[
            "subject_id",
            "exercise_name",
            "window_id",
            "raw_ax",
            "raw_ay",
            "raw_az",
            "raw_gx",
            "raw_gy",
            "raw_gz",
        ]
    )
    feats = extract_features(empty)
    assert feats.empty
