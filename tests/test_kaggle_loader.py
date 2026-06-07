"""Unit tests for ml4b.data.kaggle_loader.

Covers the abbreviation→class mapping and single-file loading (lag trim, total
acceleration reconstruction, canonical schema) on a small synthetic Kaggle-format
CSV — so the test does not depend on the (gitignored) Kaggle dataset.
"""

import numpy as np
import pandas as pd

from ml4b.data.kaggle_loader import (
    ABBREV_TO_CLASS,
    TARGET_CLASSES,
    load_kaggle_file,
)


def test_target_classes_are_three_and_sorted() -> None:
    """Exactly three target classes, in a fixed sorted order."""
    assert TARGET_CLASSES == ["bicep_curl", "row", "tricep_extension"]
    # Every mapped abbreviation points at one of the three classes.
    assert set(ABBREV_TO_CLASS.values()) == set(TARGET_CLASSES)


def test_mapping_covers_expected_abbreviations() -> None:
    """The documented abbreviation groupings (DECISIONS.md) are present."""
    assert ABBREV_TO_CLASS["IDBC"] == "bicep_curl"
    assert ABBREV_TO_CLASS["SAOCTE"] == "tricep_extension"
    assert ABBREV_TO_CLASS["NGCR"] == "row"


def _synthetic_kaggle_csv(tmp_path, n: int = 300, lag_rows: int = 30):
    """Write a tiny Kaggle-format WristMotion CSV with a leading NaN lag block."""
    t = np.arange(n) / 100.0  # 100 Hz
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "secondsElapsed": t,
            "wristMotion_accelerationX": rng.standard_normal(n) * 0.1,
            "wristMotion_accelerationY": rng.standard_normal(n) * 0.1,
            "wristMotion_accelerationZ": rng.standard_normal(n) * 0.1,
            "wristMotion_gravityX": np.zeros(n),
            "wristMotion_gravityY": np.zeros(n),
            "wristMotion_gravityZ": np.ones(n),  # ~1 g on z
            "wristMotion_rotationRateX": rng.standard_normal(n) * 0.5,
            "wristMotion_rotationRateY": rng.standard_normal(n) * 0.5,
            "wristMotion_rotationRateZ": rng.standard_normal(n) * 0.5,
        }
    )
    # Simulate the CoreMotion warm-up lag: first lag_rows have NaN sensor values.
    sensor_cols = [c for c in df.columns if c != "secondsElapsed"]
    df.loc[: lag_rows - 1, sensor_cols] = np.nan
    path = tmp_path / "010125_IDBC_W7_5_S1_R12-2025-01-01_10-00-00.csv"
    df.to_csv(path, index=False)
    return path


def test_load_kaggle_file_schema_and_lag(tmp_path) -> None:
    """A single file loads to the canonical schema with the lag rows dropped."""
    path = _synthetic_kaggle_csv(tmp_path)
    out = load_kaggle_file(path)
    assert list(out.columns) == ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]
    # No NaNs survive and the leading lag (timestamp < ~0.24 s) is removed.
    assert not out.isna().any().any()
    assert out["timestamp"].min() >= 0.24
    # Total acceleration on z ≈ user(≈0) + gravity(1 g) ≈ 1.
    assert abs(out["az"].mean() - 1.0) < 0.2
