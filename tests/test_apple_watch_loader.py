"""Unit tests for ml4b.data.apple_watch_loader.

Covers column auto-detection across the supported Sensor Logger formats, ZIP
extraction, CSV loading, the too-short-recording guard, and the end-to-end
prediction pipeline using a lightweight fake model (so the tests do not depend
on the committed model binary).
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4b.data.apple_watch_loader import (
    INTERNAL_COLUMNS,
    detect_and_normalize_columns,
    load_sensor_logger_csv,
    load_sensor_logger_zip,
    predict_from_sensor_logger,
)

N_SAMPLES = 600  # enough for several 100-sample windows


def _format_a(n: int = N_SAMPLES) -> pd.DataFrame:
    """Build a Format-A (default WristMotion.csv) frame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "time": np.arange(n),
            "seconds_elapsed": np.arange(n) / 50.0,
            "x": rng.standard_normal(n),
            "y": rng.standard_normal(n),
            "z": rng.standard_normal(n),
            "roll": rng.standard_normal(n),
            "pitch": rng.standard_normal(n),
            "yaw": rng.standard_normal(n),
        }
    )


def _format_b(n: int = N_SAMPLES) -> pd.DataFrame:
    """Build a Format-B (pre-normalized) frame."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "ax": rng.standard_normal(n),
            "ay": rng.standard_normal(n),
            "az": rng.standard_normal(n),
            "gx": rng.standard_normal(n),
            "gy": rng.standard_normal(n),
            "gz": rng.standard_normal(n),
        }
    )


class _FakeModel:
    """Minimal sklearn-like classifier returning constant predictions."""

    classes_ = np.array(["rest", "bicep_curl"])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict 'rest' for every row."""
        return np.array(["rest"] * len(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return a fixed 2-class probability for every row."""
        return np.tile([0.8, 0.2], (len(x), 1))


def _format_primary(n: int = N_SAMPLES) -> pd.DataFrame:
    """Build a PRIMARY-format frame (real WristMotion.csv: accel + gravity + rate)."""
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "time": np.arange(n),
            "seconds_elapsed": np.arange(n) / 100.0,  # ~100 Hz like Apple Watch
            "rotationRateX": rng.standard_normal(n),
            "rotationRateY": rng.standard_normal(n),
            "rotationRateZ": rng.standard_normal(n),
            "gravityX": rng.standard_normal(n),
            "gravityY": rng.standard_normal(n),
            "gravityZ": rng.standard_normal(n),
            "accelerationX": rng.standard_normal(n),
            "accelerationY": rng.standard_normal(n),
            "accelerationZ": rng.standard_normal(n),
        }
    )


def test_detect_format_a() -> None:
    """Format A maps roll/pitch/yaw to gyro and x/y/z to accel."""
    out = detect_and_normalize_columns(_format_a())
    assert list(out.columns) == INTERNAL_COLUMNS


def test_primary_restores_total_acceleration_in_g() -> None:
    """PRIMARY format reconstructs total acceleration in g; gyro stays rad/s.

    The Kaggle training data and Sensor Logger uploads are both Apple CoreMotion,
    so acceleration is canonicalized to total acceleration in g (userAccel +
    gravity) with NO m/s² conversion — see ADR-016.
    """
    df = _format_primary()
    out = detect_and_normalize_columns(df)
    # ax must equal (accelerationX + gravityX), kept in g (no scaling).
    expected_ax = df["accelerationX"].to_numpy() + df["gravityX"].to_numpy()
    assert np.allclose(out["ax"].to_numpy(), expected_ax)
    # gx must equal rotationRateX unchanged (already rad/s).
    assert np.allclose(out["gx"].to_numpy(), df["rotationRateX"].to_numpy())


def test_format_b_not_modified_by_unit_fixes() -> None:
    """Pre-normalized Format B (no gravity, not rotationRate) passes through as-is."""
    df = _format_b()
    out = detect_and_normalize_columns(df)
    assert np.allclose(out["ax"].to_numpy(), df["ax"].to_numpy())
    assert np.allclose(out["gx"].to_numpy(), df["gx"].to_numpy())


def test_detect_format_b() -> None:
    """Format B (already normalized) passes through to the internal schema."""
    out = detect_and_normalize_columns(_format_b())
    assert list(out.columns) == INTERNAL_COLUMNS


def test_detect_format_case_insensitive() -> None:
    """Column matching ignores case."""
    df = _format_b().rename(columns={"ax": "AX", "timestamp": "Timestamp"})
    out = detect_and_normalize_columns(df)
    assert list(out.columns) == INTERNAL_COLUMNS


def test_detect_unknown_columns_raises() -> None:
    """An unrecognized schema raises a clear ValueError."""
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError, match="Could not detect"):
        detect_and_normalize_columns(df)


def test_load_csv(tmp_path: Path) -> None:
    """load_sensor_logger_csv reads and normalizes a WristMotion.csv."""
    csv = tmp_path / "WristMotion.csv"
    _format_a().to_csv(csv, index=False)
    out = load_sensor_logger_csv(csv)
    assert list(out.columns) == INTERNAL_COLUMNS
    assert len(out) == N_SAMPLES


def test_load_zip_nested(tmp_path: Path) -> None:
    """load_sensor_logger_zip finds WristMotion.csv even inside a subfolder."""
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("recording/WristMotion.csv", _format_a().to_csv(index=False))
    out = load_sensor_logger_zip(zip_path)
    assert list(out.columns) == INTERNAL_COLUMNS


def test_load_zip_missing_file_raises(tmp_path: Path) -> None:
    """A ZIP without WristMotion.csv raises FileNotFoundError."""
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("HeartRate.csv", "a,b\n1,2\n")
    with pytest.raises(FileNotFoundError):
        load_sensor_logger_zip(zip_path)


def test_predict_pipeline_shape(tmp_path: Path) -> None:
    """predict_from_sensor_logger returns one row per window with the right cols."""
    csv = tmp_path / "WristMotion.csv"
    _format_a().to_csv(csv, index=False)
    feature_names = [f"f{i}" for i in range(47)]  # names are reindexed, fill 0
    results = predict_from_sensor_logger(csv, _FakeModel(), feature_names)
    assert list(results.columns) == [
        "window_id",
        "predicted_class",
        "confidence",
        "time_start_seconds",
    ]
    assert len(results) > 0
    assert (results["confidence"] == 0.8).all()


def test_predict_too_short_raises(tmp_path: Path) -> None:
    """A recording shorter than one 2 s window raises a clear ValueError.

    Uses PRIMARY format with proper seconds_elapsed so the 100 Hz resample
    yields ~50 samples (~0.5 s) — well below the 200-sample (2 s) window.
    """
    csv = tmp_path / "WristMotion.csv"
    _format_primary(n=50).to_csv(csv, index=False)  # ~0.5 s at 100 Hz
    with pytest.raises(ValueError, match="too short"):
        predict_from_sensor_logger(csv, _FakeModel(), [f"f{i}" for i in range(39)])
