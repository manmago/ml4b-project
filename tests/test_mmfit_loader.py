"""Unit tests for ml4b.data.mmfit_loader.

Builds a tiny synthetic MM-Fit workout folder (npy sensor files + labels.csv)
in a temp directory so the tests do not depend on the 1.7 GB dataset download.
Covers label parsing, the long-format schema, class mapping (including dropping
unmapped activities and the non_activity -> rest default), both-wrist loading,
and decimation to 50 Hz.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml4b.data.mmfit_loader import (
    LONG_FORMAT_COLUMNS,
    load_mmfit_labels,
    load_mmfit_workout,
)


def _make_sensor_npy(n: int, hz: float, start_frame: int = 0) -> np.ndarray:
    """Build a synthetic [frame, ts_ms, x, y, z] array at the given rate."""
    rng = np.random.default_rng(0)
    frame = np.arange(start_frame, start_frame + n, dtype=float)
    ts_ms = np.arange(n, dtype=float) * (1000.0 / hz)
    xyz = rng.standard_normal((n, 3))
    return np.column_stack([frame, ts_ms, xyz])


def _make_workout(tmp: Path, wid: str = "w99", hz: float = 100.0) -> Path:
    """Create a synthetic workout folder with both wrists + a label file."""
    d = tmp / wid
    d.mkdir()
    n = 800
    for wrist in ("l", "r"):
        np.save(d / f"{wid}_sw_{wrist}_acc.npy", _make_sensor_npy(n, hz))
        np.save(d / f"{wid}_sw_{wrist}_gyr.npy", _make_sensor_npy(n, hz))
    # Two labelled segments by FRAME index: one mapped (squats), one dropped
    # (lunges has no ML4B equivalent). The rest defaults to non_activity -> rest.
    (d / f"{wid}_labels.csv").write_text("100,300,10,squats\n400,600,10,lunges\n")
    return d


def test_load_labels(tmp_path: Path) -> None:
    """Labels parse into (start, end, activity) tuples, reps dropped."""
    d = _make_workout(tmp_path)
    labels = load_mmfit_labels(d / "w99_labels.csv")
    assert labels == [(100, 300, "squats"), (400, 600, "lunges")]


def test_workout_schema_and_mapping(tmp_path: Path) -> None:
    """Loader emits the long-format schema and maps/drops classes correctly."""
    d = _make_workout(tmp_path)
    df = load_mmfit_workout(d)
    assert list(df.columns) == LONG_FORMAT_COLUMNS
    classes = set(df["exercise_name"].unique())
    # squats -> squat is kept; non_activity -> rest is the default.
    assert "squat" in classes
    assert "rest" in classes
    # lunges has no ML4B mapping, so those samples must be dropped entirely.
    assert "lunges" not in classes
    assert df["subject_id"].unique().tolist() == ["w99"]


def test_decimation_to_50hz(tmp_path: Path) -> None:
    """A 100 Hz workout is decimated ~2x; timestamps land near 50 Hz spacing."""
    d = _make_workout(tmp_path, hz=100.0)
    df = load_mmfit_workout(d)
    # Median spacing of the (seconds) timestamp should be ~1/50 = 0.02 s.
    dt = df["timestamp"].diff().dropna()
    assert abs(dt.median() - 0.02) < 0.005


def test_single_wrist_subset(tmp_path: Path) -> None:
    """Requesting one wrist yields recording_ids only for that wrist."""
    d = _make_workout(tmp_path)
    df = load_mmfit_workout(d, wrists=("l",))
    assert df["recording_id"].str.contains("_l_").all()
    assert not df["recording_id"].str.contains("_r_").any()


def test_empty_when_no_mapped_classes(tmp_path: Path) -> None:
    """If every label is unmapped and there is no rest, output is empty-schema."""
    d = tmp_path / "w98"
    d.mkdir()
    n = 200
    for wrist in ("l", "r"):
        # All samples fall inside the single 'lunges' (dropped) interval, so no
        # rest remains either.
        arr = _make_sensor_npy(n, 100.0, start_frame=0)
        arr[:, 0] = np.arange(0, n)  # frames 0..n within the lunges interval
        np.save(d / f"w98_sw_{wrist}_acc.npy", arr)
        np.save(d / f"w98_sw_{wrist}_gyr.npy", arr)
    (d / "w98_labels.csv").write_text("0,199,10,lunges\n")
    df = load_mmfit_workout(d)
    assert list(df.columns) == LONG_FORMAT_COLUMNS
    assert df.empty


def test_default_is_rest_outside_labels(tmp_path: Path) -> None:
    """Samples outside any labelled interval are mapped to rest."""
    d = tmp_path / "w97"
    d.mkdir()
    n = 400
    for wrist in ("l", "r"):
        arr = _make_sensor_npy(n, 100.0)
        np.save(d / f"w97_sw_{wrist}_acc.npy", arr)
        np.save(d / f"w97_sw_{wrist}_gyr.npy", arr)
    # Label only frames 100-150 as squats; everything else is rest.
    (d / "w97_labels.csv").write_text("100,150,5,squats\n")
    df = load_mmfit_workout(d)
    assert isinstance(df, pd.DataFrame)
    assert df["exercise_name"].eq("rest").any()
