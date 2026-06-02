"""Tests for bout/session segmentation (:mod:`ml4b.data.session`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4b.data.canonical import OVERLAP, TARGET_HZ, WINDOW_SIZE
from ml4b.data.session import summarize_session

# Per-window time step in seconds, matching the prediction pipeline.
_STEP = WINDOW_SIZE * (1 - OVERLAP) / TARGET_HZ


def _results(labels: list[str], confidences: list[float]) -> pd.DataFrame:
    """Build a results frame like predict_from_sensor_logger emits.

    Args:
        labels: ``predicted_class`` per window in time order.
        confidences: Confidence per window (use ``nan`` for rest/unknown).

    Returns:
        DataFrame with the four pipeline output columns.
    """
    return pd.DataFrame(
        {
            "window_id": range(len(labels)),
            "predicted_class": labels,
            "confidence": confidences,
            "time_start_seconds": [i * _STEP for i in range(len(labels))],
        }
    )


def test_empty_input_returns_empty_schema():
    """An empty results frame yields an empty bout frame with the right columns."""
    out = summarize_session(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == [
        "bout_id",
        "label",
        "start_s",
        "end_s",
        "duration_s",
        "n_windows",
        "mean_confidence",
    ]


def test_rest_separates_two_bouts():
    """Rest windows split active windows into separate bouts."""
    labels = ["bicep_curl", "bicep_curl", "rest", "row", "row"]
    conf = [0.9, 0.8, np.nan, 0.7, 0.75]
    out = summarize_session(_results(labels, conf))
    assert len(out) == 2
    assert out["label"].tolist() == ["bicep_curl", "row"]
    assert out["n_windows"].tolist() == [2, 2]


def test_majority_vote_smooths_single_misclassification():
    """A single odd window does not flip the bout's majority label."""
    labels = ["bicep_curl", "row", "bicep_curl", "bicep_curl"]
    conf = [0.9, 0.6, 0.85, 0.8]
    out = summarize_session(_results(labels, conf))
    assert len(out) == 1
    assert out.loc[0, "label"] == "bicep_curl"


def test_unknown_dominated_bout_labelled_unknown():
    """A bout with no recognised exercise is labelled by its dominant non-class."""
    labels = ["unknown", "unknown", "uncertain"]
    conf = [np.nan, np.nan, np.nan]
    out = summarize_session(_results(labels, conf))
    assert len(out) == 1
    assert out.loc[0, "label"] == "unknown"


def test_trailing_bout_is_closed():
    """A recording that ends mid-exercise still produces a final bout."""
    labels = ["rest", "tricep_extension", "tricep_extension"]
    conf = [np.nan, 0.8, 0.82]
    out = summarize_session(_results(labels, conf))
    assert len(out) == 1
    assert out.loc[0, "label"] == "tricep_extension"


def test_durations_and_confidence_are_reasonable():
    """Bout duration spans the windows and confidence averages classified ones."""
    labels = ["row", "row"]
    conf = [0.6, 0.8]
    out = summarize_session(_results(labels, conf))
    row = out.loc[0]
    # Two windows: starts at 0, last starts at one step, plus one window length.
    expected_end = _STEP + WINDOW_SIZE / TARGET_HZ
    assert row["start_s"] == 0.0
    assert abs(row["end_s"] - expected_end) < 1e-6
    assert abs(row["mean_confidence"] - 0.7) < 1e-6
