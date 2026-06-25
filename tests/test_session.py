"""Tests for bout/session segmentation (:mod:`ml4b.data.session`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4b.data.canonical import OVERLAP, TARGET_HZ, WINDOW_SIZE
from ml4b.data.session import (
    count_sets,
    dominant_label,
    format_set_summary,
    summarize_session,
)

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


def test_uncertain_plurality_beats_minority_exercise():
    """A bout the model was mostly unsure about is reported as uncertain.

    This is the core fix: previously a single confident exercise window would win
    the bout even when the model was uncertain for the majority of windows.
    """
    labels = ["bicep_curl", "uncertain", "uncertain", "uncertain"]
    conf = [0.9, np.nan, np.nan, np.nan]
    out = summarize_session(_results(labels, conf))
    assert len(out) == 1
    assert out.loc[0, "label"] == "uncertain"


def test_exercise_plurality_still_wins():
    """A genuine exercise plurality is still reported as that exercise."""
    labels = ["row", "row", "row", "uncertain"]
    conf = [0.7, 0.8, 0.75, np.nan]
    out = summarize_session(_results(labels, conf))
    assert out.loc[0, "label"] == "row"


def test_dominant_label_ignores_rest_only():
    """The overall-result rule drops rest but keeps uncertain/unknown in the vote."""
    # Rest is the literal majority but must be ignored as a pause...
    assert dominant_label(["rest", "rest", "rest", "uncertain"]) == "uncertain"
    # ...while uncertain outvoting a lone exercise yields uncertain.
    assert dominant_label(["bicep_curl", "uncertain", "uncertain"]) == "uncertain"
    # A clear exercise plurality wins.
    assert dominant_label(["row", "row", "uncertain", "rest"]) == "row"
    # All-rest leaves nothing to report.
    assert dominant_label(["rest", "rest"]) is None


def test_dominant_label_tie_prefers_non_exercise():
    """On an exact tie, the non-exercise label wins over an exercise."""
    assert dominant_label(["bicep_curl", "uncertain"]) == "uncertain"


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


def test_count_sets_counts_two_bouts_of_same_exercise():
    """bicep → rest → bicep is two sets of one exercise (the headline example)."""
    labels = ["bicep_curl", "bicep_curl", "rest", "bicep_curl", "bicep_curl"]
    conf = [0.9, 0.8, np.nan, 0.85, 0.9]
    sets = summarize_session(_results(labels, conf))
    assert count_sets(sets) == [("bicep_curl", 2)]


def test_count_sets_orders_by_first_appearance():
    """Multiple exercises are counted and ordered by when they first appear."""
    labels = ["row", "rest", "bicep_curl", "rest", "row"]
    conf = [0.7, np.nan, 0.8, np.nan, 0.75]
    sets = summarize_session(_results(labels, conf))
    assert count_sets(sets) == [("row", 2), ("bicep_curl", 1)]


def test_count_sets_excludes_non_exercise_by_default():
    """uncertain/unknown sets are dropped unless explicitly included."""
    labels = ["uncertain", "uncertain", "rest", "bicep_curl", "bicep_curl"]
    conf = [np.nan, np.nan, np.nan, 0.9, 0.85]
    sets = summarize_session(_results(labels, conf))
    assert count_sets(sets) == [("bicep_curl", 1)]
    assert ("uncertain", 1) in count_sets(sets, include_non_exercise=True)


def test_format_set_summary_pluralizes_and_humanizes():
    """The headline string pluralizes and Title-Cases labels."""
    labels = ["bicep_curl", "bicep_curl", "rest", "bicep_curl", "rest", "row"]
    conf = [0.9, 0.8, np.nan, 0.85, np.nan, 0.7]
    sets = summarize_session(_results(labels, conf))
    assert format_set_summary(sets) == "2 sets of Bicep Curl · 1 set of Row"


def test_format_set_summary_empty_without_exercises():
    """No genuine-exercise sets yields an empty summary string."""
    sets = summarize_session(_results(["rest", "rest"], [np.nan, np.nan]))
    assert format_set_summary(sets) == ""
