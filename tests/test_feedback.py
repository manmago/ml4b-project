"""Tests for the human-in-the-loop feedback store and pipeline compatibility.

These cover the parts of continual learning (DECISIONS.md §8) that do not need the base
Kaggle dataset: persisting/loading corrections, the stats summary, and — crucially
— that stored corrections flow through the *same* augmentation + invariant-feature
pipeline used for training, so a retrain can consume them unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml4b.data.augmentation import augment_windows
from ml4b.data.canonical import WINDOW_SIZE
from ml4b.data.features_invariant import extract_invariant_features
from ml4b.feedback import store


def _window_df(n: int) -> pd.DataFrame:
    """Build a synthetic raw windowed frame like predict_from_sensor_logger emits."""
    rng = np.random.default_rng(0)
    rows = []
    for wid in range(n):
        row = {
            "subject_id": 0,
            "exercise_name": "unknown",
            "recording_id": 0,
            "window_id": wid,
        }
        for ch in store.RAW_CHANNELS:
            row[ch] = rng.standard_normal(WINDOW_SIZE).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def _results(labels: list[str]) -> pd.DataFrame:
    """Build a per-window results frame aligned with the windows."""
    return pd.DataFrame(
        {
            "window_id": range(len(labels)),
            "predicted_class": labels,
            "confidence": [0.9] * len(labels),
            "time_start_seconds": [float(i) for i in range(len(labels))],
        }
    )


def test_append_and_load_roundtrip(tmp_path):
    """Corrections written to the store load back with the same content."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(3)
    res = _results(["bicep_curl", "row", "tricep_extension"])
    corrected = {0: "row", 2: "row"}  # fix windows 0 and 2
    records = store.build_records(win, res, list(corrected), corrected, source="a.csv")
    assert store.append(records, path=path) == 2

    loaded = store.load(path=path)
    assert len(loaded) == 2
    assert set(loaded["corrected_label"]) == {"row"}
    assert loaded["source"].unique().tolist() == ["a.csv"]
    # Raw channels survive the round-trip at full length.
    assert len(loaded.iloc[0]["raw_ax"]) == WINDOW_SIZE


def test_stats_counts_changes(tmp_path):
    """stats() reports totals, per-label counts and how many changed a prediction."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(3)
    res = _results(["bicep_curl", "row", "row"])
    # window 0: bicep->row (changed), window 1: row->row (unchanged label kept).
    corrected = {0: "row", 1: "row"}
    store.append(
        store.build_records(win, res, list(corrected), corrected, source="a.csv"),
        path=path,
    )
    s = store.stats(path=path)
    assert s["total"] == 2
    assert s["per_label"]["row"] == 2
    assert s["n_changed"] == 1
    assert s["n_sources"] == 1


def test_empty_store(tmp_path):
    """A missing feedback file yields an empty frame and zeroed stats."""
    path = tmp_path / "nope.jsonl"
    assert store.load(path=path).empty
    assert store.stats(path=path) == {
        "total": 0,
        "per_label": {},
        "n_sources": 0,
        "n_changed": 0,
    }


def test_to_window_df_schema(tmp_path):
    """to_window_df yields the windowing schema with corrected labels as targets."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(2)
    res = _results(["bicep_curl", "bicep_curl"])
    corrected = {0: "squat", 1: "squat"}  # a brand-new class
    store.append(
        store.build_records(win, res, list(corrected), corrected, source="s.csv"),
        path=path,
    )
    wdf = store.to_window_df(store.load(path=path))
    assert list(wdf.columns) == [
        "subject_id",
        "exercise_name",
        "recording_id",
        "window_id",
        *store.RAW_CHANNELS,
    ]
    assert wdf["exercise_name"].tolist() == ["squat", "squat"]
    assert wdf["recording_id"].iloc[0] == "feedback::s.csv"


def test_feedback_flows_through_training_pipeline(tmp_path):
    """Stored corrections augment + featurise exactly like training windows do."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(4)
    res = _results(["bicep_curl"] * 4)
    corrected = {i: "row" for i in range(4)}
    store.append(
        store.build_records(win, res, list(corrected), corrected, source="s.csv"),
        path=path,
    )
    wdf = store.to_window_df(store.load(path=path))

    augmented = augment_windows(wdf, n_augment=2, random_state=42)
    assert len(augmented) == len(wdf) * 3  # original + 2 copies

    feats = extract_invariant_features(augmented)
    assert not feats.empty
    # The corrected label is carried through as the training target.
    assert set(feats["exercise_name"]) == {"row"}


def test_clear(tmp_path):
    """clear() removes the feedback file."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(1)
    res = _results(["row"])
    store.append(
        store.build_records(win, res, [0], {0: "bicep_curl"}, source="s.csv"),
        path=path,
    )
    assert not store.load(path=path).empty
    store.clear(path=path)
    assert store.load(path=path).empty


def test_build_records_only_requested_windows(tmp_path):
    """Only the listed window ids with a mapped label become records."""
    win = _window_df(3)
    res = _results(["row", "row", "row"])
    corrected = {1: "bicep_curl"}
    records = store.build_records(win, res, [1], corrected, source="s.csv")
    assert len(records) == 1
    assert records[0]["window_id"] == 1
    assert records[0]["corrected_label"] == "bicep_curl"
    assert records[0]["predicted_label"] == "row"


@pytest.mark.parametrize("conf", [float("nan"), 0.5])
def test_confidence_serialises(tmp_path, conf):
    """NaN confidence is stored as null; real confidence is preserved."""
    path = tmp_path / "feedback.jsonl"
    win = _window_df(1)
    res = _results(["row"])
    res.loc[0, "confidence"] = conf
    store.append(
        store.build_records(win, res, [0], {0: "bicep_curl"}, source="s.csv"),
        path=path,
    )
    loaded = store.load(path=path)
    if conf != conf:  # NaN
        assert loaded.iloc[0]["confidence"] is None
    else:
        assert loaded.iloc[0]["confidence"] == conf
