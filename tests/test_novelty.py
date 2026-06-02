"""Tests for the open-set novelty detector (:mod:`ml4b.data.novelty`)."""

from __future__ import annotations

import joblib
import numpy as np

from ml4b.data.novelty import NoveltyDetector


def _make_clusters(seed: int = 42):
    """Build two well-separated Gaussian clusters as a toy known distribution.

    Returns:
        Tuple ``(X, y, feature_names)`` with two classes far apart in 4-D space.
    """
    rng = np.random.default_rng(seed)
    n = 300
    # Two clusters separated by 20 units on the first axis — clearly distinct.
    cluster_a = rng.normal(loc=0.0, scale=1.0, size=(n, 4))
    cluster_b = rng.normal(loc=0.0, scale=1.0, size=(n, 4)) + np.array([20, 0, 0, 0])
    X = np.vstack([cluster_a, cluster_b])
    y = np.array(["a"] * n + ["b"] * n)
    return X, y, ["f0", "f1", "f2", "f3"]


def test_in_distribution_windows_are_known():
    """Windows drawn from a fitted cluster are classified as known."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    # The training points themselves are in-distribution.
    known = detector.is_known(X)
    # The calibration keeps ~99% known, so the vast majority must pass.
    assert known.mean() > 0.95


def test_far_away_point_is_novel():
    """A point far from every cluster centroid is rejected as novel."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    # Far from both clusters (which sit near x=0 and x=20).
    outlier = np.array([[200.0, 200.0, 200.0, 200.0]])
    assert not detector.is_known(outlier)[0]


def test_classes_and_thresholds_populated():
    """Fitting records both class centroids and per-class thresholds."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    assert detector.classes_ == ["a", "b"]
    assert set(detector.thresholds_) == {"a", "b"}
    assert all(t > 0 for t in detector.thresholds_.values())
    assert detector.feature_names == names


def test_distances_shape_matches_classes():
    """:meth:`distances` returns one column per known class."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    dist = detector.distances(X[:10])
    assert dist.shape == (10, len(detector.classes_))


def test_empty_input_returns_empty():
    """An empty feature matrix yields an empty known-mask, not an error."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    empty = np.empty((0, 4))
    assert detector.is_known(empty).shape == (0,)


def test_serialization_roundtrip(tmp_path):
    """The fitted detector pickles with joblib and behaves identically."""
    X, y, names = _make_clusters()
    detector = NoveltyDetector().fit(X, y, names)
    path = tmp_path / "detector.joblib"
    joblib.dump(detector, path)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(detector.is_known(X), reloaded.is_known(X))
