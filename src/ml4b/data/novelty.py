"""Open-set novelty detection — flags unknown exercises as ``unknown``.

The shipped classifier is a *closed-set* model: it only knows three exercises
(``bicep_curl``, ``row``, ``tricep_extension``) and is forced to pick one of them
for every active window. In a real gym recording the user also performs exercises
the model was never trained on (squats, shoulder press, …). Those windows clear
the activity gate (they are full of motion) and the model then labels them — often
*confidently* — as one of the three known classes. The confidence threshold that
produces ``uncertain`` (DECISIONS.md) only catches the low-confidence cases, not the
confident mistakes.

This module adds genuine out-of-distribution rejection (DECISIONS.md). The three
exercises form three well-separated clusters in the device-invariant feature space
(:mod:`ml4b.data.features_invariant`). For each class we fit a Gaussian model
(mean + Ledoit-Wolf shrinkage covariance) and measure how far a new window sits
from the *nearest* class centroid using the Mahalanobis distance. A window is
``known`` only if it falls within a calibrated distance of at least one class;
otherwise it is novel and the caller labels it ``unknown``.

Per-class (rather than one global) covariances are used on purpose: a single
global covariance would merge the three clusters into one diffuse blob and let
genuinely foreign motion slip through. Ledoit-Wolf shrinkage keeps the per-class
covariance well-conditioned even though the feature space has ~38 dimensions.

The detector is fit once by ``scripts/fit_novelty_detector.py`` on the same
invariant features the model is trained on, and saved next to the model as
``models/saved/novelty_detector.joblib`` so the app runs without the dataset
(same rationale as DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# Default calibration percentile: a class threshold is set so that this fraction
# of that class's own training windows is considered "known". 99 leaves a 1%
# margin, trading a tiny false-novel rate for robust rejection of foreign motion.
DEFAULT_PERCENTILE: float = 99.0


class NoveltyDetector:
    """Per-class Mahalanobis novelty detector for open-set rejection.

    Fits one Gaussian (mean + Ledoit-Wolf covariance) per known class in a
    standardized feature space and rejects windows that are far from every class
    centroid. Plain attributes only, so the fitted object pickles cleanly with
    joblib.

    Attributes:
        percentile: Calibration percentile used for the per-class thresholds.
        feature_names: Ordered feature columns the detector was fit on; used to
            sanity-check that inference passes features in the same order.
        scaler: ``StandardScaler`` fit on all training features.
        classes_: Sorted list of known class labels.
        estimators_: Mapping ``class label -> fitted LedoitWolf`` (in scaled space).
        thresholds_: Mapping ``class label -> Mahalanobis-distance threshold``.
    """

    def __init__(self, percentile: float = DEFAULT_PERCENTILE) -> None:
        """Initialise the detector.

        Args:
            percentile: Per-class calibration percentile in (0, 100]. A higher
                value makes the detector more permissive (fewer ``unknown``).
        """
        self.percentile = percentile
        self.feature_names: list[str] = []
        self.scaler: StandardScaler | None = None
        self.classes_: list[str] = []
        self.estimators_: dict[str, LedoitWolf] = {}
        self.thresholds_: dict[str, float] = {}

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> NoveltyDetector:
        """Fit one Gaussian per class and calibrate the per-class thresholds.

        Args:
            X: Feature matrix ``(n_windows, n_features)`` — the same invariant
                features fed to the classifier.
            y: Class label per window ``(n_windows,)``.
            feature_names: Ordered names of the columns in ``X``.

        Returns:
            ``self``, fitted.
        """
        self.feature_names = list(feature_names)

        # Standardize so every feature contributes comparably to the distance and
        # the per-class covariances are well-scaled before shrinkage.
        self.scaler = StandardScaler().fit(X)
        x_scaled = self.scaler.transform(X)

        # Sorted for reproducible class order in reports and serialization.
        self.classes_ = sorted({str(label) for label in y})

        y = np.asarray(y).astype(str)
        for cls in self.classes_:
            mask = y == cls
            # Ledoit-Wolf gives a well-conditioned covariance (and its inverse)
            # even with many features and limited samples per class.
            estimator = LedoitWolf().fit(x_scaled[mask])
            # .mahalanobis() returns the SQUARED distance; sqrt for an
            # interpretable, linearly-thresholded distance.
            dist = np.sqrt(estimator.mahalanobis(x_scaled[mask]))
            self.estimators_[cls] = estimator
            self.thresholds_[cls] = float(np.percentile(dist, self.percentile))

        return self

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        """Standardize ``X`` with the fitted scaler.

        Args:
            X: Raw feature matrix ``(n_windows, n_features)``.

        Returns:
            Standardized feature matrix.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.scaler is None:
            raise RuntimeError("NoveltyDetector must be fit before use.")
        return self.scaler.transform(X)

    def distances(self, X: np.ndarray) -> np.ndarray:
        """Mahalanobis distance from each window to every class centroid.

        Args:
            X: Feature matrix ``(n_windows, n_features)``.

        Returns:
            Array ``(n_windows, n_classes)`` of distances, columns ordered like
            :attr:`classes_`.
        """
        x_scaled = self._scaled(X)
        # One column per class; sqrt converts squared Mahalanobis to a distance.
        cols = [
            np.sqrt(self.estimators_[cls].mahalanobis(x_scaled))
            for cls in self.classes_
        ]
        return np.column_stack(cols)

    def is_known(self, X: np.ndarray) -> np.ndarray:
        """Decide, per window, whether it belongs to a known class.

        A window is ``known`` if its distance to at least one class centroid is
        within that class's calibrated threshold; otherwise it is novel.

        Args:
            X: Feature matrix ``(n_windows, n_features)``.

        Returns:
            Boolean array ``(n_windows,)``; ``True`` = known, ``False`` = novel.
        """
        if X.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        dist = self.distances(X)
        # Per-class threshold vector aligned to the distance columns.
        thresh = np.array([self.thresholds_[cls] for cls in self.classes_])
        # Known if the window is inside ANY class's threshold.
        return np.any(dist <= thresh, axis=1)
