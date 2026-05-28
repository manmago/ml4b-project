"""Model training module for ML4B gym exercise recognition.

Trains and compares multiple classical ML classifiers on the
preprocessed feature matrix from Phase 3. All models use
class_weight='balanced' as a second safeguard against class imbalance
(in addition to the undersampling applied in Phase 3).

The three models trained here — Random Forest, XGBoost, SVM — are compared
by macro F1 on the validation set in notebooks/04_modeling.ipynb. The winner
is serialised to models/saved/ for use in the Streamlit app.
See ADR-009 for the model selection rationale.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


class _XGBStringLabelClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper that adds string-label support to XGBClassifier.

    XGBoost raises ``ValueError: Invalid classes`` when ``sample_weight`` is
    passed together with string class labels — it requires numeric targets in
    that code path. This wrapper applies ``LabelEncoder`` internally so the
    rest of the pipeline (evaluate_model, Streamlit app) can use string labels
    at both fit and predict time without any changes.

    Attributes:
        classes_: Array of original string class names in encoder order.
        feature_importances_: Delegated to the wrapped XGBClassifier.
    """

    def __init__(self, xgb_model: Any) -> None:
        self._model = xgb_model
        self._encoder = LabelEncoder()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "_XGBStringLabelClassifier":
        """Encode string labels to integers, then fit the XGBClassifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: String class labels of shape (n_samples,)
            sample_weight: Per-sample weights for class balancing. Default None.

        Returns:
            self
        """
        # LabelEncoder is required here: XGBoost's C++ backend rejects string
        # targets when sample_weight is provided (it falls back to a code path
        # that only accepts contiguous integer labels starting at 0).
        y_encoded = self._encoder.fit_transform(y)
        self.classes_ = self._encoder.classes_  # expose for sklearn convention
        self._model.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels, decoded back to original strings.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of string class labels of shape (n_samples,)
        """
        y_encoded = self._model.predict(X).astype(int)
        return self._encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities in the order of self.classes_.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_classes) — columns match self.classes_
        """
        # XGBoost outputs columns in the same order as LabelEncoder.classes_,
        # which is alphabetical — the same order we expose in self.classes_.
        # No reordering needed.
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance scores from the underlying XGBClassifier."""
        return self._model.feature_importances_


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier as baseline model.

    Random Forest is chosen as baseline because it handles
    multi-class classification natively, is robust to feature
    scaling differences, provides feature importance scores,
    and typically performs well on tabular sensor features.
    See ADR-009 for model selection rationale.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features)
        y_train: Class labels of shape (n_samples,)
        random_state: Seed for reproducibility. Default 42.

    Returns:
        Trained RandomForestClassifier.
    """
    # n_estimators=200: balances bias-variance for 6 classes.
    # 100 can underfit; 500+ adds diminishing returns on tabular sensor data.
    # max_depth=None: grow full trees — variance is controlled by averaging
    # across 200 trees, not by limiting individual tree depth.
    # min_samples_split=5 / min_samples_leaf=2: reduce sensitivity to noise
    # at leaf level without aggressively pruning the tree.
    # class_weight='balanced': reweights each class by inverse frequency —
    # second safety net on top of the undersampling done in Phase 3.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,  # Use all CPU cores — no effect on outputs, only speed
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost classifier as second model.

    XGBoost uses sequential boosting (each tree corrects prior residuals)
    rather than bagging, which often outperforms Random Forest on tabular data.
    It natively handles multi-class via softmax and does not require feature
    scaling. See ADR-009 for model selection rationale.

    Args:
        X_train: Feature matrix
        y_train: Class labels (string or numeric)
        random_state: Seed. Default 42.

    Returns:
        Trained XGBClassifier.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier  # Deferred import — optional dependency

    # n_estimators=300, learning_rate=0.1: standard starting point for XGBoost
    # on tabular data. More trees with a lower rate generalise better than
    # fewer trees with a higher rate.
    # max_depth=6: shallower than RF since boosting corrects bias incrementally —
    # deep individual trees would overfit in the boosting context.
    # subsample=0.8, colsample_bytree=0.8: stochastic subsampling of rows and
    # columns per tree — reduces overfitting without sacrificing much accuracy.
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",  # Multi-class log-loss; avoids a verbose warning
        n_jobs=-1,
        random_state=random_state,
    )

    # XGBClassifier does not accept class_weight= like sklearn estimators.
    # Instead we compute per-sample weights so each sample is reweighted by
    # the inverse of its class frequency — equivalent to class_weight='balanced'.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # _XGBStringLabelClassifier wraps xgb and applies LabelEncoder internally
    # before fit(), then decodes predictions back to strings. This is required
    # because XGBoost's C++ backend raises ValueError when sample_weight is
    # combined with non-numeric (string) class labels.
    model = _XGBStringLabelClassifier(xgb)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Pipeline:
    """Train a Support Vector Machine classifier for comparison.

    SVMs are sensitive to feature scale — features with large numeric ranges
    dominate the decision boundary. StandardScaler is applied inside a Pipeline
    so the scaler is fitted on training data only, preventing data leakage.
    The Pipeline also ensures the scaler is automatically applied at prediction
    time in the Streamlit app. See ADR-009 for model selection rationale.

    Args:
        X_train: Feature matrix (scaled internally by the Pipeline)
        y_train: Class labels
        random_state: Seed for reproducibility. Default 42.

    Returns:
        Trained sklearn Pipeline containing [StandardScaler, SVC].
    """
    # Pipeline guarantees that scaling is always applied consistently —
    # both during training and when model.predict() is called later.
    # C=1.0 (default regularisation), kernel='rbf' (radial basis function):
    # standard starting choices for HAR benchmark tasks on tabular features.
    # probability=True enables predict_proba() for the Streamlit confidence
    # score display — comes at a small training-time cost (Platt scaling).
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,  # Required for confidence scores in Streamlit app
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline
