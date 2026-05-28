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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

    String class labels are encoded to integers internally via LabelEncoder
    (XGBoost requires numeric targets when sample_weight is provided).
    The returned object decodes predictions back to string labels transparently.

    Args:
        X_train: Feature matrix
        y_train: Class labels (string labels, e.g. 'bicep_curl', 'rest')
        random_state: Seed. Default 42.

    Returns:
        XGBWithEncoder instance — sklearn-compatible, accepts/returns string labels.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier  # Deferred import — optional dependency

    # Encode string labels to integers — XGBoost requires numeric classes
    # when sample_weight is passed; string labels cause ValueError: Invalid classes.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    # Compute sample weights on encoded labels (same result as on string labels)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_encoded)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_encoded, sample_weight=sample_weights)

    # Wrap in a simple object that decodes predictions back to string labels
    # so the rest of the pipeline (evaluate_model, Streamlit app) is unaffected.
    class XGBWithEncoder:
        """Thin wrapper exposing a sklearn-compatible interface with string labels."""

        def __init__(self, model: Any, encoder: LabelEncoder) -> None:
            self.model = model
            self.encoder = encoder
            self.classes_ = encoder.classes_  # string class names in label order

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Return string class labels."""
            return self.encoder.inverse_transform(self.model.predict(X).astype(int))

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Return class probabilities; columns match self.classes_ order."""
            return self.model.predict_proba(X)

        @property
        def feature_importances_(self) -> np.ndarray:
            """Delegate feature importances to the underlying XGBClassifier."""
            return self.model.feature_importances_

    return XGBWithEncoder(model, le)


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
