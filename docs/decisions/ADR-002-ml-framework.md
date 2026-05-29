# ADR-002: ML Framework — scikit-learn

**Status:** Accepted  
**Date:** 2026-05-15  
**Deciders:** Anshul Agrawal

---

## Context

The project classifies gym exercises from time-series sensor data (Apple Watch accelerometer + gyroscope). We need an ML framework that:

- Handles tabular/feature-engineered data well (sliding window statistics)
- Is accessible to team members with varying ML experience
- Has excellent documentation and a large community
- Integrates cleanly with pandas DataFrames and numpy arrays
- Supports model serialisation for deployment in the Streamlit app

Candidates evaluated: `scikit-learn`, `PyTorch`, `TensorFlow/Keras`, `XGBoost` (standalone).

---

## Decision

**scikit-learn** is the primary ML framework for all classical ML models.

- All preprocessing steps use scikit-learn `Pipeline` and `ColumnTransformer`
- Model selection follows the `fit` / `predict` / `score` API contract
- Models are serialised with `joblib` (scikit-learn's recommended serialiser)
- XGBoost and LightGBM may be added as additional estimators inside a scikit-learn pipeline if needed

The door remains open to adding **PyTorch** in a later phase if deep learning approaches (e.g. 1D-CNN or LSTM on raw sensor windows) are explored. This would be captured in a new ADR.

---

## Alternatives Considered

| Option | Why not chosen |
|--------|----------------|
| **PyTorch** | Built for raw sequence/deep learning; overkill for engineered tabular features and a steeper learning curve for a mixed-experience team. Kept as a future option. |
| **TensorFlow / Keras** | Same deep-learning overhead as PyTorch, heavier install, and no advantage on small tabular feature sets. |
| **XGBoost (standalone)** | Excellent gradient boosting, but using it alone would fragment the API; instead it is used *inside* the scikit-learn ecosystem as one estimator among several. |

## Rationale

The pipeline turns each sensor window into a fixed 47-feature vector, so the
problem is classical tabular classification — exactly scikit-learn's strength.
Its uniform `fit/predict` API, built-in cross-validation and model comparison,
`Pipeline` objects (which prevent data leakage), and trivial `joblib`
serialisation make it the fastest path to a reliable, well-understood model that
the Streamlit app can load directly. XGBoost still participates as a candidate
estimator (see ADR-009) without leaving the scikit-learn API.

---

## Consequences

**Positive:**
- Consistent, well-tested API for every algorithm (SVM, Random Forest, k-NN, etc.)
- Built-in cross-validation, hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`)
- Excellent documentation and tutorials; beginner-friendly
- `Pipeline` objects bundle preprocessing + model into a single artefact, preventing data leakage
- Trivial model serialisation: `joblib.dump(pipeline, path)`

**Negative / Trade-offs:**
- Not designed for raw sequence modelling — feature engineering (sliding window statistics) is required before fitting
- No GPU acceleration for classical algorithms; not a bottleneck at this dataset scale
- If deep learning is adopted later, a second framework (PyTorch) must be introduced alongside scikit-learn

**Neutral:**
- Dataset is expected to be small-to-medium (lab recordings) — scikit-learn's performance is more than sufficient
- No online/streaming learning required for this project scope
