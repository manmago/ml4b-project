# ADR-009: Model Selection Rationale — Phase 4 Classifiers
**Status:** Accepted
**Date:** 2026-05-28

## Context

Phase 3 produced a 47-feature matrix (7 per-axis statistics × 6 axes + 3 magnitude + 2 FFT features) with 21,490 training windows (balanced via undersampling) and ~13,888 validation windows (original distribution). We need to select which ML algorithm(s) to train and compare for the gym exercise recognition task with 6 classes.

The primary evaluation metric is **macro-averaged F1** (not accuracy) because the validation and test sets retain the original class distribution where `rest` dominates (~89% of windows). A model predicting `rest` for everything would achieve ~89% accuracy but near-zero macro F1.

## Decision

Train and compare three classical ML classifiers:
1. **Random Forest** — baseline model
2. **XGBoost** — gradient boosting alternative
3. **SVM (RBF kernel)** — classical HAR benchmark

All three use `class_weight='balanced'` (or equivalent sample weighting for XGBoost) as a second safeguard against class imbalance in addition to the undersampling applied in Phase 3. The best model by macro F1 on the validation set is saved to `models/saved/` for Phase 5 test evaluation and Streamlit app deployment.

## Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| Deep Learning (CNN/LSTM on raw signals) | Requires raw windowed signals, not the feature matrix; higher complexity; out of scope for a university ML project using scikit-learn (ADR-002); would need 10× more data to outperform tabular models reliably |
| k-Nearest Neighbours (k-NN) | Prediction time scales with training set size (21,490 samples × 47 features); too slow for real-time app use; also sensitive to feature scale differences across axes |
| Logistic Regression | Strong linear baseline, but exercise classification from sensor features is likely non-linear; Random Forest already provides a solid non-linear baseline with similar interpretability |
| Naive Bayes | Assumes feature independence — invalid for correlated sensor statistics (e.g. `ax_mean` and `ax_rms` are strongly correlated) |
| Decision Tree (single) | High variance; Random Forest already subsumes Decision Trees with variance reduction via bagging |

## Rationale

**Random Forest** is the canonical baseline for tabular sensor data classification:
- Handles multi-class natively and without feature scaling
- Provides feature importance scores for interpretability
- Robust to correlated features (random subspace selection)
- Well-studied in HAR literature

**XGBoost** typically outperforms Random Forest on tabular data via boosting:
- Sequential weak learners correct prior errors — stronger bias reduction
- Built-in regularisation (subsample, colsample_bytree) reduces overfitting
- Efficient implementation handles 21,000 training samples in seconds

**SVM (RBF kernel)** is included as a classical HAR benchmark:
- Widely used in activity recognition literature, enabling comparison with related work
- RBF kernel can capture non-linear decision boundaries in sensor feature space
- Must be wrapped in a `Pipeline` with `StandardScaler` because SVM is sensitive to feature scale differences — the Pipeline ensures consistent preprocessing at inference time

Comparing three models covers the standard ensemble/boosting/kernel triad for tabular classification, giving a well-rounded picture before committing to one model for test evaluation.

## Consequences

**Positive:**
- Three trained models provide a solid comparison baseline
- The Pipeline wrapping for SVM ensures consistent preprocessing at inference time (same object used in Streamlit app)
- Feature importance from Random Forest / XGBoost aids model interpretability
- ADR-009 documents the comparison so future teams can extend or replace models

**Negative:**
- SVM training on 21,000 samples with RBF kernel and `probability=True` (Platt scaling) is slow (~5–15 minutes); SVM may be dropped if compute time is unacceptable
- Hyperparameter tuning is deferred to Phase 5 — current values are reasonable starting points, not optimised
- XGBoost adds an external dependency (`xgboost>=2.0`)
