# ADR-010: Random Forest Selected as Final Model
**Status:** Accepted
**Date:** 2026-05-28

## Context

Phase 4 trained and compared three classical ML classifiers on the 47-feature matrix (21,490 balanced training windows). All three models were evaluated on the validation set (~13,888 windows, original class distribution) using macro-averaged F1 as the primary metric. The best model is serialised to `models/saved/best_model.joblib` for Phase 5 test evaluation and Streamlit app deployment.

## Decision

**Random Forest** is selected as the final model.

## Results (Validation Set)

| Model | Macro F1 | Accuracy | Training Time |
|-------|---------|---------|--------------|
| **Random Forest** | **0.8136** | **0.9618** | ~1.3 s |
| XGBoost | 0.8057 | 0.9600 | ~3.0 s |
| SVM (RBF) | 0.7478 | 0.9524 | ~8 min |

## Rationale

1. **Highest macro F1:** Random Forest achieved 0.8136 vs XGBoost 0.8057 — a difference of 0.008 in the primary metric. While the gap is small, it is consistent and decisive given equal hyperparameter effort for both models.

2. **Training speed:** Random Forest trains in ~1.3 s vs ~3.0 s for XGBoost. Faster training makes it easier to retrain on new data.

3. **Inference speed:** Random Forest loads and predicts significantly faster than XGBoost at serving time, which matters for responsive Streamlit app predictions.

4. **Interpretability:** Random Forest's `feature_importances_` are directly interpretable (mean decrease in impurity). This satisfies the explainability quality goal in ADR-002 and the Streamlit app requirement to show which features drive predictions.

5. **No label encoding overhead:** Random Forest accepts string labels natively; XGBoost required a `LabelEncoder` wrapper, adding complexity.

## Alternatives Considered

**XGBoost (macro F1 = 0.8057):** Retained as a backup model (`models/saved/xgboost.joblib`). Marginally lower macro F1. Would be preferred if the gap grew larger with hyperparameter tuning in a future iteration.

**SVM (macro F1 = 0.7478):** Rejected. The 6.6-point macro F1 gap relative to Random Forest is large enough to be decisive. SVM also required an 8-minute training time due to Platt scaling.

## Consequences

**Positive:**
- `models/saved/best_model.joblib` is a Random Forest — fast to load, predict, and interpret in Streamlit app
- `feature_importances_` available directly for Phase 5 error analysis
- Simpler serving path (no label encoding wrapper)

**Negative:**
- XGBoost may close the gap with hyperparameter tuning (deferred to a future iteration if Phase 5 test results are unsatisfactory)
- Random Forest predictions are less calibrated than SVM probabilities — confidence scores in the Streamlit app are indicative, not perfectly calibrated
