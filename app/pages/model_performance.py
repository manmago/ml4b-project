"""Model performance page — Streamlit sub-page for displaying model evaluation metrics.

This page will implement the metrics dashboard for Phase 6:
1. Load precomputed evaluation results from reports/
2. Display overall metrics: Test Accuracy (96.3%), Test Macro F1 (0.8006)
3. Show the row-normalised confusion matrix heatmap
4. Show per-class F1 scores as a bar chart
5. Show feature importance from the Random Forest model
6. Display the validation vs test comparison table and generalization gap (1.3%)

All evaluation results were computed in notebooks/05_evaluation.ipynb using the
held-out test set (first and only use of test data).

Phase 6 implementation target: replace the placeholder in streamlit_app.py with
the completed logic from this module.
"""
