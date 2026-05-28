"""Model Performance page for the ML4B Gym Exercise Recognition Streamlit app.

Displays pre-computed test set evaluation metrics: accuracy, macro F1,
per-class F1 scores, and confusion matrix heatmap.

All metrics are hardcoded from the Phase 5 evaluation notebook results
(notebooks/05_evaluation.ipynb) to avoid re-running the full test set
evaluation on every page load. The test set was held out and used exactly
once — see ADR-007 for the subject-based split rationale.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render() -> None:
    """Render the Model Performance page."""
    st.title("📊 Model Performance")
    st.markdown("""
    Evaluation results on the **held-out test set** (30,096 windows, never seen during training).
    Primary metric: **Macro F1** — used because the test set retains the original class imbalance.
    """)

    # ── Overall metrics ────────────────────────────────────────────────────
    st.subheader("Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Test Macro F1",
            "0.8006",
            help="Primary metric — average F1 across all classes",
        )
    with col2:
        st.metric("Test Accuracy", "96.3%", help="Overall correct predictions")
    with col3:
        st.metric("Val Macro F1", "0.8136", help="Validation set performance")
    with col4:
        st.metric(
            "Generalization Gap",
            "1.3%",
            help="Val F1 − Test F1 — lower is better",
        )

    st.divider()

    # ── Model comparison table (validation set) ────────────────────────────
    st.subheader("Model Comparison (Validation Set)")
    comparison_df = pd.DataFrame(
        {
            "Model": ["Random Forest ✅", "XGBoost", "SVM"],
            "Macro F1": [0.8136, 0.8057, 0.7478],
            "Accuracy": [0.9618, 0.9617, 0.9429],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-class F1 bar chart ─────────────────────────────────────────────
    st.subheader("Per-Class F1 Score (Test Set)")
    per_class_df = pd.DataFrame(
        {
            "Exercise": [
                "rest",
                "bicep_curl",
                "shoulder_press",
                "tricep_extension",
                "squat",
                "lateral_raise",
            ],
            "F1 Score": [0.9816, 0.8915, 0.8064, 0.8007, 0.7716, 0.5519],
            # Support shows how many test windows each class contributed
            "Support (windows)": [27258, 652, 495, 624, 925, 142],
        }
    )

    fig_f1 = px.bar(
        per_class_df,
        x="Exercise",
        y="F1 Score",
        color="F1 Score",
        color_continuous_scale="Blues",
        title="Per-Class F1 Score on Test Set",
        text="F1 Score",
        height=400,
    )
    fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_f1.update_layout(yaxis_range=[0, 1.1], coloraxis_showscale=False)
    # Red dashed line marks the project target threshold
    fig_f1.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="red",
        annotation_text="Target ≥ 0.80",
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()

    # ── Confusion matrix heatmap ───────────────────────────────────────────
    st.subheader("Confusion Matrix (Test Set, Row-Normalised)")
    st.markdown(
        "Each cell shows the fraction of **true** class samples predicted as each class. "
        "Diagonal = correct predictions."
    )

    classes = [
        "bicep_curl",
        "lateral_raise",
        "rest",
        "shoulder_press",
        "squat",
        "tricep_extension",
    ]
    # Row-normalised values from Phase 5 evaluation notebook
    cm = np.array(
        [
            [0.98, 0.00, 0.02, 0.00, 0.00, 0.00],
            [0.06, 0.77, 0.17, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.97, 0.01, 0.01, 0.00],
            [0.00, 0.00, 0.03, 0.96, 0.00, 0.00],
            [0.01, 0.03, 0.17, 0.00, 0.79, 0.00],
            [0.06, 0.04, 0.12, 0.01, 0.00, 0.76],
        ]
    )

    fig_cm = px.imshow(
        cm,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        title="Confusion Matrix — Random Forest (Test Set)",
        labels={"x": "Predicted", "y": "True", "color": "Rate"},
        text_auto=".2f",
        height=500,
    )
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Key findings summary ───────────────────────────────────────────────
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **Strengths**
        - rest: F1 = 0.98 — very distinctive signal
        - bicep_curl: F1 = 0.89 — clean wrist movement
        - shoulder_press: F1 = 0.81 — clearly recognizable
        - Generalization gap only 1.3% — stable model
        """)
    with col2:
        st.warning("""
        **Weaknesses**
        - lateral_raise: F1 = 0.55 — confused with bicep_curl and rest
        - squat: 21% error rate — stop phases look like rest
        - tricep_extension: 24% error rate — confused with rest
        - Only 142 lateral_raise windows in test set (low support)
        """)
