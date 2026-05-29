"""Model Performance page for the ML4B Streamlit app.

Displays the held-out test-set evaluation results computed in
``notebooks/05_evaluation.ipynb`` (and reproducible via
``scripts/train_model.py``): headline metrics, the model comparison table,
per-class F1 scores, and a row-normalized confusion matrix. All numbers are
hardcoded here so the page renders without the dataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Class order used for both the per-class F1 scores and the confusion matrix.
CLASS_NAMES = [
    "bicep_curl",
    "lateral_raise",
    "rest",
    "shoulder_press",
    "squat",
    "tricep_extension",
]

# Per-class F1 on the held-out TEST set (Phase 5 results).
PER_CLASS_F1_TEST = {
    "bicep_curl": 0.8915,
    "lateral_raise": 0.5519,
    "rest": 0.9816,
    "shoulder_press": 0.8064,
    "squat": 0.7716,
    "tricep_extension": 0.8007,
}

# Confusion matrix on the TEST set: rows = true class, cols = predicted class,
# in CLASS_NAMES order. Reproduced from notebooks/05_evaluation.ipynb.
CONFUSION_MATRIX_TEST = np.array(
    [
        [637, 0, 15, 0, 0, 0],
        [9, 109, 24, 0, 0, 0],
        [82, 94, 26550, 205, 242, 85],
        [1, 0, 15, 477, 0, 2],
        [8, 26, 156, 0, 733, 2],
        [40, 24, 78, 6, 0, 476],
    ]
)

# Validation-set macro F1 per candidate model (Phase 4 model comparison).
MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": ["Random Forest ✅", "XGBoost", "SVM (RBF)"],
        "Val Macro F1": [0.8136, 0.8057, 0.7478],
    }
)

TARGET_F1 = 0.80


def _humanize(label: str) -> str:
    """Convert a snake_case class label to Title Case for display.

    Args:
        label: Raw class label, e.g. ``"bicep_curl"``.

    Returns:
        Display string, e.g. ``"Bicep Curl"``.
    """
    return label.replace("_", " ").title()


def render() -> None:
    """Render the Model Performance page (static, dataset-free)."""
    st.title("📊 Model Performance")
    st.markdown(
        "Final evaluation of the **Random Forest** model on the held-out "
        "**test set** (subjects never seen during training). The test set was "
        "used exactly once — see `notebooks/05_evaluation.ipynb`."
    )

    # --- Headline metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Macro F1", "0.8006", "target ≥ 0.80 ✅")
    c2.metric("Test Accuracy", "96.3%")
    c3.metric("Val Macro F1", "0.8136")
    c4.metric("Generalization Gap", "1.3%", help="Val 0.8136 → Test 0.8006")

    st.divider()

    # --- Model comparison --------------------------------------------------
    st.markdown("### 🏆 Model Comparison (validation set)")
    st.markdown(
        "Three classical models were compared on the validation set by **macro F1** "
        "(see ADR-009, ADR-010). Random Forest won and became the final model."
    )
    st.dataframe(
        MODEL_COMPARISON,
        width="stretch",
        hide_index=True,
        column_config={
            "Val Macro F1": st.column_config.ProgressColumn(
                "Val Macro F1", min_value=0.0, max_value=1.0, format="%.4f"
            )
        },
    )

    st.divider()

    # --- Per-class F1 bar chart -------------------------------------------
    st.markdown("### 🎯 Per-Class F1 (test set)")
    f1_df = pd.DataFrame(
        {
            "Exercise": [_humanize(c) for c in CLASS_NAMES],
            "F1": [PER_CLASS_F1_TEST[c] for c in CLASS_NAMES],
        }
    )
    bar = px.bar(
        f1_df,
        x="Exercise",
        y="F1",
        color="F1",
        color_continuous_scale="Tealgrn",
        text="F1",
    )
    bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    # Horizontal line marking the ≥ 0.80 macro-F1 target.
    bar.add_hline(
        y=TARGET_F1,
        line_dash="dash",
        line_color="red",
        annotation_text="Target ≥ 0.80",
    )
    bar.update_layout(yaxis_range=[0, 1.05], height=420, coloraxis_showscale=False)
    st.plotly_chart(bar, width="stretch")
    st.caption(
        "**Strongest:** Rest (0.98) and Bicep Curl (0.89). "
        "**Weakest:** Lateral Raise (0.55) — it is most often confused with Rest "
        "and Bicep Curl because of its smaller, subtler motion and fewer samples."
    )

    st.divider()

    # --- Confusion matrix --------------------------------------------------
    st.markdown("### 🔢 Confusion Matrix (row-normalized, test set)")
    # Normalize each row to sum to 1 so per-class recall is comparable despite
    # the heavy class imbalance (rest dominates the raw counts).
    row_sums = CONFUSION_MATRIX_TEST.sum(axis=1, keepdims=True)
    cm_norm = CONFUSION_MATRIX_TEST / row_sums
    labels = [_humanize(c) for c in CLASS_NAMES]
    heat = px.imshow(
        cm_norm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True", "color": "Proportion"},
        text_auto=".2f",
        aspect="auto",
    )
    heat.update_layout(height=480)
    st.plotly_chart(heat, width="stretch")

    st.divider()

    # --- Key findings ------------------------------------------------------
    st.markdown("### 🔍 Key Findings")
    left, right = st.columns(2)
    with left:
        st.markdown("#### ✅ Strengths")
        st.markdown(
            "- Macro F1 target (≥ 0.80) **met** on unseen subjects\n"
            "- Tiny generalization gap (1.3%) → the model does **not** overfit\n"
            "- Excellent at detecting **Rest** (F1 = 0.98) and **Bicep Curl** (0.89)\n"
            "- Subject-based split (ADR-007) gives a realistic estimate for new users"
        )
    with right:
        st.markdown("#### ⚠️ Weaknesses")
        st.markdown(
            "- **Lateral Raise** is hardest (F1 = 0.55) — subtle motion, fewest samples\n"
            "- Minority exercises are sometimes absorbed into the dominant **Rest** class\n"
            "- Trained on RecoFit wrist data; real Apple Watch generalization still "
            "pending live data collection"
        )

    st.info(
        "ℹ️ **Why macro F1, not accuracy?** The data is highly imbalanced — `rest` "
        "is ~89% of all windows. A model that always predicts `rest` would score "
        "~89% accuracy while being useless. **Macro F1** averages F1 equally across "
        "all six classes, so it rewards correctly recognizing the *rare* exercises "
        "— exactly what matters here. See ADR-008."
    )
