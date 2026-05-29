"""Model Performance page for the ML4B Streamlit app.

Displays the held-out test-set evaluation of the Random Forest trained on the
**MM-Fit** dataset (wrist-worn smartwatch — see ADR-013), reproducible via
``scripts/build_mmfit_dataset.py`` + ``scripts/train_model.py``: headline
metrics, per-class F1 scores, a row-normalized confusion matrix, and an honest
note on real Apple Watch behaviour. All numbers are hardcoded here so the page
renders without the 1.7 GB dataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Class order used for both the per-class F1 scores and the confusion matrix.
# 7 classes: the original 6 plus push_up (MM-Fit provides it) — see ADR-013.
CLASS_NAMES = [
    "bicep_curl",
    "lateral_raise",
    "push_up",
    "rest",
    "shoulder_press",
    "squat",
    "tricep_extension",
]

# Per-class F1 on the held-out MM-Fit TEST set (workouts 09/10/11).
PER_CLASS_F1_TEST = {
    "bicep_curl": 0.8613,
    "lateral_raise": 0.9725,
    "push_up": 0.9968,
    "rest": 0.9866,
    "shoulder_press": 0.9863,
    "squat": 0.8393,
    "tricep_extension": 0.9656,
}

# Row-normalized confusion matrix on the MM-Fit TEST set: rows = true class,
# cols = predicted class, in CLASS_NAMES order (each row sums to ~1.0).
CONFUSION_MATRIX_TEST_NORM = np.array(
    [
        [0.962, 0.000, 0.000, 0.038, 0.000, 0.000, 0.000],
        [0.000, 0.993, 0.000, 0.007, 0.000, 0.000, 0.000],
        [0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
        [0.008, 0.002, 0.000, 0.975, 0.001, 0.012, 0.002],
        [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.005, 0.000, 0.995, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ]
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
        "**MM-Fit test set** (workouts never seen during training). The model is "
        "trained on wrist-worn smartwatch data (**MM-Fit**, see ADR-013) to match "
        "the Apple Watch deployment device — reproduce with "
        "`scripts/build_mmfit_dataset.py` + `scripts/train_model.py`."
    )

    # --- Headline metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Macro F1", "0.944", "target ≥ 0.80 ✅")
    c2.metric("Test Accuracy", "97.8%")
    c3.metric("Val Macro F1", "0.866")
    c4.metric("Classes", "7", help="6 original + push_up")

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
        "**All 7 classes clear the 0.80 target.** Strongest: Push Up (1.00), "
        "Rest (0.99), Shoulder Press (0.99). Weakest: Squat (0.84) and Bicep Curl "
        "(0.86) — the two that move the wrist least distinctively."
    )

    st.divider()

    # --- Confusion matrix --------------------------------------------------
    st.markdown("### 🔢 Confusion Matrix (row-normalized, test set)")
    labels = [_humanize(c) for c in CLASS_NAMES]
    heat = px.imshow(
        CONFUSION_MATRIX_TEST_NORM,
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
            "- All 7 classes exceed the macro-F1 target (≥ 0.80) on unseen workouts\n"
            "- Trained on **wrist-worn** smartwatch data (MM-Fit) — matches the "
            "Apple Watch placement (ADR-013)\n"
            "- **Push Up** and **Rest** are near-perfect (F1 ≈ 1.00 / 0.99)\n"
            "- Workout-based split = realistic estimate for new users"
        )
    with right:
        st.markdown("#### ⚠️ Weaknesses")
        st.markdown(
            "- **Bicep Curl** vs **Tricep Extension** are the hardest pair — both "
            "are elbow rotations and look alike at the wrist\n"
            "- On a *real Apple Watch*, push-ups are recognized but bicep curls are "
            "still confused with tricep extensions (residual device-orientation "
            "gap — see ADR-013/014)\n"
            "- Minority exercises can be absorbed into the dominant **Rest** class"
        )

    st.info(
        "ℹ️ **Why macro F1, not accuracy?** The data is highly imbalanced — `rest` "
        "is ~84% of all windows. A model that always predicts `rest` would score "
        "~84% accuracy while being useless. **Macro F1** averages F1 equally across "
        "all seven classes, so it rewards correctly recognizing the *rare* "
        "exercises — exactly what matters here. See ADR-008."
    )
