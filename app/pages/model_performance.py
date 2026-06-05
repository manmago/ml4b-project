"""Model Performance page for the ML4B Streamlit app.

Displays the **honest leave-one-set-out** evaluation of the Random Forest trained
on the Kaggle Gym Workout IMU dataset (Apple Watch, single subject — DECISIONS.md).
All numbers are loaded from the committed ``models/saved/model_metrics.json`` so
the page always reflects what training actually produced (no hardcoded metrics,
no dataset needed). Reproduce with ``uv run python scripts/train_model.py``.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ml4b.utils.metrics import load_model_metrics

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
    """Render the Model Performance page from the committed metrics."""
    st.title("📊 Model Performance")
    metrics = load_model_metrics()
    classes = metrics["confusion_matrix_labels"]

    st.markdown(
        "Evaluation of the **Random Forest** model with **leave-one-set-out "
        "cross-validation**: each of the dataset's exercise *sets* is held out "
        "once and predicted by a model trained on the others, so no window from a "
        "set is ever used to score itself. This is the honest, leakage-free "
        "estimate for a single-subject dataset (DECISIONS.md). Reproduce with "
        "`uv run python scripts/train_model.py`."
    )

    # --- Headline metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    macro_f1 = metrics["cv_macro_f1"]
    c1.metric(
        "Macro F1",
        f"{macro_f1:.3f}",
        f"{'≥' if macro_f1 >= TARGET_F1 else '<'} target {TARGET_F1:.2f}",
        delta_color="normal" if macro_f1 >= TARGET_F1 else "inverse",
    )
    c2.metric("Accuracy", f"{metrics['cv_accuracy']:.1%}")
    c3.metric("Classes", str(len(classes)))
    c4.metric("Training sets", str(metrics["n_sets"]))

    st.divider()

    # --- Per-class F1 bar chart -------------------------------------------
    st.markdown("### 🎯 Per-Class F1 (leave-one-set-out)")
    per_class = metrics["cv_per_class_f1"]
    f1_df = pd.DataFrame(
        {
            "Exercise": [_humanize(c) for c in classes],
            "F1": [per_class[c] for c in classes],
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
    bar.add_hline(
        y=TARGET_F1, line_dash="dash", line_color="red", annotation_text="Target ≥ 0.80"
    )
    bar.update_layout(yaxis_range=[0, 1.05], height=420, coloraxis_showscale=False)
    st.plotly_chart(bar, width="stretch")

    st.divider()

    # --- Confusion matrix (row-normalized) --------------------------------
    st.markdown("### 🔢 Confusion Matrix (row-normalized)")
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    # Row-normalize so each cell is the fraction of a true class predicted as
    # each class (rows sum to 1) — comparable across classes of different sizes.
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    labels = [_humanize(c) for c in classes]
    heat = px.imshow(
        cm_norm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "True", "color": "Proportion"},
        text_auto=".2f",
        aspect="auto",
    )
    heat.update_layout(height=460)
    st.plotly_chart(heat, width="stretch")
    st.caption(
        "Bicep curl and tricep extension are the most-confused pair — both are "
        "elbow movements that look similar at the wrist."
    )

    st.divider()

    # --- Model details -----------------------------------------------------
    st.markdown("### 🛠️ Model Details")
    gate = metrics["activity_gate_thresholds"]
    details = pd.DataFrame(
        {
            "Property": [
                "Algorithm",
                "Training anchor",
                "Classes",
                "Features",
                "Window",
                "Augmentation",
                "Evaluation",
                "Rest detection",
                "Confidence threshold",
            ],
            "Value": [
                "Random Forest (300 trees, class_weight='balanced', seed 42)",
                "Kaggle Gym Workout IMU — Apple Watch, 100 Hz, single subject (DECISIONS.md)",
                ", ".join(_humanize(c) for c in classes) + " (+ rest, uncertain)",
                f"{metrics['n_features']} device-invariant features (DECISIONS.md)",
                "200 samples = 2 s @ 100 Hz, 50% overlap",
                f"{metrics['n_augment']}× rotation+time-warp+mirror+jitter (DECISIONS.md)",
                f"{metrics['evaluation']}",
                f"energy gate: accel-std > {gate['accel_mag_std']} g OR "
                f"gyro-mean > {gate['gyro_mag_mean']} rad/s (DECISIONS.md)",
                "max class probability < 0.50 → 'uncertain' (DECISIONS.md)",
            ],
        }
    )
    st.dataframe(details, width="stretch", hide_index=True)

    st.divider()

    # --- Why macro F1 ------------------------------------------------------
    st.info(
        "ℹ️ **Why macro F1, not accuracy?** Macro F1 averages the F1 score equally "
        "across all classes, regardless of how many windows each has. That rewards "
        "the model for getting *every* exercise right — not just the most common "
        "one — which is what matters for a balanced 3-class recognizer. See DECISIONS.md."
    )

    # --- Limitations (prominent, honest) -----------------------------------
    st.warning(
        "⚠️ **Honest limitations — read before trusting these numbers.**\n\n"
        "- **Single-subject training anchor.** The Kaggle dataset is one person on "
        "one Apple Watch. True leave-one-*subject*-out evaluation is impossible, so "
        "the score above measures generalisation to an unseen *set*, **not** to a "
        "new *person*.\n"
        "- **Real-world performance will be below these numbers.** Expect lower "
        "accuracy for a different user, because the model has never seen anyone "
        "else's movement style.\n"
        "- **Augmentation substitutes for missing subject diversity.** Random "
        "rotation, time-warp, mirroring and jitter synthesise the variability that "
        "more subjects would have provided (DECISIONS.md) — a sound, documented "
        "mitigation, but not a replacement for real multi-subject data, which was "
        "not available.\n"
        "- **Methodology is correct even if the ceiling is limited:** Apple-Watch "
        "training domain, leakage-free evaluation, invariant features, an energy "
        "gate for rest, and confidence-based abstention."
    )
