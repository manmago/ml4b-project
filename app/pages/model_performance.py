"""Model & Training tab for the ML4B Streamlit app ("Daylight" design).

Displays the **honest leave-one-set-out** evaluation of the Random Forest trained
on the Kaggle Gym Workout IMU dataset (Apple Watch, single subject — DECISIONS.md).
All numbers are loaded from the committed ``models/saved/model_metrics.json`` so
the page always reflects what training actually produced (no hardcoded metrics,
no dataset needed). Reproduce with ``uv run python scripts/train_model.py``.
Presentation only — metrics come from ``src/ml4b/``.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.ui import theme, viz
from ml4b.utils.metrics import load_baseline_metrics, load_model_metrics

TARGET_F1 = 0.80
PLOTLY_CFG = {"displayModeBar": False}


def _render_model_comparison(metrics: dict, classes: list[str]) -> None:
    """Show Model 1 (Kaggle only) vs Model 2 (current) — the effect of our data.

    Reads the optional committed ``baseline_metrics.json``. When it is missing
    (older checkout, or never rebuilt) the comparison is silently skipped.

    Args:
        metrics: Current model (Model 2) metrics dict.
        classes: Fixed class order.
    """
    baseline = load_baseline_metrics()
    if baseline is None:
        return

    with st.container(border=True):
        st.markdown(
            theme.eyebrow("Effect of our own training data"), unsafe_allow_html=True
        )
        n_td = metrics.get("n_testdaten_sets", 0)
        delta = metrics["cv_macro_f1"] - baseline["cv_macro_f1"]
        theme.metric_tiles(
            [
                (
                    "Model 1 · Macro F1",
                    f"{baseline['cv_macro_f1']:.3f}",
                    f"{baseline['n_sets']} held-out Kaggle sets",
                    theme.MUTED,
                ),
                (
                    "Model 2 · Macro F1",
                    f"{metrics['cv_macro_f1']:.3f}",
                    f"Δ {delta:+.3f} · adds {n_td} of our set(s)",
                    theme.FLAME,
                ),
            ]
        )
        st.plotly_chart(
            viz.f1_compare(
                classes,
                [baseline["cv_per_class_f1"][c] for c in classes],
                [metrics["cv_per_class_f1"][c] for c in classes],
            ),
            width="stretch",
            config=PLOTLY_CFG,
        )
        st.caption(
            "Not a clean apples-to-apples gain: each model is scored over its own "
            "held-out sets, and Model 2's pool also contains our own (harder, "
            "cross-person) recordings. The same-recording comparison lives on the "
            "**Classify** tab."
        )


def render() -> None:
    """Render the Model & Training tab from the committed metrics."""
    metrics = load_model_metrics()
    classes = metrics["confusion_matrix_labels"]

    st.markdown(theme.eyebrow("Evaluation · leave-one-set-out"), unsafe_allow_html=True)
    st.markdown(
        "Evaluation of the **Random Forest** with **leave-one-set-out "
        "cross-validation**: each exercise *set* is held out once and predicted by "
        "a model trained on the others, so no window scores itself. This is the "
        "honest, leakage-free estimate for a single-subject dataset."
    )

    macro_f1 = metrics["cv_macro_f1"]
    # Per-class precision/recall from the aggregated CV confusion matrix.
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    recall_per_class = np.diag(cm) / cm.sum(axis=1)
    precision_per_class = np.diag(cm) / cm.sum(axis=0)
    macro_recall = float(recall_per_class.mean())
    theme.metric_tiles(
        [
            (
                "Macro F1",
                f"{macro_f1:.3f}",
                ("≥" if macro_f1 >= TARGET_F1 else "<") + f" target {TARGET_F1:.2f}",
                theme.FLAME if macro_f1 >= TARGET_F1 else theme.CLASS_COLORS["unknown"],
            ),
            (
                "Macro Recall",
                f"{macro_recall:.3f}",
                "averaged over classes",
                theme.STEEL,
            ),
            (
                "Accuracy",
                f"{metrics['cv_accuracy']:.1%}",
                "leave-one-set-out",
                theme.STEEL,
            ),
            (
                "Training sets",
                str(metrics["n_sets"]),
                "held out one at a time",
                theme.MUTED,
            ),
        ]
    )

    _render_model_comparison(metrics, classes)

    # Per-class quality and the confusion matrix side by side. We show the
    # precision·recall·F1 TABLE (not a second F1-only bar chart): it is strictly
    # more informative — adding precision and recall — and the F1-vs-target context
    # already lives in the Macro F1 tile above, so the bar chart was redundant.
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                theme.eyebrow("Per-class precision · recall · F1"),
                unsafe_allow_html=True,
            )
            pcf1_map = metrics["cv_per_class_f1"]
            pc_table = pd.DataFrame(
                {
                    "Exercise": [theme.humanize(c) for c in classes],
                    "Precision": [f"{p:.2f}" for p in precision_per_class],
                    "Recall": [f"{r:.2f}" for r in recall_per_class],
                    "F1": [f"{pcf1_map[c]:.2f}" for c in classes],
                }
            )
            st.dataframe(pc_table, width="stretch", hide_index=True)
            st.caption(
                f"Target macro F1 ≥ {TARGET_F1:.2f}. Precision, recall and F1 come "
                "from the aggregated leave-one-set-out confusion matrix."
            )
    with right:
        with st.container(border=True):
            st.markdown(
                theme.eyebrow("Confusion matrix (row-normalized)"),
                unsafe_allow_html=True,
            )
            cm = np.array(metrics["confusion_matrix"], dtype=float)
            # Row-normalize so each cell is the fraction of a true class predicted
            # as each class (rows sum to 1) — comparable across class sizes.
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            st.plotly_chart(
                viz.confusion(cm_norm, classes), width="stretch", config=PLOTLY_CFG
            )
            st.caption(
                "Bicep curl and tricep extension are the most-confused pair — both "
                "are elbow movements that look similar at the wrist."
            )

    with st.container(border=True):
        st.markdown(theme.eyebrow("Model details"), unsafe_allow_html=True)
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
                    "Random Forest · 300 trees · balanced · seed 42",
                    "Kaggle Gym Workout IMU — Apple Watch, 100 Hz, single subject",
                    ", ".join(theme.humanize(c) for c in classes)
                    + " (+ rest, uncertain)",
                    f"{metrics['n_features']} device-invariant features (orientation-robust)",
                    "200 samples = 2 s @ 100 Hz, 50% overlap",
                    f"{metrics['n_augment']}× rotation+time-warp+mirror+jitter",
                    f"{metrics['evaluation']}",
                    f"energy gate: accel-std > {gate['accel_mag_std']} g OR "
                    f"gyro-mean > {gate['gyro_mag_mean']} rad/s",
                    "max class probability < 0.50 → 'uncertain'",
                ],
            }
        )
        st.dataframe(details, width="stretch", hide_index=True)

    st.info(
        "**Why macro F1, not accuracy?** Macro F1 averages F1 equally across all "
        "classes regardless of size, rewarding the model for getting *every* "
        "exercise right — what matters for a balanced 3-class recognizer."
    )
    st.warning(
        "**Honest limitations — read before trusting these numbers.**\n\n"
        "- **Single-subject training anchor.** The Kaggle dataset is one person on "
        "one Apple Watch. True leave-one-*subject*-out is impossible, so the score "
        "measures generalisation to an unseen *set*, **not** a new *person*.\n"
        "- **Real-world performance will be lower** for a different user, whose "
        "movement style the model has never seen.\n"
        "- **Augmentation substitutes for missing subject diversity** (rotation, "
        "time-warp, mirror, jitter) — a sound, documented mitigation, not a "
        "replacement for real multi-subject data.\n"
        "- **Methodology is correct even if the ceiling is limited:** Apple-Watch "
        "training domain, leakage-free evaluation, invariant features, energy gate "
        "for rest, confidence-based abstention."
    )
