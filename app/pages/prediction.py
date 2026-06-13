"""Prediction page for the ML4B Streamlit app.

Accepts a Sensor Logger export (``WristMotion.csv`` directly, or the full ZIP),
runs the shared prediction pipeline from
:func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`, and visualizes
the per-window exercise predictions.

When the **baseline model** (Model 1 — trained on the Kaggle anchor only) is
available alongside the **current model** (Model 2 — Kaggle + our own uploaded
Testdaten), the page runs BOTH on the same recording and shows them side by side,
so the effect of our own training data is visible directly (DECISIONS.md). When
the baseline is missing it falls back to showing the current model only.
"""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from ml4b.data.apple_watch_loader import predict_from_sensor_logger
from ml4b.data.session import dominant_label, format_set_summary, summarize_session

# Below this many windows the recording is likely too short to be meaningful.
MIN_WINDOWS_WARNING = 5

# Fixed colours for the non-exercise outputs so they read as "not a class".
# The three real exercises keep Plotly's default colour sequence.
NON_EXERCISE_COLORS = {
    "Rest": "#c9ccd1",
    "Uncertain": "#8a8f98",
    "Unknown": "#e06c75",  # reddish: an exercise the model was not trained on
}

# Display names for the two models compared on the page.
MODEL1_NAME = "Model 1 · Kaggle only"
MODEL2_NAME = "Model 2 · Kaggle + our data"


def _humanize(label: str) -> str:
    """Convert a snake_case class label to a Title Case display string."""
    return label.replace("_", " ").title()


def _save_upload_to_tempfile(uploaded_file: Any) -> Path:
    """Persist an in-memory Streamlit upload to a temp file with its suffix."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _run_model(tmp_path: Path, model: Any, feature_names: list[str], novelty: Any):
    """Run the full prediction pipeline for one model on the uploaded recording.

    Args:
        tmp_path: Path to the materialized upload.
        model: Trained classifier.
        feature_names: Ordered feature names the model expects.
        novelty: Optional novelty detector for this model.

    Returns:
        The per-window results DataFrame, with an added ``exercise`` display
        column (Title Case label).
    """
    results = predict_from_sensor_logger(
        tmp_path, model, feature_names, novelty_detector=novelty
    )
    results = results.copy()
    results["exercise"] = results["predicted_class"].map(_humanize)
    return results


def _overall_result(results: pd.DataFrame) -> str:
    """Return the recording's single headline label (rest excluded)."""
    overall_raw = dominant_label(results["predicted_class"])
    return _humanize(overall_raw) if overall_raw else "—"


def _timeline_fig(results: pd.DataFrame):
    """Build a constant-height coloured timeline strip for one model's results."""
    df = results.copy()
    df["_band"] = 1
    fig = px.bar(
        df,
        x="time_start_seconds",
        y="_band",
        color="exercise",
        color_discrete_map=NON_EXERCISE_COLORS,
        labels={"time_start_seconds": "Time (seconds)", "exercise": "Label"},
        hover_data={"window_id": True, "confidence": ":.2f", "_band": False},
    )
    fig.update_layout(bargap=0.0, height=220, yaxis=dict(visible=False, range=[0, 1]))
    return fig


def _distribution_fig(results: pd.DataFrame):
    """Build a donut chart of the label distribution for one model's results."""
    dist = results["exercise"].value_counts().reset_index()
    dist.columns = ["exercise", "windows"]
    fig = px.pie(
        dist,
        names="exercise",
        values="windows",
        hole=0.4,
        color="exercise",
        color_discrete_map=NON_EXERCISE_COLORS,
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
    return fig


def _sets_summary_line(results: pd.DataFrame) -> str:
    """Return the human-readable set summary (e.g. '2 sets of Bicep Curl')."""
    sets = summarize_session(results)
    return format_set_summary(sets) if not sets.empty else "No active sets"


def _shared_header(results: pd.DataFrame) -> None:
    """Render the recording-level metrics shared by both models (duration, rate)."""
    total_seconds = float(results["time_start_seconds"].max()) + 2.0
    detected_hz = results.attrs.get("detected_hz", "n/a")
    c1, c2, c3 = st.columns(3)
    c1.metric("Windows", len(results))
    c2.metric("Duration", f"{total_seconds:.0f} s")
    c3.metric("Detected rate", f"{detected_hz} Hz")


def _render_single_model(results: pd.DataFrame) -> None:
    """Full single-model view (used when no baseline model is available)."""
    _shared_header(results)
    overall = _overall_result(results)
    avg_conf = (
        float(results["confidence"].mean())
        if results["confidence"].notna().any()
        else 0.0
    )
    rest_pct = float((results["predicted_class"] == "rest").mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Result", overall)
    c2.metric("Avg Confidence (classified)", f"{avg_conf:.0%}")
    c3.metric("Rest (gated)", f"{rest_pct:.0%}")

    st.divider()
    st.markdown("### 📈 Exercise Timeline")
    st.plotly_chart(_timeline_fig(results), width="stretch")

    st.markdown("### 🏋️ Detected Sets")
    summary_line = _sets_summary_line(results)
    if summary_line:
        st.markdown(f"#### 🧮 {summary_line}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("### 🥧 Exercise Distribution")
        st.plotly_chart(_distribution_fig(results), width="stretch")
    with right:
        st.markdown("### 📋 Detailed Results")
        table = results[
            ["window_id", "exercise", "confidence", "time_start_seconds"]
        ].rename(
            columns={
                "window_id": "Window",
                "exercise": "Predicted Exercise",
                "confidence": "Confidence",
                "time_start_seconds": "Start (s)",
            }
        )
        st.dataframe(
            table,
            width="stretch",
            height=340,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

    _download_button(results.rename(columns={"predicted_class": "prediction"}))


def _render_comparison(r1: pd.DataFrame, r2: pd.DataFrame) -> None:
    """Two-model comparison view (Model 1 baseline vs Model 2 current)."""
    _shared_header(r2)

    # Per-window agreement: how often the two models give the SAME label.
    merged = r1[["window_id", "predicted_class", "exercise"]].merge(
        r2[["window_id", "predicted_class", "exercise", "confidence"]],
        on="window_id",
        suffixes=("_m1", "_m2"),
    )
    agreement = float(
        (merged["predicted_class_m1"] == merged["predicted_class_m2"]).mean()
    )
    n_changed = int(
        (merged["predicted_class_m1"] != merged["predicted_class_m2"]).sum()
    )

    st.info(
        f"**{MODEL1_NAME}** was trained on the Kaggle anchor only. "
        f"**{MODEL2_NAME}** was additionally trained on our own uploaded "
        "recordings (Testdaten) — both with the same 6× augmentation and "
        "pipeline. Differences below are purely the effect of our own data."
    )

    # --- Headline: overall result of each model + agreement ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric(f"🟦 {MODEL1_NAME}", _overall_result(r1))
    c2.metric(f"🟩 {MODEL2_NAME}", _overall_result(r2))
    c3.metric("Window agreement", f"{agreement:.0%}", f"{n_changed} changed")

    d1, d2 = st.columns(2)
    d1.caption(f"Detected sets — Model 1: **{_sets_summary_line(r1)}**")
    d2.caption(f"Detected sets — Model 2: **{_sets_summary_line(r2)}**")

    st.divider()

    # --- Stacked timelines so label changes are visible at a glance --------
    st.markdown("### 📈 Exercise Timeline — both models")
    st.caption(
        "Each bar is one 2-second window. Compare the two strips: where the "
        "colours differ, our own training data changed the prediction."
    )
    st.markdown(f"**🟦 {MODEL1_NAME}**")
    st.plotly_chart(_timeline_fig(r1), width="stretch")
    st.markdown(f"**🟩 {MODEL2_NAME}**")
    st.plotly_chart(_timeline_fig(r2), width="stretch")

    st.divider()

    # --- Distributions side by side ---------------------------------------
    st.markdown("### 🥧 Exercise Distribution")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**🟦 {MODEL1_NAME}**")
        st.plotly_chart(_distribution_fig(r1), width="stretch")
    with p2:
        st.markdown(f"**🟩 {MODEL2_NAME}**")
        st.plotly_chart(_distribution_fig(r2), width="stretch")

    st.divider()

    # --- Per-window comparison table (highlight the changes) --------------
    st.markdown("### 📋 Per-Window Comparison")
    st.caption("Rows where the two models disagree are flagged in the last column.")
    table = merged.copy()
    table["changed"] = table["predicted_class_m1"] != table["predicted_class_m2"]
    table = table[
        ["window_id", "exercise_m1", "exercise_m2", "confidence", "changed"]
    ].rename(
        columns={
            "window_id": "Window",
            "exercise_m1": "Model 1 (Kaggle only)",
            "exercise_m2": "Model 2 (+ our data)",
            "confidence": "M2 Confidence",
            "changed": "Changed?",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        height=380,
        column_config={
            "M2 Confidence": st.column_config.ProgressColumn(
                "M2 Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )

    # --- Download combined predictions ------------------------------------
    download = merged[
        ["window_id", "predicted_class_m1", "predicted_class_m2", "confidence"]
    ].rename(
        columns={
            "predicted_class_m1": "model1_kaggle_only",
            "predicted_class_m2": "model2_with_our_data",
            "confidence": "model2_confidence",
        }
    )
    st.download_button(
        "⬇️ Download both models' predictions as CSV",
        data=download.to_csv(index=False).encode("utf-8"),
        file_name="ml4b_predictions_comparison.csv",
        mime="text/csv",
    )


def _download_button(results: pd.DataFrame) -> None:
    """Download button for the single-model view."""
    csv_bytes = (
        results[["window_id", "prediction", "confidence", "time_start_seconds"]]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="ml4b_predictions.csv",
        mime="text/csv",
    )


def render(
    model: Any,
    feature_names: list[str],
    novelty_detector: Any = None,
    baseline_model: Any = None,
    baseline_novelty_detector: Any = None,
) -> None:
    """Render the Predict Exercise page.

    Args:
        model: Current model (Model 2 — Kaggle + Testdaten) loaded by the entry point.
        feature_names: Ordered feature names both models expect.
        novelty_detector: Optional novelty detector for the current model.
        baseline_model: Optional baseline model (Model 1 — Kaggle only). When
            present, the page shows a two-model comparison.
        baseline_novelty_detector: Optional novelty detector for the baseline model.
    """
    st.title("🔮 Predict Exercise")
    compare = baseline_model is not None
    intro = (
        "Upload **one continuous recording** from **Sensor Logger** (Apple "
        "Watch) — you can do several exercises with rest pauses in a single "
        "session. Each 2-second window is recognized as **bicep curl, tricep "
        "extension or row**. Low-motion pauses are marked **rest**, exercises "
        "the model was not trained on are marked **unknown**, and windows the "
        "model is unsure about are marked **uncertain**."
    )
    if compare:
        intro += (
            "\n\nThe recording is run through **two models** — one trained on the "
            "Kaggle data only, one additionally trained on our own uploaded "
            "recordings — so you can see the effect of our own training data."
        )
    st.markdown(intro)

    uploaded_file = st.file_uploader(
        "Upload sensor data from Sensor Logger",
        type=["csv", "zip"],
        help=(
            "Upload either:\n"
            "• WristMotion.csv — the wrist motion file from Sensor Logger\n"
            "• A ZIP file — the full Sensor Logger export folder\n\n"
            "To export: open Sensor Logger → tap your recording → Share → "
            "Save to Files"
        ),
    )

    if uploaded_file is None:
        st.info("⬆️ Upload a `WristMotion.csv` or a Sensor Logger ZIP to begin.")
        return

    tmp_path = _save_upload_to_tempfile(uploaded_file)
    try:
        with st.spinner("Running prediction pipeline..."):
            results = _run_model(tmp_path, model, feature_names, novelty_detector)
            baseline_results = (
                _run_model(
                    tmp_path, baseline_model, feature_names, baseline_novelty_detector
                )
                if compare
                else None
            )
    except ValueError as exc:
        st.error(f"❌ Could not process the file:\n\n{exc}")
        return
    except FileNotFoundError as exc:
        st.error(f"❌ {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — show any unexpected error to the user
        st.error(f"❌ Unexpected error while processing the file: {exc}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    if results.empty:
        st.warning("No windows were produced — the recording may be too short.")
        return
    if len(results) < MIN_WINDOWS_WARNING:
        st.warning(
            f"⚠️ Only {len(results)} window(s) detected. The recording may be too "
            "short for reliable results — record at least ~15–30 seconds."
        )

    st.success(f"✅ Analyzed {len(results)} windows.")

    if compare and baseline_results is not None:
        _render_comparison(baseline_results, results)
    else:
        _render_single_model(results)

    st.caption(
        "The recording is resampled to 100 Hz before windowing. **Rest** windows "
        "are detected by an energy gate (not the model) and have no confidence; "
        "**uncertain** windows are active but below the confidence threshold."
    )
