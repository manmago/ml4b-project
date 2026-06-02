"""Prediction page for the ML4B Streamlit app.

Accepts a Sensor Logger export (``WristMotion.csv`` directly, or the full ZIP),
runs the shared prediction pipeline from
:func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`, and visualizes
the per-window exercise predictions with a timeline, distribution pie chart,
a detailed table, and a CSV download.
"""

import tempfile
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st

from ml4b.data.apple_watch_loader import predict_from_sensor_logger
from ml4b.data.session import summarize_session

# Below this many windows the recording is likely too short to be meaningful.
MIN_WINDOWS_WARNING = 5

# Fixed colours for the non-exercise outputs so they read as "not a class".
# The three real exercises keep Plotly's default colour sequence.
NON_EXERCISE_COLORS = {
    "Rest": "#c9ccd1",
    "Uncertain": "#8a8f98",
    "Unknown": "#e06c75",  # reddish: an exercise the model was not trained on
}

# Display labels that are not trained exercise classes.
NON_EXERCISE_LABELS = {"Rest", "Uncertain", "Unknown"}


def _humanize(label: str) -> str:
    """Convert a snake_case class label to a Title Case display string.

    Args:
        label: Raw class label, e.g. ``"bicep_curl"``.

    Returns:
        Display string, e.g. ``"Bicep Curl"``.
    """
    return label.replace("_", " ").title()


def _save_upload_to_tempfile(uploaded_file: Any) -> Path:
    """Persist an in-memory Streamlit upload to a temp file with its suffix.

    The loader works with file paths (and uses the suffix to choose the CSV vs
    ZIP code path), so we materialize the upload to disk first.

    Args:
        uploaded_file: The object returned by ``st.file_uploader``.

    Returns:
        Path to the temporary file containing the upload's bytes.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def render(model: Any, feature_names: list[str], novelty_detector: Any = None) -> None:
    """Render the Predict Exercise page.

    Args:
        model: Trained classifier loaded by the app entry point.
        feature_names: Ordered feature names the model expects.
        novelty_detector: Optional fitted novelty detector that flags exercises
            the model was not trained on as ``unknown`` (ADR-024). When ``None``,
            no out-of-distribution rejection is applied.
    """
    st.title("🔮 Predict Exercise")
    st.markdown(
        "Upload **one continuous recording** from **Sensor Logger** (Apple "
        "Watch) — you can do several exercises with rest pauses in a single "
        "session. Each 2-second window is recognized as **bicep curl, tricep "
        "extension or row**. Low-motion pauses are marked **rest**, exercises "
        "the model was not trained on are marked **unknown**, and windows the "
        "model is unsure about are marked **uncertain**."
    )

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

    # Run the pipeline, surfacing any error as a friendly message.
    tmp_path = _save_upload_to_tempfile(uploaded_file)
    try:
        with st.spinner("Running prediction pipeline..."):
            results = predict_from_sensor_logger(
                tmp_path, model, feature_names, novelty_detector=novelty_detector
            )
    except ValueError as exc:
        # Raised for unknown columns or a too-short recording.
        st.error(f"❌ Could not process the file:\n\n{exc}")
        return
    except FileNotFoundError as exc:
        # Raised when a ZIP has no WristMotion.csv inside.
        st.error(f"❌ {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — show any unexpected error to the user
        st.error(f"❌ Unexpected error while processing the file: {exc}")
        return
    finally:
        # Clean up the temp file regardless of success/failure.
        tmp_path.unlink(missing_ok=True)

    if results.empty:
        st.warning("No windows were produced — the recording may be too short.")
        return

    # Warn if the recording produced very few windows.
    if len(results) < MIN_WINDOWS_WARNING:
        st.warning(
            f"⚠️ Only {len(results)} window(s) detected. The recording may be too "
            "short for reliable results — record at least ~15–30 seconds."
        )

    # Add a friendly display label used throughout the visualizations.
    results = results.copy()
    results["exercise"] = results["predicted_class"].map(_humanize)

    st.success(f"✅ Analyzed {len(results)} windows.")

    # --- Summary metrics ---------------------------------------------------
    total_seconds = float(results["time_start_seconds"].max()) + 2.0
    detected_hz = results.attrs.get("detected_hz", "n/a")
    # "Most common" should reflect a real EXERCISE, not rest/uncertain.
    exercises_only = results[~results["exercise"].isin(NON_EXERCISE_LABELS)]
    most_common = (
        exercises_only["exercise"].mode().iloc[0] if not exercises_only.empty else "—"
    )
    # Confidence is NaN for gated rest windows; mean() skips them automatically.
    avg_conf = (
        float(results["confidence"].mean())
        if results["confidence"].notna().any()
        else 0.0
    )
    rest_pct = float((results["predicted_class"] == "rest").mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Windows", len(results))
    c2.metric("Duration", f"{total_seconds:.0f} s")
    c3.metric("Top Exercise", most_common)
    c4.metric("Detected rate", f"{detected_hz} Hz")

    c5, c6 = st.columns(2)
    c5.metric("Avg Confidence (classified)", f"{avg_conf:.0%}")
    c6.metric("Rest (gated)", f"{rest_pct:.0%}")

    st.caption(
        "The recording is resampled to 100 Hz before windowing. **Rest** windows "
        "are detected by an energy gate (not the model) and have no confidence; "
        "**uncertain** windows are active but below the confidence threshold."
    )

    st.divider()

    # --- Timeline chart ----------------------------------------------------
    st.markdown("### 📈 Exercise Timeline")
    st.caption(
        "Each bar is one 2-second window over time, coloured by the predicted "
        "label (rest and uncertain shown in grey)."
    )
    # Constant-height coloured strip: communicates WHAT was detected WHEN without
    # mixing the (missing) confidence of rest windows into the y-axis.
    results["_band"] = 1
    timeline = px.bar(
        results,
        x="time_start_seconds",
        y="_band",
        color="exercise",
        color_discrete_map=NON_EXERCISE_COLORS,
        labels={"time_start_seconds": "Time (seconds)", "exercise": "Label"},
        hover_data={"window_id": True, "confidence": ":.2f", "_band": False},
    )
    timeline.update_layout(
        bargap=0.0, height=260, yaxis=dict(visible=False, range=[0, 1])
    )
    st.plotly_chart(timeline, width="stretch")

    # --- Detected sets (bout summary) -------------------------------------
    st.markdown("### 🏋️ Detected Sets")
    st.caption(
        "Active windows between rest pauses are grouped into sets. Each set's "
        "exercise is the majority vote of its windows, which smooths out single "
        "misclassified windows."
    )
    sets = summarize_session(results)
    if sets.empty:
        st.info("No active sets detected — the recording looks like rest only.")
    else:
        sets_display = sets.copy()
        sets_display["Exercise"] = sets_display["label"].map(_humanize)
        sets_display = sets_display[
            ["bout_id", "Exercise", "duration_s", "n_windows", "mean_confidence"]
        ].rename(
            columns={
                "bout_id": "Set",
                "duration_s": "Duration (s)",
                "n_windows": "Windows",
                "mean_confidence": "Avg Confidence",
            }
        )
        # Sets are 1-based for humans.
        sets_display["Set"] = sets_display["Set"] + 1
        st.dataframe(
            sets_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Avg Confidence": st.column_config.ProgressColumn(
                    "Avg Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

    st.divider()

    # --- Distribution pie + table -----------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 🥧 Exercise Distribution")
        dist = results["exercise"].value_counts().reset_index()
        dist.columns = ["exercise", "windows"]
        pie = px.pie(
            dist,
            names="exercise",
            values="windows",
            hole=0.4,
            color="exercise",
            color_discrete_map=NON_EXERCISE_COLORS,
        )
        pie.update_layout(height=380)
        st.plotly_chart(pie, width="stretch")
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
            height=380,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

    # --- Download ----------------------------------------------------------
    csv_bytes = (
        results[["window_id", "predicted_class", "confidence", "time_start_seconds"]]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="ml4b_predictions.csv",
        mime="text/csv",
    )
