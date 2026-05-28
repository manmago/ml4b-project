"""Prediction page for the ML4B Gym Exercise Recognition Streamlit app.

Accepts a WristMotion.csv upload from Sensor Logger (Apple Watch),
runs it through the full prediction pipeline, and displays results.
"""

from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st

from ml4b.data.apple_watch_loader import predict_from_sensor_logger

# Human-readable labels and emoji for each model class name
EXERCISE_LABELS: dict[str, str] = {
    "bicep_curl": "💪 Bicep Curl",
    "shoulder_press": "🏋️ Shoulder Press",
    "squat": "🦵 Squat",
    "tricep_extension": "💪 Tricep Extension",
    "lateral_raise": "🤸 Lateral Raise",
    "rest": "😴 Rest",
}

# Consistent color per class for all charts on this page
EXERCISE_COLORS: dict[str, str] = {
    "bicep_curl": "#1f77b4",
    "shoulder_press": "#ff7f0e",
    "squat": "#2ca02c",
    "tricep_extension": "#d62728",
    "lateral_raise": "#9467bd",
    "rest": "#8c564b",
}


def render(model: Any, feature_names: list[str]) -> None:
    """Render the Predict Exercise page.

    Args:
        model: Trained sklearn-compatible classifier loaded at app startup.
        feature_names: Ordered list of feature names matching the model's
            training features.
    """
    st.title("🔮 Predict Exercise")
    st.markdown("""
    Upload your `WristMotion.csv` from Sensor Logger to see which exercises
    were recognized in your workout session.
    """)

    uploaded_file = st.file_uploader(
        "Upload WristMotion.csv from Sensor Logger",
        type=["csv"],
        help="Export WristMotion.csv from the Sensor Logger app after your workout",
    )

    if uploaded_file is None:
        st.info("👆 Upload a WristMotion.csv file to get started.")
        return

    # Run the full prediction pipeline and show results
    with st.spinner(
        "Running prediction pipeline (sliding window → features → model)..."
    ):
        try:
            # Save the uploaded bytes to a temp file so the pipeline can read it
            # as a Path — predict_from_sensor_logger() expects pathlib.Path
            tmp_path = Path("/tmp") / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())

            results_df = predict_from_sensor_logger(
                csv_file=tmp_path,
                model=model,
                feature_names=feature_names,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Check that WristMotion.csv has columns: "
                "timestamp, ax, ay, az, gx, gy, gz"
            )
            return

    n_windows = len(results_df)
    st.success(f"✅ Processed {n_windows} windows ({n_windows * 2} seconds of workout)")
    st.divider()

    # ── Workout summary — one metric per detected exercise class ──────────
    st.subheader("Workout Summary")
    exercise_counts = results_df["predicted_class"].value_counts()

    cols = st.columns(min(len(exercise_counts), 3))
    for i, (exercise, count) in enumerate(exercise_counts.items()):
        with cols[i % 3]:
            label = EXERCISE_LABELS.get(exercise, exercise)
            pct = count / n_windows * 100
            st.metric(label, f"{count} windows", f"{pct:.1f}% of session")

    st.divider()

    # ── Timeline chart — confidence bar per window, colored by exercise ───
    st.subheader("Exercise Timeline")
    st.markdown(
        "Each bar represents a 2-second window. "
        "Color = predicted exercise class. Height = model confidence."
    )

    fig = px.bar(
        results_df,
        x="window_id",
        y="confidence",
        color="predicted_class",
        color_discrete_map=EXERCISE_COLORS,
        labels={
            "window_id": "Window (2 seconds each)",
            "confidence": "Model Confidence",
            "predicted_class": "Exercise",
        },
        title="Predicted Exercise per 2-Second Window",
        height=400,
    )
    fig.update_layout(xaxis_title="Time Window", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Detailed results table — readable labels and formatted confidence ─
    st.subheader("Detailed Results")
    display_df = results_df.copy()
    display_df["predicted_class"] = display_df["predicted_class"].map(
        lambda x: EXERCISE_LABELS.get(x, x)
    )
    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.1%}")
    display_df.columns = ["Window ID", "Predicted Exercise", "Confidence"]
    st.dataframe(display_df, use_container_width=True)
