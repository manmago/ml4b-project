"""Home page for the ML4B Streamlit app.

Shows a plain-language project overview, the honest headline metrics (loaded
from the committed ``model_metrics.json``), the three recognizable exercises,
and step-by-step instructions for collecting and uploading Apple Watch data via
the Sensor Logger app.
"""

import streamlit as st

from ml4b.utils.metrics import load_model_metrics

# The three exercise classes the model recognizes, with friendly display names.
# rest and uncertain are NOT trained classes — see the "How It Works" section.
EXERCISES = [
    ("💪", "Bicep Curl", "elbow flexion"),
    ("🔺", "Tricep Extension", "overhead elbow extension"),
    ("🚣", "Row", "horizontal pull"),
]


def render() -> None:
    """Render the Home page (no arguments — static content + committed metrics)."""
    st.title("🏋️ Gym Exercise Recognition")
    st.subheader("ML4B SoSe 2026 · FAU Nürnberg, Lehrstuhl für Wirtschaftsinformatik")

    st.markdown(
        "This app recognizes **three gym exercises from wrist-worn sensor data** "
        "(Apple Watch accelerometer + gyroscope) using a Random Forest model "
        "trained on the **Kaggle Gym Workout IMU dataset** — recorded on an Apple "
        "Watch, the same device you upload from (see ADR-016). Record a workout "
        "with the **Sensor Logger** app, upload the file, and the app predicts "
        "which exercise you performed in each 2-second window — with a confidence "
        "score. Pauses between sets are detected automatically as **rest**."
    )

    # Honest headline metrics from the committed leave-one-set-out evaluation.
    metrics = load_model_metrics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "Random Forest")
    col2.metric(
        "Macro F1 (leave-one-set-out)",
        f"{metrics['cv_macro_f1']:.3f}",
        help="Honest cross-set estimate on a single-subject dataset — see ADR-021.",
    )
    col3.metric("Accuracy (leave-one-set-out)", f"{metrics['cv_accuracy']:.1%}")
    col4.metric("Exercises", "3 classes")

    st.divider()

    # Two-column layout: recognizable exercises + how the pipeline works.
    left, right = st.columns(2)
    with left:
        st.markdown("### 🎯 Recognizable Exercises")
        for icon, name, axis in EXERCISES:
            st.markdown(f"- {icon} **{name}** — _{axis}_")
        st.caption(
            "Plus two non-exercise outputs: **rest** (low-motion pauses, detected "
            "by an energy gate) and **uncertain** (the model is not confident "
            "enough to commit to a class)."
        )
    with right:
        st.markdown("### ⚙️ How It Works")
        st.markdown(
            "1. **Resample** the upload to 100 Hz (Apple Watch native rate)\n"
            "2. **Sliding window** — 2 s windows (200 samples, 50% overlap)\n"
            "3. **Activity gate** — low-motion windows are labelled `rest` "
            "(ADR-017)\n"
            "4. **Invariant features** — orientation-robust magnitude, shape and "
            "spectral features (ADR-018)\n"
            "5. **Prediction** — Random Forest classifies each active window; "
            "low-confidence windows become `uncertain` (ADR-020)\n\n"
            "The app uses the **exact same preprocessing code as training** "
            "(`src/ml4b/data/`), so predictions stay consistent."
        )

    st.divider()

    # Sensor Logger collection + upload instructions.
    st.markdown("### 📱 How to Collect Apple Watch Data")
    st.markdown(
        "1. Install **Sensor Logger** (free) from the iOS App Store and make sure "
        "it is installed on your **Apple Watch** too.\n"
        "2. In Sensor Logger, enable the **Wrist Motion** (Device Motion) sensor "
        "— this provides accelerometer + gyroscope.\n"
        "3. Start a recording, perform your gym exercises, then stop the recording.\n"
        "4. Tap the recording → **Share / Export** → **Save to Files** "
        "(choose CSV or ZIP).\n"
        "5. Transfer the export to this computer and upload it on the "
        "**🔮 Predict Exercise** page."
    )

    st.info(
        "📤 **What to upload:** only the single **`WristMotion.csv`** file is "
        "needed — or the **full ZIP** of the Sensor Logger export, and the app "
        "finds `WristMotion.csv` inside automatically."
    )

    st.caption(
        "New to the project? Read `docs/project/project_overview.md` and "
        "`docs/project/apple_watch_data_collection_guide.md` for full details."
    )
