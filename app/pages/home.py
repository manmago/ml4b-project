"""Home page for the ML4B Streamlit app.

Shows a plain-language project overview, the headline model metrics, the list
of recognizable exercises, and step-by-step instructions for collecting and
uploading Apple Watch data via the Sensor Logger app.
"""

import streamlit as st

# The seven exercise classes the model recognizes, with friendly display names.
EXERCISES = [
    ("💪", "Bicep Curl"),
    ("🏋️", "Shoulder Press"),
    ("🦵", "Squat"),
    ("💪", "Tricep Extension"),
    ("🙆", "Lateral Raise"),
    ("🤸", "Push Up"),
    ("😴", "Rest / No Exercise"),
]


def render() -> None:
    """Render the Home page (no arguments — static content only)."""
    st.title("🏋️ Gym Exercise Recognition")
    st.subheader("ML4B SoSe 2026 · FAU Nürnberg, Lehrstuhl für Wirtschaftsinformatik")

    st.markdown(
        "This app recognizes **gym exercises from wrist-worn sensor data** "
        "(Apple Watch accelerometer + gyroscope) using a Random Forest model "
        "trained on the **MM-Fit** smartwatch dataset (wrist-worn, matching the "
        "Apple Watch — see ADR-013). Record a workout with the Sensor Logger app, "
        "upload the file, and the app predicts which exercise you performed in "
        "each 2-second window — with a confidence score."
    )

    # Headline metrics from the held-out test set (see Model Performance page).
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "Random Forest")
    col2.metric("Test Macro F1", "0.944", help="Target was ≥ 0.80 — met ✅")
    col3.metric("Test Accuracy", "97.8%")
    col4.metric("Exercises", "7 classes")

    st.divider()

    # Two-column layout: recognizable exercises + how the pipeline works.
    left, right = st.columns(2)
    with left:
        st.markdown("### 🎯 Recognizable Exercises")
        for icon, name in EXERCISES:
            st.markdown(f"- {icon} **{name}**")
    with right:
        st.markdown("### ⚙️ How It Works")
        st.markdown(
            "1. **Sliding window** — split the signal into 2 s windows (50% overlap)\n"
            "2. **Feature extraction** — 47 features per window (statistics + FFT)\n"
            "3. **Prediction** — Random Forest classifies each window\n"
            "4. **Confidence** — the model's probability for the predicted class\n\n"
            "The app uses the **exact same preprocessing code as training** "
            "(`src/ml4b/data/`), so predictions stay consistent."
        )

    st.divider()

    # Sensor Logger collection + upload instructions.
    st.markdown("### 📱 How to Collect Apple Watch Data")
    st.markdown(
        "1. Install **Sensor Logger** (free) from the iOS App Store and make sure "
        "it is installed on your **Apple Watch** too.\n"
        "2. In Sensor Logger, enable the **Wrist Motion** sensor "
        "(accelerometer + gyroscope, recorded at 50 Hz).\n"
        "3. Start a recording, perform your gym exercises, then stop the recording.\n"
        "4. Tap the recording → **Share / Export** → **Save to Files** "
        "(choose CSV or ZIP).\n"
        "5. Transfer the export to this computer and upload it on the "
        "**🔮 Predict Exercise** page."
    )

    st.info(
        "📤 **What to upload:** either the single **`WristMotion.csv`** file, or "
        "the **full ZIP** of the Sensor Logger export — the app finds "
        "`WristMotion.csv` inside the ZIP automatically."
    )

    st.caption(
        "New to the project? Read `docs/project/project_overview.md` and "
        "`docs/project/apple_watch_data_collection_guide.md` for full details."
    )
