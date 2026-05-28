"""Home page for the ML4B Gym Exercise Recognition Streamlit app.

Displays project overview, instructions for use, and key model metrics.
"""

import streamlit as st


def render() -> None:
    """Render the Home page."""
    st.title("🏋️ Gym Exercise Recognition")
    st.subheader("ML4B SoSe 2026 | FAU Nürnberg | Lehrstuhl für Wirtschaftsinformatik")

    st.markdown("""
    This app automatically recognizes gym exercises from Apple Watch sensor data
    using a **Random Forest** model trained on the
    [RecoFit dataset](https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors)
    (Microsoft Research, 200+ participants).
    """)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "Random Forest")
    with col2:
        st.metric("Test Macro F1", "0.80")
    with col3:
        st.metric("Test Accuracy", "96.3%")
    with col4:
        st.metric("Exercise Classes", "6")

    st.divider()

    # Step-by-step usage instructions
    st.subheader("How to Use")
    st.markdown("""
    1. 📱 **Record** your gym workout using the **Sensor Logger** app on your Apple Watch
    2. 📤 **Export** the `WristMotion.csv` file from Sensor Logger
    3. 🔮 Go to **Predict Exercise** and upload the CSV file
    4. 📊 See which exercise was recognized per 2-second window with confidence scores
    """)

    st.divider()

    # One card per recognized exercise with per-class F1 for transparency
    st.subheader("Recognizable Exercises")
    exercises = {
        "💪 Bicep Curl": "F1 = 0.89",
        "🏋️ Shoulder Press": "F1 = 0.81",
        "🦵 Squat": "F1 = 0.77",
        "💪 Tricep Extension": "F1 = 0.80",
        "🤸 Lateral Raise": "F1 = 0.55",
        "😴 Rest": "F1 = 0.98",
    }
    cols = st.columns(3)
    for i, (exercise, f1) in enumerate(exercises.items()):
        with cols[i % 3]:
            st.info(f"**{exercise}**\n\n{f1}")
