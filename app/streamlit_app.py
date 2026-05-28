"""ML4B Gym Exercise Recognition — Streamlit Web Application.

Main entry point for the Streamlit app. Implements a multi-page application
with three pages:
- Home: Project overview and quickstart
- Prediction: Upload Sensor Logger CSV and get exercise predictions
- Model Performance: View model metrics, confusion matrix, feature importance

Run with: uv run streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ML4B — Gym Exercise Recognition",
    page_icon="🏋️",
    layout="wide",
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔮 Predict Exercise", "📊 Model Performance"],
)

if page == "🏠 Home":
    st.title("🏋️ Gym Exercise Recognition")
    st.subheader("ML4B SoSe 2026 | FAU Nürnberg")
    st.markdown("""
    This app recognizes gym exercises from Apple Watch sensor data
    using a Random Forest model trained on the RecoFit dataset.

    **How to use:**
    1. Record your workout with **Sensor Logger** on your Apple Watch
    2. Export the **WristMotion.csv** file
    3. Go to **Predict Exercise** and upload the file
    4. See which exercises were recognized with confidence scores
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "Random Forest")
    with col2:
        st.metric("Test Macro F1", "0.8006")
    with col3:
        st.metric("Exercises", "6 classes")

elif page == "🔮 Predict Exercise":
    st.title("🔮 Predict Exercise")
    st.info("⚙️ Full prediction pipeline coming in Phase 6.")
    uploaded_file = st.file_uploader(
        "Upload WristMotion.csv from Sensor Logger",
        type=["csv"],
    )
    if uploaded_file:
        st.warning("Prediction pipeline not yet implemented — coming soon!")

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.info("⚙️ Full metrics dashboard coming in Phase 6.")
