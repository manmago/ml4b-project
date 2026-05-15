"""
ML4B – Gym Exercise Recognition
Streamlit App Entry Point

Starten mit: uv run streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ML4B – Gym Exercise Recognition",
    page_icon="🏋️",
    layout="wide",
)

st.title("🏋️ Gym Exercise Recognition")
st.subheader("ML4B SoSe 2026 | FAU Nürnberg")

st.markdown("""
Dieses Projekt erkennt Gym-Übungen anhand von Sensordaten (z.B. Apple Watch Accelerometer/Gyroscope)
mittels Machine Learning.

### Navigation
Nutze die Seiten in der linken Sidebar:
- **Data Exploration** – Rohdaten & Feature-Visualisierung
- **Model Performance** – Evaluationsmetriken & Vergleich
- **Live Prediction** – Echtzeit-Klassifikation

### Status
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Datenpunkte", "–", help="Wird nach Datenladen aktualisiert")
with col2:
    st.metric("Übungsklassen", "–", help="Anzahl erkennbarer Übungen")
with col3:
    st.metric("Modell Accuracy", "–", help="Nach Training verfügbar")

st.info("🚧 Projekt in Entwicklung – Setup läuft.")
