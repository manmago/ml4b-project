"""Streamlit sub-pages for the ML4B app (Home, Predict Exercise, Model Performance).

Each module here exposes a ``render(...)`` function called by
``app/streamlit_app.py`` based on the sidebar navigation selection. Keeping the
pages as importable render functions (instead of relying on Streamlit's
automatic ``pages/`` folder discovery) lets the entry point pass the loaded
model and feature names into the prediction page.
"""
