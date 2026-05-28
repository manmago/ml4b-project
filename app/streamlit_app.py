"""ML4B Gym Exercise Recognition — Streamlit Web Application.

Main entry point. Loads the trained model once at startup and
passes it to all pages. Three pages:
- Home: Project overview and instructions
- Predict Exercise: Upload WristMotion.csv → get predictions
- Model Performance: Pre-computed test metrics dashboard

Run with: uv run streamlit run app/streamlit_app.py
"""

import joblib
import streamlit as st

from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR

st.set_page_config(
    page_title="ML4B — Gym Exercise Recognition",
    page_icon="🏋️",
    layout="wide",
)


# ── Model and feature names loaded once per session ────────────────────────
# st.cache_resource prevents reloading on every user interaction
@st.cache_resource
def load_model():
    """Load trained Random Forest model from disk.

    Returns:
        Trained sklearn-compatible classifier.
    """
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        st.error(
            f"Model not found at {model_path}. Run notebooks/04_modeling.ipynb first."
        )
        st.stop()
    return joblib.load(model_path)


@st.cache_resource
def load_feature_names() -> list[str]:
    """Load ordered feature names from feature_names.txt.

    Returns:
        List of feature name strings matching the model's training columns.
    """
    feature_file = DATA_PROCESSED / "feature_names.txt"
    if not feature_file.exists():
        st.error(
            f"Feature names not found at {feature_file}. "
            "Run notebooks/03_data_preparation.ipynb first."
        )
        st.stop()
    return feature_file.read_text().strip().split("\n")


model = load_model()
feature_names = load_feature_names()

# ── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔮 Predict Exercise", "📊 Model Performance"],
)

# ── Page routing ───────────────────────────────────────────────────────────
# Streamlit adds the script directory (app/) to sys.path at startup,
# so pages/ is importable as a top-level package from here.
if page == "🏠 Home":
    from pages.home import render

    render()
elif page == "🔮 Predict Exercise":
    from pages.prediction import render

    render(model, feature_names)
elif page == "📊 Model Performance":
    from pages.model_performance import render

    render()
