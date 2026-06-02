"""ML4B Gym Exercise Recognition — Streamlit Web Application.

Entry point for the app. Loads the trained Random Forest model once at startup
using ``st.cache_resource`` (loaded once per session, not per interaction).

Three pages:
  🏠 Home             — Project overview, instructions, metrics
  🔮 Predict Exercise — Upload sensor data, get exercise predictions
  📊 Model Performance — Test set evaluation metrics and visualizations

Run with: uv run streamlit run app/streamlit_app.py
Requires: models/saved/best_model.joblib and data/processed/feature_names.txt
Both files are committed to git — no dataset download needed.
"""

import sys
from pathlib import Path

# When launched via `streamlit run app/streamlit_app.py`, Streamlit puts the
# app/ directory on sys.path — NOT the project root — so `import app` fails.
# Prepend the project root so the `app.pages.*` imports resolve no matter how
# the script is launched (streamlit run, AppTest, python -m, …).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib  # noqa: E402 — imported after the sys.path bootstrap above
import streamlit as st  # noqa: E402

from ml4b.utils.config import DATA_PROCESSED, MODELS_DIR  # noqa: E402

st.set_page_config(
    page_title="ML4B — Gym Exercise Recognition",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the trained Random Forest model — cached for the session lifetime.

    Returns:
        The deserialized scikit-learn classifier from best_model.joblib.
    """
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        st.error(
            f"❌ Model not found at `{path}`.\n\n"
            "Run `uv run python scripts/train_model.py` to train the model first, "
            "or clone the full repo which includes the pre-trained model."
        )
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_novelty_detector():
    """Load the open-set novelty detector — cached for the session lifetime.

    The detector flags exercises the model was never trained on as ``unknown``
    (ADR-024). It is optional: if the artifact is missing the app still runs,
    just without out-of-distribution rejection.

    Returns:
        The deserialized ``NoveltyDetector``, or ``None`` if not available.
    """
    path = MODELS_DIR / "novelty_detector.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_feature_names() -> list[str]:
    """Load the ordered feature-name list — cached for the session lifetime.

    Returns:
        List of feature column names in the exact order the model expects.
    """
    path = DATA_PROCESSED / "feature_names.txt"
    if not path.exists():
        st.error(
            f"❌ Feature names not found at `{path}`.\n\n"
            "Run `uv run python scripts/train_model.py` first."
        )
        st.stop()
    return path.read_text().strip().split("\n")


# Load model, features and (optional) novelty detector once at startup.
model = load_model()
feature_names = load_feature_names()
novelty_detector = load_novelty_detector()

# Sidebar branding + navigation.
st.sidebar.title("🏋️ ML4B Exercise Recognition")
st.sidebar.markdown("FAU Nürnberg · SoSe 2026")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Predict Exercise", "📊 Model Performance"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Model: Random Forest · Kaggle Apple-Watch · 3 classes · "
    "leave-one-set-out Macro F1: 0.776"
)

# Route to the selected page's render function.
if page == "🏠 Home":
    from app.pages.home import render

    render()
elif page == "🔮 Predict Exercise":
    from app.pages.prediction import render

    render(model, feature_names, novelty_detector)
elif page == "📊 Model Performance":
    from app.pages.model_performance import render

    render()
