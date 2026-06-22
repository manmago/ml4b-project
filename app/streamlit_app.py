"""ML4B Gym Exercise Recognition — Streamlit Web Application (entry point).

Loads the trained models once at startup (``st.cache_resource``) and presents
three top-level tabs in the "Night Scope" wearable-telemetry design:

  Classify           — upload Apple Watch sensor data, get per-window predictions
  Model & Training   — leave-one-set-out evaluation metrics and visualizations
  About the Project  — plain-language overview, metrics and data-collection guide

All presentation lives in :mod:`app.ui`; all predictions come from ``src/ml4b/``
so the training and app pipelines stay identical (CLAUDE.md).

Run with: uv run streamlit run app/streamlit_app.py
Requires: models/saved/best_model.joblib and data/processed/feature_names.txt
(both committed to git — no dataset download needed).
"""

import sys
from pathlib import Path

# When launched via `streamlit run app/streamlit_app.py`, Streamlit puts the
# app/ directory on sys.path — NOT the project root — so `import app` fails.
# Prepend the project root so `app.*` and `ml4b.*` imports resolve no matter how
# the script is launched (streamlit run, AppTest, python -m, …).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib  # noqa: E402 — imported after the sys.path bootstrap above
import streamlit as st  # noqa: E402

from app.pages import home, model_performance, prediction  # noqa: E402
from app.ui.theme import inject_theme  # noqa: E402
from ml4b.utils.config import (  # noqa: E402
    BASELINE_MODEL_FILE,
    BASELINE_NOVELTY_FILE,
    BEST_MODEL_FILE,
    DATA_PROCESSED,
    MODELS_DIR,
    NOVELTY_FILE,
)

st.set_page_config(
    page_title="ML4B — Gym Exercise Recognition",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()


@st.cache_resource
def load_model():
    """Load the current model (Model 2 — Kaggle + Testdaten) for the session.

    Returns:
        The deserialized scikit-learn classifier from best_model.joblib.
    """
    path = MODELS_DIR / BEST_MODEL_FILE
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
    """Load the open-set novelty detector for Model 2 — cached for the session.

    The detector flags exercises the model was never trained on as ``unknown``.
    It is optional: if the artifact is missing the app still runs, just without
    out-of-distribution rejection.

    Returns:
        The deserialized ``NoveltyDetector``, or ``None`` if not available.
    """
    path = MODELS_DIR / NOVELTY_FILE
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_baseline_model():
    """Load Model 1 (baseline — Kaggle only) for the two-model comparison.

    Optional: when present, the Classify tab runs BOTH models so the user can see
    the effect of our own uploaded training data. When absent (older checkout),
    the page falls back to showing the current model only.

    Returns:
        The deserialized baseline classifier, or ``None`` if not available.
    """
    path = MODELS_DIR / BASELINE_MODEL_FILE
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_baseline_novelty_detector():
    """Load the novelty detector for Model 1 (baseline) — cached for the session.

    Returns:
        The deserialized baseline ``NoveltyDetector``, or ``None`` if not available.
    """
    path = MODELS_DIR / BASELINE_NOVELTY_FILE
    return joblib.load(path) if path.exists() else None


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


# Load both models, features and (optional) novelty detectors once at startup.
# Model 2 = current (Kaggle + Testdaten); Model 1 = baseline (Kaggle only).
model = load_model()
feature_names = load_feature_names()
novelty_detector = load_novelty_detector()
baseline_model = load_baseline_model()
baseline_novelty_detector = load_baseline_novelty_detector()

# --- Brand header ----------------------------------------------------------
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;'
    'margin-bottom:2px;">'
    "<span style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;"
    'font-size:1.7rem;color:#EAEEF6;">🏋️ Exercise Recognition</span>'
    "<span style=\"font-family:'IBM Plex Mono',monospace;font-size:0.74rem;"
    'letter-spacing:0.14em;color:#8893A7;text-transform:uppercase;">'
    "Apple Watch · wrist motion · 50 Hz</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
    'color:#5B6679;margin-bottom:14px;">ML4B · SoSe 2026 · FAU Nürnberg — '
    "Lehrstuhl für Wirtschaftsinformatik</div>",
    unsafe_allow_html=True,
)

# --- Top tab navigation ----------------------------------------------------
tab_classify, tab_model, tab_about = st.tabs(
    ["◐  Classify", "▤  Model & Training", "ℹ  About the Project"]
)

with tab_classify:
    prediction.render(
        model,
        feature_names,
        novelty_detector,
        baseline_model=baseline_model,
        baseline_novelty_detector=baseline_novelty_detector,
    )

with tab_model:
    model_performance.render()

with tab_about:
    home.render()
