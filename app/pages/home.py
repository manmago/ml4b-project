"""About-the-project tab for the ML4B Streamlit app ("Night Scope" design).

Plain-language overview, the honest headline metrics (loaded from the committed
``model_metrics.json``), the three recognizable exercises shown as animated
figures, and step-by-step instructions for collecting and uploading Apple Watch
data via the Sensor Logger app. Presentation only — metrics come from
``src/ml4b/``.
"""

import streamlit as st

from app.ui import lottie, theme
from ml4b.utils.metrics import load_model_metrics

# The three trained classes, with the wrist motion each one represents.
EXERCISES = [
    ("bicep_curl", "elbow flexion"),
    ("tricep_extension", "overhead elbow extension"),
    ("row", "horizontal pull"),
]


def render() -> None:
    """Render the About tab (static content + committed metrics)."""
    st.markdown(theme.eyebrow("About the project"), unsafe_allow_html=True)
    st.markdown(
        "This app recognizes **three gym exercises from wrist-worn sensor data** "
        "(Apple Watch accelerometer + gyroscope) using a Random Forest trained on "
        "the **Kaggle Gym Workout IMU dataset** — recorded on an Apple Watch, the "
        "same device you upload from (DECISIONS.md). Record a workout with the "
        "**Sensor Logger** app, upload it, and the app predicts the exercise in "
        "each 2-second window with a confidence score. Pauses between sets are "
        "detected automatically as **rest**."
    )

    # Honest headline metrics from the committed leave-one-set-out evaluation.
    metrics = load_model_metrics()
    theme.metric_tiles(
        [
            ("Model", "Random Forest", "300 trees · seed 42", theme.AMBER),
            (
                "Macro F1",
                f"{metrics['cv_macro_f1']:.3f}",
                "leave-one-set-out",
                theme.SKY,
            ),
            (
                "Accuracy",
                f"{metrics['cv_accuracy']:.1%}",
                "leave-one-set-out",
                "#A78BFA",
            ),
            ("Exercises", "3", "+ rest / uncertain", theme.CLASS_COLORS["rest"]),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(theme.eyebrow("Recognizable exercises"), unsafe_allow_html=True)
            cols = st.columns(3)
            for col, (label, desc) in zip(cols, EXERCISES):
                with col:
                    lottie.render_exercise(label, key=f"about-{label}", height=120)
                    st.markdown(
                        '<div style="text-align:center;">'
                        f"<div style=\"font-family:'Space Grotesk',sans-serif;"
                        f"font-weight:600;color:{theme.class_color(label)};"
                        f'font-size:0.98rem;">{theme.humanize(label)}</div>'
                        f"<div style=\"font-family:'IBM Plex Mono',monospace;"
                        f'font-size:0.68rem;color:#8893A7;">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.caption(
                "Plus two non-exercise outputs: **rest** (energy-gated low-motion "
                "pauses) and **uncertain** (model not confident enough to commit)."
            )
    with right:
        with st.container(border=True):
            st.markdown(theme.eyebrow("How it works"), unsafe_allow_html=True)
            st.markdown(
                "1. **Resample** the upload to 100 Hz (Apple Watch native rate)\n"
                "2. **Sliding window** — 2 s windows (200 samples, 50% overlap)\n"
                "3. **Activity gate** — low-motion windows labelled `rest` "
                "(DECISIONS.md)\n"
                "4. **Invariant features** — orientation-robust magnitude, shape "
                "and spectral features (DECISIONS.md)\n"
                "5. **Prediction** — Random Forest per active window; low-confidence "
                "windows become `uncertain` (DECISIONS.md)\n\n"
                "The app uses the **exact same preprocessing as training** "
                "(`src/ml4b/data/`), so predictions stay consistent."
            )

    with st.container(border=True):
        st.markdown(
            theme.eyebrow("How to collect Apple Watch data"), unsafe_allow_html=True
        )
        st.markdown(
            "1. Install **Sensor Logger** (free, iOS) and make sure it is on your "
            "**Apple Watch** too.\n"
            "2. Enable the **Wrist Motion** (Device Motion) sensor — accelerometer "
            "+ gyroscope.\n"
            "3. Start a recording, perform your gym exercises, then stop it.\n"
            "4. Tap the recording → **Share / Export** → **Save to Files** "
            "(CSV or ZIP).\n"
            "5. Transfer the export here and upload it on the **Classify** tab."
        )
        st.info(
            "📤 **What to upload:** just the single **`WristMotion.csv`** — or the "
            "**full ZIP** export; the app finds `WristMotion.csv` inside."
        )

    st.caption(
        "New to the project? Read `docs/project/project_overview.md` and "
        "`docs/project/apple_watch_data_collection_guide.md` for full details."
    )
