"""About-the-project tab for the ML4B Streamlit app ("Daylight" design).

Plain-language overview, the shared training/app pipeline, the three recognizable
exercises shown as animated figures, how the prediction works, how to read the
results, practical tips, and a privacy note. The step-by-step data-collection
walkthrough lives on the Classify tab (the onboarding journey), so it is not
repeated here. Presentation only.
"""

import streamlit as st

from app.ui import lottie, theme

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
        "same device you upload from. Record a workout with the "
        "**Sensor Logger** app, upload it, and the app predicts the exercise in "
        "each 2-second window with a confidence score. Pauses between sets are "
        "detected automatically as **rest**."
    )

    # The shared pipeline — identical preprocessing code in training and the app.
    with st.container(border=True):
        st.markdown(
            theme.eyebrow("The pipeline · training = app"), unsafe_allow_html=True
        )
        st.markdown(theme.pipeline_flow(), unsafe_allow_html=True)
        st.caption(
            "The **same preprocessing code** runs in training and in the app, so "
            "predictions stay consistent — no duplication, no divergence."
        )

    # The three recognizable exercises, full width.
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
                    f'font-size:0.68rem;color:#8E8A85;">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
        # A light-hearted look at the non-exercise output: anything that isn't one
        # of the three above (or that the model just isn't confident about) comes
        # back as `uncertain` — here's its dancing mascot.
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            lottie.render_exercise("uncertain", key="about-uncertain", height=150)
            st.markdown(
                '<div style="text-align:center;">'
                "<div style=\"font-family:'Space Grotesk',sans-serif;"
                f"font-weight:600;color:{theme.class_color('uncertain')};"
                'font-size:0.98rem;">Uncertain</div>'
                "<div style=\"font-family:'IBM Plex Mono',monospace;"
                'font-size:0.68rem;color:#8E8A85;">anything else / not sure</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Plus two non-exercise outputs: **rest** (energy-gated low-motion "
            "pauses) and **uncertain** (model not confident enough to commit)."
        )

    # How a prediction is produced, full width.
    with st.container(border=True):
        st.markdown(theme.eyebrow("How it works"), unsafe_allow_html=True)
        st.markdown(
            "1. **Resample** the upload to 100 Hz (Apple Watch native rate)\n"
            "2. **Sliding window** — 2 s windows (200 samples, 50% overlap)\n"
            "3. **Activity gate** — low-motion windows labelled `rest`\n"
            "4. **Invariant features** — orientation-robust magnitude, shape "
            "and spectral features\n"
            "5. **Prediction** — Random Forest per active window; low-confidence "
            "windows become `uncertain`\n\n"
            "The app uses the **exact same preprocessing as training**, so "
            "predictions stay consistent."
        )

    # How to read the per-window output, full width — replaces the (now
    # redundant) data-collection walkthrough that lives on the Classify tab.
    with st.container(border=True):
        st.markdown(theme.eyebrow("Understanding your results"), unsafe_allow_html=True)
        st.markdown(
            "- **Per-window predictions** — the recording is cut into 2-second "
            "windows, and each one gets its own label and **confidence (0–100%)**. "
            "A full set shows up as a run of the same label across neighbouring "
            "windows.\n"
            "- **`rest`** — not a guess. The activity gate measured very little "
            "motion (a pause between sets), so the model is not even asked.\n"
            "- **`uncertain`** — the most likely exercise was still below the "
            "confidence threshold, so the app deliberately does **not** commit.\n"
            "- **Confidence** — how strongly the Random Forest votes for the "
            "chosen class. Clean, full reps tend to score higher than half-reps "
            "or unusual movements."
        )

    # Practical tips for getting good predictions, full width.
    with st.container(border=True):
        st.markdown(theme.eyebrow("Tips for the best results"), unsafe_allow_html=True)
        st.markdown(
            "- **Wear it on your left wrist.** That matches how the training data "
            "was recorded (Apple Watch, left wrist). The features are "
            "orientation-robust, so exact strap position isn't critical — just wear "
            "the watch snug.\n"
            "- **Do full, controlled reps.** A handful of reps per set is enough "
            "for the 2-second windows to lock on.\n"
            "- **Stick to the three exercises above.** Other movements will "
            "usually come back as `uncertain` or get mislabelled — that's "
            "expected for a three-class model.\n"
            "- **Record one exercise per session.** Several sets of the *same* "
            "exercise are fine — the pauses between them are auto-detected as "
            "`rest`. We haven't tested recordings that mix *different* exercises "
            "back-to-back, so record each exercise separately for now.\n"
            "- **What to upload:** just the single **`WristMotion.csv`** — or the "
            "**full ZIP** export; the app finds `WristMotion.csv` inside."
        )

    # Privacy / offline note — true given local Streamlit + bundled assets.
    with st.container(border=True):
        st.markdown(
            theme.eyebrow("Your data stays on your device"), unsafe_allow_html=True
        )
        st.markdown(
            "Everything runs **locally on your computer**. Uploaded recordings are "
            "processed in memory by the app and are **never sent to any server**. "
            "The app needs **no internet connection** at runtime — the model, the "
            "exercise animations and all other assets are bundled with it."
        )

    st.caption(
        "New here? The **Classify** tab walks you through recording on your Apple "
        "Watch, step by step."
    )
