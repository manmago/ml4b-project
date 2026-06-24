"""Numbered onboarding "journey" for the Classify landing ("Daylight" design).

Renders a single vertical, connected timeline that walks a first-time user from
their Apple Watch to a prediction, step by step. Each step has a numbered badge,
an equal-size photo thumbnail (bundled under ``app/static/`` — see ``CREDITS.md``)
and a short instruction:

  1. Wear your Apple Watch     — Apple Watch photo
  2. Start Sensor Logger       — the Sensor Logger app icon + App Store link
  3. Do your sets              — gym photo + the three exercises as looping GIFs
  4. Export & upload           — phone-in-hand photo

The timeline is one HTML block (styled by the ``.jrn*`` rules in
:mod:`app.ui.theme`); the per-exercise figures come from
:func:`app.ui.lottie.figure_html`. Presentation only — no model logic here.
"""

from __future__ import annotations

import streamlit as st

from app.ui import lottie, theme

# The three trained classes shown in step 3, with the wrist motion each one is.
EXERCISES = [
    ("bicep_curl", "Elbow flexion"),
    ("tricep_extension", "Overhead extension"),
    ("row", "Horizontal pull"),
]

# Step thumbnails, bundled locally and served by Streamlit static serving.
IMG_WATCH = "app/static/step_watch.png"  # Apple Watch (square crop)
IMG_SENSOR_LOGGER = "app/static/step_sensorlogger.jpg"  # Sensor Logger app icon
IMG_FIGURE = "app/static/step_figure.svg"  # stick figure performing an exercise
IMG_SHARE = "app/static/step_share.svg"  # iOS-style share / export glyph

# Official, verified App Store listing for the Sensor Logger app (iOS — the
# Apple Watch flow). The badge below links here.
SENSOR_LOGGER_IOS = "https://apps.apple.com/app/sensor-logger/id1531582925"

# Standard Apple logo path, drawn white on the black badge canvas (180×54).
_APPLE_GLYPH = (
    "M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5"
    "-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C63.3 141.2 4 184.8 4 273.5"
    "q0 39.3 14.4 81.2c12.8 36.7 59 126.7 107.2 125.2 25.2-.6 43-17.9 75.8-17.9 31.8"
    " 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.3-65.2-30.7-61.7-90-61.7"
    "-91.9zm-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5"
    " 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3z"
)
_BADGE_FONT = "'Helvetica Neue',Arial,sans-serif"


def _app_store_badge() -> str:
    """Return the 'Download on the App Store' badge as an inline-SVG link tile."""
    svg = (
        '<svg viewBox="0 0 180 56" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Download on the App Store">'
        '<rect x=".75" y=".75" width="178.5" height="54.5" rx="10" fill="#000" stroke="#3a3a3a" stroke-width="1.5"/>'
        f'<path transform="translate(20,15) scale(0.0508)" fill="#fff" d="{_APPLE_GLYPH}"/>'
        f'<text x="52" y="24" fill="#fff" font-family="{_BADGE_FONT}" font-size="8.5" '
        'letter-spacing="0.2">Download on the</text>'
        f'<text x="51.5" y="43" fill="#fff" font-family="{_BADGE_FONT}" font-size="18" '
        'font-weight="600" letter-spacing="-0.4">App Store</text>'
        "</svg>"
    )
    return (
        '<div class="store-badges">'
        f'<a href="{SENSOR_LOGGER_IOS}" target="_blank" rel="noopener">{svg}</a>'
        "</div>"
    )


def _exercise_cards() -> str:
    """Return the HTML grid of the three recognizable-exercise GIF cards."""
    cards = []
    for label, sub in EXERCISES:
        color = theme.class_color(label)
        cards.append(
            '<div class="jrn-ex-card">'
            f"{lottie.figure_html(label, max_height=150)}"
            f'<div class="jrn-ex-name" style="color:{color};">'
            f"{theme.humanize(label)}</div>"
            f'<div class="jrn-ex-sub">{sub.upper()}</div>'
            "</div>"
        )
    return f'<div class="jrn-ex">{"".join(cards)}</div>'


def _step(
    number: int,
    image: str,
    title: str,
    desc: str,
    body_extra: str = "",
    extra: str = "",
) -> str:
    """Build one timeline step: badge + photo thumbnail + text (+ optional extras).

    Args:
        number: Step number shown in the badge.
        image: Static URL of the step's equal-size thumbnail photo.
        title: Step heading.
        desc: Step description (may contain inline ``<b>`` HTML).
        body_extra: Extra HTML inside the text body, under the description
            (e.g. the App Store badge).
        extra: Full-width HTML appended below the main row (e.g. exercise cards).

    Returns:
        HTML string for the step.
    """
    return (
        '<div class="jrn-step">'
        f'<div class="jrn-rail"><div class="jrn-badge">{number}</div></div>'
        '<div class="jrn-card">'
        '<div class="jrn-card-main">'
        f'<img class="jrn-thumb" src="{image}" alt="{title}" loading="lazy">'
        '<div class="jrn-body">'
        f'<div class="jrn-title">{title}</div>'
        f'<div class="jrn-desc">{desc}</div>'
        f"{body_extra}"
        "</div></div>"
        f"{extra}"
        "</div></div>"
    )


def render() -> None:
    """Render the full onboarding journey (used as the Classify empty state)."""
    steps = (
        _step(
            1,
            IMG_WATCH,
            "Wear your Apple Watch",
            "Strap it on and make sure the <b>Sensor Logger</b> app is installed on "
            "the Watch too — it reads the wrist accelerometer and gyroscope.",
        )
        + _step(
            2,
            IMG_SENSOR_LOGGER,
            "Start Sensor Logger",
            "Get the free <b>Sensor Logger</b> app, enable <b>Wrist Motion</b> "
            "(accelerometer + gyroscope) and hit record.",
            body_extra=_app_store_badge(),
        )
        + _step(
            3,
            IMG_FIGURE,
            "Do your sets",
            "Perform any of the three exercises the model recognizes. Pauses "
            "between sets are detected automatically as <b>rest</b>.",
            extra=_exercise_cards(),
        )
        + _step(
            4,
            IMG_SHARE,
            "Export & upload",
            "Tap the recording → <b>Share / Export</b> → <b>Save to Files</b> "
            "(CSV or ZIP), then drop it in the upload box above — the app "
            "predicts the exercise in each 2-second window with a confidence score.",
        )
    )
    st.markdown(
        theme.eyebrow("How it works · from watch to prediction"),
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="jrn">{steps}</div>', unsafe_allow_html=True)
