"""Lottie animation rendering for exercise figures, with graceful SVG fallback.

Premium per-exercise animations are loaded from ``app/assets/lottie/<label>.json``
and rendered with ``streamlit-lottie``. When a file is missing — or the renderer
is unavailable — the page falls back to the lightweight SVG figure from
:mod:`app.ui.theme`, so the app always works. Dropping a JSON file into the
assets folder is all it takes to upgrade a class to a polished animation; no code
change is needed (see ``app/assets/lottie/README.md`` for the naming convention).

This is presentation only — it renders animations, it has no ML logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.ui import theme

# streamlit-lottie is optional: guard the import so a missing/broken renderer
# degrades to the SVG fallback instead of crashing the whole app.
try:
    from streamlit_lottie import st_lottie

    _LOTTIE_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import failure → SVG fallback
    _LOTTIE_AVAILABLE = False

# Folder holding the bundled animations, named <raw_class_label>.json
# (e.g. bicep_curl.json, tricep_extension.json, row.json, rest.json).
LOTTIE_DIR = Path(__file__).resolve().parent.parent / "assets" / "lottie"


def _normalize(label: str) -> str:
    """Return the file-stem form of a class label (snake_case, spaces → ``_``)."""
    return label.lower().replace(" ", "_")


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict | None:
    """Read and parse a Lottie JSON file (cached). Returns ``None`` on failure."""
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — bad/missing file → fall back to SVG
        return None


def available(label: str) -> bool:
    """Return True if a usable Lottie animation exists for this class label.

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``).

    Returns:
        True when the renderer is installed and the JSON file is present.
    """
    return _LOTTIE_AVAILABLE and (LOTTIE_DIR / f"{_normalize(label)}.json").exists()


def render_exercise(label: str, key: str, height: int = 150) -> None:
    """Render the animation for a class label: Lottie if available, else SVG.

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``, ``"rest"``).
        key: Unique Streamlit element key — must differ for every call in a single
            run (e.g. ``"result"``, ``"about-bicep_curl"``).
        height: Render height in pixels.
    """
    path = LOTTIE_DIR / f"{_normalize(label)}.json"
    if _LOTTIE_AVAILABLE and path.exists():
        data = _load_json(str(path))
        if data is not None:
            st_lottie(
                data,
                height=height,
                loop=True,
                quality="high",
                key=key,
            )
            return
    # Fallback: the lightweight, dependency-free SVG figure, centred in its column.
    st.markdown(
        '<div style="display:flex;justify-content:center;">'
        f"{theme.exercise_figure(label)}</div>",
        unsafe_allow_html=True,
    )
