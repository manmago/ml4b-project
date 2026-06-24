"""Exercise figure rendering: GIF → Lottie → animated SVG, with graceful fallback.

For each class label the page shows a small looping figure. The renderer tries,
in order:

  1. a **direct GIF URL** from :data:`EXERCISE_GIFS` (e.g. a Tenor ``.gif`` link),
  2. a bundled **Lottie** animation at ``app/assets/lottie/<label>.json``,
  3. the lightweight, dependency-free **SVG** dumbbell from :mod:`app.ui.theme`.

So the app always renders something, and upgrading a class to a nicer GIF is just
pasting one URL into :data:`EXERCISE_GIFS` below — no other code change needed.

The bundled exercises point at **local files under ``app/static/``** (served by
Streamlit's static file server — see ``enableStaticServing`` in
``.streamlit/config.toml``), so the app needs no internet at runtime. A value may
be either such a local path or a **direct image URL ending in ``.gif``** (on
tenor.com: open the GIF → *Share* → *Copy GIF link*). Do **not** paste Tenor's
``<div class="tenor-gif-embed">…</div>`` + ``<script>`` embed snippet — Streamlit
strips ``<script>`` tags and never runs the embed JS, so it renders nothing; a
plain ``<img src="…">`` (what we use here) works.

This is presentation only — it renders figures, it has no ML logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.ui import theme

# ---------------------------------------------------------------------------
# Per-exercise GIF URLs. Fill in a DIRECT .gif URL (see the module docstring) to
# show that GIF instead of the built-in animation. Leave a value empty ("") to
# fall back to the Lottie/SVG figure. Keys are raw class labels.
# Example: "bicep_curl": "https://media.tenor.com/XXXXXXXX/bicep-curl.gif",
# ---------------------------------------------------------------------------
EXERCISE_GIFS: dict[str, str] = {
    # Local files under app/static/, served by Streamlit's static file server
    # (enableStaticServing in .streamlit/config.toml). Bundled in the repo so the
    # app works fully offline. Source: Tenor "mediumgif" exports (~130–260 KB each).
    "bicep_curl": "app/static/bicep_curl.gif",
    "tricep_extension": "app/static/tricep_extension.gif",
    "row": "app/static/row.gif",
    "rest": "",  # no GIF → falls back to the gently-floating SVG dumbbell
}

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


def _gif_url(label: str) -> str:
    """Return the configured GIF URL for a label, or ``""`` if none is set."""
    return EXERCISE_GIFS.get(_normalize(label), "").strip()


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict | None:
    """Read and parse a Lottie JSON file (cached). Returns ``None`` on failure."""
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — bad/missing file → fall back to SVG
        return None


def available(label: str) -> bool:
    """Return True if a usable figure (GIF or Lottie) exists for this class label.

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``).

    Returns:
        True when a GIF URL is configured, or the Lottie renderer is installed
        and the JSON file is present.
    """
    if _gif_url(label):
        return True
    return _LOTTIE_AVAILABLE and (LOTTIE_DIR / f"{_normalize(label)}.json").exists()


def figure_html(label: str, max_height: int = 150) -> str:
    """Return the figure as a self-contained HTML string: GIF ``<img>`` or SVG.

    Used wherever a figure must live inside a larger HTML block (e.g. the
    onboarding journey), so it can't be a separate Streamlit element. The Lottie
    branch is intentionally not included here — Lottie is a Streamlit component
    and only :func:`render_exercise` can place it.

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``).
        max_height: Maximum render height in pixels.

    Returns:
        HTML string (render with ``unsafe_allow_html=True``).
    """
    gif = _gif_url(label)
    if gif:
        return (
            f'<div class="ex-gif"><img src="{gif}" alt="{theme.humanize(label)}" '
            f'loading="lazy" style="max-height:{max_height}px;"></div>'
        )
    return (
        '<div style="display:flex;justify-content:center;">'
        f"{theme.exercise_figure(label)}</div>"
    )


def render_exercise(label: str, key: str, height: int = 150) -> None:
    """Render the figure for a class label: GIF if set, else Lottie, else SVG.

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``, ``"rest"``).
        key: Unique Streamlit element key — must differ for every call in a single
            run (e.g. ``"result"``, ``"about-bicep_curl"``). Used by the Lottie
            renderer; the GIF/SVG paths don't need it but accept it for symmetry.
        height: Render height in pixels.
    """
    # 1) Direct GIF URL — rendered as a styled <img> (see .ex-gif in theme.py).
    if _gif_url(label):
        st.markdown(figure_html(label, max_height=height), unsafe_allow_html=True)
        return

    # 2) Bundled Lottie animation, if available.
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

    # 3) Fallback: the lightweight, dependency-free SVG figure, centred in its column.
    st.markdown(figure_html(label, max_height=height), unsafe_allow_html=True)
