"""Design tokens, global CSS and HTML/SVG components for the "Night Scope" theme.

The visual direction is *wearable telemetry*: a dark instrument panel, like the
read-out of the watch the data comes from. One signal accent (amber phosphor),
technical monospace numerals, and a live-oscilloscope motif. All colours and
type live here so the look stays consistent and is changed in exactly one place.

Public API:
  inject_theme()                  — inject fonts + global CSS (call once per run)
  status_bar(...)                 — slim device-style header strip
  eyebrow(text)                   — small uppercase section label (returns HTML)
  metric_tiles(items)             — row of telemetry metric cards
  prob_bars(shares)               — horizontal per-class share bars
  confidence_ring(pct, color)     — animated SVG confidence ring (returns HTML)
  exercise_figure(label)          — animated dumbbell icon per exercise
  humanize(label) / class_color() — label helpers shared with the pages
"""

from __future__ import annotations

import math

import streamlit as st

# ---------------------------------------------------------------------------
# Colour tokens — "Night Scope" palette. Mirrored in CSS :root below; kept here
# as Python constants so Plotly (ui.viz) and the HTML components use the exact
# same hex values.
# ---------------------------------------------------------------------------
INK = "#0A0D13"  # app background — deep blue-black (OLED at night)
PANEL = "#121724"  # elevated surfaces / cards
PANEL_2 = "#19202E"  # inputs, nested surfaces
LINE = "#283041"  # hairline borders / oscilloscope graticule
MIST = "#8893A7"  # dimmed secondary text and labels
BONE = "#EAEEF6"  # primary text
AMBER = "#F4A52A"  # THE signal accent (used with restraint)
SKY = "#38BDF8"  # secondary data series (gyroscope)

# One colour per model output. The three trained exercises get distinct, legible
# hues; the non-exercise states read as "not a real class" (calm / muted / warn).
CLASS_COLORS: dict[str, str] = {
    "bicep_curl": "#F4A52A",  # amber
    "tricep_extension": "#A78BFA",  # violet
    "row": "#38BDF8",  # sky
    "rest": "#5B6679",  # slate — calm, low energy
    "uncertain": "#94A3B8",  # muted grey — model abstained
    "unknown": "#FB7185",  # rose — out of distribution
}


def humanize(label: str) -> str:
    """Convert a snake_case class label to a Title Case display string.

    Args:
        label: Raw label such as ``"bicep_curl"``.

    Returns:
        Display string such as ``"Bicep Curl"``.
    """
    return label.replace("_", " ").title()


def class_color(label: str) -> str:
    """Return the theme colour for a class label (falls back to amber).

    Args:
        label: Raw class label (case-insensitive, snake_case or Title Case).

    Returns:
        Hex colour string for that class.
    """
    return CLASS_COLORS.get(label.lower().replace(" ", "_"), AMBER)


# ---------------------------------------------------------------------------
# Global CSS. Plain string (NOT an f-string) so the literal { } of CSS need no
# escaping; the hex values intentionally match the Python constants above.
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0A0D13;  --panel: #121724;  --panel-2: #19202E;
    --line: #283041; --mist: #8893A7;   --bone: #EAEEF6;
    --amber: #F4A52A; --sky: #38BDF8;
    --display: 'Space Grotesk', system-ui, sans-serif;
    --body: 'Inter', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ---- App canvas: a near-black field with a faint amber instrument glow ---- */
.stApp {
    background:
        radial-gradient(1100px 520px at 78% -8%, rgba(244,165,42,0.07), transparent 60%),
        radial-gradient(900px 500px at 0% 0%, rgba(56,189,248,0.05), transparent 55%),
        var(--ink);
    color: var(--bone);
    font-family: var(--body);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1180px; }

/* We navigate with top tabs, not the sidebar — hide it for a full-width HUD. */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ---- Typography ---- */
h1, h2, h3, h4 { font-family: var(--display); letter-spacing: -0.01em; color: var(--bone); }
h1 { font-weight: 700; }
p, li, label, .stMarkdown { color: #C7CEDC; }
.mono, code { font-family: var(--mono); }
a { color: var(--amber); }

/* ---- Top tab bar styled as instrument tabs ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid var(--line); padding-bottom: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0;
    padding: 10px 18px; color: var(--mist);
    font-family: var(--display); font-weight: 600; font-size: 0.95rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(244,165,42,0.14), transparent);
    color: var(--bone);
    box-shadow: inset 0 -2px 0 var(--amber);
}

/* ---- Bordered containers (st.container(border=True)) become HUD panels ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line) !important; border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
    background: var(--panel-2); color: var(--bone);
    border: 1px solid var(--line); border-radius: 10px;
    font-family: var(--display); font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--amber); color: var(--amber);
}

/* ---- File uploader as a scope-style dropzone ---- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 1.5px dashed var(--line); border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--amber); }

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }

/* ---- Alerts: dark-tinted, accent-bordered ---- */
[data-testid="stAlert"] {
    background: var(--panel); border: 1px solid var(--line);
    border-left: 3px solid var(--amber); border-radius: 10px; color: #C7CEDC;
}

/* ---- Slim scrollbars ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 6px; }

/* ===========================  COMPONENTS  =========================== */

/* Device status bar */
.tlm-status {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    font-family: var(--mono); font-size: 0.8rem; color: var(--mist);
    border: 1px solid var(--line); border-radius: 12px;
    padding: 9px 16px; margin-bottom: 14px;
    background: linear-gradient(180deg, var(--panel), rgba(18,23,36,0.6));
}
.tlm-status .rec { color: var(--amber); font-weight: 600; }
.tlm-status .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--amber); margin-right: 7px; vertical-align: middle;
    box-shadow: 0 0 0 0 rgba(244,165,42,0.6); animation: tlm-blink 1.6s infinite;
}
.tlm-status b { color: var(--bone); font-weight: 600; }
.tlm-status .sep { color: var(--line); }

/* Eyebrow / section label */
.tlm-eyebrow {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--amber); margin: 4px 0 8px 0;
    display: flex; align-items: center; gap: 8px;
}
.tlm-eyebrow::before { content: ""; width: 14px; height: 1px; background: var(--amber); }

/* Metric tiles */
.tlm-metric {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line); border-radius: 14px; padding: 16px 16px;
    border-top: 3px solid var(--accent, var(--amber)); height: 100%;
}
.tlm-metric .lab {
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--mist);
}
.tlm-metric .val {
    font-family: var(--display); font-weight: 700; font-size: 2rem;
    line-height: 1.1; margin: 6px 0 2px; color: var(--accent, var(--bone));
    font-variant-numeric: tabular-nums;
}
.tlm-metric .sub { font-size: 0.74rem; color: var(--mist); }

/* Per-class share bars */
.tlm-bar { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
.tlm-bar .name {
    flex: 0 0 140px; font-size: 0.9rem; color: #C7CEDC; display: flex;
    align-items: center; gap: 8px;
}
.tlm-bar .swatch { width: 9px; height: 9px; border-radius: 2px; }
.tlm-bar .track { flex: 1; height: 9px; background: var(--panel-2);
    border-radius: 6px; overflow: hidden; border: 1px solid var(--line); }
.tlm-bar .fill { height: 100%; border-radius: 6px; }
.tlm-bar .pct { flex: 0 0 52px; text-align: right; font-family: var(--mono);
    font-size: 0.85rem; color: var(--bone); }

/* Result block: animated figure + name + confidence ring */
.tlm-result { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.tlm-result .meta { flex: 1; min-width: 160px; }
.tlm-result .name {
    font-family: var(--display); font-weight: 700; font-size: 2.1rem;
    line-height: 1.05; margin: 2px 0; color: var(--bone);
}
.tlm-result .tag {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--mist);
}

/* Confidence ring */
.tlm-ring { width: 118px; height: 118px; flex: 0 0 auto; }
.tlm-ring-track { fill: none; stroke: var(--line); stroke-width: 9; }
.tlm-ring-val { fill: none; stroke-width: 9; stroke-linecap: round; }
.tlm-ring-num { font-family: var(--mono); font-weight: 600; font-size: 22px;
    fill: var(--bone); }
.tlm-ring-lab { font-family: var(--mono); font-size: 8px; fill: var(--mist);
    letter-spacing: 0.12em; }

/* ---- Animated exercise icons: a dumbbell moving along each exercise's path ---- */
.tlm-fig { width: 118px; height: 118px; flex: 0 0 auto; }
.tlm-db rect { fill: var(--accent, #F4A52A); }   /* weight plates take the accent */
.tlm-db .bar { fill: #C7CEDC; }                  /* the bar stays neutral metal */
/* The dumbbell group is animated; fill-box keeps rotation around its own centre. */
.tlm-dbl { transform-box: fill-box; }
.tlm-fig--curl   .tlm-dbl { animation: db-curl   1.6s ease-in-out infinite; }
.tlm-fig--tricep .tlm-dbl { animation: db-tricep 1.7s ease-in-out infinite; }
.tlm-fig--row    .tlm-dbl { animation: db-row    1.5s ease-in-out infinite; }
.tlm-fig--rest   .tlm-dbl { animation: db-rest   3.0s ease-in-out infinite; }
.tlm-glyph { font-family: var(--display); font-weight: 700; font-size: 56px;
    fill: var(--accent, var(--mist)); animation: tlm-pulse 1.8s ease-in-out infinite; }

/* curl = arc up · tricep = straight overhead · row = horizontal pull · rest = float */
@keyframes db-curl   { 0%,100% { transform: translateY(20px) rotate(15deg); }
                       50%      { transform: translateY(-18px) rotate(-15deg); } }
@keyframes db-tricep { 0%,100% { transform: translateY(20px); }
                       50%      { transform: translateY(-28px); } }
@keyframes db-row    { 0%,100% { transform: translateX(17px); }
                       50%      { transform: translateX(-13px); } }
@keyframes db-rest   { 0%,100% { transform: translateY(2px); }
                       50%      { transform: translateY(-4px); } }
@keyframes tlm-pulse { 0%,100% { opacity: .5; transform: scale(.94); }
                       50%      { opacity: 1; transform: scale(1.07); } }
@keyframes tlm-blink { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* Accessibility: honour reduced-motion preferences. */
@media (prefers-reduced-motion: reduce) {
    .tlm-dbl, .tlm-glyph, .tlm-status .dot { animation: none !important; }
}
</style>
"""


def inject_theme() -> None:
    """Inject the global stylesheet. Call once near the top of the app run."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML/SVG component builders. Each returns a string rendered with
# ``st.markdown(..., unsafe_allow_html=True)`` (or renders directly).
# ---------------------------------------------------------------------------
def status_bar(
    file_label: str = "no signal",
    duration_s: float | None = None,
    rate_hz: float | str | None = None,
    windows: int | None = None,
    live: bool = False,
) -> None:
    """Render the slim device-style status strip at the top of a page.

    Args:
        file_label: Name of the loaded recording, or a placeholder.
        duration_s: Recording length in seconds, if known.
        rate_hz: Detected sampling rate, if known.
        windows: Number of analysed windows, if known.
        live: When True, show the pulsing ``REC`` indicator (data loaded).
    """
    dur = f"{duration_s:0.0f}s" if duration_s is not None else "--:--"
    rate = f"{rate_hz} Hz" if rate_hz is not None else "-- Hz"
    win = f"{windows}" if windows is not None else "--"
    rec = (
        '<span class="rec"><span class="dot"></span>REC</span>'
        if live
        else "<span>● IDLE</span>"
    )
    sep = '<span class="sep">|</span>'
    html = (
        f'<div class="tlm-status">{rec}{sep}'
        f"<span>FILE <b>{file_label}</b></span>{sep}"
        f"<span>DUR <b>{dur}</b></span>{sep}"
        f"<span>RATE <b>{rate}</b></span>{sep}"
        f"<span>WINDOWS <b>{win}</b></span></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def eyebrow(text: str) -> str:
    """Return an uppercase mono section label (render with unsafe_allow_html)."""
    return f'<div class="tlm-eyebrow">{text}</div>'


def metric_tiles(items: list[tuple[str, str, str, str]]) -> None:
    """Render a row of telemetry metric tiles.

    Args:
        items: List of ``(label, value, sub, hex_color)`` tuples — one per tile.
    """
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        col.markdown(
            f'<div class="tlm-metric" style="--accent:{color};">'
            f'<div class="lab">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def prob_bars(shares: dict[str, float]) -> str:
    """Build per-class horizontal share bars (sorted high → low).

    Args:
        shares: Mapping of raw class label → share in ``[0, 1]``.

    Returns:
        HTML string for all rows (render with unsafe_allow_html).
    """
    rows = []
    for label, share in sorted(shares.items(), key=lambda kv: kv[1], reverse=True):
        color = class_color(label)
        pct = max(0.0, min(1.0, share)) * 100
        rows.append(
            f'<div class="tlm-bar"><div class="name">'
            f'<span class="swatch" style="background:{color};"></span>{humanize(label)}</div>'
            f'<div class="track"><div class="fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="pct">{pct:.0f}%</div></div>'
        )
    return "".join(rows)


def confidence_ring(pct: float, color: str, label: str = "CONFIDENCE") -> str:
    """Build an animated SVG confidence ring that fills on load.

    Args:
        pct: Confidence in ``[0, 1]``.
        color: Stroke colour (hex) for the filled arc.
        label: Small caption under the percentage.

    Returns:
        SVG HTML string (render with unsafe_allow_html).
    """
    pct = max(0.0, min(1.0, pct))
    r = 52
    circ = 2 * math.pi * r
    offset = circ * (1 - pct)
    return (
        '<svg class="tlm-ring" viewBox="0 0 120 120">'
        f'<circle class="tlm-ring-track" cx="60" cy="60" r="{r}"/>'
        f'<circle class="tlm-ring-val" cx="60" cy="60" r="{r}" stroke="{color}" '
        f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}" '
        'transform="rotate(-90 60 60)">'
        f'<animate attributeName="stroke-dashoffset" from="{circ:.1f}" '
        f'to="{offset:.1f}" dur="0.9s" fill="freeze" calcMode="spline" '
        'keyTimes="0;1" keySplines="0.22 1 0.36 1"/></circle>'
        f'<text class="tlm-ring-num" x="60" y="58" text-anchor="middle">{pct * 100:.0f}%</text>'
        f'<text class="tlm-ring-lab" x="60" y="74" text-anchor="middle">{label}</text>'
        "</svg>"
    )


# A clean dumbbell icon, centred at (0,0): a neutral bar with two stacked weight
# plates per side. The whole icon is wrapped in an animated group so it can move
# along each exercise's characteristic path (see the db-* keyframes in the CSS).
def _dumbbell() -> str:
    """Return the SVG markup for a dumbbell centred at the origin."""
    return (
        '<g class="tlm-db">'
        '<rect class="bar" x="-19" y="-3.5" width="38" height="7" rx="3.5"/>'
        '<rect x="-31" y="-13" width="9" height="26" rx="4"/>'  # left outer plate
        '<rect x="-22" y="-9.5" width="6" height="19" rx="3"/>'  # left inner plate
        '<rect x="22" y="-13" width="9" height="26" rx="4"/>'  # right outer plate
        '<rect x="16" y="-9.5" width="6" height="19" rx="3"/>'  # right inner plate
        "</g>"
    )


def _figure_svg(variant: str, color: str) -> str:
    """Return the animated dumbbell SVG for one exercise.

    Args:
        variant: CSS modifier — ``curl``, ``tricep``, ``row`` or ``rest``.
        color: Accent colour for the weight plates and the soft glow.

    Returns:
        SVG HTML string.
    """
    # Outer <g> positions the icon at the centre (a presentation attribute, so the
    # CSS animation transform on the inner <g class="tlm-dbl"> composes with it
    # instead of overriding it).
    return (
        f'<svg class="tlm-fig tlm-fig--{variant}" viewBox="0 0 120 120" '
        f'style="--accent:{color};">'
        f'<circle cx="60" cy="62" r="30" fill="{color}" opacity="0.10"/>'  # soft glow
        f'<g transform="translate(60,60)"><g class="tlm-dbl">{_dumbbell()}</g></g>'
        "</svg>"
    )


def exercise_figure(label: str) -> str:
    """Return an animated dumbbell icon for a class label.

    Trained exercises get a dumbbell moving along their characteristic path; the
    non-exercise states (``rest`` floats gently, ``uncertain`` / ``unknown`` show
    a pulsing glyph).

    Args:
        label: Raw class label (e.g. ``"bicep_curl"``, ``"rest"``).

    Returns:
        SVG HTML string (render with unsafe_allow_html).
    """
    key = label.lower().replace(" ", "_")
    color = class_color(key)
    variant = {
        "bicep_curl": "curl",
        "tricep_extension": "tricep",
        "row": "row",
        "rest": "rest",
    }.get(key)
    if variant is not None:
        return _figure_svg(variant, color)
    # uncertain / unknown / anything else → a pulsing glyph instead of an icon
    glyph = "?" if key == "uncertain" else "∅"
    return (
        f'<svg class="tlm-fig" viewBox="0 0 120 120" style="--accent:{color};">'
        f'<text class="tlm-glyph" x="60" y="78" text-anchor="middle">{glyph}</text>'
        "</svg>"
    )
