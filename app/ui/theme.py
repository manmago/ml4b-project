"""Design tokens, global CSS and HTML/SVG components for the "Daylight" theme.

The visual direction is *clean, light and editorial*: a bright white canvas with
soft cards, generous whitespace and a single warm signal accent that mixes orange
into red (a flame/coral). Technical monospace numerals are kept for the metrics,
but the heavy dark "instrument" look is gone. All colours and type live here so
the look stays consistent and is changed in exactly one place.

Public API:
  inject_theme()                  — inject fonts + global CSS (call once per run)
  status_bar(...)                 — slim device-style header strip
  eyebrow(text)                   — small uppercase section label (returns HTML)
  metric_tiles(items)             — row of metric cards
  prob_bars(shares)               — horizontal per-class share bars
  confidence_ring(pct, color)     — animated SVG confidence ring (returns HTML)
  exercise_figure(label)          — animated dumbbell icon per exercise
  humanize(label) / class_color() — label helpers shared with the pages
"""

from __future__ import annotations

import math

import streamlit as st

# ---------------------------------------------------------------------------
# Colour tokens — "Daylight" palette. Mirrored in the CSS :root below; kept here
# as Python constants so Plotly (ui.viz) and the HTML components use the exact
# same hex values. Each name describes its ROLE, not a literal colour.
# ---------------------------------------------------------------------------
PAPER = "#FBF9F7"  # app canvas — warm off-white
CARD = "#FFFFFF"  # cards / elevated surfaces
SUBTLE = "#F4F1ED"  # inputs, nested surfaces, progress tracks
LINE = "#ECE6E1"  # hairline borders / chart graticule
MUTED = "#8E8A85"  # secondary / muted text and labels
INK = "#1C1B1A"  # primary text — near-black, slightly warm
FLAME = "#FF4D2E"  # THE accent — orange mixed into red (the brand colour)
EMBER = "#FF8A3D"  # orange end of the flame gradient (highlights)
STEEL = "#3E7CB1"  # secondary data series (gyroscope) — calm blue
VIOLET = "#7A5AF0"  # tertiary accent for metric tiles

# One colour per model output. The three trained exercises get distinct, legible
# hues (the flagship curl takes the brand flame); the non-exercise states read as
# "not a real class" (calm grey / warn red). The two greys are kept far apart in
# lightness so `rest` and `uncertain` stay separable on a projector (DECISIONS.md
# §10); `uncertain` additionally gets a dotted pattern in the timeline (below).
CLASS_COLORS: dict[str, str] = {
    "bicep_curl": "#FF4D2E",  # flame — the brand accent
    "tricep_extension": "#7A5AF0",  # violet
    "row": "#1FA0A0",  # teal
    "rest": "#8C857D",  # medium warm grey — calm, low energy
    "uncertain": "#D8D2CB",  # much lighter warm grey — model abstained
    "unknown": "#E5484D",  # red — out of distribution
}

# Per-class fill patterns for the Plotly timeline bars. Only `uncertain` carries a
# pattern: it makes the (intentionally pale) abstain band unmistakable next to the
# solid `rest` grey even on a low-contrast projector. Plotly pattern shape codes.
CLASS_PATTERNS: dict[str, str] = {
    "uncertain": ".",  # dotted
}

# Non-committal model outputs. `rest` is rule-based (the energy gate runs before
# the model and is identical for both models), so a label *change* between Model 1
# and Model 2 is only ever among real classes / `uncertain` / `unknown`.
ABSTAIN_LABELS: frozenset[str] = frozenset({"rest", "uncertain", "unknown"})

# Semantic colours + short labels for how Model 2's label for a window compares to
# Model 1's. Shared by the comparison headline breakdown and the per-window status
# badges so the two always tell the same story (green = gained, red = lost, amber
# = swapped, grey = unchanged/other).
STATUS_COLORS: dict[str, str] = {
    "improved": "#1B9E5A",  # green — Model 2 commits where Model 1 abstained
    "regressed": "#E5484D",  # red — Model 2 abstains where Model 1 committed
    "swap": "#E08A2B",  # amber — real class A → real class B
    "other": "#8E8A85",  # grey — abstain → a different abstain state
    "same": "#C7C2BC",  # light grey — unchanged
}
STATUS_LABELS: dict[str, str] = {
    "improved": "Rescued",
    "regressed": "Lost",
    "swap": "Swapped",
    "other": "Changed",
    "same": "Same",
}


def classify_change(m1_label: str, m2_label: str) -> str:
    """Categorise how Model 2's window label differs from Model 1's.

    The category is derived purely from the two models' own outputs — no ground
    truth is involved. A window is *improved* when Model 2 commits to a real
    exercise where Model 1 abstained (`uncertain`/`unknown`), *regressed* when the
    reverse happens, a *swap* when both commit but to different exercises, and
    otherwise unchanged (`same`) or a non-committal-to-non-committal change
    (`other`).

    Args:
        m1_label: Model 1 (baseline) raw label for the window.
        m2_label: Model 2 (current) raw label for the window.

    Returns:
        One of ``"improved"``, ``"regressed"``, ``"swap"``, ``"other"``, ``"same"``.
    """
    if m1_label == m2_label:
        return "same"
    m1_real = m1_label not in ABSTAIN_LABELS
    m2_real = m2_label not in ABSTAIN_LABELS
    if not m1_real and m2_real:
        return "improved"
    if m1_real and not m2_real:
        return "regressed"
    if m1_real and m2_real:
        return "swap"
    return "other"


def humanize(label: str) -> str:
    """Convert a snake_case class label to a Title Case display string.

    Args:
        label: Raw label such as ``"bicep_curl"``.

    Returns:
        Display string such as ``"Bicep Curl"``.
    """
    return label.replace("_", " ").title()


def class_color(label: str) -> str:
    """Return the theme colour for a class label (falls back to the flame accent).

    Args:
        label: Raw class label (case-insensitive, snake_case or Title Case).

    Returns:
        Hex colour string for that class.
    """
    return CLASS_COLORS.get(label.lower().replace(" ", "_"), FLAME)


# ---------------------------------------------------------------------------
# Global CSS. Plain string (NOT an f-string) so the literal { } of CSS need no
# escaping; the hex values intentionally match the Python constants above.
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #FBF9F7; --card: #FFFFFF; --subtle: #F4F1ED;
    --line: #ECE6E1;  --muted: #8E8A85; --ink: #1C1B1A;
    --flame: #FF4D2E; --ember: #FF8A3D; --steel: #3E7CB1; --violet: #7A5AF0;
    --display: 'Space Grotesk', system-ui, sans-serif;
    --body: 'Inter', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ---- App canvas: a bright off-white field with a faint warm glow at the top ---- */
.stApp {
    background:
        radial-gradient(1100px 520px at 82% -10%, rgba(255,77,46,0.06), transparent 60%),
        radial-gradient(900px 480px at -5% 0%, rgba(255,138,61,0.05), transparent 55%),
        var(--paper);
    color: var(--ink);
    font-family: var(--body);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.4rem 1rem 4rem; max-width: 1180px; }

/* We navigate with top tabs, not the sidebar — hide it for a full-width layout. */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ---- Typography ---- */
h1, h2, h3, h4 { font-family: var(--display); letter-spacing: -0.01em; color: var(--ink); }
h1 { font-weight: 700; }
p, li, label, .stMarkdown { color: #423F3B; }
.mono, code { font-family: var(--mono); }
a { color: var(--flame); }

/* ---- Top tab bar ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid var(--line); padding-bottom: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0;
    padding: 10px 18px; color: var(--muted);
    font-family: var(--display); font-weight: 600; font-size: 0.95rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(255,77,46,0.10), transparent);
    color: var(--ink);
    box-shadow: inset 0 -2px 0 var(--flame);
}

/* ---- Bordered containers (st.container(border=True)) become soft white cards ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line) !important; border-radius: 16px;
    box-shadow: 0 8px 24px rgba(28,27,26,0.06);
}

/* ---- Buttons: clean outline, warm accent on hover ---- */
.stButton > button, .stDownloadButton > button {
    background: var(--card); color: var(--ink);
    border: 1px solid var(--line); border-radius: 10px;
    font-family: var(--display); font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--flame); color: var(--flame);
}

/* ---- File uploader as a clean dropzone ---- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--subtle); border: 1.5px dashed var(--line); border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--flame); }

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }

/* ---- Alerts: light, warm, accent-bordered ---- */
[data-testid="stAlert"] {
    background: #FFF7F4; border: 1px solid var(--line);
    border-left: 3px solid var(--flame); border-radius: 10px; color: #423F3B;
}

/* ---- Slim scrollbars ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #DAD3CC; border-radius: 6px; }

/* ===========================  COMPONENTS  =========================== */

/* ---- Hero banner: a gym backdrop with the project title overlaid ---- */
.hero {
    position: relative; border-radius: 18px; overflow: hidden; height: 240px;
    margin: 2px 0 18px; border: 1px solid var(--line);
    box-shadow: 0 10px 30px rgba(28,27,26,0.12);
}
/* The banner photo is dense across its whole width, so object-fit:cover fills the
   wide strip edge to edge; object-position favours the equipment band (the top
   ceiling / bottom floor are cropped — full height is not needed). */
/* object-fit needs !important: Streamlit ships a more specific rule
   (.st-emotion-cache-* img { object-fit: scale-down }) that would otherwise
   letterbox the photo and leave empty bars on the sides. */
.hero-bg { position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover !important; object-position: center; display: block; }
/* Left-heavy dark scrim that still keeps the whole banner cohesive (not washed
   out on the right) and the white title legible over the photo. */
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg,
        rgba(12,8,5,0.82) 0%, rgba(12,8,5,0.5) 42%, rgba(12,8,5,0.26) 75%, rgba(12,8,5,0.2) 100%);
}
.hero-inner { position: relative; padding: 28px 32px; }
.hero-title {
    font-family: var(--display); font-weight: 700; font-size: 2.2rem; line-height: 1.1;
    color: #fff; letter-spacing: -0.01em;
}
.hero-sub { color: rgba(255,255,255,0.92); font-size: 1rem; margin-top: 8px; max-width: 660px; }
.hero-meta {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: rgba(255,255,255,0.72); margin-top: 12px;
}
@media (max-width: 560px) { .hero-title { font-size: 1.6rem; } .hero-inner { padding: 20px; } }

/* Device status bar */
.tlm-status {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    font-family: var(--mono); font-size: 0.8rem; color: var(--muted);
    border: 1px solid var(--line); border-radius: 12px;
    padding: 9px 16px; margin-bottom: 14px;
    background: var(--card);
}
/* LOADED = a recording is loaded and analysed (steady green, NOT a blinking REC). */
.tlm-status .ready { color: #1B9E5A; font-weight: 600; }
.tlm-status .idle { color: var(--muted); }
.tlm-status .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #1B9E5A; margin-right: 7px; vertical-align: middle;
}
.tlm-status b { color: var(--ink); font-weight: 600; }
.tlm-status .sep { color: var(--line); }

/* Eyebrow / section label */
.tlm-eyebrow {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--flame); margin: 4px 0 8px 0;
    display: flex; align-items: center; gap: 8px;
}
.tlm-eyebrow::before { content: ""; width: 14px; height: 1px; background: var(--flame); }

/* Metric tiles */
.tlm-metric {
    background: var(--card);
    border: 1px solid var(--line); border-radius: 14px; padding: 16px 16px;
    border-top: 3px solid var(--accent, var(--flame)); height: 100%;
    box-shadow: 0 4px 14px rgba(28,27,26,0.04);
}
.tlm-metric .lab {
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
}
.tlm-metric .val {
    font-family: var(--display); font-weight: 700; font-size: 2rem;
    line-height: 1.1; margin: 6px 0 2px; color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.tlm-metric .sub { font-size: 0.74rem; color: var(--muted); }

/* Per-class share bars */
.tlm-bar { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
.tlm-bar .name {
    flex: 0 0 140px; font-size: 0.9rem; color: #423F3B; display: flex;
    align-items: center; gap: 8px;
}
.tlm-bar .swatch { width: 9px; height: 9px; border-radius: 2px; }
.tlm-bar .track { flex: 1; height: 9px; background: var(--subtle);
    border-radius: 6px; overflow: hidden; border: 1px solid var(--line); }
.tlm-bar .fill { height: 100%; border-radius: 6px; }
.tlm-bar .pct { flex: 0 0 52px; text-align: right; font-family: var(--mono);
    font-size: 0.85rem; color: var(--ink); }

/* Result block: animated figure + name + confidence ring */
.tlm-result { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.tlm-result .meta { flex: 1; min-width: 160px; }
.tlm-result .name {
    font-family: var(--display); font-weight: 700; font-size: 2.1rem;
    line-height: 1.05; margin: 2px 0; color: var(--ink);
}
.tlm-result .tag {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted);
}

/* Confidence ring */
.tlm-ring { width: 118px; height: 118px; flex: 0 0 auto; }
.tlm-ring-track { fill: none; stroke: var(--line); stroke-width: 9; }
.tlm-ring-val { fill: none; stroke-width: 9; stroke-linecap: round; }
.tlm-ring-num { font-family: var(--mono); font-weight: 600; font-size: 22px;
    fill: var(--ink); }
.tlm-ring-lab { font-family: var(--mono); font-size: 8px; fill: var(--muted);
    letter-spacing: 0.12em; }

/* Exercise GIF (Tenor / any direct .gif URL) — a clean, rounded card */
.ex-gif { display: flex; justify-content: center; }
.ex-gif img {
    width: 100%; max-width: 230px; max-height: 175px; object-fit: contain;
    border-radius: 16px; background: var(--subtle);
    border: 1px solid var(--line); box-shadow: 0 6px 18px rgba(28,27,26,0.07);
}

/* ---- Two-model comparison cards (Classify tab) ---- */
/* Each card is a flex row: a fixed-width figure column on the left and a centred
   content block (model tag + exercise name + confidence ring) on the right.
   The CONTENT defines the card height; the figure column has a fixed width but
   stretches its HEIGHT to match the content (align-items/align-self:stretch),
   and the GIF/SVG fills that box with object-fit:contain — so it scales to the
   full content height WITHOUT distortion. The column width (~the content height)
   keeps the box near-square for our 220×220 figures. We give the figure a fixed
   width rather than deriving it from the stretched height via aspect-ratio,
   because that flex+aspect-ratio combination collapses the column to zero width.
   The content column is centred (align/justify:center), giving it equal space to
   the figure on its left and the card edge on its right. */
/* Extra, uniform breathing room inside each comparison card. Streamlit 1.57
   draws the card border and its ~15px padding on internal emotion-styled divs
   that carry no stable selector (there is no stVerticalBlockBorderWrapper
   testid in this version), so we cannot reliably override that padding. Instead
   we render the eyebrow + result inside this wrapper, which we fully control,
   and inset them further on every side — the bordered container keeps drawing
   the frame, this just pushes the content (GIF left, ring right) off the edges.
   The flex column restores the eyebrow→content spacing that separate Streamlit
   elements used to provide. */
.cmp-pad { display: flex; flex-direction: column; gap: 14px; padding: 14px; }
.cmp-pad > .tlm-eyebrow { margin: 0; }  /* spacing comes from the flex gap */
.cmp-card { display: flex; align-items: stretch; gap: 20px; }
/* No box-shadow here: this tile sits *inside* the bordered card, which has no
   overflow clipping, so a drop shadow would bleed past the card's bottom edge
   and read as the inner box spilling out. The border + subtle fill alone give a
   clean tile; the outer card already carries the elevation shadow. */
.cmp-fig {
    flex: 0 0 auto; align-self: stretch; width: 168px; position: relative;
    border-radius: 16px; background: var(--subtle); border: 1px solid var(--line);
    overflow: hidden;
}
/* Absolute fill so the figure's natural size never inflates the row — the
   content block alone drives the height, and the figure follows it. The small
   padding (with box-sizing:border-box) insets the figure into the tile so
   object-fit:contain fits it inside the *padded* area — keeping it clear of the
   rounded edges instead of sitting flush against (and seeming to spill past)
   the bottom border. */
.cmp-fig img, .cmp-fig svg {
    position: absolute; inset: 0; width: 100%; height: 100%;
    padding: 10px; box-sizing: border-box; object-fit: contain;
}
.cmp-content {
    flex: 1 1 auto; min-width: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; gap: 6px;
}
.cmp-tag {
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
}
.cmp-name {
    font-family: var(--display); font-weight: 700; font-size: 1.5rem; line-height: 1.1;
}
/* Narrow viewports: stack the figure above the text so nothing overflows. */
@media (max-width: 640px) {
    .cmp-card { flex-direction: column; align-items: center; }
    .cmp-fig { width: 150px; height: 150px; align-self: center; }
}

/* ---- Numbered onboarding journey: a vertical, connected timeline ---- */
.jrn { position: relative; margin: 6px 0 4px; }
.jrn-step { display: flex; gap: 18px; padding-bottom: 18px; }
.jrn-rail { flex: 0 0 44px; display: flex; justify-content: center; position: relative; }
.jrn-badge {
    width: 38px; height: 38px; border-radius: 50%; z-index: 1; margin-top: 2px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(180deg, var(--ember), var(--flame)); color: #fff;
    font-family: var(--display); font-weight: 700; font-size: 1rem;
    box-shadow: 0 4px 12px rgba(255,77,46,0.35);
}
/* The connector line between consecutive badges (none after the last step). */
.jrn-step:not(:last-child) .jrn-rail::after {
    content: ""; position: absolute; top: 44px; bottom: -18px; left: 50%;
    transform: translateX(-50%); width: 2px; background: #F4D6CC;
}
.jrn-card {
    flex: 1; background: var(--card); border: 1px solid var(--line);
    border-radius: 16px; padding: 15px 20px; box-shadow: 0 6px 18px rgba(28,27,26,0.05);
}
/* Main row: an equal-size photo thumbnail next to the text body. The fixed
   thumbnail size + identical left position make all four images line up in a
   neat vertical column down the timeline. */
.jrn-card-main { display: flex; gap: 16px; align-items: center; }
.jrn-thumb {
    flex: 0 0 auto; width: 88px; height: 88px; border-radius: 12px;
    object-fit: cover !important; border: 1px solid var(--line);
    box-shadow: 0 4px 12px rgba(28,27,26,0.08); background: var(--subtle);
}
.jrn-body { flex: 1 1 auto; min-width: 0; }
.jrn-title {
    font-family: var(--display); font-weight: 700; font-size: 1.05rem; color: var(--ink);
}
.jrn-desc { color: #5B5853; font-size: 0.92rem; margin-top: 4px; line-height: 1.55; }
.jrn-desc b { color: var(--ink); }
@media (max-width: 560px) { .jrn-thumb { width: 64px; height: 64px; } }

/* App Store / Google Play download badges — equal-size inline-SVG link tiles. */
.store-badges { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.store-badges a { display: inline-flex; transition: opacity 0.15s ease; }
.store-badges a:hover { opacity: 0.82; }
.store-badges svg { height: 46px; width: auto; display: block; }

/* ---- Pipeline flow (About page): connected step pills ---- */
.pipe { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 8px; margin-top: 2px; }
.pipe-step {
    background: var(--subtle); border: 1px solid var(--line); border-radius: 10px;
    padding: 8px 12px; font-size: 0.82rem; color: var(--ink); line-height: 1.2;
}
.pipe-step small {
    display: block; font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin-top: 2px;
}
.pipe-step.io {
    background: linear-gradient(180deg, rgba(255,138,61,0.16), rgba(255,77,46,0.10));
    border-color: rgba(255,77,46,0.40);
}
.pipe-arrow { color: var(--flame); font-weight: 700; }

/* The three recognizable exercises shown inside the "Do your sets" step. */
.jrn-ex { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 16px; }
.jrn-ex-card {
    text-align: center; border: 1px solid var(--line); border-radius: 14px;
    padding: 12px 10px 14px; background: var(--paper);
}
.jrn-ex-card .ex-gif img { max-height: 150px; }
.jrn-ex-name { font-family: var(--display); font-weight: 600; font-size: 0.98rem; margin-top: 8px; }
.jrn-ex-sub {
    font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-top: 2px;
}
@media (max-width: 640px) { .jrn-ex { grid-template-columns: 1fr; } }

/* ---- Animated exercise icons: a dumbbell moving along each exercise's path ---- */
.tlm-fig { width: 118px; height: 118px; flex: 0 0 auto; }
.tlm-db rect { fill: var(--accent, #FF4D2E); }   /* weight plates take the accent */
.tlm-db .bar { fill: #9AA0A6; }                  /* the bar stays neutral metal */
/* The dumbbell group is animated; fill-box keeps rotation around its own centre. */
.tlm-dbl { transform-box: fill-box; }
.tlm-fig--curl   .tlm-dbl { animation: db-curl   1.6s ease-in-out infinite; }
.tlm-fig--tricep .tlm-dbl { animation: db-tricep 1.7s ease-in-out infinite; }
.tlm-fig--row    .tlm-dbl { animation: db-row    1.5s ease-in-out infinite; }
.tlm-fig--rest   .tlm-dbl { animation: db-rest   3.0s ease-in-out infinite; }
.tlm-glyph { font-family: var(--display); font-weight: 700; font-size: 56px;
    fill: var(--accent, var(--muted)); animation: tlm-pulse 1.8s ease-in-out infinite; }

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

/* Accessibility: honour reduced-motion preferences. */
@media (prefers-reduced-motion: reduce) {
    .tlm-dbl, .tlm-glyph { animation: none !important; }
}
</style>
"""


def inject_theme() -> None:
    """Inject the global stylesheet. Call once near the top of the app run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero_banner() -> None:
    """Render the full-width hero banner: gym backdrop + project title overlay.

    The background photo is bundled under ``app/static/`` (see ``CREDITS.md``) and
    served by Streamlit's static file server, so the banner works offline.
    """
    st.markdown(
        '<div class="hero">'
        '<img class="hero-bg" src="app/static/banner_gym.jpg" alt="">'
        '<div class="hero-overlay"></div>'
        '<div class="hero-inner">'
        '<div class="hero-title">Exercise Recognition</div>'
        '<div class="hero-sub">Recognising gym exercises from Apple Watch wrist '
        "motion · 100 Hz — three exercises, two models compared "
        "(Kaggle only vs. Kaggle + our own data).</div>"
        '<div class="hero-meta">ML4B · SoSe 2026 · FAU Nürnberg — Lehrstuhl für '
        "Wirtschaftsinformatik</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


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
        live: When True, a recording is loaded and analysed → show the steady
            ``LOADED`` indicator. When False, the page is empty → ``IDLE``. (We do
            NOT show ``REC``: the app analyses an uploaded file, it never records.)
    """
    dur = f"{duration_s:0.0f}s" if duration_s is not None else "--:--"
    rate = f"{rate_hz} Hz" if rate_hz is not None else "-- Hz"
    win = f"{windows}" if windows is not None else "--"
    status = (
        '<span class="ready"><span class="dot"></span>LOADED</span>'
        if live
        else '<span class="idle">● IDLE</span>'
    )
    sep = '<span class="sep">|</span>'
    html = (
        f'<div class="tlm-status">{status}{sep}'
        f"<span>FILE <b>{file_label}</b></span>{sep}"
        f"<span>DUR <b>{dur}</b></span>{sep}"
        f"<span>RATE <b>{rate}</b></span>{sep}"
        f"<span>WINDOWS <b>{win}</b></span></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def eyebrow(text: str) -> str:
    """Return an uppercase mono section label (render with unsafe_allow_html)."""
    return f'<div class="tlm-eyebrow">{text}</div>'


def pipeline_flow() -> str:
    """Return the shared training/app pipeline as connected step pills (HTML).

    The first (input) and last (output) steps are accented; the middle steps are
    the preprocessing → model → decision stages. Render with unsafe_allow_html.
    """
    steps = [
        ("Sensor Logger CSV", "input", True),
        ("Resample", "100 Hz", False),
        ("Sliding window", "2 s · 50%", False),
        ("Activity gate", "rest", False),
        ("39 features", "invariant", False),
        ("Random Forest", "model", False),
        ("Confidence + novelty", "uncertain / unknown", False),
        ("Prediction", "output", True),
    ]
    parts = []
    for i, (label, sub, io) in enumerate(steps):
        cls = "pipe-step io" if io else "pipe-step"
        parts.append(f'<div class="{cls}">{label}<small>{sub}</small></div>')
        if i < len(steps) - 1:
            parts.append('<span class="pipe-arrow">&#8594;</span>')
    return f'<div class="pipe">{"".join(parts)}</div>'


def metric_tiles(items: list[tuple[str, str, str, str]]) -> None:
    """Render a row of metric tiles.

    Args:
        items: List of ``(label, value, sub, hex_color)`` tuples — one per tile.
            The colour is used for the tile's top accent border.
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


def confidence_ring(pct: float, color: str, label: str = "Ø CONFIDENCE") -> str:
    """Build an animated SVG confidence ring that fills on load.

    Args:
        pct: Confidence in ``[0, 1]``. Every caller passes the *mean* confidence
            over the recording's windows, hence the default ``Ø`` (average) label.
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
        f'<circle cx="60" cy="62" r="30" fill="{color}" opacity="0.12"/>'  # soft glow
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
