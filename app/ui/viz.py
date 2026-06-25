"""Light-themed Plotly figures for the "Daylight" design.

Every figure shares one clean look (transparent panel on the white card, soft
graticule grid, monospace tick labels, the class colour map from
:mod:`app.ui.theme`). The functions here are pure view helpers: they take
already-computed data and return a styled ``plotly`` figure — no model or
pipeline logic lives here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui.theme import (
    CARD,
    CLASS_COLORS,
    CLASS_PATTERNS,
    FLAME,
    INK,
    LINE,
    MUTED,
    STEEL,
    humanize,
)

# Maximum points drawn in the oscilloscope; longer signals are decimated so the
# trace stays smooth in the browser without changing its shape.
_MAX_SCOPE_POINTS = 3500


def _style(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared clean light layout to a figure.

    Args:
        fig: The figure to style.
        height: Pixel height.

    Returns:
        The same figure, styled in place.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color=MUTED, size=12),
        margin=dict(t=18, b=28, l=44, r=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
        hoverlabel=dict(bgcolor=CARD, font=dict(color=INK, family="IBM Plex Mono")),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def _decimate(t: np.ndarray, *series: np.ndarray):
    """Downsample time + parallel series to at most ``_MAX_SCOPE_POINTS`` points.

    Args:
        t: Time axis array.
        *series: One or more value arrays aligned to ``t``.

    Returns:
        Tuple of ``(t, *series)`` decimated by a constant stride.
    """
    n = len(t)
    if n <= _MAX_SCOPE_POINTS:
        return (t, *series)
    step = int(np.ceil(n / _MAX_SCOPE_POINTS))
    return (t[::step], *[s[::step] for s in series])


def _add_scope_traces(fig: go.Figure, signal: pd.DataFrame, row: int) -> None:
    """Add the accel/gyro magnitude traces to one subplot row.

    Args:
        fig: The subplot figure to add to.
        signal: Raw normalized signal ``[timestamp, ax, ay, az, gx, gy, gz]``.
        row: 1-based subplot row index for the oscilloscope.
    """
    t = signal["timestamp"].to_numpy(dtype=float)
    t = t - t[0]  # seconds since recording start
    accel_mag = np.sqrt(signal["ax"] ** 2 + signal["ay"] ** 2 + signal["az"] ** 2)
    gyro_mag = np.sqrt(signal["gx"] ** 2 + signal["gy"] ** 2 + signal["gz"] ** 2)
    t, accel, gyro = _decimate(t, accel_mag.to_numpy(), gyro_mag.to_numpy())
    fig.add_trace(
        go.Scatter(
            x=t,
            y=gyro,
            name="gyro |ω|",
            legendgroup="gyro",
            mode="lines",
            line=dict(color=STEEL, width=1),
            opacity=0.7,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=accel,
            name="accel |a|",
            legendgroup="accel",
            mode="lines",
            line=dict(color=FLAME, width=1.6),
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="|a| · |ω|", row=row, col=1)


def _add_timeline_band(
    fig: go.Figure, results: pd.DataFrame, row: int, seen: set[str]
) -> None:
    """Add one coloured per-window class band to a subplot row.

    Each label is shown in the legend only the first time it appears across all
    bands (``seen``), so two stacked bands share ONE merged legend instead of
    repeating it. ``uncertain`` gets a dotted fill so the pale abstain band stays
    visible next to the solid ``rest`` grey.

    Args:
        fig: The subplot figure to add to.
        results: Prediction results with ``predicted_class`` and
            ``time_start_seconds`` columns.
        row: 1-based subplot row index for this band.
        seen: Labels already given a legend entry (mutated in place).
    """
    df = results.copy()
    df["_band"] = 1
    for label in df["predicted_class"].unique():
        sub = df[df["predicted_class"] == label]
        marker = dict(color=CLASS_COLORS.get(label, FLAME))
        if label in CLASS_PATTERNS:
            # Dark, low-density dots (not white) so the patterned band keeps its own
            # colour and never washes out to near-white on the light card.
            marker["pattern"] = dict(
                shape=CLASS_PATTERNS[label], fgcolor=INK, size=3, solidity=0.18
            )
        fig.add_trace(
            go.Bar(
                x=sub["time_start_seconds"],
                y=sub["_band"],
                name=humanize(label),
                legendgroup=label,
                showlegend=label not in seen,
                marker=marker,
                hovertemplate=f"{humanize(label)}<br>%{{x:.0f}}s<extra></extra>",
            ),
            row=row,
            col=1,
        )
        seen.add(label)
    fig.update_yaxes(visible=False, range=[0, 1], row=row, col=1)


def scope_with_timelines(
    signal: pd.DataFrame | None,
    bands: list[tuple[str, pd.DataFrame]],
    height: int | None = None,
) -> go.Figure:
    """Oscilloscope stacked above one or more class-timeline bands on a shared x-axis.

    Stacking the raw-signal oscilloscope and the coloured per-window prediction
    band(s) vertically on ONE shared *time* axis lets a viewer drop a vertical line
    from a motion burst straight onto the window it was classified as. A single
    merged legend (deduplicated across bands) replaces the per-band legends.

    This is pure presentation: it reuses already-computed signal and prediction
    data and only arranges it — no pipeline logic here.

    Args:
        signal: Raw normalized signal for the scope, or ``None`` to omit the scope
            row (e.g. when the upload format has no readable raw trace).
        bands: ``(title, results)`` per coloured timeline band — one entry for the
            single-model view, two (Model 1 / Model 2) for the comparison view.
        height: Optional pixel height; scales with the row count by default.

    Returns:
        A styled Plotly figure with shared x-axes.
    """
    has_scope = signal is not None and not signal.empty
    n_bands = len(bands)
    n_rows = (1 if has_scope else 0) + n_bands
    # Scope row is tall; the thin colour bands split the remaining height.
    if has_scope:
        row_heights = [0.52] + [0.48 / n_bands] * n_bands
    else:
        row_heights = [1.0 / n_bands] * n_bands
    titles = (["Sensor signal · oscilloscope"] if has_scope else []) + [
        b[0] for b in bands
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    row = 1
    if has_scope:
        _add_scope_traces(fig, signal, row)
        row += 1
    seen: set[str] = set()
    for _title, results in bands:
        _add_timeline_band(fig, results, row, seen)
        row += 1

    fig = _style(fig, height or ((250 if has_scope else 0) + 116 * n_bands + 44))
    # Bars stack within each band; one shared horizontal legend sits above the top
    # subplot; only the bottom-most subplot carries the shared "time (s)" label.
    fig.update_layout(
        barmode="stack",
        bargap=0.0,
        margin=dict(t=58, b=30, l=46, r=18),
        legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"),
    )
    fig.update_xaxes(title_text="time (s)", row=n_rows, col=1)
    # Keep the subplot titles compact and on-theme (left-aligned, mono, muted).
    fig.update_annotations(
        font=dict(size=11.5, color=MUTED, family="IBM Plex Mono, monospace")
    )
    return fig


def class_donut(shares: dict[str, float], height: int = 300) -> go.Figure:
    """Donut chart of the per-class window share.

    Args:
        shares: Mapping of raw class label → window count (or share).
        height: Pixel height.

    Returns:
        A styled Plotly figure.
    """
    labels = list(shares.keys())
    fig = go.Figure(
        go.Pie(
            labels=[humanize(label) for label in labels],
            values=list(shares.values()),
            hole=0.62,
            marker=dict(
                colors=[CLASS_COLORS.get(label, FLAME) for label in labels],
                line=dict(color=CARD, width=2),
            ),
            textfont=dict(color=INK, family="IBM Plex Mono"),
        )
    )
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=INK)))
    return _style(fig, height)


def f1_compare(
    classes: list[str],
    f1_m1: list[float],
    f1_m2: list[float],
    height: int = 380,
) -> go.Figure:
    """Grouped per-class F1 bars comparing two models.

    Args:
        classes: Raw class labels in display order.
        f1_m1: Model 1 (baseline) F1 per class, aligned to ``classes``.
        f1_m2: Model 2 (current) F1 per class, aligned to ``classes``.
        height: Pixel height.

    Returns:
        A styled grouped bar figure.
    """
    disp = [humanize(c) for c in classes]
    # Same hue per exercise in BOTH models (no colour confusion); the model is told
    # apart by treatment: Model 1 = hatched + faded, Model 2 = solid.
    colors = [CLASS_COLORS.get(c, FLAME) for c in classes]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Model 1 · Kaggle only (hatched)",
            x=disp,
            y=f1_m1,
            marker=dict(
                color=colors,
                opacity=0.4,
                line=dict(color=colors, width=1.4),
                pattern=dict(shape="/", fgcolor="#FFFFFF", size=6),
            ),
            text=[f"{v:.2f}" for v in f1_m1],
            textposition="outside",
            textfont=dict(color=MUTED, size=11),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Model 2 · + our data (solid)",
            x=disp,
            y=f1_m2,
            marker=dict(color=colors),
            text=[f"{v:.2f}" for v in f1_m2],
            textposition="outside",
            textfont=dict(color=INK, size=11),
        )
    )
    # Per-class change (Model 2 − Model 1) above each group → effect of our data.
    for label, a, b in zip(disp, f1_m1, f1_m2):
        delta = b - a
        fig.add_annotation(
            x=label,
            y=max(a, b) + 0.08,
            text=f"{delta:+.2f}",
            showarrow=False,
            font=dict(size=11, color="#1B9E5A" if delta >= 0 else "#E5484D"),
        )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", y=1.16, x=0),
    )
    fig.update_yaxes(range=[0, 1.2])
    return _style(fig, height)


def confusion(cm_norm: np.ndarray, labels: list[str], height: int = 420) -> go.Figure:
    """Row-normalized confusion matrix as a flame-scale heatmap.

    Args:
        cm_norm: Row-normalized confusion matrix (rows = true classes).
        labels: Raw class labels in matrix order.
        height: Pixel height.

    Returns:
        A styled Plotly figure.
    """
    disp = [humanize(c) for c in labels]
    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=disp,
            y=disp,
            colorscale=[[0, "#FFF3EE"], [0.5, "#FFB59E"], [1, FLAME]],
            text=[[f"{v:.2f}" for v in row] for row in cm_norm],
            texttemplate="%{text}",
            textfont=dict(color=INK, family="IBM Plex Mono"),
            colorbar=dict(outlinecolor=LINE, tickfont=dict(color=MUTED)),
            hovertemplate="true %{y}<br>pred %{x}<br>%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="predicted", yaxis_title="true")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height)
