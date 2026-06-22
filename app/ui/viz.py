"""Dark-themed Plotly figures for the "Night Scope" design.

Every figure shares one instrument look (transparent panel, graticule grid,
monospace tick labels, the class colour map from :mod:`app.ui.theme`). The
functions here are pure view helpers: they take already-computed data and return
a styled ``plotly`` figure — no model or pipeline logic lives here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.ui.theme import AMBER, BONE, CLASS_COLORS, LINE, MIST, PANEL_2, SKY, humanize

# Maximum points drawn in the oscilloscope; longer signals are decimated so the
# trace stays smooth in the browser without changing its shape.
_MAX_SCOPE_POINTS = 3500


def _style(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared dark instrument layout to a figure.

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
        font=dict(family="IBM Plex Mono, monospace", color=MIST, size=12),
        margin=dict(t=18, b=28, l=44, r=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=BONE)),
        hoverlabel=dict(bgcolor=PANEL_2, font=dict(color=BONE, family="IBM Plex Mono")),
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


def oscilloscope(signal: pd.DataFrame, height: int = 250) -> go.Figure:
    """Plot accelerometer (amber) and gyroscope (sky) magnitude over time.

    This is the page's signature element: the raw wrist-motion trace, drawn like
    an oscilloscope read-out. The signal is loaded with the same pipeline loader
    used for prediction, so it is exactly what the model sees.

    Args:
        signal: Raw normalized signal with columns
            ``[timestamp, ax, ay, az, gx, gy, gz]``.
        height: Pixel height of the figure.

    Returns:
        A styled Plotly figure.
    """
    t = signal["timestamp"].to_numpy(dtype=float)
    t = t - t[0]  # seconds since recording start
    accel_mag = np.sqrt(signal["ax"] ** 2 + signal["ay"] ** 2 + signal["az"] ** 2)
    gyro_mag = np.sqrt(signal["gx"] ** 2 + signal["gy"] ** 2 + signal["gz"] ** 2)
    t, accel, gyro = _decimate(t, accel_mag.to_numpy(), gyro_mag.to_numpy())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=gyro,
            name="gyro |ω|",
            mode="lines",
            line=dict(color=SKY, width=1),
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=accel,
            name="accel |a|",
            mode="lines",
            line=dict(color=AMBER, width=1.6),
        )
    )
    fig.update_layout(
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis_title="time (s)",
    )
    return _style(fig, height)


def class_timeline(results: pd.DataFrame, height: int = 150) -> go.Figure:
    """Coloured per-window strip: which class each 2-second window got.

    Args:
        results: Prediction results with ``predicted_class`` and
            ``time_start_seconds`` columns.
        height: Pixel height of the strip.

    Returns:
        A styled Plotly figure (a thin stacked band over time).
    """
    df = results.copy()
    df["_band"] = 1
    fig = go.Figure()
    for label in df["predicted_class"].unique():
        sub = df[df["predicted_class"] == label]
        fig.add_trace(
            go.Bar(
                x=sub["time_start_seconds"],
                y=sub["_band"],
                name=humanize(label),
                marker_color=CLASS_COLORS.get(label, AMBER),
                hovertemplate=f"{humanize(label)}<br>%{{x:.0f}}s<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        bargap=0.0,
        legend=dict(orientation="h", y=1.3, x=0),
        xaxis_title="time (s)",
    )
    fig.update_yaxes(visible=False, range=[0, 1])
    return _style(fig, height)


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
                colors=[CLASS_COLORS.get(label, AMBER) for label in labels],
                line=dict(color="#0A0D13", width=2),
            ),
            textfont=dict(color=BONE, family="IBM Plex Mono"),
        )
    )
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=BONE)))
    return _style(fig, height)


def per_class_f1(
    classes: list[str], f1: list[float], target: float, height: int = 360
) -> go.Figure:
    """Per-class F1 bar chart with a target line.

    Args:
        classes: Raw class labels in display order.
        f1: F1 score per class, aligned to ``classes``.
        target: Target F1 to draw as a reference line.
        height: Pixel height.

    Returns:
        A styled Plotly figure.
    """
    fig = go.Figure(
        go.Bar(
            x=[humanize(c) for c in classes],
            y=f1,
            marker_color=[CLASS_COLORS.get(c, AMBER) for c in classes],
            text=[f"{v:.2f}" for v in f1],
            textposition="outside",
            textfont=dict(color=BONE),
        )
    )
    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color=MIST,
        annotation_text=f"target ≥ {target:.2f}",
        annotation_font_color=MIST,
    )
    fig.update_yaxes(range=[0, 1.08])
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
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Model 1 · Kaggle only", x=disp, y=f1_m1, marker_color=MIST)
    )
    fig.add_trace(
        go.Bar(name="Model 2 · + our data", x=disp, y=f1_m2, marker_color=AMBER)
    )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", y=1.14, x=0),
    )
    fig.update_yaxes(range=[0, 1.08])
    return _style(fig, height)


def confusion(cm_norm: np.ndarray, labels: list[str], height: int = 420) -> go.Figure:
    """Row-normalized confusion matrix as an amber-scale heatmap.

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
            colorscale=[[0, "#0E1320"], [0.5, "#7a5a1f"], [1, AMBER]],
            text=[[f"{v:.2f}" for v in row] for row in cm_norm],
            texttemplate="%{text}",
            textfont=dict(color=BONE, family="IBM Plex Mono"),
            colorbar=dict(outlinecolor=LINE, tickfont=dict(color=MIST)),
            hovertemplate="true %{y}<br>pred %{x}<br>%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="predicted", yaxis_title="true")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height)
