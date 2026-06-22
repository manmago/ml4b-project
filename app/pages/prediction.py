"""Classify page for the ML4B Streamlit app ("Night Scope" design).

Accepts a Sensor Logger export (``WristMotion.csv`` directly, or the full ZIP),
runs the shared prediction pipeline from
:func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`, and visualizes
the per-window exercise predictions as a wearable-telemetry read-out: the raw
sensor oscilloscope, a recognized-exercise result with an animated figure and a
confidence ring, a per-window timeline and the label distribution.

When the **baseline model** (Model 1 — Kaggle anchor only) is available next to
the **current model** (Model 2 — Kaggle + our own uploaded Testdaten), the page
runs BOTH on the same recording and shows them side by side, so the effect of
our own training data is visible directly (DECISIONS.md). When the baseline is
missing it falls back to the single-model view.

This module is presentation only — every prediction comes from ``src/ml4b/``.
"""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui import lottie, theme, viz
from ml4b.data.apple_watch_loader import (
    load_sensor_logger_csv,
    load_sensor_logger_zip,
    predict_from_sensor_logger,
)
from ml4b.data.session import dominant_label, format_set_summary, summarize_session

# Below this many windows the recording is likely too short to be meaningful.
MIN_WINDOWS_WARNING = 5

# The three trained exercises shown as an animated teaser on the empty state.
TRAINED_EXERCISES = [
    ("bicep_curl", "elbow flexion"),
    ("tricep_extension", "overhead extension"),
    ("row", "horizontal pull"),
]

PLOTLY_CFG = {"displayModeBar": False}


def _save_upload_to_tempfile(uploaded_file: Any) -> Path:
    """Persist an in-memory Streamlit upload to a temp file with its suffix."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _load_signal(tmp_path: Path) -> pd.DataFrame | None:
    """Load the raw normalized signal for the oscilloscope (best-effort).

    Reuses the same pipeline loaders as prediction, so the trace is exactly what
    the model consumes. Returns ``None`` if the signal can't be read (the rest of
    the page still works without the scope).

    Args:
        tmp_path: Path to the materialized upload.

    Returns:
        DataFrame ``[timestamp, ax, ay, az, gx, gy, gz]`` or ``None``.
    """
    try:
        if tmp_path.suffix.lower() == ".zip":
            return load_sensor_logger_zip(tmp_path)
        return load_sensor_logger_csv(tmp_path)
    except Exception:  # noqa: BLE001 — scope is optional, never block prediction
        return None


def _run_model(
    tmp_path: Path, model: Any, feature_names: list[str], novelty: Any
) -> pd.DataFrame:
    """Run the full prediction pipeline for one model on the uploaded recording.

    Args:
        tmp_path: Path to the materialized upload.
        model: Trained classifier.
        feature_names: Ordered feature names the model expects.
        novelty: Optional novelty detector for this model.

    Returns:
        The per-window results DataFrame.
    """
    return predict_from_sensor_logger(
        tmp_path, model, feature_names, novelty_detector=novelty
    )


def _overall(results: pd.DataFrame) -> str:
    """Return the recording's single dominant label (rest excluded)."""
    return dominant_label(results["predicted_class"]) or "rest"


def _avg_conf(results: pd.DataFrame) -> float:
    """Return mean confidence over classified windows (0.0 if none)."""
    conf = results["confidence"]
    return float(conf.mean()) if conf.notna().any() else 0.0


def _shares(results: pd.DataFrame) -> dict[str, float]:
    """Return the per-class window count as a label → count mapping."""
    return results["predicted_class"].value_counts().to_dict()


def _sets_line(results: pd.DataFrame) -> str:
    """Return the human-readable set summary (e.g. '2 sets of Bicep Curl')."""
    sets = summarize_session(results)
    return format_set_summary(sets) if not sets.empty else "No complete sets detected"


# ---------------------------------------------------------------------------
# Shared panels
# ---------------------------------------------------------------------------
def _signal_panel(signal: pd.DataFrame | None) -> None:
    """Render the oscilloscope panel (the page signature)."""
    with st.container(border=True):
        st.markdown(
            theme.eyebrow("Sensor signal · oscilloscope"), unsafe_allow_html=True
        )
        if signal is None or signal.empty:
            st.caption("Raw signal preview unavailable for this file format.")
            return
        st.plotly_chart(viz.oscilloscope(signal), width="stretch", config=PLOTLY_CFG)
        st.caption(
            "Total acceleration |a| (amber) and rotation rate |ω| (sky) over the "
            "whole recording — exactly the signal the model windows and classifies."
        )


def _timeline_panel(results: pd.DataFrame, title: str = "Per-window timeline") -> None:
    """Render the coloured per-window class timeline."""
    with st.container(border=True):
        st.markdown(theme.eyebrow(title), unsafe_allow_html=True)
        st.plotly_chart(viz.class_timeline(results), width="stretch", config=PLOTLY_CFG)
        st.caption(
            "Each segment is one 2-second window, coloured by its predicted label."
        )


def _detail_expander(results: pd.DataFrame) -> None:
    """Detailed per-window table plus a CSV download (inside an expander)."""
    with st.expander("Detailed per-window results & CSV export"):
        table = results[
            ["window_id", "predicted_class", "confidence", "time_start_seconds"]
        ].copy()
        table["predicted_class"] = table["predicted_class"].map(theme.humanize)
        table = table.rename(
            columns={
                "window_id": "Window",
                "predicted_class": "Exercise",
                "confidence": "Confidence",
                "time_start_seconds": "Start (s)",
            }
        )
        st.dataframe(
            table,
            width="stretch",
            height=320,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        st.download_button(
            "⬇  Download predictions (CSV)",
            data=results[
                ["window_id", "predicted_class", "confidence", "time_start_seconds"]
            ]
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="ml4b_predictions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Result hero — animation + name + confidence ring
# ---------------------------------------------------------------------------
def _result_hero(label: str, confidence: float, key: str) -> None:
    """Render the headline result: exercise animation, name and confidence ring.

    Args:
        label: Raw predicted class label for the recording.
        confidence: Confidence in ``[0, 1]`` shown in the ring.
        key: Unique element key for the animation in this run.
    """
    color = theme.class_color(label)
    c_anim, c_meta, c_ring = st.columns([1.1, 2.2, 1.1], gap="medium")
    with c_anim:
        lottie.render_exercise(label, key=key, height=150)
    with c_meta:
        st.markdown(
            '<div style="padding-top:22px;">'
            "<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
            'letter-spacing:0.16em;text-transform:uppercase;color:#8893A7;">'
            "Recognized exercise</div>"
            f"<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;"
            f'font-size:2.1rem;line-height:1.1;margin:2px 0;color:{color};">'
            f"{theme.humanize(label)}</div>"
            "<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
            'color:#8893A7;">dominant label across the recording</div></div>',
            unsafe_allow_html=True,
        )
    with c_ring:
        st.markdown(theme.confidence_ring(confidence, color), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Single-model view
# ---------------------------------------------------------------------------
def _render_single(results: pd.DataFrame, signal: pd.DataFrame | None) -> None:
    """Full single-model view (used when no baseline model is available)."""
    overall = _overall(results)
    avg_conf = _avg_conf(results)
    total_s = float(results["time_start_seconds"].max()) + 2.0
    rest_pct = float((results["predicted_class"] == "rest").mean())

    with st.container(border=True):
        st.markdown(theme.eyebrow("Recognition"), unsafe_allow_html=True)
        _result_hero(overall, avg_conf, key="result-current")
    theme.metric_tiles(
        [
            ("Windows", str(len(results)), "2 s each", theme.AMBER),
            ("Duration", f"{total_s:.0f} s", "recording length", theme.SKY),
            ("Avg confidence", f"{avg_conf:.0%}", "classified windows", "#A78BFA"),
            ("Rest", f"{rest_pct:.0%}", "energy-gated", theme.CLASS_COLORS["rest"]),
        ]
    )

    _signal_panel(signal)
    _timeline_panel(results)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(theme.eyebrow("Label distribution"), unsafe_allow_html=True)
            st.markdown(theme.prob_bars(_shares(results)), unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown(theme.eyebrow("Detected sets"), unsafe_allow_html=True)
            st.markdown(
                f"<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:600;"
                f'font-size:1.15rem;color:#EAEEF6;">{_sets_line(results)}</div>',
                unsafe_allow_html=True,
            )

    _detail_expander(results)


# ---------------------------------------------------------------------------
# Two-model comparison view
# ---------------------------------------------------------------------------
def _compact_result(model_name: str, results: pd.DataFrame, key: str) -> None:
    """Compact result card (animation + name + ring) for one model in a column.

    Args:
        model_name: Short tag shown above the exercise name (e.g. ``"baseline"``).
        results: That model's per-window results.
        key: Unique element key for the animation in this run.
    """
    label = _overall(results)
    conf = _avg_conf(results)
    color = theme.class_color(label)
    c_anim, c_meta = st.columns([1, 1.5], gap="small")
    with c_anim:
        lottie.render_exercise(label, key=key, height=120)
    with c_meta:
        st.markdown(
            "<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.7rem;"
            "letter-spacing:0.14em;text-transform:uppercase;color:#8893A7;"
            f'padding-top:10px;">{model_name}</div>'
            f"<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;"
            f'font-size:1.5rem;line-height:1.1;color:{color};margin-bottom:6px;">'
            f"{theme.humanize(label)}</div>"
            f"{theme.confidence_ring(conf, color)}",
            unsafe_allow_html=True,
        )


def _render_comparison(
    r1: pd.DataFrame, r2: pd.DataFrame, signal: pd.DataFrame | None
) -> None:
    """Two-model comparison view (Model 1 baseline vs Model 2 current)."""
    merged = r1[["window_id", "predicted_class"]].merge(
        r2[["window_id", "predicted_class", "confidence"]],
        on="window_id",
        suffixes=("_m1", "_m2"),
    )
    agreement = float(
        (merged["predicted_class_m1"] == merged["predicted_class_m2"]).mean()
    )
    n_changed = int(
        (merged["predicted_class_m1"] != merged["predicted_class_m2"]).sum()
    )
    total_s = float(r2["time_start_seconds"].max()) + 2.0

    st.info(
        "**Model 1** was trained on the Kaggle anchor only; **Model 2** adds our "
        "own uploaded recordings (Testdaten) — same pipeline and augmentation. "
        "Differences below are purely the effect of our own data."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(theme.eyebrow("Model 1 · Kaggle only"), unsafe_allow_html=True)
            _compact_result("Model 1 · baseline", r1, key="cmp-m1")
    with c2:
        with st.container(border=True):
            st.markdown(theme.eyebrow("Model 2 · + our data"), unsafe_allow_html=True)
            _compact_result("Model 2 · + our data", r2, key="cmp-m2")

    theme.metric_tiles(
        [
            ("Windows", str(len(r2)), "2 s each", theme.AMBER),
            ("Duration", f"{total_s:.0f} s", "recording length", theme.SKY),
            (
                "Agreement",
                f"{agreement:.0%}",
                f"{n_changed} windows changed",
                "#A78BFA",
            ),
            ("Avg conf · M2", f"{_avg_conf(r2):.0%}", "current model", "#A78BFA"),
        ]
    )

    _signal_panel(signal)

    _timeline_panel(r1, title="Timeline · Model 1 (Kaggle only)")
    _timeline_panel(r2, title="Timeline · Model 2 (+ our data)")

    with st.container(border=True):
        st.markdown(theme.eyebrow("Per-window comparison"), unsafe_allow_html=True)
        table = merged.copy()
        table["changed"] = table["predicted_class_m1"] != table["predicted_class_m2"]
        table["predicted_class_m1"] = table["predicted_class_m1"].map(theme.humanize)
        table["predicted_class_m2"] = table["predicted_class_m2"].map(theme.humanize)
        table = table.rename(
            columns={
                "window_id": "Window",
                "predicted_class_m1": "Model 1",
                "predicted_class_m2": "Model 2",
                "confidence": "M2 confidence",
                "changed": "Changed?",
            }
        )
        st.dataframe(
            table,
            width="stretch",
            height=340,
            column_config={
                "M2 confidence": st.column_config.ProgressColumn(
                    "M2 confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        st.download_button(
            "⬇  Download both models' predictions (CSV)",
            data=merged.rename(
                columns={
                    "predicted_class_m1": "model1_kaggle_only",
                    "predicted_class_m2": "model2_with_our_data",
                    "confidence": "model2_confidence",
                }
            )
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="ml4b_predictions_comparison.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
def _render_empty_state() -> None:
    """Animated teaser + brief instructions shown before any upload."""
    with st.container(border=True):
        st.markdown(theme.eyebrow("Recognizable exercises"), unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (label, desc) in zip(cols, TRAINED_EXERCISES):
            with col:
                lottie.render_exercise(label, key=f"empty-{label}", height=140)
                st.markdown(
                    '<div style="text-align:center;">'
                    f"<div style=\"font-family:'Space Grotesk',sans-serif;"
                    f"font-weight:600;color:{theme.class_color(label)};"
                    f'font-size:1.05rem;">{theme.humanize(label)}</div>'
                    f"<div style=\"font-family:'IBM Plex Mono',monospace;"
                    f'font-size:0.72rem;color:#8893A7;">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
        st.caption(
            "Plus **rest** (low-motion pauses, energy-gated) and **uncertain** / "
            "**unknown** for windows the model abstains on or has never seen."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(
    model: Any,
    feature_names: list[str],
    novelty_detector: Any = None,
    baseline_model: Any = None,
    baseline_novelty_detector: Any = None,
) -> None:
    """Render the Classify tab.

    Args:
        model: Current model (Model 2 — Kaggle + Testdaten).
        feature_names: Ordered feature names both models expect.
        novelty_detector: Optional novelty detector for the current model.
        baseline_model: Optional baseline model (Model 1 — Kaggle only). When
            present, the page shows a two-model comparison.
        baseline_novelty_detector: Optional novelty detector for the baseline.
    """
    compare = baseline_model is not None

    uploaded_file = st.file_uploader(
        "Upload a Sensor Logger recording (Apple Watch)",
        type=["csv", "zip"],
        help=(
            "Upload either WristMotion.csv (the wrist-motion file from Sensor "
            "Logger) or the full ZIP export — the app finds WristMotion.csv "
            "inside. Export: open Sensor Logger → tap your recording → Share → "
            "Save to Files."
        ),
    )

    if uploaded_file is None:
        theme.status_bar(live=False)
        _render_empty_state()
        return

    tmp_path = _save_upload_to_tempfile(uploaded_file)
    try:
        with st.spinner("Running prediction pipeline…"):
            signal = _load_signal(tmp_path)
            results = _run_model(tmp_path, model, feature_names, novelty_detector)
            baseline_results = (
                _run_model(
                    tmp_path, baseline_model, feature_names, baseline_novelty_detector
                )
                if compare
                else None
            )
    except (ValueError, FileNotFoundError) as exc:
        theme.status_bar(file_label=uploaded_file.name, live=False)
        st.error(f"❌ Could not process the file:\n\n{exc}")
        return
    except Exception as exc:  # noqa: BLE001 — surface any unexpected error
        theme.status_bar(file_label=uploaded_file.name, live=False)
        st.error(f"❌ Unexpected error while processing the file: {exc}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    if results.empty:
        theme.status_bar(file_label=uploaded_file.name, live=False)
        st.warning("No windows were produced — the recording may be too short.")
        return

    total_s = float(results["time_start_seconds"].max()) + 2.0
    theme.status_bar(
        file_label=uploaded_file.name,
        duration_s=total_s,
        rate_hz=results.attrs.get("detected_hz", "n/a"),
        windows=len(results),
        live=True,
    )
    if len(results) < MIN_WINDOWS_WARNING:
        st.warning(
            f"⚠️ Only {len(results)} window(s) detected — record at least "
            "~15–30 seconds for reliable results."
        )

    if compare and baseline_results is not None:
        _render_comparison(baseline_results, results, signal)
    else:
        _render_single(results, signal)

    st.caption(
        "The recording is resampled to 100 Hz before windowing. **Rest** windows "
        "are detected by an energy gate (not the model) and have no confidence; "
        "**uncertain** windows are active but below the confidence threshold."
    )
