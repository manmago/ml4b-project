"""Classify page for the ML4B Streamlit app ("Daylight" design).

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

from app.ui import journey, lottie, theme, viz
from ml4b.data.apple_watch_loader import (
    load_sensor_logger_csv,
    load_sensor_logger_zip,
    predict_from_sensor_logger,
)
from ml4b.data.session import dominant_label, format_set_summary, summarize_session

# Below this many windows the recording is likely too short to be meaningful.
MIN_WINDOWS_WARNING = 5

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
def _scope_timeline_panel(
    signal: pd.DataFrame | None, bands: list[tuple[str, pd.DataFrame]]
) -> None:
    """Oscilloscope + coloured prediction band(s) on one shared, x-aligned time axis.

    Args:
        signal: Raw normalized signal for the scope (``None`` → scope omitted).
        bands: ``(title, results)`` per timeline band — one for the single-model
            view, two (Model 1 / Model 2) for the comparison view.
    """
    with st.container(border=True):
        st.markdown(
            theme.eyebrow("Signal & prediction · shared time axis"),
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            viz.scope_with_timelines(signal, bands), width="stretch", config=PLOTLY_CFG
        )
        if signal is None or signal.empty:
            st.caption(
                "Raw signal preview unavailable for this file format — showing the "
                "prediction timeline only. Each segment is one 2-second window."
            )
        else:
            st.caption(
                "Acceleration |a| (amber) and rotation |ω| (sky) over the recording, "
                "x-aligned with the prediction band(s) below: drop a vertical line "
                "from a motion burst straight onto the window it was classified as."
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
            "Download predictions (CSV)",
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
            'letter-spacing:0.16em;text-transform:uppercase;color:#8E8A85;">'
            "Recognized exercise</div>"
            f"<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;"
            f'font-size:2.1rem;line-height:1.1;margin:2px 0;color:{color};">'
            f"{theme.humanize(label)}</div>"
            "<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
            'color:#8E8A85;">dominant label across the recording</div></div>',
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
            ("Windows", str(len(results)), "2 s each", theme.FLAME),
            ("Duration", f"{total_s:.0f} s", "recording length", theme.STEEL),
            ("Avg confidence", f"{avg_conf:.0%}", "classified windows", theme.VIOLET),
            ("Rest", f"{rest_pct:.0%}", "energy-gated", theme.CLASS_COLORS["rest"]),
        ]
    )

    _scope_timeline_panel(signal, [("Per-window timeline", results)])

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
                f'font-size:1.15rem;color:#1C1B1A;">{_sets_line(results)}</div>',
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
            "letter-spacing:0.14em;text-transform:uppercase;color:#8E8A85;"
            f'padding-top:10px;">{model_name}</div>'
            f"<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:700;"
            f'font-size:1.5rem;line-height:1.1;color:{color};margin-bottom:6px;">'
            f"{theme.humanize(label)}</div>"
            f"{theme.confidence_ring(conf, color)}",
            unsafe_allow_html=True,
        )


def _change_counts(merged: pd.DataFrame) -> dict[str, int]:
    """Count per-window change categories between Model 1 and Model 2.

    Args:
        merged: Per-window join with ``predicted_class_m1`` / ``predicted_class_m2``.

    Returns:
        Mapping of each :func:`theme.classify_change` category to its window count.
    """
    counts = {key: 0 for key in theme.STATUS_LABELS}
    for m1, m2 in zip(merged["predicted_class_m1"], merged["predicted_class_m2"]):
        counts[theme.classify_change(m1, m2)] += 1
    return counts


def _change_chips(counts: dict[str, int]) -> str:
    """Build the coloured rescued / lost / swapped breakdown chips (HTML)."""
    chips = [("improved", "↑ Rescued"), ("regressed", "↓ Lost"), ("swap", "⇄ Swapped")]
    spans = []
    for key, text in chips:
        color = theme.STATUS_COLORS[key]
        spans.append(
            '<span style="display:inline-flex;align-items:center;gap:6px;'
            f"background:{color}14;color:{color};border:1px solid {color}33;"
            "border-radius:999px;padding:5px 13px;margin:0 8px 6px 0;"
            "font-family:'IBM Plex Mono',monospace;font-size:0.84rem;"
            f'font-weight:600;">{text} · {counts[key]}</span>'
        )
    return '<div style="margin:8px 0 2px;">' + "".join(spans) + "</div>"


def _fmt_conf(value: float) -> str:
    """Format a window confidence as a percent, or '—' when absent (rest/abstain)."""
    return "—" if pd.isna(value) else f"{value:.0%}"


def _status_css(category: str) -> str:
    """Return the per-window status-badge cell CSS for a change category."""
    color = theme.STATUS_COLORS[category]
    # 8-digit hex (color + alpha) gives a soft tinted badge; the text is the hue.
    return f"background-color:{color}1A;color:{color};font-weight:600;"


def _render_comparison(
    r1: pd.DataFrame, r2: pd.DataFrame, signal: pd.DataFrame | None
) -> None:
    """Two-model comparison view (Model 1 baseline vs Model 2 current)."""
    # Join both models per window, keeping BOTH confidences so the table can show
    # Model 1's confidence next to Model 2's.
    merged = r1[["window_id", "predicted_class", "confidence"]].merge(
        r2[["window_id", "predicted_class", "confidence"]],
        on="window_id",
        suffixes=("_m1", "_m2"),
    )
    counts = _change_counts(merged)
    n_changed = sum(counts[k] for k in ("improved", "regressed", "swap", "other"))
    m1_conf = _avg_conf(r1)
    m2_conf = _avg_conf(r2)

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

    # Lead with the rescue story: where Model 1 abstained, Model 2 now commits to a
    # real exercise. Phrase it positively and directionally, with a small per-
    # category breakdown (categories inferred from the two models' own outputs).
    if counts["improved"]:
        headline = (
            f"Model 2 commits to {counts['improved']} window(s) that Model 1 left "
            "uncertain or unknown — turning abstentions into a real exercise."
        )
    elif n_changed:
        headline = (
            f"Model 2 re-labels {n_changed} of {len(merged)} windows vs. Model 1."
        )
    else:
        headline = "Model 2 and Model 1 agree on every window."

    with st.container(border=True):
        st.markdown(
            theme.eyebrow("Effect of our own data · window changes"),
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style=\"font-family:'Space Grotesk',sans-serif;font-weight:600;"
            'font-size:1.12rem;line-height:1.35;color:#1C1B1A;margin:2px 0;">'
            f"{headline}</div>",
            unsafe_allow_html=True,
        )
        cc_left, cc_right = st.columns([3, 1.3], gap="large")
        with cc_left:
            st.markdown(_change_chips(counts), unsafe_allow_html=True)
        with cc_right:
            cc_right.metric(
                "Ø confidence · Model 2",
                f"{m2_conf:.0%}",
                delta=f"{(m2_conf - m1_conf) * 100:+.1f} pp vs. Model 1",
            )
        st.caption(
            "Categories are inferred from the two models' own outputs (no ground "
            "truth): **Rescued** = Model 1 abstained, Model 2 committed; **Lost** = "
            "the reverse; **Swapped** = both committed, but to different exercises."
        )

    # Oscilloscope + both prediction bands share one x-aligned time axis.
    _scope_timeline_panel(
        signal,
        [("Model 1 · Kaggle only", r1), ("Model 2 · + our data", r2)],
    )

    with st.container(border=True):
        st.markdown(theme.eyebrow("Per-window comparison"), unsafe_allow_html=True)
        categories = [
            theme.classify_change(m1, m2)
            for m1, m2 in zip(
                merged["predicted_class_m1"], merged["predicted_class_m2"]
            )
        ]
        # Order: Window | Model 1 | M1 conf | Model 2 | M2 conf | Status — with a
        # coloured Status badge instead of a checkbox (which looked interactive).
        table = pd.DataFrame(
            {
                "Window": merged["window_id"],
                "Model 1": merged["predicted_class_m1"].map(theme.humanize),
                "M1 conf": merged["confidence_m1"].map(_fmt_conf),
                "Model 2": merged["predicted_class_m2"].map(theme.humanize),
                "M2 conf": merged["confidence_m2"].map(_fmt_conf),
                "Status": [theme.STATUS_LABELS[c] for c in categories],
            }
        )
        styled = table.style.apply(
            lambda _col: [_status_css(c) for c in categories], subset=["Status"]
        )
        st.dataframe(styled, width="stretch", height=340, hide_index=True)
        st.download_button(
            "Download both models' predictions (CSV)",
            data=merged.rename(
                columns={
                    "predicted_class_m1": "model1_kaggle_only",
                    "confidence_m1": "model1_confidence",
                    "predicted_class_m2": "model2_with_our_data",
                    "confidence_m2": "model2_confidence",
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
    """Numbered step-by-step onboarding journey shown before any upload."""
    journey.render()
    st.caption(
        "Besides the three exercises, the app also outputs **rest** (low-motion "
        "pauses, energy-gated) and **uncertain** / **unknown** for windows the "
        "model abstains on or has never seen."
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
    spinner_msg = (
        "Running prediction pipeline (both models)…"
        if compare
        else "Running prediction pipeline…"
    )
    try:
        with st.spinner(spinner_msg):
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
        st.error(f"Could not process the file:\n\n{exc}")
        return
    except Exception as exc:  # noqa: BLE001 — surface any unexpected error
        theme.status_bar(file_label=uploaded_file.name, live=False)
        st.error(f"Unexpected error while processing the file: {exc}")
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
            f"Only {len(results)} window(s) detected — record at least "
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
