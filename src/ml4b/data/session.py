"""Session segmentation — group per-window predictions into exercise sets (bouts).

The prediction pipeline (:func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`)
emits one label per 2-second window over a whole continuous recording. For a real
gym session — where the user records once and performs several exercises with rest
pauses between sets — a flat per-window list is hard to read. This module folds
that timeline into **bouts**: maximal runs of consecutive *active* windows, split
by the ``rest`` windows the activity gate detected (ADR-017, ADR-025).

Each bout is the natural unit of a gym "set". It is summarised by a single label
chosen by majority vote over the genuine-exercise windows in the bout, which also
smooths out isolated per-window misclassifications. A bout that contains no
confident exercise window is labelled ``unknown`` (an unrecognised exercise) or
``uncertain`` (active but below the confidence threshold), whichever dominates.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ml4b.data.activity_gate import REST_LABEL
from ml4b.data.apple_watch_loader import NOVEL_LABEL, UNCERTAIN_LABEL
from ml4b.data.canonical import OVERLAP, TARGET_HZ, WINDOW_SIZE

# Labels that are NOT a recognised exercise; excluded from the majority vote.
_NON_EXERCISE = {REST_LABEL, NOVEL_LABEL, UNCERTAIN_LABEL}


def summarize_session(
    results: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
    target_hz: int = TARGET_HZ,
) -> pd.DataFrame:
    """Fold per-window predictions into per-bout (per-set) summaries.

    A bout is a maximal run of consecutive non-``rest`` windows. ``rest`` windows
    act as separators and never appear in a bout.

    Args:
        results: Output of
            :func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`, with
            columns ``[window_id, predicted_class, confidence, time_start_seconds]``.
        window_size: Samples per window (to compute each window's duration).
        overlap: Window overlap fraction (to compute the per-window time step).
        target_hz: Sampling rate the windows were formed at, in Hz.

    Returns:
        One row per bout with columns
        ``[bout_id, label, start_s, end_s, duration_s, n_windows, mean_confidence]``,
        ordered by start time. Empty (with that schema) if there are no active
        windows.
    """
    columns = [
        "bout_id",
        "label",
        "start_s",
        "end_s",
        "duration_s",
        "n_windows",
        "mean_confidence",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    # A window spans window_size samples; consecutive windows advance by the
    # step (window_size * (1 - overlap)). Both expressed in seconds.
    window_seconds = window_size / target_hz
    step_seconds = window_size * (1 - overlap) / target_hz

    # Sort defensively so contiguity reflects real time order.
    ordered = results.sort_values("time_start_seconds").reset_index(drop=True)

    bouts: list[dict] = []
    current: list[pd.Series] = []

    def _flush() -> None:
        """Summarise the accumulated active windows into one bout row."""
        if not current:
            return
        bout = pd.DataFrame(current)
        labels = bout["predicted_class"].tolist()
        # Majority vote over genuine-exercise windows only.
        exercise_labels = [lbl for lbl in labels if lbl not in _NON_EXERCISE]
        if exercise_labels:
            label = Counter(exercise_labels).most_common(1)[0][0]
        else:
            # No recognised exercise: report whichever non-exercise label
            # dominates (unknown vs uncertain).
            label = Counter(labels).most_common(1)[0][0]

        start_s = float(bout["time_start_seconds"].iloc[0])
        # The bout ends one window-length after the last window starts.
        end_s = float(bout["time_start_seconds"].iloc[-1]) + window_seconds
        # Confidence is NaN for unknown/uncertain windows; mean() skips them.
        mean_conf = bout["confidence"].mean()
        bouts.append(
            {
                "bout_id": len(bouts),
                "label": label,
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
                "duration_s": round(end_s - start_s, 2),
                "n_windows": len(bout),
                "mean_confidence": (
                    float(mean_conf) if pd.notna(mean_conf) else float("nan")
                ),
            }
        )
        current.clear()

    # Walk the timeline; rest windows close the running bout, active windows
    # extend it. step_seconds documents the cadence but the contiguity test uses
    # the rest label, which is the gate's ground truth for a pause.
    _ = step_seconds
    for _, row in ordered.iterrows():
        if row["predicted_class"] == REST_LABEL:
            _flush()
        else:
            current.append(row)
    _flush()  # close the final bout if the recording ended mid-exercise

    return pd.DataFrame(bouts, columns=columns)
