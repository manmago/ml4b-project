"""Session segmentation — group per-window predictions into exercise sets (bouts).

The prediction pipeline (:func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`)
emits one label per 2-second window over a whole continuous recording. For a real
gym session — where the user records once and performs several exercises with rest
pauses between sets — a flat per-window list is hard to read. This module folds
that timeline into **bouts**: maximal runs of consecutive *active* windows, split
by the ``rest`` windows the activity gate detected (DECISIONS.md).

Each bout is the natural unit of a gym "set". It is summarised by a single label
chosen by **plurality vote over all of its windows** (``rest`` excluded): the
single most frequent label wins. This deliberately lets ``uncertain`` or
``unknown`` win a bout when they are the most common output — we must NOT report a
confident-looking exercise for a set the model was, window-for-window, mostly
unsure about. Only when a genuine exercise is the actual plurality is that
exercise reported. The same rule smooths isolated per-window misclassifications.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd

from ml4b.data.activity_gate import REST_LABEL
from ml4b.data.apple_watch_loader import NOVEL_LABEL, UNCERTAIN_LABEL
from ml4b.data.canonical import OVERLAP, TARGET_HZ, WINDOW_SIZE

# Post-hoc outputs that are NOT a trained exercise class. Used only for the
# conservative tie-break below — they are still counted in the plurality vote.
_NON_EXERCISE = {REST_LABEL, NOVEL_LABEL, UNCERTAIN_LABEL}


def dominant_label(
    labels: Iterable[str],
    ignore: frozenset[str] = frozenset({REST_LABEL}),
) -> str | None:
    """Return the single most frequent label (the plurality), ``rest`` excluded.

    This is the shared "overall result" rule used both for the per-bout label and
    for the app's whole-recording summary, so the two never disagree. ``rest`` is
    ignored by default because it is a pause, not a result; **``uncertain`` and
    ``unknown`` ARE counted** — if the model was unsure (or saw an untrained
    movement) for the plurality of windows, that is the honest answer and we do
    not promote a minority exercise over it.

    Ties are broken conservatively: if a non-exercise output (``uncertain`` /
    ``unknown``) ties with an exercise, the non-exercise label wins so we never
    over-report an exercise the model did not clearly predict. Remaining ties fall
    back to alphabetical order for determinism.

    Args:
        labels: Per-window class labels, in any order.
        ignore: Labels excluded from the vote entirely. Defaults to ``{rest}``.

    Returns:
        The winning label, or ``None`` if every label was ignored (e.g. a
        recording that is entirely ``rest``).
    """
    counts = Counter(lbl for lbl in labels if lbl not in ignore)
    if not counts:
        return None
    top_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return tied[0]
    # Conservative tie-break: a non-exercise output beats an exercise so an
    # uncertain/unknown plurality is never overruled by an equally-frequent class.
    non_exercise_tied = sorted(lbl for lbl in tied if lbl in _NON_EXERCISE)
    if non_exercise_tied:
        return non_exercise_tied[0]
    return sorted(tied)[0]


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
        # Plurality vote over ALL of the bout's windows (they are all non-rest by
        # construction, so nothing extra is ignored here). Counting uncertain /
        # unknown alongside the exercises means a set the model was mostly unsure
        # about is reported as "uncertain", never as a minority exercise. Falls
        # back to a deterministic label if somehow empty.
        label = dominant_label(labels, ignore=frozenset()) or UNCERTAIN_LABEL

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


def count_sets(
    sets: pd.DataFrame,
    include_non_exercise: bool = False,
) -> list[tuple[str, int]]:
    """Count how many sets were detected per exercise label.

    Folds the per-bout table from :func:`summarize_session` into a per-label set
    count — the answer to "how many sets of each exercise did I do?". A recording
    of *bicep curl → rest → bicep curl* yields ``[("bicep_curl", 2)]`` (two sets,
    one exercise), which is exactly the "2 sets of bicep curl" headline the app
    shows.

    Args:
        sets: The per-bout DataFrame returned by :func:`summarize_session`
            (one row per set, carrying a ``label`` column).
        include_non_exercise: If ``False`` (default), drop ``rest`` / ``uncertain``
            / ``unknown`` sets so only genuine exercises are counted. If ``True``,
            keep them (e.g. to also report "1 uncertain set").

    Returns:
        A list of ``(label, count)`` pairs ordered by each label's first
        appearance in time. Empty if there are no qualifying sets.
    """
    if sets.empty:
        return []
    # dict preserves insertion order (3.7+); bouts are time-ordered, so the
    # result is ordered by when each exercise was first performed.
    counts: dict[str, int] = {}
    for label in sets["label"]:
        if not include_non_exercise and label in _NON_EXERCISE:
            continue
        counts[label] = counts.get(label, 0) + 1
    return list(counts.items())


def format_set_summary(sets: pd.DataFrame) -> str:
    """Render a one-line, human-readable set summary of a recording.

    Args:
        sets: The per-bout DataFrame from :func:`summarize_session`.

    Returns:
        A display string like ``"2 sets of Bicep Curl · 1 set of Row"``, or an
        empty string if no genuine-exercise sets were detected.
    """
    parts = []
    for label, n in count_sets(sets):
        nice = label.replace("_", " ").title()
        unit = "set" if n == 1 else "sets"
        parts.append(f"{n} {unit} of {nice}")
    return " · ".join(parts)
