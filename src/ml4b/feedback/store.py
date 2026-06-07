"""Persistent store for user label corrections (DECISIONS.md §8).

Each correction records the **raw window samples** (the six canonical-unit
channels) together with the label the user says is correct, plus light metadata
(the model's original prediction, its confidence, the source file, a timestamp).

Why store the raw window and not the extracted features? Features are an
implementation detail that can change (e.g. a new feature set). The raw window is
the ground truth; re-deriving features from it through the shared pipeline at
retrain time guarantees corrections stay compatible with whatever feature code is
current — the project's single-pipeline rule applied to feedback.

The on-disk format is newline-delimited JSON (``feedback.jsonl``): append-only,
dependency-free, human-inspectable, and trivial to back up or delete.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml4b.utils.config import DATA_FEEDBACK

# The six raw signal channels stored per window (canonical units: accel in g
# total, gyro in rad/s) — identical names to windowing.apply_sliding_window.
RAW_CHANNELS: tuple[str, ...] = (
    "raw_ax",
    "raw_ay",
    "raw_az",
    "raw_gx",
    "raw_gy",
    "raw_gz",
)

# Default location of the feedback log (created on first write).
FEEDBACK_FILE: Path = DATA_FEEDBACK / "feedback.jsonl"


def _feedback_file(path: Path | None = None) -> Path:
    """Resolve the feedback file path, defaulting to the configured location."""
    return path if path is not None else FEEDBACK_FILE


def build_records(
    window_df: pd.DataFrame,
    results: pd.DataFrame,
    window_ids: list[int],
    corrected_labels: dict[int, str],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Assemble feedback records for a set of corrected windows.

    Args:
        window_df: The raw windowed frame returned by
            :func:`ml4b.data.apple_watch_loader.predict_from_sensor_logger`
            (``return_windows=True``); row position equals ``window_id``.
        results: The per-window prediction frame from the same call (carries
            ``predicted_class`` and ``confidence``).
        window_ids: The ``window_id`` values the user corrected.
        corrected_labels: Map ``window_id -> corrected label``.
        source: Identifier of the uploaded recording (e.g. the file name).

    Returns:
        A list of JSON-serialisable record dicts, one per corrected window.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    for wid in window_ids:
        corrected = corrected_labels.get(wid)
        if corrected is None:
            continue
        win = window_df.iloc[wid]
        res = results.iloc[wid]
        record: dict[str, Any] = {
            "ts": ts,
            "source": source,
            "window_id": int(wid),
            "corrected_label": str(corrected),
            "predicted_label": str(res["predicted_class"]),
            # Confidence is NaN for gated/abstained windows — store None then.
            "confidence": (
                float(res["confidence"]) if pd.notna(res["confidence"]) else None
            ),
        }
        # Store each channel as a plain list of floats.
        for ch in RAW_CHANNELS:
            record[ch] = [float(v) for v in win[ch]]
        records.append(record)
    return records


def build_labelled_records(
    window_df: pd.DataFrame,
    label: str,
    *,
    source: str,
    window_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build feedback records labelling whole windows of a clean recording.

    Where :func:`build_records` captures *corrections* of the model's
    predictions, this labels a recording whose exercise the user already knows
    (e.g. one clean set), so it can be folded straight into training via
    ``scripts/update_model.py``. This is the recommended way to add your own data
    to the single-subject base dataset (DECISIONS.md §6, §8). There is no model
    prediction here, so ``predicted_label`` is recorded as ``"(labelled import)"``.

    Args:
        window_df: A windowed frame from
            :func:`ml4b.data.apple_watch_loader.load_and_window_recording`
            (carries the six ``raw_*`` channels per row).
        label: The exercise label that applies to every selected window.
        source: Identifier of the recording (e.g. the file name).
        window_ids: Row positions to keep. ``None`` (default) labels every row;
            pass only the active rows to drop rest windows.

    Returns:
        A list of JSON-serialisable record dicts, one per selected window.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ids = range(len(window_df)) if window_ids is None else window_ids
    records: list[dict[str, Any]] = []
    for wid in ids:
        win = window_df.iloc[wid]
        record: dict[str, Any] = {
            "ts": ts,
            "source": source,
            "window_id": int(wid),
            "corrected_label": str(label),
            "predicted_label": "(labelled import)",
            "confidence": None,
        }
        # Store each channel as a plain list of floats (same schema as corrections).
        for ch in RAW_CHANNELS:
            record[ch] = [float(v) for v in win[ch]]
        records.append(record)
    return records


def append(records: list[dict[str, Any]], path: Path | None = None) -> int:
    """Append correction records to the feedback log, creating it if needed.

    Args:
        records: Records produced by :func:`build_records`.
        path: Optional override of the feedback file location (for tests).

    Returns:
        The number of records written.
    """
    if not records:
        return 0
    file = _feedback_file(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return len(records)


def load(path: Path | None = None) -> pd.DataFrame:
    """Load all stored corrections.

    Args:
        path: Optional override of the feedback file location (for tests).

    Returns:
        One row per correction. Empty (with the expected columns) if no feedback
        has been recorded yet.
    """
    file = _feedback_file(path)
    columns = [
        "ts",
        "source",
        "window_id",
        "corrected_label",
        "predicted_label",
        "confidence",
        *RAW_CHANNELS,
    ]
    if not file.exists():
        return pd.DataFrame(columns=columns)
    rows = [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def stats(path: Path | None = None) -> dict[str, Any]:
    """Summarise the feedback store for display in the app / CLI.

    Returns:
        Dict with the total count, per-label counts, number of source files, and
        how many corrections actually changed the model's prediction.
    """
    df = load(path)
    if df.empty:
        return {"total": 0, "per_label": {}, "n_sources": 0, "n_changed": 0}
    per_label = df["corrected_label"].value_counts().to_dict()
    n_changed = int((df["corrected_label"] != df["predicted_label"]).sum())
    return {
        "total": int(len(df)),
        "per_label": {str(k): int(v) for k, v in per_label.items()},
        "n_sources": int(df["source"].nunique()),
        "n_changed": n_changed,
    }


def to_window_df(feedback: pd.DataFrame) -> pd.DataFrame:
    """Convert stored corrections into a windowing-compatible DataFrame.

    The result matches :func:`ml4b.data.windowing.apply_sliding_window`'s schema
    so it flows straight into augmentation and invariant-feature extraction. Each
    correction's source file becomes a distinct ``recording_id`` (prefixed
    ``feedback::``) so augmented copies stay grouped with their origin and never
    collide with the integer set ids of the base dataset.

    Args:
        feedback: Frame from :func:`load`.

    Returns:
        DataFrame with columns
        ``[subject_id, exercise_name, recording_id, window_id, raw_a*, raw_g*]``.
    """
    columns = [
        "subject_id",
        "exercise_name",
        "recording_id",
        "window_id",
        *RAW_CHANNELS,
    ]
    if feedback.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            "subject_id": 0,
            # The corrected label IS the training target.
            "exercise_name": feedback["corrected_label"].astype(str),
            "recording_id": "feedback::" + feedback["source"].astype(str),
            "window_id": range(len(feedback)),
        }
    )
    for ch in RAW_CHANNELS:
        out[ch] = feedback[ch].tolist()
    return out[columns]


def clear(path: Path | None = None) -> None:
    """Delete the feedback log (used by tests and an explicit user reset)."""
    file = _feedback_file(path)
    file.unlink(missing_ok=True)
