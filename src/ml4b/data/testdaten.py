"""Discover and label the committed Apple-Watch recordings under ``data/Testdaten/``.

Single source of truth for the Testdaten folder convention, shared by
``scripts/rebuild_from_testdaten.py`` (training) and ``scripts/calibrate_gate.py``
(gate calibration) so the two never drift apart. Only the **folder name** decides
the label, so filenames are free-form (typos like ``rowsr_...`` inside ``Rows/`` are
fine):

    Biceps_Curls*        -> bicep_curl        (training class)
    Rows*                -> row               (training class)
    Triceps_Extensions*  -> tricep_extension  (training class)
    Rest*                -> NOT a class — validates / calibrates the energy gate
    Uncertain*           -> NOT a class — validates open-set ``unknown`` rejection
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ml4b.data.activity_gate import gate_window_df
from ml4b.data.apple_watch_loader import load_and_window_recording
from ml4b.utils.config import PROJECT_ROOT

# Root holding the committed recordings, one subfolder per category.
TESTDATEN_DIR: Path = PROJECT_ROOT / "data" / "Testdaten"

# Folder-name prefix (lower-cased) -> training label. Prefix match keeps us robust
# to per-recording suffixes; only the folder sets the label.
EXERCISE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("biceps_curls", "bicep_curl"),
    ("rows", "row"),
    ("triceps_extensions", "tricep_extension"),
)
# Non-training folders, used for validation / calibration only (never classes).
REST_PREFIXES: tuple[str, ...] = ("rest",)
UNCERTAIN_PREFIXES: tuple[str, ...] = ("uncertain",)


def label_for(folder_name: str) -> str | None:
    """Return the training label for a subfolder, or None if it is not an exercise
    folder (``Rest`` / ``Uncertain`` / anything unrecognised).
    """
    name = folder_name.lower()
    for prefix, label in EXERCISE_PREFIXES:
        if name.startswith(prefix):
            return label
    return None


def recordings_in(folder: Path) -> list[Path]:
    """List the Sensor Logger recordings inside a subfolder.

    A recording is a ``<rec>/WristMotion.csv`` directory export or a ``.zip``
    placed directly in the folder.
    """
    return sorted(folder.glob("*/WristMotion.csv")) + sorted(folder.glob("*.zip"))


def recording_name(rec: Path) -> str:
    """Human-readable id for a recording (its export-folder name, or zip stem)."""
    return rec.parent.name if rec.suffix.lower() == ".csv" else rec.stem


def iter_category(prefixes: tuple[str, ...]) -> Iterator[tuple[str, Path]]:
    """Yield ``(folder_name, recording_path)`` for folders matching ``prefixes``.

    Args:
        prefixes: Lower-cased folder-name prefixes (e.g. :data:`REST_PREFIXES`).
    """
    if not TESTDATEN_DIR.is_dir():
        return
    for sub in sorted(p for p in TESTDATEN_DIR.iterdir() if p.is_dir()):
        if sub.name.lower().startswith(prefixes):
            for rec in recordings_in(sub):
                yield sub.name, rec


def load_exercise_windows(
    active_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Window every labelled Testdaten **exercise** recording into one frame.

    Each recording is loaded through the same front half of the prediction pipeline
    and tagged with the folder's label and a unique ``recording_id`` (so leave-one-
    set-out CV treats each recording as one held-out set).

    Args:
        active_only: If True (default), keep only the active (motion) windows — a
            clean labelled set is mostly motion and rest windows add noise.

    Returns:
        ``(window_df, per_label_counts)`` — the combined windowed frame (empty if no
        recordings) and how many recordings contributed per label.
    """
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    if not TESTDATEN_DIR.is_dir():
        return pd.DataFrame(), counts

    for sub in sorted(p for p in TESTDATEN_DIR.iterdir() if p.is_dir()):
        label = label_for(sub.name)
        if label is None:
            continue  # Rest / Uncertain / unrecognised: not a training class.
        for rec in recordings_in(sub):
            name = recording_name(rec)
            try:
                window_df, _hz, _n = load_and_window_recording(rec)
            except ValueError as exc:
                print(f"    ! skip {sub.name}/{name}: {exc}")
                continue
            if active_only:
                window_df = window_df[gate_window_df(window_df).to_numpy()].copy()
            if window_df.empty:
                print(f"    ! {sub.name}/{name}: all windows gated as rest")
                continue
            # Overwrite the loader's placeholder label/group with the real ones.
            window_df["exercise_name"] = label
            window_df["recording_id"] = f"testdaten::{label}::{name}"
            frames.append(window_df)
            counts[label] = counts.get(label, 0) + 1

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, counts
