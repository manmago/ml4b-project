"""Calibrate the activity-gate thresholds from REST + EXERCISE motion energy.

Read-only analysis — it does **not** change any code. The energy gate
(``src/ml4b/data/activity_gate.py``) separates rest from exercise with two
thresholds on per-window motion energy:

    a window is ACTIVE  iff  accel_mag_std > ACCEL_MAG_STD_THRESHOLD
                         OR  gyro_mag_mean > GYRO_MAG_MEAN_THRESHOLD

A good threshold sits in the GAP between the energy of genuine rest (low) and real
exercise (high). This tool measures BOTH distributions and reports where the gap is,
how much margin each threshold has to either side, and a recommended operating point.

It does **not** replace the existing calibration — the two bounds are complementary:

* EXERCISE energy (Kaggle clean sets) gives the **upper** bound: the threshold must
  stay BELOW real exercise, or real exercise gets filtered out.
* committed ``Testdaten/Rest/`` recordings give the **lower** bound: the threshold must
  stay ABOVE genuine rest, or pauses leak through and get classified as an exercise.

Rest data only adds the lower bound; the exercise upper bound stays as before. With no
``Testdaten/Rest/`` recordings the lower bound is unverified and the tool says so.

Apply a recommendation by editing the two constants in ``activity_gate.py``, then
re-run ``make update`` to retrain + validate (and ``make calibrate`` to confirm).

Run:
    uv run python scripts/calibrate_gate.py        (make calibrate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4b.data.activity_gate import (
    ACCEL_MAG_STD_THRESHOLD,
    GYRO_MAG_MEAN_THRESHOLD,
    window_energy,
)
from ml4b.data.apple_watch_loader import load_and_window_recording
from ml4b.data.canonical import OVERLAP, WINDOW_SIZE
from ml4b.data.kaggle_loader import load_kaggle_3class
from ml4b.data.testdaten import REST_PREFIXES, iter_category, recording_name
from ml4b.data.windowing import apply_sliding_window


def _energies(window_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-window ``(accel_mag_std, gyro_mag_mean)`` for a windowed frame."""
    accel: list[float] = []
    gyro: list[float] = []
    for _, row in window_df.iterrows():
        a_std, g_mean = window_energy(
            row["raw_ax"],
            row["raw_ay"],
            row["raw_az"],
            row["raw_gx"],
            row["raw_gy"],
            row["raw_gz"],
        )
        accel.append(a_std)
        gyro.append(g_mean)
    return np.asarray(accel), np.asarray(gyro)


def _gate_active_fraction(
    accel: np.ndarray, gyro: np.ndarray, t_accel: float, t_gyro: float
) -> float:
    """Fraction of windows the gate marks ACTIVE (accel > t_accel OR gyro > t_gyro)."""
    if len(accel) == 0:
        return 0.0
    return float(((accel > t_accel) | (gyro > t_gyro)).mean())


def _exercise_energies() -> tuple[np.ndarray, np.ndarray]:
    """Energy of clean Kaggle exercise windows (the upper bound)."""
    raw = load_kaggle_3class()
    win = apply_sliding_window(
        raw, window_size=WINDOW_SIZE, overlap=OVERLAP, sampling_rate=100
    )
    return _energies(win)


def _rest_energies() -> tuple[np.ndarray, np.ndarray, int]:
    """Energy of every committed Testdaten/Rest/ window (the lower bound).

    Returns ``(accel, gyro, n_recordings)``; empty arrays if there is no rest data.
    """
    accel_parts: list[np.ndarray] = []
    gyro_parts: list[np.ndarray] = []
    n = 0
    for _folder, rec in iter_category(REST_PREFIXES):
        try:
            win, _hz, _n = load_and_window_recording(rec)
        except ValueError as exc:
            print(f"  ! skip {recording_name(rec)}: {exc}")
            continue
        a, g = _energies(win)
        accel_parts.append(a)
        gyro_parts.append(g)
        n += 1
        print(f"  rest take: {recording_name(rec)} ({len(a)} windows)")
    if not accel_parts:
        return np.array([]), np.array([]), 0
    return np.concatenate(accel_parts), np.concatenate(gyro_parts), n


def _gap_threshold(
    rest_vals: np.ndarray, ex_vals: np.ndarray
) -> tuple[float | None, float, float]:
    """Recommend a single-feature threshold in the rest↔exercise gap.

    Returns ``(threshold_or_None, exercise_p01, rest_p99)``. The threshold is the
    geometric mid-point of the gap, or ``None`` if the two distributions overlap on
    this feature (then the other feature must carry separation via the OR rule).
    """
    ex_p01 = float(np.percentile(ex_vals, 1))
    rest_p99 = float(np.percentile(rest_vals, 99)) if len(rest_vals) else float("nan")
    if len(rest_vals) and rest_p99 > 0 and ex_p01 > 0 and rest_p99 < ex_p01:
        return float(np.sqrt(rest_p99 * ex_p01)), ex_p01, rest_p99
    return None, ex_p01, rest_p99


def _report_feature(
    name: str, rest_vals: np.ndarray, ex_vals: np.ndarray, current: float
) -> None:
    """Print the rest/exercise spread and the gap (or overlap) for one feature."""
    ex_p01, ex_p05 = np.percentile(ex_vals, [1, 5])
    print(f"\n[{name}]  current threshold = {current}")
    print(f"  exercise low : p01={ex_p01:.4f}  p05={ex_p05:.4f}  (stay BELOW)")
    if len(rest_vals) == 0:
        print("  rest         : (no Testdaten/Rest/ data — lower bound UNVERIFIED)")
        return
    rest_p95, rest_p99 = np.percentile(rest_vals, [95, 99])
    print(f"  rest high    : p95={rest_p95:.4f}  p99={rest_p99:.4f}  (stay ABOVE)")
    rec, _ex, _rest = _gap_threshold(rest_vals, ex_vals)
    if rec is not None:
        print(f"  GAP ok -> recommended ≈ {rec:.4f}")
        print(
            f"           margin: {ex_p01 / rec:.1f}x below exercise, "
            f"{rec / rest_p99:.1f}x above rest"
        )
    else:
        print(
            f"  OVERLAP: rest p99 ({rest_p99:.4f}) ≥ exercise p01 ({ex_p01:.4f}) — this "
            "feature alone cannot separate; the other feature carries it (OR rule)."
        )


def main() -> None:
    """Measure rest vs exercise energy and recommend gate thresholds."""
    print("=" * 70)
    print("ML4B — Activity-gate calibration (rest vs exercise energy)")
    print("=" * 70)

    print("Loading exercise energy (Kaggle clean sets — upper bound)...")
    ex_a, ex_g = _exercise_energies()
    print(f"  {len(ex_a):,} exercise windows")

    print("Loading rest energy (Testdaten/Rest/ — lower bound)...")
    rest_a, rest_g, n_rest = _rest_energies()

    # How the CURRENT thresholds score on both sides.
    print("\nCurrent thresholds:")
    print(f"  ACCEL_MAG_STD_THRESHOLD = {ACCEL_MAG_STD_THRESHOLD}")
    print(f"  GYRO_MAG_MEAN_THRESHOLD = {GYRO_MAG_MEAN_THRESHOLD}")
    ex_active = _gate_active_fraction(
        ex_a, ex_g, ACCEL_MAG_STD_THRESHOLD, GYRO_MAG_MEAN_THRESHOLD
    )
    print(f"  -> {ex_active:.1%} of exercise kept ACTIVE (want ~100%)")
    if n_rest:
        rest_gated = 1.0 - _gate_active_fraction(
            rest_a, rest_g, ACCEL_MAG_STD_THRESHOLD, GYRO_MAG_MEAN_THRESHOLD
        )
        print(f"  -> {rest_gated:.1%} of rest correctly GATED (want ~100%)")

    # Per-feature spread + gap.
    _report_feature("accel_std", rest_a, ex_a, ACCEL_MAG_STD_THRESHOLD)
    _report_feature("gyro_mean", rest_g, ex_g, GYRO_MAG_MEAN_THRESHOLD)

    if not n_rest:
        print("\nNo rest recordings yet. Add clean pauses under Testdaten/Rest/ (watch")
        print("held still, fidgeting, drinking) and re-run to verify/tune the lower")
        print("bound. Until then the thresholds sit safely below exercise energy only.")
        print("=" * 70)
        return

    # Recommended joint operating point: per-feature gap mid-point where a gap exists,
    # otherwise keep the current threshold and let the other feature separate.
    rec_a = _gap_threshold(rest_a, ex_a)[0]
    rec_g = _gap_threshold(rest_g, ex_g)[0]
    t_a = rec_a if rec_a is not None else ACCEL_MAG_STD_THRESHOLD
    t_g = rec_g if rec_g is not None else GYRO_MAG_MEAN_THRESHOLD
    ex_active_rec = _gate_active_fraction(ex_a, ex_g, t_a, t_g)
    rest_gated_rec = 1.0 - _gate_active_fraction(rest_a, rest_g, t_a, t_g)

    print("\nRecommended thresholds (placed in the gap, both sides):")
    print(f"  ACCEL_MAG_STD_THRESHOLD = {t_a:.4f}")
    print(f"  GYRO_MAG_MEAN_THRESHOLD = {t_g:.4f}")
    print(
        f"  -> {ex_active_rec:.1%} exercise kept ACTIVE | {rest_gated_rec:.1%} rest GATED"
    )
    print(
        "\nTo apply: edit the two constants in src/ml4b/data/activity_gate.py, then "
        "re-run `make update` (retrain + validate) and `make calibrate` to confirm.\n"
        "Only change them if the recommendation clearly beats the current scores."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
