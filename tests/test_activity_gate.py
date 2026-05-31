"""Unit tests for ml4b.data.activity_gate.

Verifies that a still (rest) window is gated out while an energetic exercise
window is kept active, and that gate_window_df returns an index-aligned mask.
"""

import numpy as np
import pandas as pd

from ml4b.data.activity_gate import gate_window_df, is_active, window_energy

WIN = 200


def _still_window() -> dict:
    """A near-still window: ~1 g gravity on z, tiny noise, almost no rotation."""
    rng = np.random.default_rng(1)
    return {
        "raw_ax": (rng.standard_normal(WIN) * 0.005).tolist(),
        "raw_ay": (rng.standard_normal(WIN) * 0.005).tolist(),
        "raw_az": (1.0 + rng.standard_normal(WIN) * 0.005).tolist(),
        "raw_gx": (rng.standard_normal(WIN) * 0.01).tolist(),
        "raw_gy": (rng.standard_normal(WIN) * 0.01).tolist(),
        "raw_gz": (rng.standard_normal(WIN) * 0.01).tolist(),
    }


def _exercise_window() -> dict:
    """An energetic window: large acceleration swings and clear rotation."""
    t = np.linspace(0, 4 * np.pi, WIN)
    return {
        "raw_ax": (0.6 * np.sin(t)).tolist(),
        "raw_ay": (0.5 * np.cos(t)).tolist(),
        "raw_az": (1.0 + 0.4 * np.sin(t)).tolist(),
        "raw_gx": (1.5 * np.sin(t)).tolist(),
        "raw_gy": (1.2 * np.cos(t)).tolist(),
        "raw_gz": (0.8 * np.sin(t)).tolist(),
    }


def test_still_window_is_rest() -> None:
    """A still window falls below both thresholds -> not active."""
    w = _still_window()
    a_std, g_mean = window_energy(
        w["raw_ax"], w["raw_ay"], w["raw_az"], w["raw_gx"], w["raw_gy"], w["raw_gz"]
    )
    assert not is_active(a_std, g_mean)


def test_exercise_window_is_active() -> None:
    """An energetic window clears the thresholds -> active."""
    w = _exercise_window()
    a_std, g_mean = window_energy(
        w["raw_ax"], w["raw_ay"], w["raw_az"], w["raw_gx"], w["raw_gy"], w["raw_gz"]
    )
    assert is_active(a_std, g_mean)


def test_gate_mask_aligned() -> None:
    """gate_window_df returns a boolean mask aligned to the input index."""
    df = pd.DataFrame([_still_window(), _exercise_window()])
    mask = gate_window_df(df)
    assert list(mask) == [False, True]
    assert list(mask.index) == list(df.index)
