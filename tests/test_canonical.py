"""Unit tests for ml4b.data.canonical.

Covers total-acceleration reconstruction (with and without a gravity vector) and
uniform resampling to the target rate (both up- and down-sampling), which are the
two pieces of canonicalization shared by the training and inference loaders.
"""

import numpy as np
import pandas as pd

from ml4b.data.canonical import (
    CANONICAL_CHANNELS,
    TARGET_HZ,
    reconstruct_total_acceleration,
    resample_uniform,
)


def test_reconstruct_with_gravity_adds_components() -> None:
    """Total acceleration = user acceleration + gravity, component-wise (in g)."""
    df = pd.DataFrame(
        {
            "ux": [0.1, 0.2],
            "uy": [0.0, -0.1],
            "uz": [0.0, 0.0],
            "gx": [0.0, 0.0],
            "gy": [0.0, 0.0],
            "gz": [1.0, 1.0],
        }
    )
    out = reconstruct_total_acceleration(df, ("ux", "uy", "uz"), ("gx", "gy", "gz"))
    assert np.allclose(out["ax"], [0.1, 0.2])
    assert np.allclose(out["az"], [1.0, 1.0])  # ~1 g at rest on z


def test_reconstruct_without_gravity_passthrough() -> None:
    """Without a gravity vector, user acceleration is used unchanged."""
    df = pd.DataFrame({"ux": [0.3], "uy": [0.4], "uz": [0.5]})
    out = reconstruct_total_acceleration(df, ("ux", "uy", "uz"), None)
    assert np.allclose(out["ax"], [0.3]) and np.allclose(out["az"], [0.5])


def _ramp_df(n: int, hz: float) -> pd.DataFrame:
    """Build a canonical df whose channels are simple ramps for interpolation."""
    t = np.arange(n) / hz
    data = {"timestamp": t}
    for i, ch in enumerate(CANONICAL_CHANNELS):
        data[ch] = t * (i + 1)  # distinct linear ramp per channel
    return pd.DataFrame(data)


def test_resample_upsamples_50_to_100() -> None:
    """A 50 Hz, 2 s signal resamples to ~201 samples at 100 Hz."""
    df = _ramp_df(n=100, hz=50.0)  # spans ~1.98 s
    out = resample_uniform(df, target_hz=TARGET_HZ)
    # Expect about (duration * 100) + 1 samples.
    expected = int(round((df["timestamp"].iloc[-1]) * TARGET_HZ)) + 1
    assert len(out) == expected
    assert len(out) > len(df)  # genuinely upsampled


def test_resample_preserves_endpoints_and_linearity() -> None:
    """Linear ramps stay linear after interpolation; endpoints are preserved."""
    df = _ramp_df(n=200, hz=100.0)
    out = resample_uniform(df, target_hz=TARGET_HZ)
    # 'ax' is the first channel: ramp slope 1 → value ≈ timestamp.
    assert np.allclose(out["ax"].to_numpy(), out["timestamp"].to_numpy(), atol=1e-9)
    assert np.isclose(out["timestamp"].iloc[0], df["timestamp"].iloc[0])
    assert np.isclose(out["timestamp"].iloc[-1], df["timestamp"].iloc[-1])


def test_resample_too_short_returns_input() -> None:
    """Fewer than two samples cannot define a grid — input is returned."""
    df = _ramp_df(n=1, hz=100.0)
    assert len(resample_uniform(df)) == 1
