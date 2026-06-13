"""Unit tests for ml4b.models.pipeline (the shared two-model training core).

These lock the contract that BOTH shipped models go through identical
augmentation + feature extraction + Random Forest + novelty fitting, differing
only in the data they see (DECISIONS.md §9). They use a tiny synthetic windowed
DataFrame so they run fast and need no dataset.
"""

import numpy as np
import pandas as pd

from ml4b.models.pipeline import (
    build_metrics_payload,
    build_training_matrix,
    fit_model_and_novelty,
    gate_calibration,
)

WIN = 200


def _windows(n_per_class: int = 4) -> pd.DataFrame:
    """Build a small windowed DataFrame matching apply_sliding_window output.

    Two classes across two recordings each, so leave-one-set-out grouping and a
    multi-class Random Forest both have something to work with.
    """
    rng = np.random.default_rng(7)
    rows = []
    wid = 0
    for cls, offset in (("bicep_curl", 0.0), ("row", 1.5)):
        for rec in range(2):
            for _ in range(n_per_class):
                rows.append(
                    {
                        "subject_id": "s0",
                        "exercise_name": cls,
                        "recording_id": f"{cls}_{rec}",
                        "window_id": wid,
                        # Offset the two classes so they are separable in feature space.
                        "raw_ax": (rng.standard_normal(WIN) + offset).tolist(),
                        "raw_ay": rng.standard_normal(WIN).tolist(),
                        "raw_az": rng.standard_normal(WIN).tolist(),
                        "raw_gx": (rng.standard_normal(WIN) + offset).tolist(),
                        "raw_gy": rng.standard_normal(WIN).tolist(),
                        "raw_gz": rng.standard_normal(WIN).tolist(),
                    }
                )
                wid += 1
    return pd.DataFrame(rows)


def test_build_training_matrix_marks_augmented_copies() -> None:
    """Augmentation keeps originals first; the mask is a clean position threshold."""
    df = _windows(4)  # 2 classes * 2 recs * 4 = 16 originals
    tm = build_training_matrix(df, n_augment=5, random_state=42)

    assert tm.n_orig == len(df)
    assert len(tm.X) == len(df) * 6  # 1 original + 5 augmented copies
    assert tm.is_augmented[: tm.n_orig].sum() == 0  # originals first
    assert tm.is_augmented[tm.n_orig :].all()  # augmented after
    assert len(tm.feature_names) > 0
    assert tm.X.shape == (len(tm.y), len(tm.feature_names))


def test_build_training_matrix_is_deterministic() -> None:
    """A fixed seed reproduces the exact same feature matrix."""
    df = _windows(3)
    a = build_training_matrix(df, n_augment=2, random_state=42)
    b = build_training_matrix(df, n_augment=2, random_state=42)
    np.testing.assert_allclose(a.X, b.X)


def test_fit_model_and_novelty_predicts_and_keeps_known() -> None:
    """The fitted model predicts known classes; novelty keeps originals known."""
    df = _windows(4)
    tm = build_training_matrix(df, n_augment=3, random_state=42)
    model, detector = fit_model_and_novelty(tm, random_state=42)

    preds = model.predict(tm.X)
    assert set(preds).issubset({"bicep_curl", "row"})
    # The novelty detector is calibrated on the originals, so the vast majority of
    # them must read as "known". With this tiny fixture the per-class 99th-percentile
    # threshold sits just below the single most-extreme training point, so we allow a
    # small rejection margin (on the real datasets this is ~99%).
    known_frac = detector.is_known(tm.X[~tm.is_augmented]).mean()
    assert known_frac >= 0.8


def test_gate_calibration_reports_active_fraction() -> None:
    """Gate calibration returns the expected statistics keys in range."""
    stats = gate_calibration(_windows(3))
    assert 0.0 <= stats["fraction_kept_active"] <= 1.0
    assert "accel_std_median" in stats
    assert "gyro_mean_median" in stats


def test_build_metrics_payload_merges_extra_keys() -> None:
    """The metrics payload carries the core keys plus any Testdaten extras."""
    cv = {
        "macro_f1": 0.5,
        "accuracy": 0.6,
        "per_class_f1": {"bicep_curl": 0.4, "row": 0.6},
        "confusion_matrix": np.array([[1, 0], [0, 1]]),
    }
    payload = build_metrics_payload(
        cv,
        class_names=["bicep_curl", "row"],
        n_sets=4,
        n_orig=16,
        n_features=39,
        n_augment=5,
        gate_stats={"fraction_kept_active": 1.0},
        evaluation="unit-test",
        extra={"n_testdaten_sets": 2},
    )
    assert payload["cv_macro_f1"] == 0.5
    assert payload["n_testdaten_sets"] == 2
    assert payload["confusion_matrix"] == [[1, 0], [0, 1]]
