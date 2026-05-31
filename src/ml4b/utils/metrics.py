"""Load the committed model-evaluation metrics.

``scripts/train_model.py`` writes the honest leave-one-set-out results to
``models/saved/model_metrics.json`` (committed, so the Streamlit app can show
real numbers after a fresh clone). Both the Home and Model Performance pages read
them through this single helper so the displayed metrics can never drift from
what training actually produced.
"""

from __future__ import annotations

import json
from typing import Any

from ml4b.utils.config import MODELS_DIR

# Path to the committed metrics file produced by the training script.
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def load_model_metrics() -> dict[str, Any]:
    """Load the committed model metrics dictionary.

    Returns:
        The parsed contents of ``models/saved/model_metrics.json`` (classes,
        leave-one-set-out macro F1 / accuracy, per-class F1, confusion matrix,
        activity-gate calibration, etc.).

    Raises:
        FileNotFoundError: If the metrics file is missing (run the training
            script to regenerate it).
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics not found at {METRICS_PATH}. "
            "Run `uv run python scripts/train_model.py` to regenerate them."
        )
    return json.loads(METRICS_PATH.read_text())
