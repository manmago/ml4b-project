"""Model evaluation module for ML4B gym exercise recognition.

Provides functions to evaluate trained classifiers and generate
standardized evaluation artifacts (metrics dict, confusion matrix,
classification report) that are used both in notebooks and the
Streamlit app.

Primary metric throughout: macro-averaged F1 score.
Accuracy is reported for completeness but is NOT the primary metric
because val/test sets retain the original class distribution where
'rest' dominates — a model predicting 'rest' for everything would
achieve ~89% accuracy but near-zero macro F1.
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    class_names: list[str],
    save_dir: Path | None = None,
) -> dict:
    """Evaluate a trained classifier and return standardized metrics.

    Computes accuracy, macro F1, per-class F1, and confusion matrix.
    Optionally saves a normalised confusion matrix plot to save_dir.
    Primary metric is macro F1 — not accuracy — because val/test sets
    retain original class imbalance where 'rest' dominates.

    Args:
        model: Trained sklearn-compatible classifier (or Pipeline)
        X: Feature matrix of shape (n_samples, n_features)
        y: True class labels of shape (n_samples,)
        model_name: Human-readable name used in plot titles and result keys
        class_names: Ordered list of class name strings matching label values
        save_dir: If provided, saves confusion matrix PNG here

    Returns:
        Dict with keys: model_name, accuracy, macro_f1, per_class_f1
        (dict of class→f1), confusion_matrix (np.ndarray),
        classification_report (str).
    """
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)

    # Macro F1 averages F1 equally across all 6 classes, regardless of
    # class size. This penalises models that ignore rare exercise classes.
    macro_f1 = f1_score(y, y_pred, average="macro", labels=class_names)

    # Per-class F1 reveals which individual exercises are hardest to classify.
    per_class_f1_values = f1_score(
        y, y_pred, average=None, labels=class_names, zero_division=0
    )
    per_class_f1 = dict(zip(class_names, per_class_f1_values.tolist()))

    # Confusion matrix uses the same class order as class_names so axes match
    # the per-class F1 dict above.
    cm = confusion_matrix(y, y_pred, labels=class_names)

    report = classification_report(
        y, y_pred, labels=class_names, target_names=class_names, zero_division=0
    )

    if save_dir is not None:
        _save_confusion_matrix(cm, class_names, model_name, save_dir)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_dir: Path,
) -> None:
    """Save a normalised confusion matrix heatmap as a PNG figure.

    Args:
        cm: Raw confusion matrix array from sklearn.metrics.confusion_matrix
        class_names: Class label strings used for axis tick labels
        model_name: Used in the plot title and output filename
        save_dir: Directory to write the PNG into (created if absent)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Normalise row-wise: each cell shows the fraction of true-class samples
    # predicted as each other class. Easier to compare across classes of
    # different sizes than raw counts.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}\n(row-normalised: fraction of true class)"
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Sanitise model name for use in a filename (spaces → underscores)
    safe_name = model_name.lower().replace(" ", "_")
    save_path = save_dir / f"confusion_matrix_{safe_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple evaluate_model results.

    Args:
        results: List of dicts returned by evaluate_model()

    Returns:
        DataFrame with one row per model, columns:
        [model_name, accuracy, macro_f1, f1_<class> for each class].
        Sorted by macro_f1 descending so the best model is first.
    """
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "model_name": r["model_name"],
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
        }
        # Flatten per-class F1 into individual columns for tabular comparison
        for cls, f1 in r["per_class_f1"].items():
            row[f"f1_{cls}"] = round(f1, 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by primary metric so the winning model appears at the top
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def save_model(
    model: Any,
    model_name: str,
    save_dir: Path,
) -> Path:
    """Save a trained model to disk as a .joblib file.

    joblib is preferred over pickle for sklearn objects because it handles
    large numpy arrays more efficiently via memory-mapping.

    Args:
        model: Trained sklearn-compatible classifier or Pipeline
        model_name: Used to construct the filename (spaces → underscores)
        save_dir: Target directory (models/saved/). Created if absent.

    Returns:
        Absolute Path to the saved .joblib file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    save_path = save_dir / f"{safe_name}.joblib"
    joblib.dump(model, save_path)
    return save_path
