"""Generate the curated set of handover figures under ``reports/figures/``.

These figures summarise the shipped models for the hand-over team and the report.
They are rendered **from the committed artifacts only** — the two metrics JSON
files and the saved Random Forest — so the script is fast, deterministic and needs
no dataset download:

    models/saved/model_metrics.json      -> current model (Kaggle + Testdaten)
    models/saved/baseline_metrics.json   -> baseline model (Kaggle only)
    models/saved/best_model.joblib       -> Random Forest (feature importances)
    data/processed/feature_names.txt     -> ordered feature names

Figures produced (all written to ``reports/figures/``):

  1. confusion_matrix_current.png    — current model, leave-one-set-out CV
  2. confusion_matrix_baseline.png   — baseline model, leave-one-set-out CV
  3. per_class_f1_comparison.png     — baseline vs current per-class + macro F1
  4. feature_importance_top15.png    — 15 most important invariant features
  5. dataset_composition.png         — training sets per class (Kaggle + Testdaten)

Run:
    uv run python scripts/generate_report_figures.py
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml4b.utils.config import BEST_MODEL_FILE, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

# Kaggle anchor sets per class (from the dataset mapping, dataset_evaluation.md §3.1).
# Testdaten counts are read from the metrics JSON so they stay in sync with rebuilds.
KAGGLE_SETS_PER_CLASS: dict[str, int] = {
    "bicep_curl": 24,
    "row": 21,
    "tricep_extension": 30,
}


def _load_json(name: str) -> dict:
    """Load a metrics JSON file from ``models/saved/``."""
    return json.loads((MODELS_DIR / name).read_text())


def _save(fig: plt.Figure, filename: str) -> None:
    """Write a figure to ``reports/figures/`` at a consistent resolution."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPORTS_DIR.parents[1])}")


def confusion_matrix_figure(metrics: dict, title: str, filename: str) -> None:
    """Render a row-normalised confusion-matrix heatmap from a metrics payload.

    Args:
        metrics: Parsed metrics JSON (must hold ``confusion_matrix`` and
            ``confusion_matrix_labels``).
        title: Plot title.
        filename: Output filename under ``reports/figures/``.
    """
    labels = metrics["confusion_matrix_labels"]
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    # Row-normalise so each cell is the fraction of a true class — comparable
    # across classes of different sizes.
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"{title}\n(leave-one-set-out CV, row-normalised)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, filename)


def per_class_f1_figure(baseline: dict, current: dict) -> None:
    """Grouped bar chart comparing per-class + macro F1 of both models."""
    classes = list(current["cv_per_class_f1"].keys())
    metrics_order = classes + ["macro"]

    def values(m: dict) -> list[float]:
        return [m["cv_per_class_f1"][c] for c in classes] + [m["cv_macro_f1"]]

    base_vals = values(baseline)
    cur_vals = values(current)

    x = np.arange(len(metrics_order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(
        x - width / 2, base_vals, width, label="Baseline (Kaggle only)", color="#9ecae1"
    )
    b2 = ax.bar(
        x + width / 2,
        cur_vals,
        width,
        label="Current (Kaggle + Testdaten)",
        color="#3182bd",
    )
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class and macro F1 — leave-one-set-out CV")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_order])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "per_class_f1_comparison.png")


def feature_importance_figure(top_n: int = 15) -> None:
    """Top-N Random Forest feature importances for the shipped (current) model."""
    model = joblib.load(MODELS_DIR / BEST_MODEL_FILE)
    names = (DATA_PROCESSED / "feature_names.txt").read_text().split()
    importances = np.asarray(model.feature_importances_)

    order = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in order][::-1]
    top_vals = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names, top_vals, color="#31a354")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} device-invariant features — current Random Forest")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, "feature_importance_top15.png")


def dataset_composition_figure(current: dict) -> None:
    """Stacked bar of training sets per class: Kaggle anchor + our Testdaten."""
    td = current.get("testdaten_sets_per_class", {})
    classes = list(KAGGLE_SETS_PER_CLASS.keys())
    kaggle = [KAGGLE_SETS_PER_CLASS[c] for c in classes]
    testdaten = [td.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x, kaggle, label="Kaggle anchor (Apple Watch)", color="#3182bd")
    b2 = ax.bar(
        x,
        testdaten,
        bottom=kaggle,
        label="Our Testdaten (Apple Watch)",
        color="#fd8d3c",
    )
    ax.bar_label(b1, label_type="center", fontsize=9, color="white")
    for rect, k, t in zip(b2, kaggle, testdaten):
        if t:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                k + t / 2,
                str(t),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            k + t + 0.6,
            str(k + t),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Number of labelled sets")
    ax.set_title("Training data composition per class")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "dataset_composition.png")


def main() -> None:
    """Render every handover figure into ``reports/figures/``."""
    current = _load_json("model_metrics.json")
    baseline = _load_json("baseline_metrics.json")

    print(f"Writing figures to {REPORTS_DIR} ...")
    confusion_matrix_figure(
        current,
        "Confusion Matrix — Current (Kaggle + Testdaten)",
        "confusion_matrix_current.png",
    )
    confusion_matrix_figure(
        baseline,
        "Confusion Matrix — Baseline (Kaggle only)",
        "confusion_matrix_baseline.png",
    )
    per_class_f1_figure(baseline, current)
    feature_importance_figure()
    dataset_composition_figure(current)
    print("Done.")


if __name__ == "__main__":
    main()
