"""Retrain the model from base data + user corrections (ADR-027).

Continual-learning entry point. Reads the corrections collected in the app (or
via the API) from ``data/feedback/feedback.jsonl`` and retrains the Random Forest
on the base Kaggle data plus those corrections, using the identical pipeline as
``scripts/train_model.py``. The originally-shipped model is backed up to
``models/saved/best_model_base.joblib`` before the first overwrite.

Run with:
    uv run python scripts/update_model.py                 # retrain with feedback
    uv run python scripts/update_model.py --restore-base  # undo: restore base model
    uv run python scripts/update_model.py --repeat 5      # weight feedback more
"""

from __future__ import annotations

import argparse
import json

from ml4b.feedback import retrain, store


def main() -> None:
    """Parse CLI args and run the requested retrain / restore action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat",
        type=int,
        default=retrain.FEEDBACK_REPEAT,
        help="Times each correction is repeated before augmentation (weighting).",
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=retrain.N_AUGMENT,
        help="Augmented copies per window (default matches initial training).",
    )
    parser.add_argument(
        "--restore-base",
        action="store_true",
        help="Restore the originally-shipped model and exit (undo retraining).",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("ML4B Continual Learning — retrain from feedback (ADR-027)")
    print("=" * 64)

    if args.restore_base:
        ok = retrain.restore_base_model()
        print("Base model restored." if ok else "No base-model backup found.")
        return

    fb = store.stats()
    print(
        f"Feedback corrections: {fb['total']} "
        f"({fb['n_changed']} changed a prediction) "
        f"across {fb['n_sources']} recording(s)"
    )
    print(f"Per label: {fb['per_label']}")

    if not retrain.base_available():
        print(
            "\nBase dataset not found at data/raw/kaggle_gym_imu/. Download it to "
            "retrain (your feedback is safely stored)."
        )
        return
    if fb["total"] == 0:
        print("\nNo corrections recorded yet — nothing to learn from.")
        return

    print(
        f"\nRetraining (feedback repeated {args.repeat}x, "
        f"{args.n_augment}x augmentation)..."
    )
    manifest = retrain.retrain_model(
        feedback_repeat=args.repeat, n_augment=args.n_augment
    )
    print("\nDone. Manifest:")
    print(json.dumps(manifest, indent=2))
    print("=" * 64)


if __name__ == "__main__":
    main()
