"""Add a clean, labelled recording to the continual-learning feedback store.

Use this when you record a single known exercise (one clean set) and want to fold
it straight into training — the recommended way to add your *own* data to the
single-subject base dataset (DECISIONS.md §6, §8). Each active 2-second window of
the recording is stored under your label; afterwards run
``scripts/update_model.py`` to retrain on the base data **plus** your additions.

Recording protocol (one exercise per file): see
``docs/project/apple_watch_data_collection_guide.md``.

Run:
    uv run python scripts/add_labelled_recording.py PATH --label bicep_curl
    uv run python scripts/add_labelled_recording.py rec.zip --label squat --keep-rest

A brand-new label (e.g. ``squat``) becomes a new class on the next retrain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml4b.data.activity_gate import gate_window_df
from ml4b.data.apple_watch_loader import load_and_window_recording
from ml4b.feedback import store


def main() -> None:
    """Parse CLI args, window the recording, and store its labelled windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", type=Path, help="Sensor Logger WristMotion.csv or ZIP export."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Exercise label that applies to the whole recording (e.g. bicep_curl).",
    )
    parser.add_argument(
        "--keep-rest",
        action="store_true",
        help="Also store low-motion (rest) windows; by default they are dropped.",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        raise SystemExit(f"File not found: {args.recording}")

    # Same load -> resample -> window front half as the prediction pipeline.
    window_df, detected_hz, _ = load_and_window_recording(args.recording)
    n_total = len(window_df)

    # Keep only active windows unless the user asks to keep rest too — a clean set
    # is mostly motion, and rest windows add little but noise to a labelled set.
    if args.keep_rest:
        ids = list(range(n_total))
    else:
        active = gate_window_df(window_df).to_numpy()
        ids = [int(i) for i in range(n_total) if active[i]]

    if not ids:
        print(
            "No windows to add — every window was gated as rest. If the recording "
            "really is the exercise, re-run with --keep-rest."
        )
        return

    records = store.build_labelled_records(
        window_df, args.label, source=args.recording.name, window_ids=ids
    )
    n = store.append(records)
    print(f"Detected ~{detected_hz} Hz; {n_total} windows total.")
    print(f"Stored {n} window(s) labelled '{args.label}'.")
    print(f"Feedback store now: {store.stats()}")
    print(
        "\nNext: fold these into the model with\n"
        "    uv run python scripts/update_model.py"
    )


if __name__ == "__main__":
    main()
