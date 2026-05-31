# ADR-016: Final Target Classes — 3 Exercises on the Apple-Watch Dataset
**Status:** Accepted
**Date:** 2026-05-31

## Context
The project's deployment target is a person uploading Apple Watch (Sensor
Logger) data. Earlier models trained on MM-Fit (a non-Apple smartwatch) scored
highly on their own test set but generalised poorly to real Apple-Watch uploads
— a device-domain shift. We switched the training anchor to the Kaggle "Gym
Workout IMU Dataset", which is recorded on an Apple Watch SE at 100 Hz (left
wrist), the same device family as deployment (see `docs/data_understanding/
dataset_evaluation.md`). That dataset contains 21 exercises but is single
subject; we must pick a small set of classes that are well covered and reliably
distinguishable.

## Decision
Recognise **three** exercise classes, each formed by grouping biomechanically
equivalent Kaggle abbreviations:

| Class | Kaggle abbreviations | Sets | Movement axis |
|-------|----------------------|------|---------------|
| `bicep_curl` | AIDBC, IDBC, PREC | 24 | Elbow flexion |
| `tricep_extension` | CGOCTE, MTE, SAOCTE, SAODTE | 30 | Elbow extension (overhead) |
| `row` | CGCR, NGCR, MGTBR | 21 | Horizontal pull |

`rest` and `uncertain` are **not** trained classes — they are produced by the
activity gate (ADR-017) and the confidence threshold (ADR-020).

## Alternatives Considered
- **Include `push_up`** — the original third target. The Kaggle dataset contains
  **no** push-ups (its `APULL` is an *assisted pull-up*, a pulling movement).
  Sourcing push-ups from MM-Fit would reintroduce the device-domain shift for
  that one class. Rejected.
- **`lateral_raise`** (SACLR 14, DLR 2) — well covered, but its motion (shoulder
  abduction) overlaps in the wrist signal with the start of a curl and adds a
  fourth class with weaker separability. Rejected to keep classes maximally
  distinct.
- **`shoulder_press`** (DSP 7, MSP 3) — fewer sets and a vertical-push pattern
  that is easily confused with overhead triceps extensions. Rejected.
- **More than three classes** — every added class lowers per-class separability
  on a single-subject anchor; three distinct axes is the sweet spot.

## Rationale
The three chosen classes (1) maximise per-class coverage among well-distinct
options, (2) span **three different movement axes** — flexion, overhead
extension, horizontal pull — which gives the classifier the cleanest decision
boundaries, and (3) are gym-reproducible: anyone can perform a curl, an overhead
triceps extension, and a row to test the app. Keeping the count at three
maximises reliability given a single-subject training anchor.

## Consequences
- **Positive:** device-domain match (Apple Watch → Apple Watch), good coverage,
  distinct classes, honest reproducibility.
- **Negative:** `push_up` is dropped versus the original plan; the model is
  trained on a single subject (see ADR-021 for the evaluation limitation). The
  three classes are the scope the app advertises; other exercises will be mapped
  to the nearest class or surfaced as `uncertain`.
