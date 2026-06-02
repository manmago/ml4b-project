# ADR-024: Open-Set Novelty Detection → "unknown" Output
**Status:** Accepted
**Date:** 2026-06-02

## Context
The shipped model is a closed-set 3-class classifier (`bicep_curl`, `row`,
`tricep_extension`). In a real gym session the user records one continuous stream
and also performs exercises the model was never trained on (squats, shoulder
press, …). Those windows are full of motion, so they clear the activity gate
(ADR-017) and reach the model, which is forced to pick one of its three classes —
often with *high* confidence. The confidence threshold (ADR-020) only catches the
*low*-confidence cases, so foreign exercises are routinely mislabelled as a known
class with high confidence. A smoke test confirmed this: a synthetic high-energy
foreign movement was labelled `tricep_extension` for all 31 active windows.

## Decision
Add an optional per-class **novelty detector**
(`src/ml4b/data/novelty.py`, `NoveltyDetector`) to the inference pipeline. In the
standardized device-invariant feature space, fit one Gaussian per known class
(mean + Ledoit-Wolf shrinkage covariance) and measure the Mahalanobis distance of
each new window to the nearest class centroid. A window is `known` only if it is
within a calibrated per-class threshold (99th percentile of that class's training
distances); otherwise it is rejected as `unknown` and never reaches the model.
`unknown` is **not** a trained class. The detector is fit by
`scripts/fit_novelty_detector.py` on the same Kaggle 3-class invariant features
(no augmentation) and committed as `models/saved/novelty_detector.joblib`. It is
optional: if the artifact is absent the pipeline behaves exactly as before.

## Alternatives Considered
- **Confidence threshold only (status quo, ADR-020)** — keeps confidently-wrong
  labels on foreign exercises; does not solve the problem.
- **IsolationForest / one-class SVM (global)** — a single global model blends the
  three clusters into one diffuse region and lets foreign motion that lands
  *between* clusters slip through; less interpretable thresholds.
- **Train a real "other" class** — needs a large, labelled, open-ended set of
  non-target exercises we do not have (single-subject dataset, ADR-021); cannot
  cover the open space of all possible movements.
- **Global (not per-class) Mahalanobis** — same blending problem as a global
  one-class model.

## Rationale
The three exercises form well-separated clusters in the invariant feature space,
so a *per-class* Gaussian with Mahalanobis distance models "is this like one of
the exercises I know?" directly. Ledoit-Wolf shrinkage keeps the ~38-dimensional
covariance well-conditioned with limited samples. The empirical-percentile
threshold needs no Gaussian-tail assumption and no new data. The detector is
trained on the identical feature pipeline the model uses, so it cannot drift from
the classifier. With the same smoke test, all 31 foreign windows flipped from a
confident `tricep_extension` to `unknown`, while 99.3 % of genuine training
windows remained `known` (no over-rejection of real exercise).

## Consequences
- **Positive:** unseen exercises are rejected as `unknown` instead of being
  confidently mislabelled; the user can record a whole multi-exercise session and
  trust the per-set labels. Decoupled from the model (separate artifact); optional
  and reproducible.
- **Negative:** one more artifact to keep in sync with `feature_names.txt` /
  the model (must re-run `fit_novelty_detector.py` if features change); the
  thresholds are a heuristic calibration (99th percentile), not learned end-to-end;
  novelty is judged purely on feature distance, so an unseen exercise whose motion
  happens to mimic a known one can still pass.
