# ADR-020: Confidence Threshold → "uncertain" Output
**Status:** Accepted
**Date:** 2026-05-31

## Context
The model is a 3-class classifier; `predict` always returns one of the three
exercises, even for a window that is none of them (an exercise outside scope, a
transition, or noisy motion that passed the activity gate). On real uploads this
produces confident-looking but wrong labels and erodes trust in the app.

## Decision
Apply a confidence threshold in the inference pipeline
(`CONFIDENCE_THRESHOLD = 0.50` in `src/ml4b/data/canonical.py`). For each active
window, take the model's top class probability; if it is below the threshold,
output `uncertain` instead of forcing one of the three classes. `uncertain` is
**not** a trained class — it is a post-hoc abstention.

## Alternatives Considered
- **Always output the argmax class** — current default; produces overconfident
  wrong predictions on out-of-scope motion. Rejected.
- **Add a trained "other" class** — needs labelled out-of-scope examples we do
  not have, and would not cover the open-ended space of non-target motions.
- **Higher threshold (e.g. 0.7)** — abstains too often given only three classes
  (chance level is 0.33); 0.50 is a balanced starting point and is a single,
  documented constant that can be tuned.

## Rationale
Abstaining when unsure is the honest behaviour for a 3-class model deployed on
open-ended real motion. A probability threshold needs no extra data, is
transparent, and is trivial to adjust. Combined with the activity gate (ADR-017),
the model only commits to a class on movement that is both energetic *and*
confidently one of the three.

## Consequences
- **Positive:** fewer confidently-wrong labels; clearer UX (the app shows
  `uncertain`); tunable via one constant.
- **Negative:** some genuine exercise windows near a decision boundary are
  reported `uncertain`, slightly lowering apparent recall; the threshold value is
  a heuristic, not learned.
