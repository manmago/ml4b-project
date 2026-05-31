# ADR-021: Leave-One-Set-Out Evaluation (Single-Subject Limitation)
**Status:** Accepted
**Date:** 2026-05-31

## Context
The Kaggle Apple-Watch anchor (ADR-016) is **single subject**: 75 set files from
one person on one watch. We need an honest estimate of how the model will do on a
*new* recording. The danger is leakage — windows from the same set (or augmented
copies of them) appearing in both train and test inflate the score, exactly the
trap that made the old MM-Fit model look like 0.96 macro F1 yet fail in reality.

## Decision
Evaluate with **leave-one-set-out cross-validation** (`LeaveOneGroupOut`,
grouping by the per-file `recording_id`) in `scripts/train_model.py`:
- For each held-out set, train on all *other* sets including their augmented
  copies, and test on **only the held-out set's original (non-augmented)
  windows**.
- Augmented copies of the held-out set are excluded from training, so there is no
  leakage through augmentation.
- Aggregate the held-out predictions into macro F1, per-class F1 and a confusion
  matrix (saved to `reports/leave_one_set_out_results.json`).

The shipped model is then retrained on **all** sets + augmentation; the honest
performance estimate is the cross-validation number, not the final model's fit.

## Alternatives Considered
- **Leave-one-subject-out (LOSO)** — the gold standard, but **impossible** here:
  there is only one subject. Documented as a limitation rather than faked.
- **Random window split / k-fold on windows** — leaks same-set (and augmented)
  windows across folds; produces dishonest, inflated scores. Rejected.
- **GroupKFold by set (k=5)** — also leak-free, but leave-one-set-out uses every
  set as a test set once and is the most thorough given only 75 groups.

## Rationale
Leave-one-set-out is the strongest leak-free estimate available for a
single-subject dataset. It honestly measures generalisation to an unseen *set*,
while being explicit that generalisation to an unseen *person* cannot be measured
from this data.

## Consequences
- **Positive:** trustworthy, leak-free metric (leave-one-set-out macro F1 ≈
  0.78); reproducible; the reported number will not collapse in deployment the
  way the old inflated score did.
- **Negative:** it still cannot estimate **cross-subject** generalisation —
  real-world performance on a *new person* is expected to be **below** this
  number. This is the project's central limitation and is documented prominently
  (README, project overview, architecture). Augmentation (ADR-019) and invariant
  features (ADR-018) are the mitigations.
