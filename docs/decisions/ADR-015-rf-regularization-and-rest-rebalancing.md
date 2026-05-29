# ADR-015: Random Forest Regularization and `rest` Rebalancing

**Status:** Accepted
**Date:** 2026-05-29
**Deciders:** Anshul Agrawal

---

## Context

Two problems were observed with the first MM-Fit model (ADR-013):

1. **Over-confident / over-fit trees.** The Random Forest reached **macro F1
   ≈ 1.00 on TRAIN** while validation was 0.88, with individual trees grown to
   depth ~28 (`max_depth=None`). Full-depth trees memorize the training set,
   which produces over-confident probabilities (shown as "confidence" in the
   app) and tends to generalize worse across devices.

2. **`rest` over-prediction on real Apple Watch data.** On real recordings,
   exercise windows were frequently labelled `rest` (e.g. a bicep-curl
   recording predicted 75% `rest`). `rest` was still the single largest training
   class (capped at 2× the largest exercise class, ADR-008), so the model's
   `rest` region in feature space was disproportionately large.

## Decision

1. **Regularize the Random Forest** (`src/ml4b/models/train.py`):
   `n_estimators` 200 → **300**, `max_depth` None → **20**,
   `min_samples_leaf` 2 → **4** (keep `min_samples_split=5`,
   `class_weight='balanced'`).
2. **Rebalance `rest`** (`scripts/build_mmfit_dataset.py`): undersample
   multiplier 2.0 → **1.5**, bringing `rest` close to the largest exercise class.

## Alternatives Considered

| Option | Result |
|--------|--------|
| **mult=1.5 + depth≤20 + leaf=4 (chosen)** | Val 0.866 / Test 0.944; real Apple Watch `rest` dropped (sample_1 75→66%, sample_2 37→33%, push_up 25→19%). |
| Keep mult=2.0, no depth cap | Val 0.880 / Test 0.961 but full-depth trees and the most `rest` over-prediction. |
| mult=1.0 (fully balanced) | Cut `rest` more but dropped Val to 0.853 and hurt `squat`/`push_up` in-domain — too aggressive. |
| Deeper regularization (depth 12, leaf 10) | No validation gain — RF fits these features near-perfectly regardless; only added bias. |

## Rationale

A grid search on the validation set showed depth/leaf limits cost almost no
validation F1, so capping depth is a low-risk way to reduce memorization and
improve probability calibration. Reducing the `rest` cap to 1.5× measurably
lowers `rest` over-prediction on real Apple Watch recordings while keeping every
in-domain class ≥ 0.84 F1. mult=1.0 was rejected as too aggressive (it hurt
minority exercises).

**Honest note on "overfitting":** a Random Forest scoring ~1.0 on TRAIN is
normal — RF fits separable tabular features almost perfectly. The healthy
held-out **Test macro F1 = 0.944** shows generalization within MM-Fit is sound;
the Val/Test spread reflects which workouts are harder, not data leakage (splits
are by workout). The regularization here is about calibration and cross-device
robustness, not a broken model.

## Consequences

**Positive:**
- Trees capped at depth 20; less memorization, better-calibrated confidence.
- `rest` over-prediction on real Apple Watch data reduced across all 3 samples.
- All 7 classes still ≥ 0.84 F1 on the held-out test set.

**Negative / Trade-offs:**
- Small in-domain cost: Val 0.880 → 0.866, Test 0.961 → 0.944.
- `rest` is still the top prediction on the gentlest real recording
  (`bicep_curl_sample_1`, ~66%), which genuinely contains long rest periods; the
  residual bicep/tricep confusion (ADR-013/014) is unchanged by this tuning.
