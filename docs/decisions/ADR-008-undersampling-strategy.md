# ADR-008: Class Imbalance — Undersampling Strategy

**Status:** Accepted
**Date:** 2026-05-28

## Context

After Phase 3 windowing, the class distribution was severely imbalanced:

| Class | Windows | Share |
|-------|---------|-------|
| rest | 138,220 | 88.8% |
| squat | 5,746 | 3.7% |
| bicep_curl | 3,749 | 2.4% |
| shoulder_press | 3,402 | 2.2% |
| tricep_extension | 3,130 | 2.0% |
| lateral_raise | 1,351 | 0.9% |

The imbalance arises because participants spend significantly more time resting than exercising. Without correction, a model predicting `rest` for every window would achieve ~89% accuracy — making accuracy a misleading metric and the model useless in practice (the app is supposed to detect *which exercise* is being performed, not that the user is mostly at rest).

This requires a decision before Phase 4 modelling begins, because:
1. Which metric to use as primary (accuracy vs. macro F1).
2. Whether to correct the imbalance in the data, in the model, or both.
3. Whether to apply correction to all splits or only training.

## Decision

Three-part fix applied together:

1. **Undersample `rest` in the training set**: cap `rest` at `2×` the size of the largest exercise class (`squat`, ~5,746 windows). Implementation: `undersample_majority_class()` in `src/ml4b/data/splitting.py`.
2. **Val and test sets are left unchanged** to reflect real-world class distribution.
3. **Use `class_weight='balanced'` in all Phase 4 models** as a second safeguard during training.
4. **Use macro-averaged F1 as the primary evaluation metric** — it weights all classes equally regardless of support.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| No undersampling — rely only on `class_weight='balanced'` | Simple; no data discarded | Insufficient for 89:1 imbalance; training loss is still dominated by the majority class |
| Oversample minority classes (SMOTE) | No data discarded; increases minority count | SMOTE on statistical/FFT features can create unrealistic synthetic samples; adds complexity |
| Undersample `rest` to match the **smallest** class (lateral_raise, 1,351) | Perfectly balanced training data | Loses ~96% of rest data; model unlikely to learn rest pattern well enough for real-world use |
| Remove `rest` class entirely | Perfectly balanced among exercise classes | Makes the app useless — rest detection is critical for the deployment scenario (Apple Watch on a real workout) |
| **Cap at `2× largest minority` (chosen)** | Preserves enough rest data; drastically reduces dominance | Training set is smaller than without undersampling; slight information loss from majority class |

## Rationale

- A multiplier of 2.0 targets a roughly 2:1 imbalance ratio (rest vs. next largest class), which is manageable with `class_weight='balanced'` as a second safeguard.
- Matching the smallest class (1,351) would discard ~99% of rest windows — rest detection robustness would suffer.
- Keeping val and test unchanged ensures that evaluation metrics represent the real-world scenario: most inference windows will indeed be `rest`, and the model must handle that gracefully.
- Macro-averaged F1 is unaffected by class support differences, so it gives an honest view of per-class performance regardless of the imbalance in the test set.

## Consequences

**Positive**
- Training set is far more balanced — a model can no longer trivially win by predicting `rest`.
- Val/test metrics reflect real-world distribution → honest generalization estimate.
- Macro F1 becomes a meaningful primary metric for Phase 4 and Phase 5.

**Negative**
- Training set is smaller (~11,500 + minority rows vs. ~155,000 original).
- The undersampling is non-deterministic between different random seeds — must use `random_state=42` consistently for reproducibility.
- If the RecoFit data changes (e.g. more subjects added), the cap must be recomputed.
