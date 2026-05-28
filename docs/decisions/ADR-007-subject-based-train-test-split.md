# ADR-007: Subject-Based Train / Validation / Test Split

**Status:** Accepted
**Date:** 2026-05-28

## Context
After feature extraction in Phase 3, the windowed feature matrix must be partitioned into train, validation, and test sets. The choice of partitioning strategy directly determines whether reported metrics reflect **real-world generalization** or merely memorisation of person-specific quirks.

The RecoFit dataset has roughly 94 subjects; each subject contributes many windows from multiple exercise sessions. Two windows from the same subject share a person-specific motion signature: handedness, joint geometry, characteristic acceleration profile, watch tightness, dominant arm. A model trained on subject S's data can learn to recognise S — and if S's windows also appear in the test set, the test metric measures *that* recognition, not the exercise classification task we actually care about.

The project's stated deployment scenario is "live prediction on a brand-new individual using an Apple Watch." Any evaluation strategy that does not simulate this scenario produces misleading numbers.

## Decision
**Subject-based splitting**: every window from a given `subject_id` is assigned in full to exactly one of `train`, `val`, or `test`. No subject ever appears in more than one split.

Implementation: `subject_based_split()` in `src/ml4b/data/splitting.py`. Default ratios:

* Train: ~70% of subjects
* Validation: ~10% of subjects
* Test: ~20% of subjects
* Random seed: 42 (project-wide reproducibility convention)

Notebook `notebooks/03_data_preparation.ipynb` cell 7 contains an explicit `isdisjoint` assertion that prevents any future regression of this invariant.

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| Random row-level splitting (`train_test_split` on rows) | Trivially balanced classes per split; standard scikit-learn workflow | **Data leakage**: same subject in train and test ⇒ inflated metrics; unrelated to deployment performance |
| Stratified random splitting (stratify by `exercise_name`) | Guarantees class balance per split | Still leaks subjects across splits — same problem as above; the apparent class balance only hides the leakage, doesn't fix it |
| **Subject-based splitting — chosen** | Honest generalization estimate; mirrors the actual deployment scenario | Slightly more variance in class balance across splits; risk of an unlucky shuffle leaving a class entirely out of one split (mitigated by validation in the notebook) |
| Leave-One-Subject-Out cross-validation | Most rigorous estimate of cross-subject generalization | ~94× the compute; overkill for Phase 3 prototyping. May revisit in Phase 5 |

## Rationale
The RecoFit paper and the broader human-activity-recognition literature unanimously use subject-disjoint evaluation. With sensor data, person-level leakage is the most common single reason for reported accuracy numbers that do not survive a demo with a new user.

Because the dataset has ~94 subjects, a 70/10/20 split leaves ~65 training subjects — enough variability for a classical model (Random Forest, SVM) to generalize, while keeping ~19 test subjects (a statistically meaningful sample size for per-class F1).

A fixed `random_state=42` ensures the split is reproducible across machines and runs, which is essential for hyperparameter tuning in Phase 4 (otherwise val-set performance differences could be noise from a different split).

## Consequences
**Positive**
* Reported metrics on the test set genuinely estimate "how well does this model work on a person we have never seen?" — the only number that matters for the Apple Watch deployment.
* Aligns directly with the two-dataset validation strategy in `docs/architecture/architecture.md` § 1.
* Removes a common silent-failure mode where good validation numbers do not survive contact with a real user.

**Negative**
* Effective training-set size is smaller than a row-level split would give (fewer subjects' data is seen in training).
* Train/val/test class balance is no longer guaranteed — an unlucky shuffle could under-represent a class in the test set. The notebook's sanity-check cell asserts that every split contains all 6 classes; if this ever fails we will either re-seed or fall back to stratified subject sampling.
* Cannot use `sklearn.model_selection.train_test_split` directly; the custom splitter must be kept in sync with the data schema.
