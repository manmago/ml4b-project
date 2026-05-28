# ADR-006: Sliding Window Parameters (Size and Overlap)

**Status:** Accepted
**Date:** 2026-05-28

## Context
Phase 3 (Data Preparation) segments each continuous RecoFit sensor recording into fixed-length windows; one window becomes one training sample for the ML model. Two parameters define this segmentation:

* **Window size** — how many consecutive samples (and therefore how many seconds of motion) make up one training sample.
* **Overlap** — how much consecutive windows share, expressed as a fraction of window size.

Together these parameters control:

1. The temporal resolution of the resulting feature vector — too short a window cannot capture one full repetition of slow exercises; too long mixes transitions between reps and rest.
2. The number of training samples produced — overlap multiplies the sample count without requiring more raw data.
3. The amount of redundancy between adjacent samples — too much overlap and the model is effectively trained on duplicates.

A decision is needed before the windowing module (`src/ml4b/data/windowing.py`) is written, because every downstream artefact (features, splits, models, evaluation) depends on the parameter choice.

## Decision
* **Window size: 100 samples** = 2 seconds at the RecoFit sampling rate of 50 Hz.
* **Overlap: 50%** = step size of 50 samples between consecutive windows.

Implementation: defaults of `apply_sliding_window(window_size=100, overlap=0.5, sampling_rate=50)` in `src/ml4b/data/windowing.py`.

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| 1 s windows (50 samples) | More training samples; finer temporal resolution | Too short to contain one full repetition of slow exercises (squats, shoulder press) — features become noisy and class-discrimination drops |
| **2 s windows (100 samples) — chosen** | Captures one full repetition phase for typical gym exercises; matches the time scale used in related work | Slightly fewer windows per minute than 1 s |
| 4 s windows (200 samples) | Captures multiple repetitions; very smooth statistics | Mixes transitions between reps and rest — class boundaries become ambiguous, particularly for shorter recordings; fewer windows overall |
| 25% overlap | Less redundancy between consecutive samples | Only ~33% more samples than non-overlapping — leaves data on the table |
| **50% overlap — chosen** | Doubles the sample count vs non-overlapping with moderate redundancy; standard choice in human-activity-recognition literature | Adjacent windows share half their content — modest information redundancy |
| 75% overlap | 4x the sample count | Heavy redundancy; very correlated samples violate the i.i.d. assumption of classical ML evaluation more strongly |

## Rationale
2 seconds is the canonical window length in the wrist-worn HAR literature (Morris et al. 2014 — the RecoFit paper itself — and follow-up work) because it is long enough to contain one full repetition of the slowest common gym exercise (squats and shoulder presses are typically 1.5–2 s per rep at moderate cadence) yet short enough that a single window rarely contains a transition between exercise and rest.

50% overlap is the standard compromise: it doubles the number of training samples without producing the strong redundancy that 75% overlap creates, and it is the overlap used in most published HAR benchmarks, so our results will be directly comparable.

The two choices interact: a 2 s window with 50% overlap gives one new sample every second, which is also the timescale at which the Streamlit deployment will produce predictions — keeping training and inference cadence aligned.

## Consequences
**Positive**
* Approximately 2× more training samples than non-overlapping windows, improving the small-data regime of classical ML.
* Window length aligns with the natural repetition rate of most target exercises.
* Direct comparability with published HAR results.
* Inference cadence (one prediction per second) matches the Streamlit app's target UX.

**Negative**
* Adjacent windows share 50% of their raw samples, so they are not statistically independent. We mitigate the inflation this causes in evaluation metrics by combining it with **subject-based splitting** (ADR-007) — no two windows that share content can land in different splits, because they always belong to the same subject.
* Recordings shorter than 100 samples (2 s) are dropped entirely. In practice the RecoFit recordings are tens of seconds each, so this is negligible.
