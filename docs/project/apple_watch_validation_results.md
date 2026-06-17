# Apple Watch Sanity-Check Results (one-shot, honest)

> **Purpose & method.** A one-time, no-tuning sanity check of the final 3-class
> model on the **real Apple Watch** recordings in `data/raw/apple_watch/test_samples/`.
> These files were **not** used for training or tuning, and **nothing was changed
> in response to these results** — they are reported exactly as produced. They are
> a qualitative check, not a benchmark (n = 3 recordings, single subject).
>
> Reproduce: `uv run python scripts/test_apple_watch_prediction.py` or upload each
> file on the app's Predict page.

Date: 2026-06-01 · Model: `best_model.joblib` as committed at that date (the
Kaggle-only baseline, leave-one-set-out macro F1 0.776; the current shipped model
adds our Testdaten → 0.792).

---

## Results (as produced — not optimized)

| File | True exercise | Detected rate | Windows | Top exercise (model) | Correct? | Per-window distribution | Avg conf (classified) |
|------|---------------|---------------|---------|----------------------|----------|--------------------------|-----------------------|
| `bicep_curl_sample_1.csv` | bicep curl | 100 Hz | 598 | **bicep_curl** | ✅ yes | bicep 122 · tricep 115 · row 23 · rest 265 · uncertain 73 | 0.62 |
| `bicep_curl_sample_2.csv` | bicep curl | 100 Hz | 67 | **bicep_curl** | ✅ yes | bicep 42 · tricep 2 · row 1 · rest 12 · uncertain 10 | 0.67 |
| `push_up_sample.csv` | push-up (**out of scope**) | 100 Hz | 32 | tricep_extension | ⚠️ n/a | tricep 18 · bicep 3 · rest 2 · uncertain 9 | 0.51 |

---

## Honest interpretation

- **Bicep curls are recognized correctly on real Apple-Watch data.** In both
  curl recordings the **majority exercise label is `bicep_curl`** — decisively so
  in sample 2 (42 of 67 windows). This is the key cross-recording success: a real
  Apple-Watch curl session is correctly identified.
- **Window-level bicep/triceps confusion persists** in sample 1 (122 bicep vs 115
  triceps). Both are elbow movements that look similar at the wrist — it is the
  most-confused pair in the leave-one-set-out confusion matrix too. The
  **majority vote is still correct**, but per-window confidence is moderate
  (~0.62), and a window-level timeline shows both labels.
- **Rest gating works:** 265/598 windows in sample 1 are gated as `rest` (the
  watch was idle for much of that long recording), and `uncertain` absorbs
  low-confidence windows — neither is forced into a wrong exercise class.
- **Push-ups are out of scope and behave as expected.** `push_up` is **not** a
  trained class (it is absent from the Kaggle dataset — DECISIONS.md). The model maps
  the motion to its nearest in-scope class (`tricep_extension`, which also
  involves elbow extension) at **low confidence (0.51)** with 9/32 windows
  `uncertain`. This is a faithful "I don't have this class" signal, not a claim
  that push-ups are recognized.

## What this does and does not show

- **Does show:** the Apple-Watch training domain (DECISIONS.md) transfers — real
  Apple-Watch curls are correctly identified, unlike the previous MM-Fit model;
  the rest gate and confidence threshold behave sensibly on real data.
- **Does not show:** cross-subject performance. These samples are from a single
  user; with only 3 recordings this is a qualitative check. As documented in
  DECISIONS.md, true cross-person accuracy is expected to be **below** the
  leave-one-set-out macro F1 of 0.776 and cannot be measured from the
  single-subject anchor.

No parameters, thresholds, features, or model settings were changed in response
to these results, per the project's honesty rule.
