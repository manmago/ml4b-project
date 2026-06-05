# Night Log — Autonomous Overnight Session (2026-06-05 → 06-06)

Worked autonomously overnight on three independent tasks, each on its **own
feature branch off `develop`**. **Nothing is merged to `main` or `develop`** —
every branch is left for review. This log is committed on each branch; the
section for that branch is the authoritative record of what changed there.

## Overview of the three branches

| Branch | Task | Status |
|--------|------|--------|
| `feature/uncertain-overall-result` | Make the app report **uncertain** as the overall/per-set result when it is the most common label (rest excluded) instead of forcing a minority exercise. | ✅ Done |
| `feature/consolidate-decisions` | Replace the 25 fine-grained ADRs with **one consolidated decisions file**, and update every reference across the repo. | ⏳ See that branch |
| `feature/continuous-learning` | Add a **human-in-the-loop correction + continual-learning loop**: correct predictions in the app, persist feedback, retrain the model on base data + corrections. | ⏳ See that branch |

How to review:
```bash
git branch                       # see the three feature/* branches
git log develop..feature/<name>  # commits added by that branch
git diff develop...feature/<name>
```

---

## Branch 1 — `feature/uncertain-overall-result`

### Problem
The app could show a specific exercise as the headline result even when most
windows were classified **uncertain**. Both summary views voted over
*genuine-exercise windows only*, stripping out `uncertain`/`unknown`/`rest`
before counting — so one confident exercise window could beat a majority of
uncertain windows. Per your instruction, only `rest` should be ignored; if
`uncertain` has the highest share we must say "uncertain".

### What changed
- **`src/ml4b/data/session.py`**
  - New shared helper `dominant_label(labels, ignore={rest})`: the single most
    frequent label wins; `rest` is excluded, but `uncertain`/`unknown` are
    counted. Ties between a non-exercise output and an exercise resolve to the
    non-exercise label (never over-claim). One source of truth for both views.
  - `summarize_session` per-bout label now uses this plurality over **all** the
    bout's windows, so a mostly-uncertain set is labelled `uncertain`, not the
    minority exercise.
- **`app/pages/prediction.py`**
  - The summary metric "Top Exercise" → **"Overall Result"**, computed with
    `dominant_label` over the whole recording. It can now read *Bicep Curl*,
    *Row*, *Tricep Extension*, *Uncertain* or *Unknown* (rest excluded). Removed
    the now-unused `NON_EXERCISE_LABELS` constant.
- **`tests/test_session.py`** — added 4 tests: uncertain plurality beats a
  minority exercise; a real exercise plurality still wins; `dominant_label`
  ignores only rest; tie prefers the non-exercise label.
- **`docs/decisions/ADR-026-uncertain-as-overall-result.md`** — records the
  decision (kept the ADR convention on this branch; Branch 2 is the one that
  consolidates ADRs).

### Verification
- `uv run ruff format` + `uv run ruff check` — clean.
- `uv run pytest` — **61 passed** (10 in `test_session.py`, incl. the 4 new).

### Notes / trade-offs
- A set with a true exercise plurality but many boundary/uncertain windows can
  now read as "uncertain". That is the intended conservative behaviour.
- `rest` remains a pause and is never a "result" (matches your instruction).
