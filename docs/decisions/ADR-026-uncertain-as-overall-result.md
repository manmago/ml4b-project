# ADR-026: "uncertain" Can Be the Overall / Per-Set Result
**Status:** Accepted
**Date:** 2026-06-05

## Context
The app summarises a recording two ways: a whole-recording **Overall Result**
metric and a per-set **Detected Sets** table. Both previously reported a label by
voting over **genuine-exercise windows only** — `rest`, `uncertain` and `unknown`
were stripped out before the vote (`prediction.py` used `mode()` over exercises;
`session.py` voted over exercise labels and only fell back to a non-exercise
label when a bout had *zero* exercise windows).

The consequence was dishonest: a recording (or set) where the model output
`uncertain` for the large majority of windows but `bicep_curl` for one confident
window would still be summarised as **Bicep Curl**. The headline result claimed a
specific exercise the model was, window-for-window, mostly unsure about — exactly
the over-claiming the confidence threshold (ADR-020) and novelty gate (ADR-024)
exist to prevent.

## Decision
Summaries use a single shared **plurality rule** (`session.dominant_label`): the
most frequent label wins, with **`rest` excluded** (it is a pause, not a result)
but **`uncertain` and `unknown` counted**. So:
- if a genuine exercise is the plurality → that exercise is reported;
- if `uncertain`/`unknown` is the plurality → that is reported honestly;
- if everything is `rest` → no result (`—` / empty).

Ties between a non-exercise output and an exercise are broken in favour of the
non-exercise label (never over-claim an exercise). Both the Overall Result metric
and the per-bout label call the same function, so the two views cannot disagree.

## Alternatives Considered
- **Status quo (vote over exercises only)** — over-reports an exercise whenever
  any exercise window exists, even amid mostly-uncertain windows. The bug we are
  fixing. Rejected.
- **Exclude `rest` *and* `unknown`, keep only exercises vs `uncertain`** — would
  hide genuinely out-of-distribution sets; `unknown` is a meaningful honest
  result and should be able to win. Rejected.
- **Confidence-weighted vote** — weight each window by its probability. More
  complex, and `uncertain`/`unknown` windows have no comparable confidence, so it
  reintroduces the bias toward the few confident exercise windows. Rejected for
  the transparent count-based rule.

## Consequences
- **Positive:** headline and per-set results are honest — the app says
  "uncertain" when it is uncertain instead of guessing an exercise; one shared
  rule keeps the Overall Result and Detected Sets consistent.
- **Negative:** a set with a true exercise plurality but many boundary/uncertain
  windows can read as "uncertain"; this is the intended, conservative trade-off.
  The plurality threshold is implicit (most frequent) and not tunable without a
  code change.
