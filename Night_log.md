# Night Log — Autonomous Overnight Session (2026-06-05 → 06-06)

Worked autonomously overnight on three independent tasks, each on its **own
feature branch off `develop`**. **Nothing is merged to `main` or `develop`.**
This copy of the log lives on `feature/consolidate-decisions`; its Branch 2
section is the authoritative record for this branch.

## Overview of the three branches

| Branch | Task | Status |
|--------|------|--------|
| `feature/uncertain-overall-result` | Report **uncertain** as the overall/per-set result when it is the most common label (rest excluded) instead of forcing a minority exercise. | ✅ Done |
| `feature/consolidate-decisions` | Replace the 24 fine-grained ADRs with **one consolidated `docs/DECISIONS.md`**, and update every reference across the repo. | ✅ Done (this branch) |
| `feature/continuous-learning` | Human-in-the-loop **correction + continual-learning loop**. | ⏳ See that branch |

---

## Branch 2 — `feature/consolidate-decisions`

### Goal
You asked to drop the per-decision ADR scheme (24 files: `ADR-001…025`) in
favour of **one file with the most important decisions, not too fine-grained**,
and to update **all references** across the repo.

### What changed
- **New `docs/DECISIONS.md`** — a single living decision log, organised into
  **7 themed sections** (Tooling, ML Framework & Model, Dataset & Artifacts,
  Preprocessing & Features, Inference Safeguards, Evaluation, Documentation &
  Reproducibility). Each entry is concise (decision + why + main alternative
  rejected). Superseded choices (RecoFit/MM-Fit, subject split, undersampling)
  are summarised as history rather than dropped. Ends with a **traceability
  table** mapping every old `ADR-001…025` to its section so nothing is lost.
- **Deleted all 24 `docs/decisions/ADR-*.md`** and the now-empty
  `docs/decisions/` folder.
- **Updated every reference repo-wide** (41 files): doc prose, README, STRUCTURE,
  architecture, CRISP-DM log, data dictionary, setup guides, the 5 notebooks, and
  all `src/`, `scripts/`, `tests/` inline comments/docstrings. File-path
  references (`docs/decisions/ADR-xxx.md`, `../decisions/…`) now point to
  `docs/DECISIONS.md`; inline `ADR-0NN` tokens collapse to `DECISIONS.md` (with
  §-section pointers added by hand in the high-traffic docs).
- The bulk rewrite was done with a one-off script (`scripts/_consolidate_refs.py`)
  that was **deleted after use** — it is not part of the deliverable.

### Verification
- `grep -rIn "ADR"` over all tracked files → **only** `docs/DECISIONS.md`
  (its traceability table) still mentions the old ADR numbers, by design.
- All 5 notebooks re-validated as parseable JSON.
- `uv run ruff check` clean; `uv run pytest` → **57 passed**.

### Note for the maintainer
- `TEAM_PRAESENTATION.md` is **untracked / local-only** (it is git-ignored) and
  still explains the old ADR concept at length. The script does not touch
  untracked files, and rewriting a presentation's "what is an ADR" prose is a
  content decision for you. Update it by hand if you keep it.
- `CLAUDE.md` (also local-only) still describes the per-ADR "Decision
  Documentation Rule". If you adopt the consolidated log, update that rule to
  "add an entry to `docs/DECISIONS.md`".
- If you merge **both** Branch 1 and Branch 2: Branch 1 adds
  `docs/decisions/ADR-026-…md` (the uncertain-result decision). Its content is
  **already folded into `DECISIONS.md` §5** here, so after merging Branch 1 you
  can delete that leftover ADR-026 file.
