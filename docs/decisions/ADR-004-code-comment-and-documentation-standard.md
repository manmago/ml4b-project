# ADR-004: Code Comment and Documentation Standard

**Status:** Accepted
**Date:** 2026-05-20

## Context

The project must be understandable by a new team with no prior knowledge. The domain — gym exercise recognition from wrist-worn sensor time-series data — involves non-trivial signal processing concepts (windowing, feature extraction, sampling rates) and ML concepts (pipeline construction, data leakage prevention, cross-validation) that are not self-evident from code alone.

A documentation standard must be established to ensure consistent quality across all contributors and AI-assisted sessions.

## Decision

Enforce the following documentation standard on all Python code in this project:

1. **Module-level docstring** at the top of every `.py` file — explains what the file does and its role in the project pipeline
2. **Google-style docstring** on every function and class — includes Args, Returns, Raises sections
3. **Inline comments** on every non-trivial code block — explain WHAT the block does and WHY that approach was chosen
4. **Notebook markdown cells** before every code cell — plain-language explanation of what the next cell does and why

## Alternatives Considered

**Minimal comments only:** Add comments only where code is truly ambiguous, rely on well-named identifiers to convey intent. Standard in production codebases with experienced teams. Insufficient here because: (a) the team includes students with varying ML experience, (b) the university requires the project to be explainable to non-ML stakeholders, (c) AI-assisted development means code may be generated without reasoning visible in the code itself.

**README-per-module:** Write a README.md in each `src/ml4b/` subdirectory explaining the module. Supplements code but does not replace inline documentation — the reasoning for individual decisions remains invisible at the code level.

## Rationale

Gym exercise recognition involves domain-specific choices (window size, sampling rate, feature selection) whose reasoning must be transparent for academic review and for future teams continuing the work. Explicit inline comments make the decision chain visible at the point of implementation, not just in ADRs. Google-style docstrings are machine-readable and compatible with IDEs and documentation generators. The university evaluation requires the project to be explainable to non-technical stakeholders.

## Consequences

**Positive:**
- New team members can understand any code block without prior context
- University reviewers can follow the reasoning behind every technical choice
- AI-assisted sessions produce consistent documentation quality
- Handover readiness criterion 3 (understand every decision) is met at the code level

**Negative / Trade-offs:**
- More writing required per function and code block
- Comments can drift from code if not maintained — requires discipline and review enforcement
- Slightly longer files; offset by the reviewer agent catching missing documentation
