# ADR-003: Multi-Agent Documentation Strategy

**Status:** Accepted
**Date:** 2026-05-20

## Context

The project requires continuous decision documentation, handover-ready documentation at all times, and consistent quality across ML code, documentation, and review. A single CLAUDE.md instruction file was becoming unwieldy and produced unfocused output — the AI would attempt all roles simultaneously with lower quality in each.

The project must be fully handover-ready so that a new team with no prior knowledge can understand the codebase, reproduce results, and continue development. This requires specialist focus for each type of work.

## Decision

Use multiple specialist Claude Code agent instruction files stored in `agents/`:
- `agents/data_scientist.md` — focused on ML work, feature engineering, model training, notebook authoring
- `agents/documenter.md` — focused on documentation maintenance, arc42, ADRs, CRISP-DM log, setup guides
- `agents/reviewer.md` — focused on pre-commit review using a structured checklist

## Alternatives Considered

**Single CLAUDE.md for everything:** Keep all instructions in one file and rely on Claude to select the right mode based on context. Simpler setup, but results in less focused output because the agent must juggle competing concerns (ML rigor vs. documentation completeness vs. review strictness) in every response.

**No agent specialization:** Let Claude operate without structured instruction files and rely on in-context prompts. Most flexible but produces inconsistent quality and no persistent standards.

## Rationale

Specialist agents produce more focused output with fewer competing concerns per response. The reviewer agent enforces a structured checklist that cannot be forgotten. The documenter agent maintains the handover readiness criteria continuously. The data scientist agent enforces ML best practices (no data leakage, reproducibility, result persistence) without those constraints polluting documentation tasks.

## Consequences

**Positive:**
- Each agent has clear, non-overlapping responsibilities
- Review quality improves because the reviewer checklist is explicit and structured
- Documentation stays current because the documenter has explicit handover criteria
- ML code quality improves because the data scientist has explicit ML best-practice requirements

**Negative / Trade-offs:**
- Slightly more overhead in selecting the right agent for each task
- Agent files must be kept up to date as the project evolves
- New team members must read `agents/` directory to understand available agents
