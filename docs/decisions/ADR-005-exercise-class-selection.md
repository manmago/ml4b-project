# ADR-005: Final Exercise Class Selection
**Status:** Accepted
**Date:** 2026-05-16

## Context
The initial project plan defined 6 target exercise classes based on common gym exercises
(Bicep Curl, Shoulder Press, Lateral Raise, Squat, Bench Press, Deadlift).
After exploring the RecoFit dataset in Phase 2, the actual subject coverage per class
was analyzed to determine which classes have sufficient data for robust model training.

## Decision
Replace Bench Press and Deadlift with Tricep Extension as the 6th exercise class.
Final classes: bicep_curl, shoulder_press, squat, tricep_extension, lateral_raise, rest

## Alternatives Considered
1. Keep original 6 classes (including Bench Press + Deadlift) despite low participant counts
2. Combine datasets to get more data for underrepresented classes
3. Reduce to 5 classes only

## Rationale
- Bench Press (Chest Press rack) and Deadlift (Dumbbell Deadlift Row) each had only
  ~20 participants — below the 50% subject threshold visible in the coverage plot
- Tricep Extension had ~42 participants — well above threshold
- Combining datasets was rejected due to high preprocessing complexity and time cost
- Data-driven class selection is methodologically stronger than arbitrary selection

## Consequences
- Model training is more robust due to balanced class sizes (30-90 participants each)
- Tricep Extension replaces Bench Press/Deadlift in personal Apple Watch test recordings
- Business Understanding document updated to reflect data-driven rationale
