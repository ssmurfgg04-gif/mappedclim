# Submissions

## Current (Pipeline Locked — Bug-Fixed)

| File | Pipeline | Features | R² | Risk | Description |
|------|----------|----------|-----|------|-------------|
| `lean_submission.csv` | Phase 2 (3-model) | 67 | 0.976 | LOW | Primary — proven, fast |
| `expanded_submission.csv` | Integrated 10x (5-model) | 68 | 0.967 | MEDIUM | Secondary — hedge with interactions |

Both include:
- **Bug Fix #1**: `is_perfectly_mapped` indicator (72,260 tracts)
- **Bug Fix #2**: `building_gap` clipped to [-4, 1]
- **Deterministic fix**: Train on `gap_only`, rural penalty at inference only
- **17 interaction features**: climate/fire × gap interactions

### Inference Formula
```
final_score = model.predict(X) - 1.0 * rural_penalty
```

### Decision Rule (Aug 28)
- Submit both to public LB
- If expanded > lean → interactions are real signal
- If lean > expanded → interactions are proxy-specific, stay lean
- Retrain winner on full data for final submission

## Archive

Previous submission versions kept for reference. These are **NOT** the current locked submissions.
