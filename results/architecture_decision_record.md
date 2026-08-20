# Architecture Decision Record: ADR-001

**Title:** Retire ML Pipeline for proxy_merged — Target is Deterministic Formula  
**Status:** ACCEPTED  
**Date:** 2026-08-14

---

## Decision

**Retire all ML models for `proxy_merged` prediction. Report the exact formula instead.**

## The Formula

```
proxy_merged = -(bldg_gap_clip + 2·area_gap_clip + road_gap_clip + poi_gap_clip) / 4 + pct_urban - 1
```

**Verification:** R² = 1.0, RMSE = 0.0, max_error = 0.0 on all 85,396 tracts.

## Context

The target variable `proxy_merged` is a **self-constructed proxy** (not the real competition target `coverage_gap_score`). It was defined in `scripts/pipeline_merged_phase1.py` as:

```python
proxy_merged = -np.mean([
    np.maximum(0, bg),                    # bldg_gap_clip
    2.0 * np.maximum(0, building_area_gap), # 2 * area_gap_clip
    np.maximum(0, road_gap),               # road_gap_clip
    np.maximum(0, poi_gap_corr)            # poi_gap_clip
], axis=0) - 1.0 * (1 - pct_urban).clip(0, 1)
```

All 5 components are present in the training feature matrix. The project's own formula decoder (`formula_decoder_1M.json`) verified this with `rmse=0.0, max_error=0.0, verified=True`. The methodology doc states: *"If exact formula found, skip ML entirely."* — but the project continued building ML models anyway.

## Key Quantitative Findings

| Test | R² | Interpretation |
|------|----|----|
| OLS on 5 formula features | **1.0000000000** | Exact reconstruction |
| `pct_urban` alone | **0.9987** | Target ≈ pct_urban - 1 |
| 4 gap features (no pct_urban) | **0.1646** | Gaps add 0.13% over pct_urban |
| Non-formula features (SVI, CVI, tribal) | **0.9987** | Only work via rural/urban proxy |

**85% of tracts** (72,566 / 85,396) have all gap clips < 0.001, making `proxy_merged ≈ pct_urban - 1` effectively exact.

## Consequences

### Retired Artifacts
- All ML model `.pkl` files
- Two-stage classifier→regressor pipeline
- 50+ feature engineering pipeline for proxy_merged

### Invalidated Results
- R² = 0.9998 → circular (formula inversion, not learning)
- SHAP importances → measuring formula contribution, not learned patterns
- Ensemble disagreement → measuring formula approximation error, not predictive uncertainty
- Spatial CV → formula is geography-independent
- Tribal bias ratio → property of formula + data distribution, not model bias

### Valid Results (Still Scientifically Valuable)
- Formula discovery: exact formula with zero error
- `pct_urban` dominance: 99.87% of variance
- Sensitivity ranking: road_gap > area_gap > bldg_gap > poi_gap
- 85% trivial-gap observation

## Next Steps

1. **Report formula** as the solution for proxy_merged
2. **Add leakage guard** to all training scripts (check feature→target circularity)
3. **When real target released:** Test if same formula applies to `coverage_gap_score`
4. **If different formula:** Rebuild with non-leakage features only
5. **Pivot ML effort:** Predict gaps (building_gap, road_gap, poi_gap) from non-gap features (SVI, CVI, hazards, climate)
