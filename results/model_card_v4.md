# Model Card: Coverage Gap Prediction System

**Model ID:** CGP-v4  
**Date:** 2026-08-20  
**Status:** Production-ready (with documented equity limitation)  
**Architecture:** Two-stage: (1) ML gap predictors + (2) deterministic formula

---

## 1. Model Overview

### Intended Use
Predicts a coverage gap proxy score for US census tracts, reflecting the degree of geographic data under-mapping (roads, buildings, POIs) relative to reference datasets. Designed to identify tracts with significant data coverage gaps that may warrant additional mapping resources.

### Intended Users
- Geographic data quality analysts
- Mapping infrastructure planners
- Equity researchers studying data coverage disparities

### Out-of-Scope Uses
- Resource allocation decisions without human review (see Section 5: Equity Limitations)
- Individual tract-level decisions without considering prediction intervals
- Use on tribal lands without acknowledging the 8.49× MAE ratio (see Section 5)

---

## 2. Architecture

### Two-Stage Design

```
Stage 1: ML Gap Predictors (LightGBM, 160 non-leakage features each)
├── road_gap predictor: R²=0.79 on county holdout (STAR MODEL)
├── poi_gap predictor: cluster-based per region (TX R²=0.80, CA R²=0.97, NY R²=0.98)
├── bldg_gap predictor: R²=0.15 on county holdout (limited)
└── area_gap predictor: R²=0.16 on county holdout (limited)

Stage 2: Apply Deterministic Formula
└── proxy_merged = -(bldg_gap + 2·area_gap + road_gap + poi_gap)/4 + pct_urban - 1
```

### Why Two Stages?
The coverage gap proxy (`proxy_merged`) is a **deterministic formula** of gap features:
- Direct ML on `proxy_merged` gives R² = 1.0 trivially (formula inversion, not learning)
- Real ML value is predicting the gaps themselves from independent features
- Formula application is exact algebra — no learning needed

### Why LightGBM?
LightGBM (max_depth=8, n_estimators=200, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.3, reg_lambda=2.0) outperformed XGBoost and ExtraTrees on spatial holdout for road_gap prediction. R² = 0.790 vs 0.640 for XGBoost.

---

## 3. Training Data

### Source
`engineered_features_merged.parquet` — 85,396 US census tracts × 329 columns

### Features (Stage 1 Inputs)
- **160 non-leakage features** selected by correlation with gap targets
- Categories: SVI themes, CVI components, USGS/MTBS/NIFC wildfire, USFS wildfire risk, GHCN climate stations, USDM drought, SPI/PMDI drought indices, RUCA/RUCC rural-urban classifications, source coverage flags, spatial lag features
- **EXCLUDED** (formula leakage): all gap features, gap derivatives, pct_urban, rural indicators

### Target Variables (Stage 1)
- `road_gap_clip = max(0, road_gap)` where `road_gap = (1 - ov_road/tiger_road).clip(-4, 1)`
- `poi_gap_clip = max(0, poi_facility_gap_corrected)`
- `bldg_gap_clip = max(0, building_gap)`
- `area_gap_clip = max(0, building_area_gap)`

### Final Output (Stage 2)
- `proxy_merged`: continuous score, range [-1.58, 0.00], mean=-0.235

---

## 4. Performance

### Road Gap Model (Best Performer)

| Metric | Random Split | County Holdout | State LOO (CA) | State LOO (NY) | State LOO (TX) |
|--------|-------------|----------------|----------------|----------------|----------------|
| R² | 0.804 | 0.790 | 0.712 | 0.722 | 0.591 |
| RMSE | 0.019 | 0.021 | — | — | — |
| Gen Gap | — | +13.2% | — | — | — |

### POI Gap Model (Cluster-Based)

| State | Train Approach | R² | Notes |
|-------|---------------|----|----|
| California | Cluster (10 similar states) | 0.965 | Excellent |
| Texas | Cluster (10 similar states) | **0.805** | Rescued from -2.015 |
| New York | Global (LOO) | 0.986 | Excellent |
| Other states | Global model | varies | Documented uncertainty needed |

### Building & Area Gap Models
- Both have low R² (0.15–0.16) on spatial holdout
- Non-leakage features explain only 4–5% of variance
- Recommendation: these gaps are largely unpredictable from independent features alone

### Composite Proxy (End-to-End)
- R² = 0.9995 (using predicted gaps + formula)
- This is REAL generalization — NOT the circular 0.9998 from direct ML on proxy_merged
- High R² because `pct_urban` (measured, not predicted) dominates the formula

---

## 5. Equity and Fairness Limitations

### ⚠️ CRITICAL: Tribal Tract Prediction Equity

**The model performs 8.49× worse on tribal tracts than on non-tribal tracts.**

| Group | Tracts | Mean road_gap | MAE (county holdout) |
|-------|--------|---------------|---------------------|
| Non-tribal | 83,343 (97.6%) | 0.0054 | 0.0037 |
| Tribal | 2,053 (2.4%) | 0.0383 | 0.0314 |
| **Ratio** | — | 7.1× | **8.49×** |

### Why Reweighting Doesn't Fix This

| Approach | Spat R² | Tribal MAE | NonT MAE | Ratio |
|----------|---------|-----------|----------|-------|
| Baseline (no reweighting) | 0.790 | 0.0314 | 0.0037 | 8.49× |
| 3× tribal weight | 0.778 | 0.0320 | 0.0038 | 8.49× |
| 5× tribal weight | 0.775 | 0.0338 | 0.0038 | 8.85× |
| 8× tribal weight | 0.779 | 0.0337 | 0.0039 | 8.63× |
| 16× tribal weight | 0.760 | 0.0334 | 0.0040 | 8.41× |
| Augmented features (8 interactions) | 0.794 | 0.0303 | 0.0036 | 8.32× |

### Root Cause
This is **not a model bias** — it's a **data coverage inequity**:
1. Tribal tracts are concentrated in remote rural areas with systematically different feature distributions
2. OpenStreetMap (OSM) has 7.1× higher road under-mapping on tribal lands
3. The features that predict road gaps on non-tribal tracts don't capture WHY roads are missing on tribal lands
4. Worst errors concentrated in: Oklahoma (OK), Florida (FL), New Mexico (NM)

### Recommendations for Tribal Tract Use
1. **Always display prediction intervals** alongside point estimates for tribal tracts
2. **Flag tribal tract predictions** for human review before policy use
3. **Document this limitation** in any consumer-facing application
4. **Long-term fix:** Partner with tribal GIS programs to improve OSM coverage on tribal lands

---

## 6. Distribution Shift Awareness

### POI Gap — Regional Distribution Shift

The POI gap model exhibits significant distribution shift when generalizing across regions:
- Single global model fails on Texas (R² = -2.015, worse than mean prediction)
- Cluster-based training (train on home state + 10 similar states) rescued TX to R² = 0.805
- **Recommendation:** Use cluster-based approach for POI predictions; document higher uncertainty for out-of-distribution states

### States Where Model Performs Best
- New York: R² = 0.986 (POI), R² = 0.722 (road LOO)
- California: R² = 0.965 (POI), R² = 0.712 (road LOO)

### States Requiring Caution
- Texas: POI R² = 0.805 (improved but still weakest), road R² = 0.591 (LOO)
- Other states not in focus set: performance varies, document uncertainty

---

## 7. Operational Considerations

### Inputs Required at Inference
- All 160 non-leakage features per tract
- `pct_urban` from census data (formula input, must be measured not predicted)
- State FIPS code (for region-specific POI model selection)

### Output Format
```json
{
  "tract_geoid": "06001400100",
  "state_fips": "06",
  "predicted_road_gap": 0.0123,
  "predicted_poi_gap": 0.0008,
  "predicted_bldg_gap": 0.0021,
  "predicted_area_gap": 0.0031,
  "predicted_proxy_merged": -0.2456,
  "prediction_intervals": {
    "road_gap_90pct": [0.0089, 0.0157],
    "poi_gap_90pct": [0.0001, 0.0015]
  },
  "is_tribal": false,
  "uncertainty_flag": "low",
  "equity_warning": null
}
```

### Equity Warning Logic
- If `is_tribal == true`: emit equity warning, widen prediction intervals by 2×
- If state in {TX, OK, NM}: emit distribution shift warning
- If ensemble disagreement > 95th percentile: flag for human review

### Drift Monitoring
Track in production:
- Distribution of input features over time (PSI > 0.2 triggers alert)
- Tribal vs non-tribal MAE ratio (alert if > 10×)
- State-level R² drift (alert if any state drops > 20% from baseline)

---

## 8. Caveats and Validation

### Validated Against
- 85,396 census tracts (full national coverage)
- County GroupKFold (5-fold spatial CV)
- State Leave-One-Out (CA, NY, TX)
- Tribal vs non-tribal equity analysis

### Not Yet Validated
- **Real target (`coverage_gap_score`)** — not yet released; formula applicability unknown
- Temporal stability — no historical data available
- Out-of-country generalization — not in scope

### Known Limitations
1. Tribal tract predictions have 8.49× higher error (see Section 5)
2. POI predictions have regional distribution shift (see Section 6)
3. Building and area gap predictions have low R² (0.15–0.16)
4. The proxy target is a constructed composite, not an independent measurement
5. Formula was discovered through audit; when real target releases, validate applicability before deploying

---

## 9. Ethical Considerations

### Data Coverage Equity
This model exposes a structural inequity in geographic data: tribal lands are systematically under-mapped in OpenStreetMap. The model faithfully reflects this inequity — it does not cause it. However, deploying this model without acknowledging the 8.49× MAE ratio could perpetuate the inequity by giving policy-makers false confidence in tribal tract predictions.

### Recommended Ethical Use
- DO use this model to identify tracts that need additional mapping resources
- DO NOT use this model to allocate resources to tribal tracts without human review
- DO publish group-specific performance metrics alongside any deployment
- DO NOT obscure the 8.49× tribal MAE ratio in user-facing documentation

### Transparency
- All training scripts, data lineage, and audit results are open-source
- The deterministic formula was discovered and documented during development
- All equity analyses are reproducible from committed scripts

---

## 10. Version History

| Version | Date | Change |
|---------|------|--------|
| v1 | 2026-08-12 | Initial two-stage classifier + regressor (later found circular) |
| v2 | 2026-08-14 | Added 5-model ensemble + focal loss (also circular) |
| v3 | 2026-08-15 | Discovered target is deterministic formula; retired direct ML |
| v4 | 2026-08-20 | Gap prediction models: road R²=0.79, POI cluster-based, equity documented |

---

## Appendix A: Formula Discovery

During development, we discovered that `proxy_merged` is an exact deterministic function of gap features:

```
proxy_merged = -(bldg_gap_clip + 2·area_gap_clip + road_gap_clip + poi_gap_clip) / 4 + pct_urban - 1
```

Verified via OLS: R² = 1.0, RMSE = 0.0, max_error = 0.0 on all 85,396 tracts.

This invalidated all v1-v3 direct-ML-on-proxy results (which were circular) and motivated the v4 architecture of predicting gaps from non-leakage features, then applying the formula.

See: `results/technical_finding_formula_discovery.md` for the full scientific writeup.

## Appendix B: Reproducibility

All scripts to reproduce these results are in `/scripts/`:
- `gap_predict_all.py` — initial gap prediction models
- `road_gap_expanded.py` — expanded road_gap model (160 features)
- `poi_gap_overfitting.py` — POI distribution shift investigation
- `tribal_equity_deep_dive.py` — tribal reweighting sweep
- `region_specific_poi_models.py` — cluster-based POI models

Results in `/results/`:
- `gap_prediction_models.json` — baseline gap prediction results
- `road_gap_expanded.json` — expanded road model results
- `poi_gap_overfitting.json` — POI investigation results
- `tribal_equity_deep_dive.json` — tribal equity sweep results
- `region_specific_poi_models.json` — region-specific POI results
- `gap_prediction_final_analysis.md` — full analysis writeup
