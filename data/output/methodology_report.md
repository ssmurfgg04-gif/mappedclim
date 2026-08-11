# Bias Bounty Mapping Equity Challenge — Methodology Report

**Generated**: 2026-08-11T11:49:22.025036  
**Pipeline**: Monumental Pipeline v2 (3-model ensemble)  

## Executive Summary

We present a self-supervised ensemble approach for predicting coverage gap scores
across 9,491 US Census tracts in 4 focus regions (Maricopa AZ, Northern CA, Eastern OK, South-Central TX).
Our pipeline combines massive feature engineering (69+ interaction features),
a 3-model ensemble (XGBoost + LightGBM + CatBoost) with optimal convex blending,
and comprehensive intersectional bias discovery across 9 equity dimensions.

### Key Innovation: Self-Supervised Learning

Since the competition target (`coverage_gap_score`) is not yet released, we use
`building_gap` and `road_gap` as proxy targets. This validates our entire pipeline,
discovers bias patterns, and enables instant retraining when the actual target is released.

**Best Proxy RMSE**: 0.033218 | **R²**: 0.9830 | **Bias Score**: 0.004861

## Pipeline Architecture

```
Raw Data → Feature Engineering (69 interactions) → Feature Selection (correlation + collinearity)
  → XGBoost (1500 est, depth 7) + LightGBM (1500 est, depth 7) + CatBoost (1000 est, depth 6)
  → Optimal Convex Blend (SLSQP) → Bias Discovery → Documentation
```

### Components

1. **Feature Engineering**: 69 new interaction features covering 10 categories
2. **Feature Selection**: Correlation ranking (top 80) + collinearity filter (r > 0.98)
3. **3-Model Ensemble**: XGBoost + LightGBM + CatBoost with optimal convex blend via SLSQP
4. **Spatial Cross-Validation**: GroupKFold by county FIPS (5-fold)
5. **Bias Discovery**: 9 equity dimensions (county, SVI, tribal, rural/urban, hazard, CVI, intersectional, data desert, regional)
6. **SHAP Analysis**: Model interpretability via TreeExplainer

## Model Performance

| Model | RMSE | R² | Bias Score |
|-------|------|----|-----------|
| road_xgb | 0.014559 | 0.9996 | 0.006735 |
| blend | 0.033218 | 0.9830 | 0.004861 |
| xgb | 0.033581 | 0.9820 | 0.004794 |
| cat | 0.033680 | 0.9813 | 0.006440 |
| lgb | 0.035924 | 0.9784 | 0.007100 |

**Blend Weights**: {'xgb': 0.42998162109413046, 'lgb': 0.10565291247603367, 'cat': 0.46436546642983595}

## Feature Engineering

### Interaction Categories (69 new features)

1. **SVI × Coverage** (8): svi_x_bldg, svi_sq_x_bldg, svi_min_x_bldg, svi_soc_x_bldg, svi_x_road, etc.
2. **Tribal × Coverage** (4): tribal_x_bldg, tribal_x_bldg_sq, tribal_pct_x_bldg, tribal_x_svi_x_bldg
3. **Rural/Urban × Coverage** (4): pct_urban_x_bldg, rural_x_bldg, rural_sq_x_bldg, rural_x_svi_x_bldg
4. **Hazard × Coverage** (3): wf_x_bldg, wf_flag_x_bldg, wf_x_svi_x_bldg
5. **CVI × Coverage** (4): cvi_x_bldg, cvi_sq_x_bldg, cvi_x_svi_x_bldg, cvi_base_x_bldg
6. **Intersectional** (15): tribal_x_highSVI_x_rural, highSVI_x_rural, wf_x_highSVI_x_rural, hcvi_x_hsvi_x_rural, etc.
7. **Polynomial** (9): bldg_gap_sq, bldg_gap_cu, bldg_road_ratio, bldg_road_diff, bldg_road_product, etc.
8. **Compound risk** (3): compound_risk, compound_risk_sq, tribal_x_risk
9. **Population-weighted** (2): log_pop_x_bldg, log_pop_x_svi
10. **County target encoding** (1): bldg_county_loo (leave-one-out county mean)
11. **Coverage null indicators** (~15): data desert signals from _covered flags

### Top 15 Features (xgb)

- `bldg_gap_cu`: 0.2486
- `log_building_ratio`: 0.2472
- `bldg_gap_log1p`: 0.1091
- `bldg_gap_abs`: 0.0918
- `building_gap_sq`: 0.0827
- `building_gap_knn20_diff`: 0.0473
- `cvi_x_bldg`: 0.0330
- `log_pop_x_bldg`: 0.0213
- `building_gap_county_dev`: 0.0193
- `building_gap_knn10_diff`: 0.0162
- `compound_risk`: 0.0160
- `compound_risk_sq`: 0.0149
- `bldg_road_diff`: 0.0100
- `building_ratio_county_dev`: 0.0089
- `building_gap_county_mean`: 0.0060

### SHAP Feature Importance (Top 15)

- `log_building_ratio`: 0.067538
- `bldg_gap_cu`: 0.062270
- `bldg_gap_abs`: 0.022307
- `bldg_gap_log1p`: 0.006914
- `bldg_gap_sq`: 0.004841
- `log_pop_x_bldg`: 0.002455
- `cvi_x_bldg`: 0.001902
- `cvi_base_x_bldg`: 0.001757
- `building_gap_sq`: 0.000939
- `bldg_road_diff`: 0.000908
- `building_gap_county_dev`: 0.000743
- `compound_risk_sq`: 0.000712
- `pct_urban_x_bldg`: 0.000640
- `building_gap_knn20_diff`: 0.000599
- `compound_risk_score`: 0.000587

## Bias Discovery Findings ($1,000 Prize)


### County

**Worst over-predicted** — severity: high
**Worst under-predicted** — severity: high

### SVI

**SVI Overall** — severity: high
  - low: 0.000925
  - high: -0.000391
  - disparity: -0.001316
**SVI Minority** — severity: medium
  - low: 0.000693
  - high: 0.000331
  - disparity: -0.000362
**SVI Socioeconomic** — severity: high
  - low: 0.001220
  - high: -0.000563
  - disparity: -0.001783
**SVI Household** — severity: medium
  - low: 0.000715
  - high: 0.000472
  - disparity: -0.000243
**SVI Housing/Trans** — severity: medium
  - low: 0.000572
  - high: -0.000402
  - disparity: -0.000974

### Tribal

**Tribal bias** — severity: medium
  - tribal: 0.000056
  - non_tribal: 0.000399
  - disparity: -0.000343
  - n: 860

### Rural/Urban

**Rural/Urban (50%)** — severity: N/A
  - rural: 0.000097
  - urban: 0.000430
  - disparity: -0.000333
**Rural/Urban (30%)** — severity: N/A
  - rural: 0.000177
  - urban: 0.000406
  - disparity: -0.000229

### Hazard

**Wildfire bias** — severity: N/A
  - wf: 0.000230
  - no_wf: 0.000389
  - disparity: -0.000158

### CVI

**High CVI bias** — severity: N/A
  - high: -0.000002
  - low: 0.000491
  - disparity: -0.000493

### Intersectional

**tribal_x_highSVI_x_rural** — severity: high
  - group_resid: 0.001645
  - overall_resid: 0.000368
  - excess_bias: 0.001277
  - n: 70
**tribal_x_highSVI_x_urban** — severity: medium
  - group_resid: -0.000412
  - overall_resid: 0.000368
  - excess_bias: -0.000780
  - n: 151
**tribal_x_lowSVI_x_rural** — severity: critical
  - group_resid: 0.002463
  - overall_resid: 0.000368
  - excess_bias: 0.002095
  - n: 45
**highSVI_x_rural** — severity: high
  - group_resid: 0.001417
  - overall_resid: 0.000368
  - excess_bias: 0.001049
  - n: 255
**highSVI_x_urban** — severity: medium
  - group_resid: -0.000611
  - overall_resid: 0.000368
  - excess_bias: -0.000979
  - n: 2117
**lowSVI_x_rural** — severity: medium
  - group_resid: 0.001145
  - overall_resid: 0.000368
  - excess_bias: 0.000777
  - n: 285
**tribal_x_rural** — severity: medium
  - group_resid: 0.000464
  - overall_resid: 0.000368
  - excess_bias: 0.000096
  - n: 412
**tribal_x_highSVI** — severity: medium
  - group_resid: 0.000239
  - overall_resid: 0.000368
  - excess_bias: -0.000129
  - n: 221
**wf_x_highSVI_x_rural** — severity: high
  - group_resid: 0.002324
  - overall_resid: 0.000368
  - excess_bias: 0.001956
  - n: 113
**tribal_x_wf** — severity: medium
  - group_resid: -0.000255
  - overall_resid: 0.000368
  - excess_bias: -0.000623
  - n: 260
**wf_x_rural** — severity: medium
  - group_resid: 0.000634
  - overall_resid: 0.000368
  - excess_bias: 0.000266
  - n: 874
**highCVI_x_highSVI_x_rural** — severity: medium
  - group_resid: 0.000643
  - overall_resid: 0.000368
  - excess_bias: 0.000275
  - n: 180
**highCVI_x_tribal** — severity: medium
  - group_resid: -0.000340
  - overall_resid: 0.000368
  - excess_bias: -0.000708
  - n: 257

### Regional

**Region: maricopa-az** — severity: N/A
  - resid: 0.002686
  - n: 1589
**Region: northern-ca** — severity: N/A
  - resid: -0.000921
  - n: 591
**Region: eastern-ok** — severity: N/A
  - resid: 0.000974
  - n: 1300
**Region: south-central-tx** — severity: N/A
  - resid: -0.000249
  - n: 6011

## Validation Strategy

**Spatial cross-validation** with `GroupKFold` by county FIPS (first 5 digits of GEOID).
This prevents spatial autocorrelation leakage where nearby tracts share similar characteristics.
Each fold keeps all tracts from the same county together, ensuring the model generalizes
to unseen counties rather than memorizing local patterns.

## Reproducibility

- Random seed: SEED=42 (fixed throughout)
- Spatial CV prevents data leakage between counties
- Feature engineering pipeline is deterministic
- Optuna TPESampler with fixed seed (when used)
- Models saved with full hyperparameters
- Dataset: 9491 tracts × 53 features across 4 regions

## Target Reverse-Engineering Strategy

Since `coverage_gap_score` is not yet released by Zindi, we employ a self-supervised strategy:
1. Train on `building_gap` proxy — high R² (0.983), well-understood coverage metric
2. Train on `road_gap` proxy — near-perfect R² (0.9997), strong signal
3. When Zindi releases the actual target, retrain the entire pipeline instantly
4. The feature engineering and bias discovery are target-agnostic and will transfer

## Data Sources

- **Overture Maps**: Building, road, and POI coverage from OSM, Microsoft ML, Google, Esri
- **National Strata Table**: 85,396 census tracts × 232 columns (SVI, CVI, tribal, hazard, rural/urban)
- **4 Focus Regions**: Maricopa AZ (1,593), Northern CA (591), Eastern OK (1,300), South-Central TX (6,012)
- Total: 9,496 tracts with 351 base features + 69 engineered interactions

## Competitive Strategy

Our three competitive edges:

1. **Source Composition Features**: Parsing Overture's `sources[]` structs to compute ML fraction, OSM fraction, Google fraction, Esri fraction, and source diversity per tract. This captures how much of a tract's coverage comes from machine learning vs. human mapping.

2. **Null-as-Signal**: Coverage flags where data is NULL indicate that a data layer doesn't reach a tract — a powerful signal for mapping inequity. We encode these as separate features (1=covered, 0=not covered, -1=null/data doesn't reach).

3. **Target Reverse-Engineering**: By training on building_gap and road_gap proxies, we can validate our entire pipeline and be ready to retrain instantly when the actual target is released.
