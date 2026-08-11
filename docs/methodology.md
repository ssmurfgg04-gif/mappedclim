# Methodology: Bias Bounty Mapping Equity Challenge

## 1. Problem Understanding

Our task is to predict coverage gap scores for U.S. Census tracts, where the coverage
gap measures the completeness of Overture Maps data relative to authoritative reference
datasets (Census TIGER/Line Roads, Microsoft Building Footprints, HIFLD critical
facilities). The evaluation metric is Root Mean Squared Error (RMSE).

**Key insight**: The coverage gap is a deterministic function of the input data — it
is computed by the organizers, not subjectively assigned. This means reverse-engineering
the exact formula is the highest-leverage activity.

## 2. Data Sources

| Source | Role | Key Variables |
|--------|------|---------------|
| Overture Maps (buildings, roads, POIs) | Coverage target | geometry, sources[], confidence |
| Census TIGER/Line Roads | Road reference | geometry, road class |
| Microsoft Building Footprints | Building reference | geometry |
| HIFLD (hospitals, fire stations, EMS, schools) | Facility reference | geometry, facility type |
| Census ACS Housing | Housing reference | housing_units |
| National Strata Table (232 cols) | Feature backbone | SVI, CVI, rural/urban, hazard indices |

## 3. Feature Engineering

### 3.1 Coverage Gap Features (Primary)
- `building_count_ratio` = overture_building_count / microsoft_building_count
- `building_area_ratio` = overture_building_area / microsoft_building_area
- `road_length_ratio` = overture_road_length / tiger_road_length
- `road_count_ratio` = overture_road_segment_count / tiger_road_segment_count
- `poi_to_facility_ratio` = overture_poi_count / hifld_facility_count
- `buildings_per_housing_unit` = overture_building_count / acs_housing_units

### 3.2 Source Composition Features (Competitive Advantage)
Parsed from the nested `sources[]` column in Overture data:
- `ml_derived_fraction`: % of features from ML models (no human verification)
- `osm_fraction`: % of features from OpenStreetMap
- `source_diversity`: number of unique source datasets
- `mean_osm_staleness_days`: average days since last OSM update
- `mean_poi_confidence`: average Overture confidence score for POIs
- `low_confidence_fraction`: % of POIs with confidence < 0.5

### 3.3 Null Flag Features (Signal, Not Missing)
- `is_conus`: whether tract is in CONUS
- `has_wildfire_data`, `has_heat_data`, `has_drought_data`: data availability indicators
- `data_coverage_depth`: number of data layers that reach this tract

### 3.4 Spatial Lag Features
- `spatial_lag_k10_mean_*`: mean value among 10 nearest neighboring tracts
- `county_mean_*`: county-level aggregate
- `county_dev_*`: tract deviation from county mean
- `dist_to_tribal_boundary`: distance to nearest tribal land
- `tribal_overlap_fraction`: fraction of tract overlapping tribal lands

### 3.5 Vulnerability Interaction Features
- `svi_x_rural`: SVI x rural indicator
- `svi_x_wildfire_risk`: SVI x wildfire risk
- `compound_risk_score`: heat + wildfire + drought risk
- `is_tribal_x_high_svi`: tribal x high-SVI intersection

## 4. Validation Strategy

### 4.1 Spatial Cross-Validation
Standard random K-fold gives optimistic estimates due to spatial autocorrelation.
We use GroupKFold by County to prevent data leakage.

### 4.2 Public/Private Split Awareness
- Public LB: ~30% of test data
- Private LB: ~70% of test data (determines final ranking)
- We select 2 submissions based on strong local CV, NOT public LB position

## 5. Model Architecture

### 5.1 Base Models
| Model | Strength |
|-------|----------|
| XGBoost | Industry standard, robust |
| LightGBM | Fast, excellent with large feature sets |
| CatBoost | Superior categorical handling |

### 5.2 Ensembling
- Weighted Averaging with optimized blend weights
- Stacking with Ridge meta-learner

## 6. Bias Discovery (for $1,000 prize)

Analyzed model residuals by individual strata and their intersections.
Key intersectional gaps not captured by the automated API:
- Rural x high-SVI x tribal tracts
- Tribal x high-wildfire tracts
- Border x high-SVI (colonias)

## 7. Reproducibility

- Random seed: 42
- One-command execution: `python scripts/run_pipeline.py --phase all`

---

Generated: 2026-08-11 04:28:20
