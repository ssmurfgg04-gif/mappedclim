#!/usr/bin/env python3
"""
National-scale feature engineering pipeline for Zindi competition.
Extends regional features (9,496 tracts) to all 85,396 national tracts
using strata-based imputation and LightGBM gap prediction.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import h3
from pathlib import Path

t0 = time.time()
SEED = 42
np.random.seed(SEED)

PROJ = Path(__file__).resolve().parent.parent
STRATA_PATH = PROJ / "kaggle_dataset/national-strata-tract-table.parquet"
REGIONAL_PATH = PROJ / "data/output/engineered_features_v3.parquet"
OUT_PATH = PROJ / "data/features/national_tract_features.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Step 1: Load national strata ───────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading national strata table")
print("=" * 70)
strata = pd.read_parquet(STRATA_PATH)
print(f"  National strata: {strata.shape[0]:,} rows × {strata.shape[1]} cols")
print(f"  GEOIDs: {strata['GEOID'].nunique():,} unique")
assert strata.shape[0] == 85396, f"Expected 85,396 rows, got {strata.shape[0]}"

# Ensure lat/lon are numeric
for c in ['INTPTLAT', 'INTPTLON']:
    if c in strata.columns and strata[c].dtype == object:
        strata[c] = pd.to_numeric(strata[c], errors='coerce')

# ─── Step 2: Load regional features ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading regional features")
print("=" * 70)
regional = pd.read_parquet(REGIONAL_PATH)
print(f"  Regional features: {regional.shape[0]:,} rows × {regional.shape[1]} cols")

# Deduplicate regional by GEOID (keep first)
n_before = len(regional)
regional = regional.drop_duplicates(subset='GEOID', keep='first')
if len(regional) < n_before:
    print(f"  Deduplicated: {n_before} → {len(regional)} rows (removed {n_before - len(regional)} dupes)")

# ─── Step 3: Identify overlap and gaps ──────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Identifying overlap and gaps")
print("=" * 70)
strata_geoids = set(strata['GEOID'])
regional_geoids = set(regional['GEOID'])
overlap = strata_geoids & regional_geoids
only_regional = regional_geoids - strata_geoids
only_national = strata_geoids - regional_geoids
print(f"  GEOIDs in both:           {len(overlap):,}")
print(f"  GEOIDs only in regional:  {len(only_regional):,}")
print(f"  GEOIDs only in national:  {len(only_national):,}")
print(f"  National tracts needing synthetic features: {len(only_national):,}")

# ─── Step 4: Build national feature table ───────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Building national feature table")
print("=" * 70)

# Start with all national strata as the base
national = strata.copy()

# Merge regional features (left join to keep all national tracts)
print("  Merging regional features onto national strata (left join on GEOID)...")
# Only bring columns from regional that are NOT already in strata (except GEOID)
strata_cols_no_geoid = set(c for c in strata.columns if c != 'GEOID')
regional_unique_cols = ['GEOID'] + [c for c in regional.columns if c != 'GEOID' and c not in strata_cols_no_geoid]
regional_to_merge = regional[regional_unique_cols].copy()

national = national.merge(regional_to_merge, on='GEOID', how='left')

# Mark which tracts have regional features
national['has_regional_features'] = national['GEOID'].isin(regional_geoids).astype(int)
has_reg = national['has_regional_features'] == 1
needs_synth = national['has_regional_features'] == 0
print(f"  Tracts with regional features:    {has_reg.sum():,}")
print(f"  Tracts needing synthetic features: {needs_synth.sum():,}")
print(f"  Total rows:                       {len(national):,}")
assert len(national) == 85396, f"Expected 85,396 rows after merge, got {len(national)}"

# ─── Step 5: Prepare strata features for gap prediction ─────────────────
print("\n" + "=" * 70)
print("STEP 5: Preparing strata features for LightGBM gap prediction")
print("=" * 70)

# Define numeric strata columns to use as predictors
# These are columns available in the national strata
strata_predictor_candidates = [
    'svi_overall', 'svi_socioeconomic', 'svi_household', 'svi_minority',
    'svi_housing_transport', 'svi_pop',
    'cvi_overall', 'cvi_baseline', 'cvi_climate',
    'cvi_baseline_health', 'cvi_baseline_socioeconomic',
    'cvi_baseline_infrastructure', 'cvi_baseline_environment',
    'cvi_climate_health', 'cvi_climate_socioeconomic', 'cvi_climate_extreme_events',
    'pct_urban', 'pop_total', 'pop_urban', 'pop_rural',
    'tribal_pct', 'tribal_legal_pct', 'tribal_stat_pct',
    'usgs_wildfire_burned_pct_area', 'usgs_wildfire_pct_area_2plus',
    'mtbs_wildfire_burned_pct_land', 'mtbs_wildfire_burned_km2',
    'nifc_wildfire_burned_pct_land', 'nifc_wildfire_burned_km2',
    'ruca_pop_density', 'ruca_primary', 'ruca_secondary',
    'usfs_WHP_mean', 'usfs_Exposure_mean', 'usfs_BP_mean',
    'usfs_BuildingDensity_mean', 'usfs_PopDen_mean',
    'usfs_BuildingCount_sum', 'usfs_PopCount_sum',
    'usdm_summer_dsci', 'usdm_summer_pct_d0plus',
    'usfs_HURisk_mean', 'usfs_HUImpact_mean',
    'INTPTLAT', 'INTPTLON',
]

# Filter to only columns that exist AND are numeric in national
strata_predictor_cols = []
for c in strata_predictor_candidates:
    if c in national.columns and pd.api.types.is_numeric_dtype(national[c]):
        strata_predictor_cols.append(c)

# Add numeric _covered columns
covered_cols = [c for c in national.columns
                if c.endswith('_covered')
                and pd.api.types.is_numeric_dtype(national[c])]
for c in covered_cols:
    if c not in strata_predictor_cols:
        strata_predictor_cols.append(c)

# Add boolean columns as int
bool_predictor_cols = [c for c in ['tribal_any', 'usgs_wildfire_ever', 'mtbs_wildfire_ever', 'nifc_wildfire_ever']
                       if c in national.columns and national[c].dtype == bool]
for c in bool_predictor_cols:
    int_col = c + '_int'
    national[int_col] = national[c].astype(int)
    strata_predictor_cols.append(int_col)

print(f"  Predictor columns: {len(strata_predictor_cols)}")

# Prepare training data from regional tracts
train_mask = has_reg & national['building_gap'].notna()
print(f"  Training tracts (with building_gap): {train_mask.sum():,}")

X_train = national.loc[train_mask, strata_predictor_cols].copy()
y_bldg_gap = national.loc[train_mask, 'building_gap'].copy()

train_mask_road = has_reg & national['road_gap'].notna()
print(f"  Training tracts (with road_gap):    {train_mask_road.sum():,}")
X_train_road = national.loc[train_mask_road, strata_predictor_cols].copy()
y_road_gap = national.loc[train_mask_road, 'road_gap'].copy()

# Fill NaNs with -999 for LightGBM
FILL_VAL = -999
X_train_filled = X_train.fillna(FILL_VAL).replace([np.inf, -np.inf], FILL_VAL)
X_train_road_filled = X_train_road.fillna(FILL_VAL).replace([np.inf, -np.inf], FILL_VAL)

# Remove zero-variance columns
variances = X_train_filled.var(numeric_only=True)
good_cols = variances[variances > 1e-10].index.tolist()
print(f"  Usable predictors (non-zero var): {len(good_cols)}")

X_train_filled = X_train_filled[good_cols]
X_train_road_filled = X_train_road_filled[good_cols]

# ─── Step 6: Train LightGBM models for gap prediction ───────────────────
print("\n" + "=" * 70)
print("STEP 6: Training LightGBM models for gap prediction")
print("=" * 70)

# Building gap model
print("\n  Training building_gap model...")
lgb_bldg = lgb.LGBMRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
    boosting_type='gbdt', random_state=SEED, verbose=-1,
    n_jobs=-1
)
lgb_bldg.fit(X_train_filled, y_bldg_gap)
bldg_pred_train = lgb_bldg.predict(X_train_filled)
bldg_rmse = np.sqrt(np.mean((y_bldg_gap - bldg_pred_train) ** 2))
bldg_r2 = 1 - np.sum((y_bldg_gap - bldg_pred_train) ** 2) / np.sum((y_bldg_gap - y_bldg_gap.mean()) ** 2)
print(f"    building_gap model: RMSE={bldg_rmse:.6f}, R²={bldg_r2:.4f}")

# Road gap model
print("\n  Training road_gap model...")
lgb_road = lgb.LGBMRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
    boosting_type='gbdt', random_state=SEED, verbose=-1,
    n_jobs=-1
)
lgb_road.fit(X_train_road_filled, y_road_gap)
road_pred_train = lgb_road.predict(X_train_road_filled)
road_rmse = np.sqrt(np.mean((y_road_gap - road_pred_train) ** 2))
road_r2 = 1 - np.sum((y_road_gap - road_pred_train) ** 2) / np.sum((y_road_gap - y_road_gap.mean()) ** 2)
print(f"    road_gap model: RMSE={road_rmse:.6f}, R²={road_r2:.4f}")

# Feature importances
print("\n  Top 15 features for building_gap:")
imp = pd.Series(lgb_bldg.feature_importances_, index=good_cols).sort_values(ascending=False)
for i, (feat_name, feat_imp) in enumerate(imp.head(15).items()):
    print(f"    {i+1:2d}. {feat_name:40s} {feat_imp}")

# ─── Step 7: Predict gaps for national tracts ───────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Predicting gaps for national tracts without regional features")
print("=" * 70)

X_national = national.loc[needs_synth, good_cols].fillna(FILL_VAL).replace([np.inf, -np.inf], FILL_VAL)

bldg_gap_pred = lgb_bldg.predict(X_national)
road_gap_pred = lgb_road.predict(X_national)

print(f"  Predicted building_gap for {len(bldg_gap_pred):,} tracts")
print(f"    mean={bldg_gap_pred.mean():.4f}, std={bldg_gap_pred.std():.4f}, "
      f"min={bldg_gap_pred.min():.4f}, max={bldg_gap_pred.max():.4f}")
print(f"  Predicted road_gap for {len(road_gap_pred):,} tracts")
print(f"    mean={road_gap_pred.mean():.4f}, std={road_gap_pred.std():.4f}, "
      f"min={road_gap_pred.min():.4f}, max={road_gap_pred.max():.4f}")

# Fill in predicted gaps
national.loc[needs_synth, 'building_gap'] = bldg_gap_pred
national.loc[needs_synth, 'road_gap'] = road_gap_pred

# ─── Step 8: Compute derived features for synthetic tracts ──────────────
print("\n" + "=" * 70)
print("STEP 8: Computing derived features for all national tracts")
print("=" * 70)

# Compute ratio features from gaps
print("  Computing ratio features...")
national['building_ratio'] = national['building_ratio'].fillna(1 + national['building_gap'])
national['road_ratio'] = national['road_ratio'].fillna(1 + national['road_gap'])

# ─── Core strata-derived values ─────────────────────────────────────────
svi = national['svi_overall'].fillna(0.5)
svi_soc = national['svi_socioeconomic'].fillna(0.5)
svi_hh = national['svi_household'].fillna(0.5)
svi_min = national['svi_minority'].fillna(0.5)
svi_ht = national['svi_housing_transport'].fillna(0.5)
cvi = national['cvi_overall'].fillna(0.5)
cvi_base = national['cvi_baseline'].fillna(0.5)
cvi_clim = national['cvi_climate'].fillna(0.5)
tribal_pct = national['tribal_pct'].fillna(0)
tribal_any_f = national['tribal_any'].astype(float)
pct_urban = national['pct_urban'].fillna(0.5)
pop_total = national['pop_total'].astype(float)
bldg_gap = national['building_gap'].fillna(0)
road_gap = national['road_gap'].fillna(0)
bldg_ratio = national['building_ratio'].fillna(1 + bldg_gap)
rural = (1 - pct_urban).clip(0, 1)
wf_area = national['usgs_wildfire_burned_pct_area'].fillna(0)
wf_ever = national['usgs_wildfire_ever'].astype(float)

print("  Computing interaction features...")

# ─── SVI interactions ───────────────────────────────────────────────────
national['svi_x_rural'] = national['svi_x_rural'].fillna(svi * rural)
national['svi_x_building_gap'] = national['svi_x_building_gap'].fillna(svi * bldg_gap)
national['svi_x_road_gap'] = national['svi_x_road_gap'].fillna(svi * road_gap)

# SVI × building_gap (using bldg_gap = building_gap)
national['svi_x_bldg'] = national['svi_x_bldg'].fillna(svi * bldg_gap)
national['svi_sq_x_bldg'] = national['svi_sq_x_bldg'].fillna(svi ** 2 * bldg_gap)
national['svi_abs_x_bldg_abs'] = national['svi_abs_x_bldg_abs'].fillna(svi.abs() * bldg_gap.abs())
national['svi_x_bldg_sq'] = national['svi_x_bldg_sq'].fillna(svi * bldg_gap ** 2)
national['svi_cubed_x_bldg'] = national['svi_cubed_x_bldg'].fillna(svi ** 3 * bldg_gap)
national['svi_min_x_bldg'] = national['svi_min_x_bldg'].fillna(svi_min * bldg_gap)
national['svi_soc_x_bldg'] = national['svi_soc_x_bldg'].fillna(svi_soc * bldg_gap)
national['svi_hh_x_bldg'] = national['svi_hh_x_bldg'].fillna(svi_hh * bldg_gap)
national['svi_ht_x_bldg'] = national['svi_ht_x_bldg'].fillna(svi_ht * bldg_gap)

# SVI × road_gap
national['svi_x_road'] = national['svi_x_road'].fillna(svi * road_gap)
national['svi_sq_x_road'] = national['svi_sq_x_road'].fillna(svi ** 2 * road_gap)
national['svi_min_x_road'] = national['svi_min_x_road'].fillna(svi_min * road_gap)

# ─── Tribal interactions ────────────────────────────────────────────────
national['tribal_x_hazard'] = national['tribal_x_hazard'].fillna(tribal_any_f * (wf_area + cvi) / 2)
national['tribal_x_building_gap'] = national['tribal_x_building_gap'].fillna(tribal_any_f * bldg_gap)
national['tribal_x_bldg'] = national['tribal_x_bldg'].fillna(tribal_any_f * bldg_gap)
national['tribal_pct_x_bldg'] = national['tribal_pct_x_bldg'].fillna(tribal_pct * bldg_gap)
national['tribal_x_road'] = national['tribal_x_road'].fillna(tribal_any_f * road_gap)
national['tribal_x_bldg_sq'] = national['tribal_x_bldg_sq'].fillna(tribal_any_f * bldg_gap ** 2)
national['tribal_x_svi_x_bldg'] = national['tribal_x_svi_x_bldg'].fillna(tribal_any_f * svi * bldg_gap)
national['tribal_x_cvi_x_bldg'] = national['tribal_x_cvi_x_bldg'].fillna(tribal_any_f * cvi * bldg_gap)

# ─── Urban/rural interactions ───────────────────────────────────────────
national['pct_urban_x_bldg'] = national['pct_urban_x_bldg'].fillna(pct_urban * bldg_gap)
national['rural_x_bldg'] = national['rural_x_bldg'].fillna(rural * bldg_gap)
national['rural_sq_x_bldg'] = national['rural_sq_x_bldg'].fillna(rural ** 2 * bldg_gap)
national['rural_x_road'] = national['rural_x_road'].fillna(rural * road_gap)
national['rural_x_svi_x_bldg'] = national['rural_x_svi_x_bldg'].fillna(rural * svi * bldg_gap)
national['urban_x_svi_x_bldg'] = national['urban_x_svi_x_bldg'].fillna(pct_urban * svi * bldg_gap)

# ─── Wildfire interactions ──────────────────────────────────────────────
national['wf_x_bldg'] = national['wf_x_bldg'].fillna(wf_area * bldg_gap)
national['wf_flag_x_bldg'] = national['wf_flag_x_bldg'].fillna(wf_ever * bldg_gap)
national['wf_area_x_bldg'] = national['wf_area_x_bldg'].fillna(wf_area * bldg_gap)
national['wf_x_svi_x_bldg'] = national['wf_x_svi_x_bldg'].fillna(wf_area * svi * bldg_gap)
national['wf_x_tribal'] = national['wf_x_tribal'].fillna(wf_area * tribal_any_f)

# ─── CVI interactions ───────────────────────────────────────────────────
national['cvi_x_bldg'] = national['cvi_x_bldg'].fillna(cvi * bldg_gap)
national['cvi_sq_x_bldg'] = national['cvi_sq_x_bldg'].fillna(cvi ** 2 * bldg_gap)
national['cvi_x_svi_x_bldg'] = national['cvi_x_svi_x_bldg'].fillna(cvi * svi * bldg_gap)
national['cvi_base_x_bldg'] = national['cvi_base_x_bldg'].fillna(cvi_base * bldg_gap)
national['cvi_clim_x_bldg'] = national['cvi_clim_x_bldg'].fillna(cvi_clim * bldg_gap)

# ─── Compound risk ──────────────────────────────────────────────────────
print("  Computing compound risk and proxy features...")
national['compound_risk_score'] = national['compound_risk_score'].fillna(
    (svi + cvi + tribal_any_f + wf_area) / 4
)

# ─── Gap transformations ────────────────────────────────────────────────
national['building_gap_sq'] = national['building_gap_sq'].fillna(bldg_gap ** 2)
national['road_gap_sq'] = national['road_gap_sq'].fillna(road_gap ** 2)
national['bldg_gap_sq'] = national['bldg_gap_sq'].fillna(bldg_gap ** 2)
national['bldg_gap_cu'] = national['bldg_gap_cu'].fillna(bldg_gap ** 3)
national['bldg_gap_abs'] = national['bldg_gap_abs'].fillna(bldg_gap.abs())
national['bldg_gap_log1p_abs'] = national['bldg_gap_log1p_abs'].fillna(np.log1p(bldg_gap.abs()))
national['road_gap_abs'] = national['road_gap_abs'].fillna(road_gap.abs())

# ─── Ratio transformations ──────────────────────────────────────────────
road_ratio_filled = national['road_ratio'].fillna(1 + road_gap)
national['bldg_road_ratio'] = national['bldg_road_ratio'].fillna(
    bldg_ratio / road_ratio_filled.replace(0, np.nan).fillna(1)
)
national['bldg_road_diff'] = national['bldg_road_diff'].fillna(bldg_gap - road_gap)
national['bldg_road_product'] = national['bldg_road_product'].fillna(bldg_gap * road_gap)
national['log_building_ratio'] = national['log_building_ratio'].fillna(np.log1p(bldg_ratio.clip(lower=0)))
national['log_road_ratio'] = national['log_road_ratio'].fillna(np.log1p(road_ratio_filled.clip(lower=0)))
national['log_bldg_ratio'] = national['log_bldg_ratio'].fillna(np.log1p(bldg_ratio.clip(lower=0)))
national['bldg_ratio_sq'] = national['bldg_ratio_sq'].fillna(bldg_ratio ** 2)

# ─── Compound risk derivatives ──────────────────────────────────────────
compound_risk = national['compound_risk_score'].fillna(0)
national['compound_risk'] = national['compound_risk'].fillna(compound_risk)
national['compound_risk_sq'] = national['compound_risk_sq'].fillna(compound_risk ** 2)
national['tribal_x_risk'] = national['tribal_x_risk'].fillna(tribal_any_f * compound_risk)

# ─── Log population features ────────────────────────────────────────────
log_pop = np.log1p(pop_total)
national['log_pop'] = national['log_pop'].fillna(log_pop)
national['log_pop_x_bldg'] = national['log_pop_x_bldg'].fillna(log_pop * bldg_gap)
national['log_pop_x_svi'] = national['log_pop_x_svi'].fillna(log_pop * svi)

# ─── High-SVI and hazard-stratified interactions ────────────────────────
high_svi = (svi > svi.quantile(0.75)).astype(float)
low_svi = (svi < svi.quantile(0.25)).astype(float)
high_cvi = (cvi > cvi.quantile(0.75)).astype(float)

national['highsvi_x_rural'] = national['highsvi_x_rural'].fillna(high_svi * rural)
national['tribal_x_highsvi_x_rural'] = national['tribal_x_highsvi_x_rural'].fillna(tribal_any_f * high_svi * rural)
national['tribal_x_lowsvi_x_rural'] = national['tribal_x_lowsvi_x_rural'].fillna(tribal_any_f * low_svi * rural)
national['tribal_x_rural'] = national['tribal_x_rural'].fillna(tribal_any_f * rural)
national['tribal_hsvi_rural_x_bldg'] = national['tribal_hsvi_rural_x_bldg'].fillna(tribal_any_f * high_svi * rural * bldg_gap)
national['hsvi_rural_x_bldg'] = national['hsvi_rural_x_bldg'].fillna(high_svi * rural * bldg_gap)
national['wf_x_rural_x_hsvi'] = national['wf_x_rural_x_hsvi'].fillna(wf_area * rural * high_svi)
national['hcvi_x_hsvi_x_rural'] = national['hcvi_x_hsvi_x_rural'].fillna(high_cvi * high_svi * rural)
national['hcvi_x_tribal'] = national['hcvi_x_tribal'].fillna(high_cvi * tribal_any_f)

# ─── Coverage null features ─────────────────────────────────────────────
print("  Computing coverage null features...")
covered_null_cols = [c for c in national.columns
                     if c.endswith('_covered')
                     and pd.api.types.is_numeric_dtype(national[c])]
for c in covered_null_cols:
    null_col = c.replace('_covered', '_covered_null')
    if null_col in national.columns:
        national[null_col] = national[null_col].fillna((national[c] == 0).astype(float))

# ─── Data coverage features ─────────────────────────────────────────────
if 'data_coverage_depth' in national.columns:
    national['data_coverage_depth'] = national['data_coverage_depth'].fillna(
        national[covered_null_cols].sum(axis=1)
    )
else:
    national['data_coverage_depth'] = national[covered_null_cols].sum(axis=1)

if 'data_coverage_fraction' in national.columns:
    national['data_coverage_fraction'] = national['data_coverage_fraction'].fillna(
        national['data_coverage_depth'] / max(len(covered_null_cols), 1)
    )
else:
    national['data_coverage_fraction'] = national['data_coverage_depth'] / max(len(covered_null_cols), 1)

# ─── Proxy features ─────────────────────────────────────────────────────
print("  Computing proxy features...")
national['proxy_simple_avg'] = national['proxy_simple_avg'].fillna(
    (svi + cvi + tribal_any_f + rural + wf_area) / 5
)
national['proxy_svi_weighted'] = national['proxy_svi_weighted'].fillna(
    svi * 0.4 + cvi * 0.3 + tribal_any_f * 0.2 + rural * 0.1
)
national['proxy_max_gap'] = national['proxy_max_gap'].fillna(
    np.maximum(bldg_gap.abs(), road_gap.abs())
)
national['proxy_pop_weighted'] = national['proxy_pop_weighted'].fillna(
    (svi * 0.3 + cvi * 0.3 + bldg_gap.abs() * 0.2 + tribal_any_f * 0.1 + rural * 0.1)
    * np.log1p(pop_total) / np.log1p(pop_total).median()
)

# ─── County FIPS ────────────────────────────────────────────────────────
if 'county_fips' not in national.columns:
    national['county_fips'] = national['GEOID'].str[:5]
else:
    national['county_fips'] = national['county_fips'].fillna(national['GEOID'].str[:5])

if 'state_fips' not in national.columns:
    national['state_fips'] = national['GEOID'].str[:2]
else:
    national['state_fips'] = national['state_fips'].fillna(national['GEOID'].str[:2])

# ─── Region dummies (all 0 for non-regional tracts) ─────────────────────
for reg in ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']:
    col = f'region_{reg}'
    if col not in national.columns:
        national[col] = 0.0
    else:
        national[col] = national[col].fillna(0.0)

# ─── Fill remaining KNN/county features with 0 for synthetic tracts ─────
# These cannot be computed without the full regional spatial data
knn_cols = [c for c in national.columns if '_knn' in c]
county_stat_cols = [c for c in national.columns if '_county_' in c]
for c in knn_cols + county_stat_cols:
    if c in national.columns and national[c].isna().any():
        national[c] = national[c].fillna(0)

# ─── Fill remaining building/source features ────────────────────────────
source_cols = ['bldg_total_sources', 'bldg_source_diversity', 'bldg_osm_fraction',
               'bldg_ms_ml_fraction', 'bldg_google_fraction', 'bldg_esri_fraction',
               'poi_count_by_conf', 'poi_mean_confidence', 'poi_low_conf_fraction',
               'poi_very_high_conf_fraction']
for c in source_cols:
    if c in national.columns:
        national[c] = national[c].fillna(0)

# gap_x_ml_fraction
if 'gap_x_ml_fraction' in national.columns:
    ml_frac = national.get('bldg_ms_ml_fraction', pd.Series(0, index=national.index)).fillna(0)
    national['gap_x_ml_fraction'] = national['gap_x_ml_fraction'].fillna(bldg_gap * ml_frac)

# ─── Fill remaining numeric features with 0 ─────────────────────────────
remaining_null = national.isnull().sum()
remaining_null_cols = remaining_null[remaining_null > 0].index.tolist()
print(f"\n  Remaining columns with nulls: {len(remaining_null_cols)}")
if remaining_null_cols:
    for c in remaining_null_cols:
        if pd.api.types.is_numeric_dtype(national[c]):
            national[c] = national[c].fillna(0)
        elif national[c].dtype == bool:
            national[c] = national[c].fillna(False)
        elif national[c].dtype == object:
            national[c] = national[c].fillna('')

# ─── Step 9: Compute national-level county statistics ───────────────────
print("\n" + "=" * 70)
print("STEP 9: Computing national-level county statistics")
print("=" * 70)

# For tracts without regional features, compute county-level stats
# from ALL national tracts in that county
county_key_cols = ['building_gap', 'road_gap', 'svi_overall', 'svi_socioeconomic',
                   'svi_household', 'svi_minority', 'svi_housing_transport', 'svi_pop']

for col in county_key_cols:
    if col not in national.columns:
        continue
    mean_col = f'{col}_county_mean'
    std_col = f'{col}_county_std'
    dev_col = f'{col}_county_dev'

    # Compute county-level aggregates from all national tracts
    county_agg = national.groupby('county_fips')[col].agg(['mean', 'std'])
    county_agg.columns = ['mean', 'std']

    if mean_col in national.columns:
        # Fill NaN county means with national-level aggregates
        mapping_mean = county_agg['mean'].to_dict()
        national[mean_col] = national[mean_col].fillna(national['county_fips'].map(mapping_mean))
    else:
        national[mean_col] = national['county_fips'].map(county_agg['mean'].to_dict())

    if std_col in national.columns:
        mapping_std = county_agg['std'].to_dict()
        national[std_col] = national[std_col].fillna(national['county_fips'].map(mapping_std))
    else:
        national[std_col] = national['county_fips'].map(county_agg['std'].to_dict())

    if dev_col in national.columns:
        national[dev_col] = national[dev_col].fillna(national[col] - national[mean_col])
    # Don't create dev_col if it doesn't exist - it's a derived feature

n_county_filled = needs_synth.sum()
print(f"  Computed county stats for {n_county_filled:,} synthetic tracts from national data")

# ─── Step 10: Final statistics and save ─────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Final statistics and saving")
print("=" * 70)

print(f"\n  Final shape: {national.shape[0]:,} rows × {national.shape[1]} cols")
assert national.shape[0] == 85396, f"Expected 85,396 rows, got {national.shape[0]}"

# Check for remaining nulls
total_nulls = national.isnull().sum().sum()
print(f"  Total remaining nulls: {total_nulls:,}")
if total_nulls > 0:
    null_summary = national.isnull().sum()
    null_cols = null_summary[null_summary > 0]
    print(f"  Columns with nulls (top 10):")
    for c, n in null_cols.sort_values(ascending=False).head(10).items():
        print(f"    {c}: {n:,}")

# Stats by has_regional_features
for flag in [1, 0]:
    mask = national['has_regional_features'] == flag
    label = "regional" if flag == 1 else "synthetic"
    print(f"\n  {label.capitalize()} tracts ({mask.sum():,}):")
    if 'building_gap' in national.columns:
        bg = national.loc[mask, 'building_gap']
        print(f"    building_gap: mean={bg.mean():.4f}, std={bg.std():.4f}, "
              f"min={bg.min():.4f}, max={bg.max():.4f}")
    if 'road_gap' in national.columns:
        rg = national.loc[mask, 'road_gap']
        print(f"    road_gap:     mean={rg.mean():.4f}, std={rg.std():.4f}, "
              f"min={rg.min():.4f}, max={rg.max():.4f}")
    if 'svi_x_bldg' in national.columns:
        sx = national.loc[mask, 'svi_x_bldg']
        print(f"    svi_x_bldg:   mean={sx.mean():.4f}, std={sx.std():.4f}")

# Column summary
print(f"\n  Key feature groups:")
feature_groups = {
    'strata_base': [c for c in national.columns if c in strata.columns and c != 'GEOID'],
    'gap_features': [c for c in national.columns if 'gap' in c.lower()],
    'ratio_features': [c for c in national.columns if 'ratio' in c.lower()],
    'interaction_features': [c for c in national.columns if '_x_' in c],
    'knn_features': [c for c in national.columns if '_knn' in c],
    'county_features': [c for c in national.columns if '_county_' in c],
    'coverage_features': [c for c in national.columns if '_covered' in c or '_null' in c or 'null_' in c],
    'proxy_features': [c for c in national.columns if c.startswith('proxy_')],
    'log_features': [c for c in national.columns if c.startswith('log_')],
}
for group_name, group_cols in feature_groups.items():
    print(f"    {group_name:25s}: {len(group_cols):3d} columns")

# Save
print(f"\n  Saving to: {OUT_PATH}")
national.to_parquet(OUT_PATH, index=False)
file_size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"  File size: {file_size_mb:.1f} MB")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"COMPLETE in {elapsed:.1f}s")
print(f"  National tracts: {national.shape[0]:,}")
print(f"  Features:        {national.shape[1]}")
print(f"  With regional:   {(national['has_regional_features']==1).sum():,}")
print(f"  Synthetic:       {(national['has_regional_features']==0).sum():,}")
print(f"{'=' * 70}")
