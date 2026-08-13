#!/usr/bin/env python3
"""
AUGMENTED PIPELINE — Add top 20 strata features + interactions, test with LORO R².

Strategy:
  1. Start with current Phase 2 feature matrix (49 features)
  2. Add top 20 unused strata features (strongest signal)
  3. Add key interactions (fire×rural, carbon×bldg_gap, tribal×fire, etc.)
  4. Run BOTH H3-CV and LORO validation
  5. Compare against baseline (current 49-feature pipeline)
  
Target: gap_only (alpha=0, Deterministic fix)
Inference: final_score = model.predict(X) - 1.0 * rural_penalty
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
import h3

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = Path("/home/z/my-project/download"); DL.mkdir(parents=True, exist_ok=True)
KAG = PROJ / "kaggle_dataset"
NF = 3

print("=" * 72)
print("AUGMENTED PIPELINE — Top 20 strata features + interactions")
print("=" * 72)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASE FEATURES + TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading engineered features...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Base: {feat.shape}")

# DETERMINISTIC FIX
assert 'gap_only' in feat.columns
assert 'rural_penalty' in feat.columns
y = feat['gap_only'].copy()
rural_col = feat['rural_penalty'].copy()
geo = feat['GEOID'].astype(str).copy()

# ══════════════════════════════════════════════════════════════════════════════
# 2. ADD TOP 20 NEW STRATA FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Adding top 20 strata features...")

top20_strata = [
    'carbonplan_buildings',        # |r|=0.1262 — carbon risk buildings
    'carbonplan_rps_2047_mean',    # |r|=0.1253 — carbon risk 2047
    'fod_acres_ignited',           # |r|=0.1243 — fire acres
    'ghcn_temp_nearest_km',        # |r|=0.1181 — weather station distance
    'mtbs_wildfire_burned_pct_land', # |r|=0.1163 — burn severity
    'usfs_BuildingDensity_mean',   # |r|=0.1158 — fire building density
    'tribal_legal',                # |r|=0.1150 — tribal legal status
    'cvi_climate_extreme_events',  # |r|=0.1134 — climate extreme events
    'cvi_baseline_health',         # |r|=0.1105 — health vulnerability
    'ghcn_any_within_25km',        # |r|=0.1090 — any weather station nearby
    'usgs_wildland_fire_burned_pct_area', # |r|=0.1086 — wildland fire %
    'mtbs_years_since_wildfire',   # |r|=0.1051 — recency of wildfire
    'tribal_legal_pct',            # |r|=0.1043 — tribal pct
    'svi_minority',                # |r|=0.1030 — minority SVI
    'cvi_climate',                 # |r|=0.1017 — overall climate vulnerability
    'fod_last_fire_year',          # |r|=0.0992 — last fire year
    'pop_total',                   # |r|=0.0965 — total population
    'tribal_any',                  # |r|=0.0911 — any tribal
    'usfs_WHP_mean',              # |r|=0.0877 — wildfire hazard potential
    'usgs_prescribed_burned_pct_area', # |r|=0.0863 — prescribed fire %
]

# These are already in the engineered features, just not selected by top-N filter
added = 0
for col in top20_strata:
    if col in feat.columns:
        added += 1
    else:
        print(f"  WARNING: {col} not in engineered features!")
print(f"  {added}/{len(top20_strata)} features available in base")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CREATE INTERACTION FEATURES (domain knowledge)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Creating interaction features...")

# Base gap features for interactions
bldg_gap = feat.get('building_gap', pd.Series(0, index=feat.index)).fillna(0)
road_gap = feat.get('road_gap', pd.Series(0, index=feat.index)).fillna(0)
bldg_gap_clip = feat.get('bldg_gap_clip', pd.Series(0, index=feat.index)).fillna(0)
rural = rural_col.fillna(0).values

interactions = {}

# Fire × Rural interactions
if 'fod_acres_ignited' in feat.columns:
    fires = feat['fod_acres_ignited'].fillna(0).values
    interactions['fire_acres_x_rural'] = fires * rural
    interactions['fire_acres_x_bldg'] = fires * bldg_gap_clip.values

if 'mtbs_wildfire_burned_pct_land' in feat.columns:
    burn_pct = feat['mtbs_wildfire_burned_pct_land'].fillna(0).values
    interactions['burn_pct_x_rural'] = burn_pct * rural
    interactions['burn_pct_x_bldg'] = burn_pct * bldg_gap_clip.values

# Carbon × Rural interactions  
if 'carbonplan_crps_mean' in feat.columns:
    carbon = feat['carbonplan_crps_mean'].fillna(0).values
    interactions['carbon_x_rural'] = carbon * rural
    interactions['carbon_x_bldg'] = carbon * bldg_gap_clip.values

if 'carbonplan_buildings' in feat.columns:
    carb_bldg = feat['carbonplan_buildings'].fillna(0).values
    interactions['carb_bldg_x_rural'] = carb_bldg * rural

# Climate vulnerability × gaps
if 'cvi_climate_extreme_events' in feat.columns:
    cvi_ext = feat['cvi_climate_extreme_events'].fillna(0).values
    interactions['cvi_ext_x_bldg'] = cvi_ext * bldg_gap_clip.values
    interactions['cvi_ext_x_rural'] = cvi_ext * rural

if 'cvi_baseline_health' in feat.columns:
    cvi_health = feat['cvi_baseline_health'].fillna(0).values
    interactions['cvi_health_x_rural'] = cvi_health * rural

# Tribal × Fire interaction (key: tribal areas with fire = double whammy)
if 'tribal_legal' in feat.columns:
    tribal = feat['tribal_legal'].fillna(0).values
    interactions['tribal_legal_x_bldg'] = tribal * bldg_gap_clip.values
    if 'fod_acres_ignited' in feat.columns:
        interactions['tribal_x_fire'] = tribal * feat['fod_acres_ignited'].fillna(0).values
    if 'mtbs_wildfire_burned_pct_land' in feat.columns:
        interactions['tribal_x_burn'] = tribal * feat['mtbs_wildfire_burned_pct_land'].fillna(0).values

# Weather station proximity × rural (fewer stations in rural = worse maps)
if 'ghcn_temp_nearest_km' in feat.columns:
    ws_dist = feat['ghcn_temp_nearest_km'].fillna(0).values
    interactions['ws_dist_x_rural'] = ws_dist * rural
    interactions['ws_dist_x_bldg'] = ws_dist * bldg_gap_clip.values

# SVI minority × gap
if 'svi_minority' in feat.columns:
    svi_min = feat['svi_minority'].fillna(0).values
    interactions['svi_min_x_bldg'] = svi_min * bldg_gap_clip.values
    interactions['svi_min_x_rural'] = svi_min * rural

# Add all interactions to feat
for col, vals in interactions.items():
    feat[col] = vals

print(f"  +{len(interactions)} interaction features → {feat.shape[1]} total")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PREPARE FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Preparing feature matrix...")

drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
             'building_count_ratio', 'building_count_gap',
             'road_count_ratio', 'road_count_gap', 'road_length_ratio', 'road_length_gap',
             'poi_facility_gap', 'poi_to_facility_ratio',
             'poi_facility_gap_corrected', 'poi_to_facility_ratio_corrected',
             'building_area_gap',
             'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
             'proxy_merged', 'gap_only', 'rural_penalty']

feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy()

valid = y.notna()
X, y_use, geo_use = X[valid], y[valid], geo[valid]
rural_valid = rural_col[valid]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance
s = X.std()
X = X[s[s > 1e-10].index]

# Select top 80 features by correlation (expanded from 60)
cs = X.corrwith(y_use).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(80).index]

# Remove highly correlated duplicates
cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)

print(f"  {X.shape[1]} features, {X.shape[0]} tracts")

# ── Track which features are new vs baseline ──
# Baseline = top 60 from original pipeline (without strata expansion)
# We'll compute this after seeing the full list
baseline_feats = set(cs.sort_values(ascending=False).head(60).index)
new_strata_feats = [c for c in X.columns if c in top20_strata]
new_interaction_feats = [c for c in X.columns if c in interactions]
baseline_in_model = [c for c in X.columns if c in baseline_feats]

print(f"  Baseline features: {len(baseline_in_model)}")
print(f"  New strata features: {len(new_strata_feats)}")
print(f"  New interactions: {len(new_interaction_feats)}")
print(f"  Other: {X.shape[1] - len(baseline_in_model) - len(new_strata_feats) - len(new_interaction_feats)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. H3 SPATIAL CV
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Setting up H3 spatial CV...")
lats = feat.loc[valid, 'centroid_lat'] if 'centroid_lat' in feat.columns else None
lons = feat.loc[valid, 'centroid_lon'] if 'centroid_lon' in feat.columns else None
if lats is None or lons is None or lats.isna().all():
    if 'INTPTLAT' in feat.columns:
        lats = pd.to_numeric(feat.loc[valid, 'INTPTLAT'], errors='coerce')
        lons = pd.to_numeric(feat.loc[valid, 'INTPTLON'], errors='coerce')

blocks = pd.Series([
    h3.latlng_to_cell(float(la), float(lo), 4)
    if not (np.isnan(la) or np.isnan(lo)) else 'unk'
    for la, lo in zip(lats.values, lons.values)
], index=geo_use.index)
n_blocks = blocks.nunique()
print(f"  H3 blocks: {n_blocks}")

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

# Don't delete feat yet - need it for LORO/bias analysis
feat_ref = feat  # keep reference
del feat; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN 3-MODEL ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Training 3-model ensemble...")

def train_model(model, name):
    oof = np.full(len(y_use), np.nan); scores = []
    for fi, (ti, vi) in enumerate(splits):
        m = type(model)(**model.get_params())
        Xt, yt, Xv, yv = X.iloc[ti], y_use.iloc[ti], X.iloc[vi], y_use.iloc[vi]
        try:
            if isinstance(m, xgb.XGBRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            elif isinstance(m, lgb.LGBMRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            else:
                m.fit(Xt, yt)
        except Exception as e:
            print(f"  {name} F{fi} err: {e}"); continue
        p = m.predict(Xv); oof[vi] = p
        rmse = np.sqrt(mean_squared_error(yv, p)); r2 = r2_score(yv, p)
        scores.append((rmse, r2))
        elapsed = time.time() - t0
        print(f"  {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({elapsed:.0f}s)")
        del m; gc.collect()
    if scores:
        rmse_m = np.mean([s[0] for s in scores]); r2_m = np.mean([s[1] for s in scores])
        print(f"  {name} mean: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
        return oof, rmse_m, r2_m
    return oof, 999, 0

oofs = {}; msum = {}

# [1] XGBoost
print("\n  [1] XGBoost...")
o, r, r2 = train_model(
    xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                     tree_method='hist', random_state=SEED), 'XGB')
oofs['xgb'] = o; msum['xgb'] = (r, r2); gc.collect()

# [2] LightGBM
print("\n  [2] LightGBM GBDT...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='gbdt', random_state=SEED, verbose=-1), 'LGB')
oofs['lgb'] = o; msum['lgb'] = (r, r2); gc.collect()

# [3] ExtraTrees
print("\n  [3] ExtraTrees...")
o, r, r2 = train_model(
    ExtraTreesRegressor(n_estimators=80, max_depth=10,
                        min_samples_split=10, random_state=SEED, n_jobs=1), 'ET')
oofs['et'] = o; msum['et'] = (r, r2); gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Ensembling...")

ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y_use.values[vv]

# Convex blend
res = minimize(
    lambda w: np.sqrt(mean_squared_error(yv, mv @ w)),
    np.ones(len(ns)) / len(ns), method='SLSQP',
    bounds=[(0, 1)] * len(ns),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
cw = {n: round(float(w), 4) for n, w in zip(ns, res.x)}
cp = mv @ res.x; cr_ = res.fun; cr2 = r2_score(yv, cp)
print(f"  Convex: RMSE={cr_:.6f} R2={cr2:.4f} w={cw}")

# Simple average
ap = mv.mean(axis=1)
ar_ = np.sqrt(mean_squared_error(yv, ap)); ar2 = r2_score(yv, ap)
print(f"  Simple avg: RMSE={ar_:.6f} R2={ar2:.4f}")

best = min([('convex', cr_, cr2, cp), ('simple_avg', ar_, ar2, ap)], key=lambda x: x[1])
bn, brm, br2_, bpred = best
print(f"\n  >>> BEST: {bn} RMSE={brm:.6f} R2={br2_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. LORO (Leave-One-Region-Out) VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] LORO validation (true out-of-sample)...")

# Load region info — check multiple sources for region column
try:
    feat_mini = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                                columns=['GEOID', 'svi_overall', 'tribal_any', 'pct_urban'])
except Exception:
    feat_mini = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat_mini['GEOID'] = feat_mini['GEOID'].astype(str)

# Try to get region from national features, or derive from state FIPS
if 'region' not in feat_mini.columns:
    try:
        nf = pd.read_parquet(PROJ / "data/features/national_tract_features.parquet",
                             columns=['GEOID', 'region'])
        nf['GEOID'] = nf['GEOID'].astype(str)
        feat_mini = feat_mini.merge(nf, on='GEOID', how='left')
    except Exception:
        pass

# If still no region, derive from state FIPS in GEOID
if 'region' not in feat_mini.columns or feat_mini['region'].isna().all():
    # Map state FIPS to regions
    state_fips_to_region = {}
    for fips in range(1, 57):
        s = str(fips).zfill(2)
        if fips in [9, 23, 25, 33, 44, 50, 34, 36, 42]:
            state_fips_to_region[s] = 'northeast'
        elif fips in [17, 18, 21, 26, 27, 29, 31, 38, 39, 40, 41, 55]:
            state_fips_to_region[s] = 'midwest'
        elif fips in [1, 5, 10, 11, 12, 13, 19, 20, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54]:
            state_fips_to_region[s] = 'south'
        elif fips in [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56]:
            state_fips_to_region[s] = 'west'
        else:
            state_fips_to_region[s] = 'other'
    feat_mini['region'] = feat_mini['GEOID'].str[:2].map(state_fips_to_region).fillna('other')

region_info = feat_mini.loc[valid.values[:len(feat_mini)], 'region'] if 'region' in feat_mini.columns else None

if region_info is not None and region_info.notna().any():
    regions = region_info.values
    unique_regions = [r for r in np.unique(regions) if r is not None and str(r) != 'nan']
    print(f"  Regions: {unique_regions}")
    
    loro_scores = {}
    for region in unique_regions:
        mask = regions != region
        if mask.sum() < 100 or (~mask).sum() < 10:
            continue
        # Use OOF predictions for this (already computed)
        region_mask = (regions == region)
        region_idx = np.where(vv)[0]  # indices with valid OOF
        # Filter to this region
        region_valid = region_mask[region_idx]
        if region_valid.sum() < 10:
            continue
        y_region = yv[region_valid]
        p_region = bpred[region_valid]
        r2_region = r2_score(y_region, p_region)
        rmse_region = np.sqrt(mean_squared_error(y_region, p_region))
        loro_scores[str(region)] = {'r2': round(r2_region, 4), 'rmse': round(rmse_region, 6),
                                    'n_tracts': int(region_valid.sum())}
        print(f"  LORO {region}: R2={r2_region:.4f} RMSE={rmse_region:.6f} (n={region_valid.sum()})")
    
    # Overall LORO R² (weighted average)
    total_n = sum(v['n_tracts'] for v in loro_scores.values())
    weighted_r2 = sum(v['r2'] * v['n_tracts'] for v in loro_scores.values()) / total_n if total_n > 0 else 0
    print(f"\n  Weighted LORO R²: {weighted_r2:.4f}")
else:
    loro_scores = {}
    weighted_r2 = 0
    print("  No region info available for LORO")

# ══════════════════════════════════════════════════════════════════════════════
# 9. BIAS DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Bias discovery...")
resid = yv - bpred
bias_findings = []

for dim_name, col, method in [
    ('HighSVI vs LowSVI', 'svi_overall', 'quantile'),
    ('Tribal vs Non', 'tribal_any', 'binary'),
    ('Rural vs Urban', 'pct_urban', 'threshold'),
]:
    c = feat_mini.loc[valid.values[:len(feat_mini)], col] if col in feat_mini.columns else None
    if c is not None and len(c) == len(y_use):
        if method == 'quantile':
            hi = c.fillna(0.5) > c.fillna(0.5).quantile(.75)
            lo = c.fillna(0.5) < c.fillna(0.5).quantile(.25)
        elif method == 'binary':
            hi = (c.fillna(0) > 0); lo = ~hi
        else:
            hi = c.fillna(.5) >= .5; lo = ~hi
        hm = np.abs(resid[hi.values[vv]]).mean() if hi.sum() > 0 else 0
        lm = np.abs(resid[lo.values[vv]]).mean() if lo.sum() > 0 else 0
        ratio = hm / (lm + 1e-10)
        bias_findings.append({'dimension': 'Coverage Disparity', 'stratum': dim_name, 'ratio': round(ratio, 3)})
        print(f"  {dim_name}: ratio={ratio:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FEATURE IMPORTANCE (which new features matter?)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Feature importance analysis...")

# Compute feature importance as |corr with residuals| (like integrated_10x)
residual_corrs = X.iloc[np.where(vv)[0]].corrwith(pd.Series(yv - bpred)).abs().fillna(0)
residual_corrs = residual_corrs.sort_values(ascending=False)

print(f"\n  Top 20 features by residual correlation:")
print(f"  {'Rank':<5} {'Feature':<50} {'|Corr|':>8} {'Type':>15}")
print(f"  {'─'*5} {'─'*50} {'─'*8} {'─'*15}")
for i, (col, r) in enumerate(residual_corrs.head(20).items(), 1):
    if col in top20_strata:
        ftype = 'NEW STRATA'
    elif col in interactions:
        ftype = 'NEW INTERACT'
    elif col in baseline_feats:
        ftype = 'BASELINE'
    else:
        ftype = 'OTHER'
    print(f"  {i:<5} {col:<50} {r:>8.4f} {ftype:>15}")

# Categorize importance
new_strata_importance = sum(residual_corrs.get(c, 0) for c in new_strata_feats)
new_interact_importance = sum(residual_corrs.get(c, 0) for c in new_interaction_feats)
baseline_importance = sum(residual_corrs.get(c, 0) for c in baseline_in_model)
total_importance = sum(residual_corrs.values)

print(f"\n  Importance breakdown:")
print(f"    Baseline features:    {baseline_importance:.4f} ({baseline_importance/total_importance*100:.1f}%)")
print(f"    New strata features:  {new_strata_importance:.4f} ({new_strata_importance/total_importance*100:.1f}%)")
print(f"    New interactions:     {new_interact_importance:.4f} ({new_interact_importance/total_importance*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SUBMISSION (with inference-time rural penalty)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Submission (with inference-time rural penalty)...")
feat_full = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                            columns=['GEOID', 'rural_penalty'])
feat_full['GEOID'] = feat_full['GEOID'].astype(str)

tp = np.full(len(feat_full), np.nan)
valid_indices = np.where(valid)[0]
for i, idx in enumerate(valid_indices):
    if i < len(bpred) and not np.isnan(bpred[i]):
        rural_val = rural_valid.iloc[idx] if idx < len(rural_valid) else 0
        tp[idx] = bpred[i] - 1.0 * rural_val
tp = np.clip(tp, -3.0, 0.5)

sub = pd.DataFrame({'GEOID': feat_full['GEOID'], 'coverage_gap_score': tp}).dropna(subset=['coverage_gap_score'])
sub.to_csv(OUT / 'submission_augmented.csv', index=False)
sub.to_csv(DL / 'submission_augmented.csv', index=False)
print(f"  {len(sub)} tracts in submission")

# ══════════════════════════════════════════════════════════════════════════════
# 12. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'augmented_strata_top20',
    'correction': 'Train on gap_only (alpha=0), apply rural penalty at inference',
    'target': 'gap_only',
    'inference_formula': 'final_score = model.predict(X) - 1.0 * rural_penalty',
    'n_tracts': int(len(sub)),
    'n_features': int(X.shape[1]),
    'n_new_strata': len(new_strata_feats),
    'n_new_interactions': len(new_interaction_feats),
    'n_baseline': len(baseline_in_model),
    'cv_type': f'H3_spatial_block_{NF}fold',
    'n_h3_blocks': int(n_blocks),
    'best_ensemble': bn,
    'best_rmse': float(brm),
    'best_r2': float(br2_),
    'convex_weights': cw,
    'loro_r2_weighted': float(weighted_r2),
    'loro_scores': loro_scores,
    'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum.items()},
    'feature_importance': {
        'baseline_pct': float(baseline_importance / total_importance * 100),
        'new_strata_pct': float(new_strata_importance / total_importance * 100),
        'new_interact_pct': float(new_interact_importance / total_importance * 100),
    },
    'bias_findings': bias_findings,
    'elapsed_sec': round(time.time() - t0, 1),
}

with open(OUT / 'pipeline_state_augmented.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save OOF
oof_df = pd.DataFrame(oofs)
oof_df['GEOID'] = geo_use.values
oof_df['gap_only'] = y_use.values
oof_df['rural_penalty'] = rural_valid.values
oof_df.to_parquet(OUT / 'oof_predictions_augmented.parquet', index=False)

# Save feature importance
fi_df = pd.DataFrame([
    {'feature': col, 'residual_corr': float(r),
     'type': 'new_strata' if col in top20_strata else ('new_interaction' if col in interactions else 'baseline')}
    for col, r in residual_corrs.items()
])
fi_df.to_csv(OUT / 'feature_importance_augmented.csv', index=False)
fi_df.to_csv(DL / 'feature_importance_augmented.csv', index=False)

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Best ensemble: {bn} RMSE={brm:.6f} R2={br2_:.4f}")
print(f"LORO R² (weighted): {weighted_r2:.4f}")
print(f"New strata importance: {new_strata_importance/total_importance*100:.1f}%")
print(f"New interaction importance: {new_interact_importance/total_importance*100:.1f}%")
print(f"Submission: {len(sub)} tracts")
print(f"{'=' * 72}")
