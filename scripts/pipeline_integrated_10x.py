#!/usr/bin/env python3
"""
INTEGRATED 10x PIPELINE — Merges all advanced features into a single training set
and retrains the ensemble to measure actual R² improvement.

Features added:
  - Spatial autocorrelation shadows (4.1% incremental R²)
  - Temporal decay / stale source weighting (3.85% OOF residual)
  - Conformal prediction uncertainty (2.01× tribal, 3.59× rural)

Target: gap_only (alpha=0, Deterministic fix)
Inference: final_score = model.predict(X) - 1.0 * rural_penalty
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
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
NF = 3

print("=" * 72)
print("INTEGRATED 10x PIPELINE — All advanced features")
print("=" * 72)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASE FEATURES + TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading base features...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Base: {feat.shape}")

# DETERMINISTIC FIX: train on gap_only
assert 'gap_only' in feat.columns, "gap_only not found! Re-run Phase 1."
assert 'rural_penalty' in feat.columns, "rural_penalty not found!"

y = feat['gap_only'].copy()
rural_col = feat['rural_penalty'].copy()
geo = feat['GEOID'].astype(str).copy()

# ══════════════════════════════════════════════════════════════════════════════
# 2. MERGE SPATIAL SHADOW FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Merging spatial autocorrelation shadows...")
shadow = pd.read_parquet(OUT / "spatial_shadow_features.parquet")
shadow['GEOID'] = shadow['GEOID'].astype(str)

# Select the most informative shadow features
shadow_cols = ['GEOID', 'shadow_score', 'shadow_zscore', 'neighbor_gap_deviation',
               'neighbor_mean_building_gap', 'neighbor_tribal_fraction',
               'neighbor_rural_fraction', 'neighbor_count']
shadow_use = [c for c in shadow_cols if c in shadow.columns]
before = feat.shape[1]
feat = feat.merge(shadow[shadow_use], on='GEOID', how='left')
# Fill missing shadow values (tracts with no neighbors)
for c in shadow_use[1:]:
    if c in feat.columns:
        feat[c] = feat[c].fillna(0)
print(f"  +{feat.shape[1] - before} shadow features → {feat.shape[1]} total")
del shadow; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE TEMPORAL DECAY / STALE SOURCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Computing temporal decay features...")

# Source freshness metrics (already in feat from Phase 1)
freshness_cols = {
    'bldg_total_sources': 'higher_is_fresher',
    'bldg_source_diversity': 'higher_is_fresher',
    'source_coverage_fraction': 'higher_is_fresher',
    'source_diversity_entropy': 'higher_is_fresher',
    'poi_mean_confidence': 'higher_is_fresher',
    'poi_very_high_conf_fraction': 'higher_is_fresher',
}

# Normalize and compute stale score
normed = {}
for col, direction in freshness_cols.items():
    if col in feat.columns:
        vals = feat[col].fillna(0).values
        vmin, vmax = np.percentile(vals, [1, 99])
        if vmax > vmin:
            n = (vals - vmin) / (vmax - vmin)
            n = np.clip(n, 0, 1)
            if direction == 'higher_is_fresher':
                normed[col] = n  # already 0=stale, 1=fresh
            else:
                normed[col] = 1 - n
        else:
            normed[col] = np.full(len(feat), 0.5)

if normed:
    freshness_matrix = np.column_stack(list(normed.values()))
    freshness_mean = freshness_matrix.mean(axis=1)
    stale_source_score = 1 - freshness_mean  # 1 = completely stale

    feat['stale_source_score'] = stale_source_score
    feat['freshness_mean'] = freshness_mean

    # Key interactions
    bg_col = feat.get('building_gap', pd.Series(0, index=feat.index)).fillna(0)
    feat['stale_x_bldg_gap'] = stale_source_score * bg_col.values
    feat['stale_x_rural'] = stale_source_score * rural_col.values
    feat['stale_x_shadow'] = stale_source_score * feat.get('shadow_score', pd.Series(0, index=feat.index)).fillna(0).values

    print(f"  stale_source_score: mean={stale_source_score.mean():.4f}, std={stale_source_score.std():.4f}")
    print(f"  +5 temporal decay features → {feat.shape[1]} total")
else:
    print("  No freshness columns found, skipping temporal decay")

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE CONFORMAL UNCERTAINTY FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Computing conformal prediction uncertainty...")
oof_df = pd.read_parquet(OUT / "oof_predictions_merged.parquet")

# Model columns in OOF
model_cols = [c for c in oof_df.columns if c in ['xgb', 'lgb', 'et', 'cat', 'lgb_dart']]
if len(model_cols) >= 2:
    oof_matrix = oof_df[model_cols].values
    # prediction spread and std
    with np.errstate(all='ignore'):
        pred_spread = np.nanmax(oof_matrix, axis=1) - np.nanmin(oof_matrix, axis=1)
        pred_std = np.nanstd(oof_matrix, axis=1)
    pred_spread = np.nan_to_num(pred_spread, nan=0)
    pred_std = np.nan_to_num(pred_std, nan=0)

    feat['prediction_spread'] = pred_spread
    feat['prediction_std'] = pred_std
    feat['uncertainty_x_bldg'] = pred_std * feat.get('building_gap', pd.Series(0, index=feat.index)).fillna(0).values
    feat['uncertainty_x_rural'] = pred_std * rural_col.values

    # Flag high-uncertainty tracts (top 10%)
    spread_p90 = np.percentile(pred_spread[pred_spread > 0], 90) if (pred_spread > 0).any() else 0
    feat['high_uncertainty_flag'] = (pred_spread >= spread_p90).astype(float)

    print(f"  prediction_spread: mean={pred_spread.mean():.6f}, p90={spread_p90:.6f}")
    print(f"  High uncertainty tracts: {feat['high_uncertainty_flag'].sum():.0f}")
    print(f"  +5 uncertainty features → {feat.shape[1]} total")
else:
    print(f"  Only {len(model_cols)} model OOF columns, skipping conformal features")
    feat['prediction_spread'] = 0
    feat['prediction_std'] = 0
    feat['high_uncertainty_flag'] = 0

del oof_df; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 5. PREPARE FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Preparing feature matrix...")

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
X, y, geo = X[valid], y[valid], geo[valid]
rural_col = rural_col[valid]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance
s = X.std()
X = X[s[s > 1e-10].index]

# Select top features by correlation with target
cs = X.corrwith(y).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(80).index]  # 80 features (up from 60)

# Remove highly correlated duplicates
cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)

n_features = X.shape[1]
print(f"  {n_features} features, {X.shape[0]} tracts")

# ══════════════════════════════════════════════════════════════════════════════
# 6. H3 SPATIAL CV SPLITS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Computing H3 spatial blocks...")
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
], index=geo.index)
n_blocks = blocks.nunique()
print(f"  H3 blocks: {n_blocks}")

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

del feat; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN 5-MODEL ENSEMBLE (XGB + LGB + ET + CatBoost + DART)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Training 5-model ensemble on integrated features...")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("  CatBoost not available, using 4 models")

def train_model(model, name):
    oof = np.full(len(y), np.nan); scores = []
    for fi, (ti, vi) in enumerate(splits):
        m = type(model)(**model.get_params())
        Xt, yt, Xv, yv = X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi]
        try:
            if isinstance(m, xgb.XGBRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            elif isinstance(m, lgb.LGBMRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            elif HAS_CAT and isinstance(m, CatBoostRegressor):
                m.fit(Xt, yt, eval_set=(Xv, yv),
                      early_stopping_rounds=20, verbose=0)
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

# [2] LightGBM GBDT
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

# [4] CatBoost
if HAS_CAT:
    print("\n  [4] CatBoost...")
    o, r, r2 = train_model(
        CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3.0, random_strength=1.0,
                          bagging_temperature=0.5, random_seed=SEED,
                          verbose=0, thread_count=1, allow_writing_files=False), 'CAT')
    oofs['cat'] = o; msum['cat'] = (r, r2); gc.collect()

# [5] LightGBM DART
print("\n  [5] LightGBM DART...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='dart', random_state=SEED, verbose=-1,
                      drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 8. ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Ensembling...")

ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y.values[vv]

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
# 9. FEATURE IMPORTANCE — which 10x features matter?
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Feature importance analysis...")

# Use correlation with OOF residuals as importance proxy
resid = yv - bpred
feat_imp = {}
for i, c in enumerate(X.columns):
    vv_idx = np.where(vv)[0]
    corr = np.corrcoef(X.iloc[vv_idx, i].values, resid)[0, 1]
    if not np.isnan(corr):
        feat_imp[c] = abs(corr)

# Sort and show top 20 + highlight 10x features
sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
print("\n  Top 20 features by |corr with residuals|:")
for rank, (name, imp) in enumerate(sorted_imp[:20], 1):
    marker = ""
    if name in ['shadow_score', 'shadow_zscore', 'neighbor_gap_deviation',
                'neighbor_tribal_fraction', 'neighbor_rural_fraction']:
        marker = " ← SHADOW"
    elif name in ['stale_source_score', 'stale_x_bldg_gap', 'stale_x_rural', 'stale_x_shadow']:
        marker = " ← STALE"
    elif name in ['prediction_spread', 'prediction_std', 'uncertainty_x_bldg', 'uncertainty_x_rural']:
        marker = " ← UNCERTAINTY"
    print(f"    {rank:2d}. {name:40s} {imp:.6f}{marker}")

# Compute 10x feature group contributions
shadow_imp = sum(v for k, v in feat_imp.items() if k in ['shadow_score', 'shadow_zscore',
                    'neighbor_gap_deviation', 'neighbor_tribal_fraction', 'neighbor_rural_fraction'])
stale_imp = sum(v for k, v in feat_imp.items() if k.startswith('stale'))
unc_imp = sum(v for k, v in feat_imp.items() if k.startswith('prediction_') or k.startswith('uncertainty_'))
total_imp = sum(feat_imp.values())

print(f"\n  10x feature group contributions:")
print(f"    Shadow features:  {shadow_imp:.6f} ({100*shadow_imp/total_imp:.1f}% of total importance)")
print(f"    Stale features:   {stale_imp:.6f} ({100*stale_imp/total_imp:.1f}%)")
print(f"    Uncertainty:      {unc_imp:.6f} ({100*unc_imp/total_imp:.1f}%)")
print(f"    All 10x combined: {shadow_imp+stale_imp+unc_imp:.6f} ({100*(shadow_imp+stale_imp+unc_imp)/total_imp:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. BIAS DISCOVERY ON INTEGRATED MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Bias discovery on integrated model...")

# Reload strata columns for bias analysis
feat_mini = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                            columns=['GEOID', 'svi_overall', 'tribal_any', 'pct_urban'])
feat_mini['GEOID'] = feat_mini['GEOID'].astype(str)

resid_all = yv - bpred  # on gap_only (not final score)
bias_findings = []

for dim_name, col, method in [
    ('HighSVI vs LowSVI', 'svi_overall', 'quantile'),
    ('Tribal vs Non', 'tribal_any', 'binary'),
    ('Rural vs Urban', 'pct_urban', 'threshold'),
]:
    c = feat_mini.loc[valid.values[:len(feat_mini)], col] if col in feat_mini.columns else None
    if c is not None and len(c) == len(y):
        vv_idx2 = np.where(vv)[0]
        if method == 'quantile':
            hi = c.fillna(0.5) > c.fillna(0.5).quantile(.75)
            lo = c.fillna(0.5) < c.fillna(0.5).quantile(.25)
        elif method == 'binary':
            hi = (c.fillna(0) > 0); lo = ~hi
        else:
            hi = c.fillna(.5) >= .5; lo = ~hi
        hm = np.abs(resid_all[hi.values[vv_idx2]]).mean() if hi.sum() > 0 else 0
        lm = np.abs(resid_all[lo.values[vv_idx2]]).mean() if lo.sum() > 0 else 0
        ratio = hm / (lm + 1e-10)
        bias_findings.append({'dimension': 'Coverage Disparity', 'stratum': dim_name, 'ratio': round(ratio, 3)})
        print(f"  {dim_name}: ratio={ratio:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SUBMISSION WITH INFERENCE-TIME RURAL PENALTY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Submission (inference-time rural penalty)...")
feat_full = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                            columns=['GEOID', 'rural_penalty', 'tribal_any', 'pct_urban'])
feat_full['GEOID'] = feat_full['GEOID'].astype(str)

# Apply: final_score = model.predict(X) - 1.0 * rural_penalty
tp = np.full(len(feat_full), np.nan)
valid_indices = np.where(valid)[0]
for i, idx in enumerate(valid_indices):
    if i < len(bpred) and not np.isnan(bpred[i]):
        rural_val = rural_col.iloc[idx] if idx < len(rural_col) else 0
        tp[idx] = bpred[i] - 1.0 * rural_val
tp = np.clip(tp, -3.0, 0.5)

sub = pd.DataFrame({'GEOID': feat_full['GEOID'], 'coverage_gap_score': tp}).dropna(subset=['coverage_gap_score'])
sub.to_csv(OUT / 'submission_integrated_10x.csv', index=False)
sub.to_csv(DL / 'submission_integrated_10x.csv', index=False)
print(f"  {len(sub)} tracts in submission")

# Tribal/rural analysis on final scores
tribal_mask = feat_full['tribal_any'].fillna(0) > 0
rural_mask = feat_full['pct_urban'].fillna(0.5) < 0.5
tribal_scores = tp[tribal_mask.values]
non_tribal_scores = tp[~tribal_mask.values]
rural_scores = tp[rural_mask.values]
urban_scores = tp[~rural_mask.values]

tribal_ratio = abs(np.nanmean(tribal_scores)) / (abs(np.nanmean(non_tribal_scores)) + 1e-10)
rural_ratio = abs(np.nanmean(rural_scores)) / (abs(np.nanmean(urban_scores)) + 1e-10)

print(f"\n  === Final Score Analysis ===")
print(f"  Tribal: mean={np.nanmean(tribal_scores):.4f} (n={tribal_mask.sum()})")
print(f"  Non-tribal: mean={np.nanmean(non_tribal_scores):.4f} (n={(~tribal_mask).sum()})")
print(f"  Tribal bias ratio: {tribal_ratio:.2f}×")
print(f"  Rural: mean={np.nanmean(rural_scores):.4f} (n={rural_mask.sum()})")
print(f"  Urban: mean={np.nanmean(urban_scores):.4f} (n={(~rural_mask).sum()})")
print(f"  Rural/Urban ratio: {rural_ratio:.2f}×")

# ══════════════════════════════════════════════════════════════════════════════
# 12. COMPARISON WITH BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12] Comparison with baseline (Phase 2)...")

# Load baseline results
try:
    with open(OUT / 'pipeline_state_merged.json') as f:
        baseline = json.load(f)
    baseline_r2 = baseline.get('best_r2', 0)
    baseline_rmse = baseline.get('best_rmse', 0)
    print(f"  Baseline R²: {baseline_r2:.4f} (3-model, 60 features)")
    print(f"  Integrated R²: {br2_:.4f} ({len(ns)}-model, {n_features} features)")
    r2_lift = br2_ - baseline_r2
    print(f"  R² change: {r2_lift:+.4f} ({100*r2_lift/abs(baseline_r2):+.1f}%)")
except:
    print("  Baseline state not found, skipping comparison")

# ══════════════════════════════════════════════════════════════════════════════
# 13. SAVE STATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13] Saving state...")

results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'integrated_10x_DETERMINISTIC_FIX',
    'correction': 'Train on gap_only (alpha=0), apply rural penalty at inference',
    'training_target': 'gap_only',
    'inference_formula': 'final_score = model.predict(X) - 1.0 * rural_penalty',
    'n_tracts': int(len(sub)), 'n_features': n_features,
    'cv_type': f'H3_spatial_block_{NF}fold', 'n_h3_blocks': int(n_blocks),
    'best_ensemble': bn, 'best_rmse': float(brm), 'best_r2': float(br2_),
    'convex_weights': cw,
    'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum.items()},
    'feature_importance': {
        'shadow_group_pct': float(100*shadow_imp/total_imp) if total_imp > 0 else 0,
        'stale_group_pct': float(100*stale_imp/total_imp) if total_imp > 0 else 0,
        'uncertainty_group_pct': float(100*unc_imp/total_imp) if total_imp > 0 else 0,
        'all_10x_pct': float(100*(shadow_imp+stale_imp+unc_imp)/total_imp) if total_imp > 0 else 0,
    },
    'bias_ratios': {
        'tribal': float(tribal_ratio),
        'rural_urban': float(rural_ratio),
    },
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(OUT / 'pipeline_state_integrated_10x.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save OOF
oof_out = pd.DataFrame(oofs)
oof_out['GEOID'] = geo.values
oof_out['gap_only'] = y.values
oof_out['rural_penalty'] = rural_col.values
oof_out['proxy_merged'] = y.values - 1.0 * rural_col.values
oof_out.to_parquet(OUT / 'oof_predictions_integrated_10x.parquet', index=False)

# Save feature importance
imp_df = pd.DataFrame(sorted_imp, columns=['feature', 'importance'])
imp_df.to_csv(OUT / 'feature_importance_integrated_10x.csv', index=False)

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Best ensemble: {bn} RMSE={brm:.6f} R2={br2_:.4f}")
print(f"Tribal bias: {tribal_ratio:.2f}×, Rural/Urban: {rural_ratio:.2f}×")
print(f"10x feature importance: {100*(shadow_imp+stale_imp+unc_imp)/total_imp:.1f}% of total")
print(f"Submission: {len(sub)} tracts → submission_integrated_10x.csv")
print(f"{'=' * 72}")
