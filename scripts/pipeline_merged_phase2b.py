#!/usr/bin/env python3
"""
MERGED PROXY PIPELINE — Phase 2b: Add CatBoost + DART to existing 3-model OOF predictions
Loads existing OOF and adds 2 more models, then re-ensembles.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42
NF = 3

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"
DL = Path("/home/z/my-project/download")

print("=" * 72)
print("PHASE 2b: Add CatBoost + DART -> Full 5-model ensemble")
print("=" * 72)
t0 = time.time()

# Load existing OOF
print("\n[1] Loading existing OOF predictions...")
oof_df = pd.read_parquet(OUT / "oof_predictions_merged.parquet")
print(f"  OOF shape: {oof_df.shape}")

y = oof_df['proxy_merged'].values
geo = oof_df['GEOID'].values

# Load engineered features for training
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)

drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
             'building_count_ratio', 'building_count_gap',
             'road_count_ratio', 'road_count_gap', 'road_length_ratio', 'road_length_gap',
             'poi_facility_gap', 'poi_to_facility_ratio',
             'poi_facility_gap_corrected', 'poi_to_facility_ratio_corrected',
             'building_area_gap',
             'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
             'proxy_merged']

feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy()

valid = pd.Series(y).notna()
X = X[valid]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)
s = X.std(); X = X[s[s > 1e-10].index]

cs = X.corrwith(pd.Series(y)).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(60).index]

cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)
print(f"  {X.shape[1]} features, {X.shape[0]} tracts")

# H3 blocks
import h3
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
], index=range(len(y)))

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

del feat; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN CATBOOST + DART
# ══════════════════════════════════════════════════════════════════════════════
y_series = pd.Series(y)

def train_model(model, name):
    oof = np.full(len(y), np.nan); scores = []
    for fi, (ti, vi) in enumerate(splits):
        m = type(model)(**model.get_params())
        Xt, yt, Xv, yv = X.iloc[ti], y_series.iloc[ti], X.iloc[vi], y_series.iloc[vi]
        try:
            if isinstance(m, lgb.LGBMRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            elif isinstance(m, CatBoostRegressor):
                m.fit(Xt, yt, eval_set=(Xv, yv),
                      early_stopping_rounds=20, verbose=0)
        except Exception as e:
            print(f"  {name} F{fi} err: {e}"); continue
        p = m.predict(Xv); oof[vi] = p
        rmse = np.sqrt(mean_squared_error(yv, p)); r2 = r2_score(yv, p)
        scores.append((rmse, r2))
        print(f"  {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({time.time()-t0:.0f}s)")
        del m; gc.collect()
    if scores:
        rmse_m = np.mean([s[0] for s in scores]); r2_m = np.mean([s[1] for s in scores])
        print(f"  {name} mean: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
        return oof, rmse_m, r2_m
    return oof, 999, 0

oofs = {}
# Load existing OOF
for m in ['xgb', 'lgb', 'et']:
    if m in oof_df.columns:
        oofs[m] = oof_df[m].values

msum_existing = {}
with open(OUT / 'pipeline_state_merged.json') as f:
    state = json.load(f)
for k, v in state['models'].items():
    msum_existing[k] = (v['rmse'], v['r2'])

# [4] CatBoost
print("\n  [4] CatBoost...")
o, r, r2 = train_model(
    CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                      l2_leaf_reg=3.0, random_strength=1.0,
                      bagging_temperature=0.5, random_seed=SEED,
                      verbose=0, thread_count=1, allow_writing_files=False), 'CAT')
oofs['cat'] = o; msum_existing['cat'] = (r, r2); gc.collect()

# [5] DART
print("\n  [5] LightGBM DART...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='dart', random_state=SEED, verbose=-1,
                      drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum_existing['lgb_dart'] = (r, r2); gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# FULL 5-MODEL ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Full 5-model ensemble...")

ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y[vv]

# Convex
res = minimize(
    lambda w: np.sqrt(mean_squared_error(yv, mv @ w)),
    np.ones(len(ns)) / len(ns), method='SLSQP',
    bounds=[(0, 1)] * len(ns),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
cw = {n: round(float(w), 4) for n, w in zip(ns, res.x)}
cp = mv @ res.x; cr_ = res.fun; cr2 = r2_score(yv, cp)
print(f"  Convex: RMSE={cr_:.6f} R2={cr2:.4f} w={cw}")

# Simple avg
ap = mv.mean(axis=1)
ar_ = np.sqrt(mean_squared_error(yv, ap)); ar2 = r2_score(yv, ap)
print(f"  Simple avg: RMSE={ar_:.6f} R2={ar2:.4f}")

best = min([('convex', cr_, cr2, cp), ('simple_avg', ar_, ar2, ap)], key=lambda x: x[1])
bn, brm, br2_, bpred = best
print(f"\n  >>> BEST: {bn} RMSE={brm:.6f} R2={br2_:.4f}")

# Update submission
print("\n[4] Updating submission...")
feat_full = pd.read_parquet(OUT / "engineered_features_merged.parquet", columns=['GEOID'])
feat_full['GEOID'] = feat_full['GEOID'].astype(str)

tp = np.full(len(feat_full), np.nan)
valid_mask = valid.values
for i in range(len(bpred)):
    if not np.isnan(bpred[i]):
        # Find the ith valid index
        pass

# Simpler: just re-create submission from OOF
sub_df = pd.DataFrame({'GEOID': geo, 'coverage_gap_score': np.clip(bpred, -3.0, 0.5)})
sub_df.to_csv(OUT / 'submission_merged_v2.csv', index=False)
sub_df.to_csv(DL / 'submission_merged_v2.csv', index=False)
print(f"  {len(sub_df)} tracts in submission")

# Save state
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'merged_proxy_honest_5model',
    'proxy_formula': 'proxy = -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg)) - 1.0*(1-pct_urban)',
    'n_tracts': int(len(sub_df)), 'n_features': int(X.shape[1]),
    'cv_type': f'H3_spatial_block_{NF}fold', 'n_h3_blocks': int(blocks.nunique()),
    'best_ensemble': bn, 'best_rmse': float(brm), 'best_r2': float(br2_),
    'convex_weights': cw,
    'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum_existing.items()},
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(OUT / 'pipeline_state_merged_v2.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save all OOF
oof_all = pd.DataFrame(oofs)
oof_all['GEOID'] = geo; oof_all['proxy_merged'] = y
oof_all.to_parquet(OUT / 'oof_predictions_merged_v2.parquet', index=False)

el = time.time() - t0
print(f"\nDONE in {el:.0f}s | {bn} RMSE={brm:.6f} R2={br2_:.4f} | 5 models | {len(sub_df)} tracts")
