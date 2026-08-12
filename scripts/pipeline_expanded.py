#!/usr/bin/env python3
"""Expanded feature pipeline — add 68 strong strata features and retrain"""
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
OUT = PROJ / "data/output"
DL = Path("/home/z/my-project/download")
NF = 3

print("=" * 72)
print("EXPANDED FEATURE PIPELINE — Add strong strata features")
print("=" * 72)
t0 = time.time()

# Load current features
print("\n[1] Loading features...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Base: {feat.shape}")

# Load strata
strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
strata['GEOID'] = strata['GEOID'].astype(str)

# Find strong features
merged = strata.merge(feat[['GEOID', 'gap_only']], on='GEOID', how='inner')
numeric_strata = [c for c in strata.columns if pd.api.types.is_numeric_dtype(strata[c]) and c != 'GEOID']
strong_features = []
for col in numeric_strata:
    if col in merged.columns and merged[col].notna().sum() > 1000:
        valid = merged[[col, 'gap_only']].dropna()
        if len(valid) > 1000:
            corr = valid[col].corr(valid['gap_only'])
            if not np.isnan(corr) and abs(corr) > 0.10:
                na_rate = strata[col].isna().mean()
                if na_rate < 0.95:
                    strong_features.append(col)

print(f"  Found {len(strong_features)} strong strata features (|r|>0.10, NaN<95%)")

# Merge into feat
existing = set(feat.columns)
new_cols = [c for c in strong_features if c not in existing]
if new_cols:
    before = feat.shape[1]
    to_merge = strata[['GEOID'] + new_cols].copy()
    feat = feat.merge(to_merge, on='GEOID', how='left')
    for c in new_cols:
        feat[c] = feat[c].fillna(-999)
    print(f"  +{feat.shape[1] - before} new strata features → {feat.shape[1]} total")

# Merge shadow features
shadow = pd.read_parquet(OUT / "spatial_shadow_features.parquet")
shadow['GEOID'] = shadow['GEOID'].astype(str)
shadow_use = [c for c in ['shadow_score', 'shadow_zscore', 'neighbor_gap_deviation',
                           'neighbor_tribal_fraction', 'neighbor_rural_fraction']
              if c in shadow.columns and c not in feat.columns]
if shadow_use:
    to_merge = shadow[['GEOID'] + shadow_use].copy()
    before = feat.shape[1]
    feat = feat.merge(to_merge, on='GEOID', how='left')
    for c in shadow_use:
        feat[c] = feat[c].fillna(0)
    print(f"  +{feat.shape[1] - before} shadow features → {feat.shape[1]} total")

# Save expanded features
feat.to_parquet(OUT / "engineered_features_expanded.parquet", index=False)
print(f"  Saved: {feat.shape}")

# Train
print("\n[2] Preparing feature matrix...")
y = feat['gap_only'].copy()
rural_col = feat['rural_penalty'].copy()
geo = feat['GEOID'].astype(str).copy()

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

s = X.std()
X = X[s[s > 1e-10].index]

cs = X.corrwith(y).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(60).index]  # Top 60 features (memory-safe)

cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)

n_features = X.shape[1]
print(f"  {n_features} features, {X.shape[0]} tracts")

# H3 blocks
lats = feat.loc[valid, 'centroid_lat'] if 'centroid_lat' in feat.columns else None
lons = feat.loc[valid, 'centroid_lon'] if 'centroid_lon' in feat.columns else None
if lats is None or lons is None or lats.isna().all():
    lats = pd.to_numeric(feat.loc[valid, 'INTPTLAT'], errors='coerce')
    lons = pd.to_numeric(feat.loc[valid, 'INTPTLON'], errors='coerce')

blocks = pd.Series([
    h3.latlng_to_cell(float(la), float(lo), 4)
    if not (np.isnan(la) or np.isnan(lo)) else 'unk'
    for la, lo in zip(lats.values, lons.values)
], index=geo.index)
n_blocks = blocks.nunique()

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

del feat; gc.collect()

# Train 5-model ensemble
print(f"\n[3] Training 5-model ensemble ({n_features} features)...")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False

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
                m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=20, verbose=0)
            else:
                m.fit(Xt, yt)
        except Exception as e:
            print(f"  {name} F{fi} err: {e}"); continue
        p = m.predict(Xv); oof[vi] = p
        rmse = np.sqrt(mean_squared_error(yv, p)); r2 = r2_score(yv, p)
        scores.append((rmse, r2))
        print(f"  {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({time.time()-t0:.0f}s)")
        del m; gc.collect()
    if scores:
        return oof, np.mean([s[0] for s in scores]), np.mean([s[1] for s in scores])
    return oof, 999, 0

oofs = {}; msum = {}

o, r, r2 = train_model(
    xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                     tree_method='hist', random_state=SEED), 'XGB')
oofs['xgb'] = o; msum['xgb'] = (r, r2); gc.collect()

o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='gbdt', random_state=SEED, verbose=-1), 'LGB')
oofs['lgb'] = o; msum['lgb'] = (r, r2); gc.collect()

o, r, r2 = train_model(
    ExtraTreesRegressor(n_estimators=50, max_depth=10,
                        min_samples_split=10, random_state=SEED, n_jobs=1), 'ET')
oofs['et'] = o; msum['et'] = (r, r2); gc.collect()

if HAS_CAT:
    o, r, r2 = train_model(
        CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3.0, random_seed=SEED,
                          verbose=0, thread_count=1, allow_writing_files=False), 'CAT')
    oofs['cat'] = o; msum['cat'] = (r, r2); gc.collect()

o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='dart', random_state=SEED, verbose=-1,
                      drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); gc.collect()

# Ensemble
print("\n[4] Ensembling...")
ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y.values[vv]

res = minimize(
    lambda w: np.sqrt(mean_squared_error(yv, mv @ w)),
    np.ones(len(ns)) / len(ns), method='SLSQP',
    bounds=[(0, 1)] * len(ns),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
cw = {n: round(float(w), 4) for n, w in zip(ns, res.x)}
cp = mv @ res.x; cr2 = r2_score(yv, cp)
print(f"  Convex: R2={cr2:.4f} w={cw}")

# Submission
print("\n[5] Submission...")
feat_full = pd.read_parquet(OUT / "engineered_features_expanded.parquet",
                            columns=['GEOID', 'rural_penalty', 'tribal_any', 'pct_urban'])
feat_full['GEOID'] = feat_full['GEOID'].astype(str)

bpred = cp
tp = np.full(len(feat_full), np.nan)
valid_indices = np.where(valid)[0]
for i, idx in enumerate(valid_indices):
    if i < len(bpred) and not np.isnan(bpred[i]):
        rural_val = rural_col.iloc[idx] if idx < len(rural_col) else 0
        tp[idx] = bpred[i] - 1.0 * rural_val
tp = np.clip(tp, -3.0, 0.5)

sub = pd.DataFrame({'GEOID': feat_full['GEOID'], 'coverage_gap_score': tp}).dropna(subset=['coverage_gap_score'])
sub.to_csv(OUT / 'submission_expanded.csv', index=False)
sub.to_csv(DL / 'submission_expanded.csv', index=False)

tribal_mask = feat_full['tribal_any'].fillna(0) > 0
tribal_ratio = abs(np.nanmean(tp[tribal_mask.values])) / (abs(np.nanmean(tp[~tribal_mask.values])) + 1e-10)

# Top features
print("\n[6] Top 20 features by correlation with target:")
cs_full = X.corrwith(y).abs().fillna(0).sort_values(ascending=False)
for rank, (name, imp) in enumerate(cs_full.head(20).items(), 1):
    print(f"  {rank:2d}. {name:45s} |r|={imp:.4f}")

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Features: {n_features} (was 49 baseline)")
print(f"Convex R2: {cr2:.4f}, weights: {cw}")
print(f"Tribal bias: {tribal_ratio:.2f}x")
print(f"Submission: {len(sub)} tracts → submission_expanded.csv")
print(f"{'=' * 72}")
