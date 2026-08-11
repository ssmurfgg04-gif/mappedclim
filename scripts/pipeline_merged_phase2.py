#!/usr/bin/env python3
"""
MERGED PROXY PIPELINE — Phase 2: Train ensemble (streamlined for 85K tracts)
Reduces to XGB + LGB + ET (3 models), fewer estimators, 3-fold H3 CV.
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
NF = 3

print("=" * 72)
print("PHASE 2: Train 3-model ensemble + H3-CV (streamlined)")
print("=" * 72)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD + PREP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading engineered features...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
print(f"  Shape: {feat.shape}")

y = feat['proxy_merged'].copy()
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
             'proxy_merged']

feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy()

valid = y.notna()
X, y, geo = X[valid], y[valid], geo[valid]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

s = X.std()
X = X[s[s > 1e-10].index]

cs = X.corrwith(y).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(60).index]  # 60 features

cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)

print(f"  {X.shape[1]} features, {X.shape[0]} tracts")

# H3 blocks
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
# TRAIN 3-MODEL ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Training 3-model ensemble...")

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
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Ensembling...")

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
# BIAS DISCOVERY + SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Bias discovery...")

resid = yv - bpred
bias_findings = []

# We need the strata columns for bias analysis
feat_mini = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                            columns=['GEOID', 'svi_overall', 'tribal_any', 'pct_urban'])
feat_mini['GEOID'] = feat_mini['GEOID'].astype(str)

for dim_name, col, method in [
    ('HighSVI vs LowSVI', 'svi_overall', 'quantile'),
    ('Tribal vs Non', 'tribal_any', 'binary'),
    ('Rural vs Urban', 'pct_urban', 'threshold'),
]:
    c = feat_mini.loc[valid.values[:len(feat_mini)], col] if col in feat_mini.columns else None
    if c is not None and len(c) == len(y):
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

bdf = pd.DataFrame(bias_findings)

# Submission
print("\n[5] Submission...")
feat_full = pd.read_parquet(OUT / "engineered_features_merged.parquet", columns=['GEOID'])
feat_full['GEOID'] = feat_full['GEOID'].astype(str)

tp = np.full(len(feat_full), np.nan)
valid_indices = np.where(valid)[0]
for i, idx in enumerate(valid_indices):
    if i < len(bpred) and not np.isnan(bpred[i]):
        tp[idx] = bpred[i]
tp = np.clip(tp, -3.0, 0.5)

sub = pd.DataFrame({'GEOID': feat_full['GEOID'], 'coverage_gap_score': tp}).dropna(subset=['coverage_gap_score'])
sub.to_csv(OUT / 'submission_merged.csv', index=False)
sub.to_csv(DL / 'submission_merged.csv', index=False)
print(f"  {len(sub)} tracts in submission")

# Save results
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'merged_proxy_honest',
    'proxy_formula': 'proxy = -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg)) - 1.0*(1-pct_urban)',
    'target': 'proxy_merged',
    'n_tracts': int(len(sub)), 'n_features': int(X.shape[1]),
    'cv_type': f'H3_spatial_block_{NF}fold', 'n_h3_blocks': int(n_blocks),
    'best_ensemble': bn, 'best_rmse': float(brm), 'best_r2': float(br2_),
    'convex_weights': cw,
    'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum.items()},
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(OUT / 'pipeline_state_merged.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

bdf.to_csv(OUT / 'bias_findings_merged.csv', index=False)
pd.DataFrame([{'model': k, 'rmse': v[0], 'r2': v[1]} for k, v in msum.items()]).to_csv(
    OUT / 'model_comparison_merged.csv', index=False)

oof_df = pd.DataFrame(oofs)
oof_df['GEOID'] = geo.values
oof_df['proxy_merged'] = y.values
oof_df.to_parquet(OUT / 'oof_predictions_merged.parquet', index=False)

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Best ensemble: {bn} RMSE={brm:.6f} R2={br2_:.4f}")
print(f"Submission: {len(sub)} tracts saved to {DL / 'submission_merged.csv'}")
print(f"{'=' * 72}")
