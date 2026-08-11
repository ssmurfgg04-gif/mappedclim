#!/usr/bin/env python3
"""v3 FINAL pipeline: XGB+LGB+CAT+ET+DART, H3-CV, 70/30 blend, stacking"""
import numpy as np, pandas as pd, json, time, sys, gc
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import h3
sys.stdout.reconfigure(line_buffering=True)
SEED = 42; np.random.seed(SEED)
PROJ = Path(__file__).resolve().parent.parent  # project root from scripts/
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = PROJ.parent / "download"; DL.mkdir(parents=True, exist_ok=True)  # sibling of project root
NF = 3

print("v3 FINAL: 5-model ensemble + H3-CV + 70/30 blend")
t0 = time.time()

feat = pd.read_parquet(OUT / "engineered_features_v3.parquet")
print(f"Features: {feat.shape}")

drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
        'building_gap','road_gap','building_ratio','road_ratio','building_count_ratio',
        'building_count_gap','road_count_ratio','road_count_gap','road_length_ratio',
        'road_length_gap','poi_facility_gap','poi_to_facility_ratio',
        'coverage_gap_score','coverage_gap','gap_score','coverage_score']
feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy(); y = feat['building_gap'].copy(); geo = feat['GEOID'].astype(str).copy()
v = y.notna(); X, y, geo = X[v], y[v], geo[v]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)
s = X.std(); X = X[s[s > 1e-10].index]
cs = X.corrwith(y).abs().fillna(0); X = X[cs.sort_values(ascending=False).head(80).index]
cm = X.corr().abs(); up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
td = [c for c in up.columns if any(up[c] > 0.98)]; X = X.drop(columns=td)
print(f"{X.shape[1]} feats, {X.shape[0]} tracts")

lats = feat.loc[v, 'centroid_lat']; lons = feat.loc[v, 'centroid_lon']
blocks = pd.Series([h3.latlng_to_cell(float(la), float(lo), 4) if not (np.isnan(la) or np.isnan(lo)) else 'unk'
                    for la, lo in zip(lats.values, lons.values)], index=geo.index)
print(f"H3: {blocks.nunique()} blocks")

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i%NF for i, b in enumerate(ub)}; sf = blocks.map(fa).values
splits = [(np.where(sf!=f)[0], np.where(sf==f)[0]) for f in range(NF)]

def train_model(model, name):
    oof = np.full(len(y), np.nan); scores = []
    for fi, (ti, vi) in enumerate(splits):
        m = type(model)(**model.get_params())
        Xt, yt, Xv, yv = X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi]
        try:
            if isinstance(m, xgb.XGBRegressor): m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            elif isinstance(m, lgb.LGBMRegressor): m.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
            elif isinstance(m, CatBoostRegressor): m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=30, verbose=0)
            else: m.fit(Xt, yt)
        except Exception as e: print(f"  {name} F{fi} err: {e}"); continue
        p = m.predict(Xv); oof[vi] = p
        rmse = np.sqrt(mean_squared_error(yv, p)); r2 = r2_score(yv, p)
        scores.append((rmse, r2)); print(f"  {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({time.time()-t0:.0f}s)")
        del m; gc.collect()
    rmse_m = np.mean([s[0] for s in scores]); r2_m = np.mean([s[1] for s in scores])
    print(f"  {name}: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
    return oof, rmse_m, r2_m

oofs = {}; msum = {}

print("\n[1] XGBoost...")
o, r, r2 = train_model(xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, tree_method='hist', random_state=SEED), 'XGB')
oofs['xgb'] = o; msum['xgb'] = (r, r2); gc.collect()

print("\n[2] LightGBM GBDT...")
o, r, r2 = train_model(lgb.LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10, boosting_type='gbdt', random_state=SEED, verbose=-1), 'LGB')
oofs['lgb'] = o; msum['lgb'] = (r, r2); gc.collect()

print("\n[3] CatBoost...")
o, r, r2 = train_model(CatBoostRegressor(iterations=600, depth=7, learning_rate=0.03, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5, random_seed=SEED, verbose=0, thread_count=1, allow_writing_files=False), 'CAT')
oofs['cat'] = o; msum['cat'] = (r, r2); gc.collect()

print("\n[4] ExtraTrees...")
o, r, r2 = train_model(ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_split=5, random_state=SEED, n_jobs=1), 'ET')
oofs['et'] = o; msum['et'] = (r, r2); gc.collect()

print("\n[5] LightGBM DART...")
o, r, r2 = train_model(lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10, boosting_type='dart', random_state=SEED, verbose=-1, drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); gc.collect()

# Ensemble
print("\nENSEMBLE")
ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y.values[vv]
res = minimize(lambda w: np.sqrt(mean_squared_error(yv, mv@w)), np.ones(len(ns))/len(ns), method='SLSQP', bounds=[(0,1)]*len(ns), constraints={'type':'eq','fun':lambda w:sum(w)-1})
cw = {n: round(float(w),4) for n, w in zip(ns, res.x)}; cp = mv@res.x; cr_ = res.fun; cr2 = r2_score(yv, cp)
print(f"  Convex: RMSE={cr_:.6f} R2={cr2:.4f} w={cw}")

sn = sorted(cw.items(), key=lambda x: -x[1]); t1, t2 = sn[0][0], sn[1][0]
a, b = oofs[t1][vv], oofs[t2][vv]
sh = min(a.min(), b.min())
a_s = a - sh + 1e-8 if sh < 0 else a + 1e-8
b_s = b - sh + 1e-8 if sh < 0 else b + 1e-8
gp = np.sqrt(a_s * b_s)
if sh < 0: gp = gp + sh - 1e-8
hp = 0.70 * gp + 0.30 * ((a + b) / 2)
hr_ = np.sqrt(mean_squared_error(yv, hp)); hr2 = r2_score(yv, hp)
print(f"  Hybrid 70/30: RMSE={hr_:.6f} R2={hr2:.4f}")

bv = blocks[vv].values; ub2 = list(set(bv)); np.random.seed(SEED); np.random.shuffle(ub2)
fa2 = {b: i%NF for i, b in enumerate(ub2)}; sf2 = np.array([fa2.get(b, 0) for b in bv])
s2 = [(np.where(sf2!=f)[0], np.where(sf2==f)[0]) for f in range(NF)]
so = np.full(len(yv), np.nan)
for fi, (ti, vi) in enumerate(s2):
    meta = Ridge(alpha=1.0, random_state=SEED); meta.fit(mv[ti], yv[ti]); so[vi] = meta.predict(mv[vi])
v2 = ~np.isnan(so); sr_ = np.sqrt(mean_squared_error(yv[v2], so[v2])); sr2 = r2_score(yv[v2], so[v2])
print(f"  Stack(Ridge): RMSE={sr_:.6f} R2={sr2:.4f}")

best = min([('convex', cr_, cr2, cp), ('hybrid_70/30', hr_, hr2, hp), ('stack', sr_, sr2, so)], key=lambda x: x[1])
bn, brm, br2_, bpred = best
print(f"\n  >>> BEST: {bn} RMSE={brm:.6f} R2={br2_:.4f}")

# Bias
print("\nBias Discovery...")
resid = yv - bpred; out = []
fd = feat[v]
for dim, col, q in [('HighSVI vs LowSVI', 'svi_overall', True), ('Tribal vs Non', 'tribal_any', False), ('Rural vs Urban', 'pct_urban', False)]:
    c = fd.get(col)
    if c is not None:
        if q: hi = c.fillna(0.5) > c.fillna(0.5).quantile(.75); lo = c.fillna(0.5) < c.fillna(0.5).quantile(.25)
        elif col == 'tribal_any': hi = (c.fillna(0) > 0); lo = ~hi
        else: hi = c.fillna(.5) >= .5; lo = ~hi
        hm, lm = np.abs(resid[hi]).mean(), np.abs(resid[lo]).mean()
        out.append({'dim': 'CovDisp', 'stratum': dim, 'ratio': round(hm/(lm+1e-10), 3)})
print(f"  {len(out)} findings")
bdf = pd.DataFrame(out)

# Submission
print("\nSubmission...")
tp = np.full(len(feat), np.nan)
for i, idx in enumerate(np.where(v)[0]):
    if i < len(bpred) and not np.isnan(bpred[i]): tp[idx] = bpred[i]
tp = np.clip(tp, -3.0, 0.5)
sub = pd.DataFrame({'GEOID': feat['GEOID'].astype(str), 'coverage_gap_score': tp}).dropna(subset=['coverage_gap_score'])
sub.to_csv(OUT / 'submission.csv', index=False); sub.to_csv(DL / 'submission.csv', index=False)
print(f"  {len(sub)} tracts")

leak = json.load(open(OUT / 'leakage_analysis.json'))
res_final = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'pipeline': 'ultimate_v3', 'target': 'building_gap',
    'cv_type': 'H3_spatial_block_3fold', 'n_blocks': leak['n_h3_blocks'], 'n_features': int(X.shape[1]),
    'n_tracts': len(sub), 'best_ensemble': bn, 'best_rmse': float(brm), 'best_r2': float(br2_),
    'convex_weights': cw, 'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum.items()},
    'leakage_check': leak, 'elapsed_sec': round(time.time()-t0, 1)
}
with open(OUT / 'pipeline_state.json', 'w') as f: json.dump(res_final, f, indent=2, default=str)
bdf.to_csv(OUT / 'comprehensive_bias_findings.csv', index=False)
pd.DataFrame([{'model': k, 'rmse': v[0], 'r2': v[1]} for k, v in msum.items()]).to_csv(OUT / 'model_comparison.csv', index=False)

el = time.time() - t0
print(f"\nDONE in {el:.0f}s | {bn} RMSE={brm:.6f} R2={br2_:.4f}")
print(f"Leakage: County R2={leak['county_cv_r2']:.4f} -> H3 R2={leak['h3_cv_r2']:.4f}")
print(f"Submission: {len(sub)} tracts saved")
