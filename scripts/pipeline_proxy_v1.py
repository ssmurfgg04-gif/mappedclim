#!/usr/bin/env python3
"""
Implement proxy_v1 directional target and retrain the full 5-model ensemble.

proxy_v1 = -mean(building_gap, road_gap, poi_facility_gap_corrected) - 2.0 * svi_overall

This is directionally correct:
  - Under-mapped tracts (negative gaps) → more negative proxy → worse score
  - High SVI (vulnerable) → more negative proxy → worse score
  - Well-mapped + low SVI → less negative → better score

Retrains XGB + LGB + CAT + ET + DART with H3 spatial block 3-fold CV.
Saves models and produces submission on proxy_v1.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import h3

SEED = 42; np.random.seed(SEED)
NF = 3  # number of H3 CV folds

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = Path("/home/z/my-project/download"); DL.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJ / "models"; MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("PROXY_V1 PIPELINE: Directional target + 5-model ensemble + H3-CV")
print("proxy_v1 = -mean(building_gap, road_gap, poi_corrected) - 2.0*svi_overall")
print("=" * 72)
t0 = time.time()

# ── Load features ─────────────────────────────────────────────────────────────
print("\n[1] Loading features...")
feat = pd.read_parquet(OUT / "engineered_features_v3.parquet")
print(f"    Shape: {feat.shape}")

# ── Compute proxy_v1 ──────────────────────────────────────────────────────────
print("\n[2] Computing proxy_v1 directional target...")

# Merge SVI from strata if not in features
if 'svi_overall' not in feat.columns or feat['svi_overall'].isna().all():
    print("    Merging SVI from national strata table...")
    strata_path = PROJ / "kaggle_dataset/national-strata-tract-table.parquet"
    strata = pd.read_parquet(strata_path, columns=['GEOID', 'svi_overall'])
    svi_map = strata.set_index('GEOID')['svi_overall']
    feat['svi_overall'] = feat['GEOID'].astype(str).map(svi_map)
    print(f"    SVI merged: non-null={feat['svi_overall'].notna().sum()}")

# Get gap columns
building_gap = feat['building_gap'].fillna(0) if 'building_gap' in feat.columns else pd.Series(0, index=feat.index)
road_gap = feat['road_gap'].fillna(0) if 'road_gap' in feat.columns else pd.Series(0, index=feat.index)

# Use corrected POI gap if available, else fall back
if 'poi_facility_gap_corrected' in feat.columns:
    poi_gap = feat['poi_facility_gap_corrected'].fillna(0)
    print("    Using poi_facility_gap_corrected")
elif 'poi_facility_gap' in feat.columns:
    poi_gap = feat['poi_facility_gap'].fillna(0)
    print("    Using poi_facility_gap (no corrected version)")
else:
    poi_gap = road_gap.copy()
    print("    WARNING: No POI gap available, using road_gap as fallback")

svi_overall = feat['svi_overall'].fillna(0.5)  # default to median SVI

# proxy_v1 = -mean(building_gap, road_gap, poi_corrected) - 2.0 * svi_overall
gap_mean = np.mean([building_gap.values, road_gap.values, poi_gap.values], axis=0)
proxy_v1 = -gap_mean - 2.0 * svi_overall.values

feat['proxy_v1'] = proxy_v1

print(f"\n    proxy_v1 statistics:")
print(f"      mean  = {proxy_v1.mean():.4f}")
print(f"      std   = {proxy_v1.std():.4f}")
print(f"      min   = {proxy_v1.min():.4f}")
print(f"      max   = {proxy_v1.max():.4f}")
print(f"      range = [{proxy_v1.min():.4f}, {proxy_v1.max():.4f}]")

# Validate: check a few example tracts
print(f"\n    Sample tracts:")
for i in [0, len(feat)//4, len(feat)//2, 3*len(feat)//4]:
    bg = building_gap.iloc[i]
    rg = road_gap.iloc[i]
    pg = poi_gap.iloc[i]
    sv = svi_overall.iloc[i]
    pv = proxy_v1[i]
    print(f"      i={i}: b_gap={bg:.3f}, r_gap={rg:.3f}, poi_gap={pg:.3f}, "
          f"svi={sv:.3f} → proxy_v1={pv:.3f}")

# ── Prepare features for modeling ─────────────────────────────────────────────
print("\n[3] Preparing feature matrix...")

drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
        'building_gap','road_gap','building_ratio','road_ratio','building_count_ratio',
        'building_count_gap','road_count_ratio','road_count_gap','road_length_ratio',
        'road_length_gap','poi_facility_gap','poi_to_facility_ratio',
        'poi_facility_gap_corrected','poi_to_facility_ratio_corrected',
        'coverage_gap_score','coverage_gap','gap_score','coverage_score',
        'proxy_simple_avg','proxy_svi_weighted','proxy_max_gap','proxy_pop_weighted',
        'proxy_v1']

feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop and pd.api.types.is_numeric_dtype(feat[c])]

X = feat[fc].copy()
y = pd.Series(proxy_v1, index=feat.index)
geo = feat['GEOID'].astype(str).copy()

v = y.notna()
X, y, geo = X[v], y[v], geo[v]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance features
s = X.std()
X = X[s[s > 1e-10].index]

# Select top features by correlation with target
cs = X.corrwith(y).abs().fillna(0)
top_n = min(80, len(cs))
X = X[cs.sort_values(ascending=False).head(top_n).index]

# Remove highly correlated features
cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
td = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=td)

print(f"    {X.shape[1]} features, {X.shape[0]} tracts")

# ── H3 Spatial Block CV ──────────────────────────────────────────────────────
print("\n[4] Computing H3 spatial blocks (resolution 4)...")

lats = feat.loc[v, 'centroid_lat']
lons = feat.loc[v, 'centroid_lon']
blocks = pd.Series(
    [h3.latlng_to_cell(float(la), float(lo), 4) if not (np.isnan(la) or np.isnan(lo)) else 'unk'
     for la, lo in zip(lats.values, lons.values)],
    index=geo.index
)
print(f"    H3: {blocks.nunique()} blocks")

ub = list(blocks.unique())
np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

# ── Train models ──────────────────────────────────────────────────────────────
def train_model(model, name):
    oof = np.full(len(y), np.nan)
    scores = []
    for fi, (ti, vi) in enumerate(splits):
        m = type(model)(**model.get_params())
        Xt, yt, Xv, yv = X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi]
        try:
            if isinstance(m, xgb.XGBRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            elif isinstance(m, lgb.LGBMRegressor):
                m.fit(Xt, yt, eval_set=[(Xv, yv)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            elif isinstance(m, CatBoostRegressor):
                m.fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=30,
                      verbose=0)
            else:
                m.fit(Xt, yt)
        except Exception as e:
            print(f"  {name} F{fi} err: {e}")
            continue
        p = m.predict(Xv)
        oof[vi] = p
        rmse = np.sqrt(mean_squared_error(yv, p))
        r2 = r2_score(yv, p)
        scores.append((rmse, r2))
        print(f"  {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({time.time()-t0:.0f}s)")
        del m; gc.collect()

    rmse_m = np.mean([s[0] for s in scores]) if scores else float('nan')
    r2_m = np.mean([s[1] for s in scores]) if scores else float('nan')
    print(f"  {name}: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
    return oof, rmse_m, r2_m

oofs = {}
msum = {}

# XGBoost
print("\n[5] XGBoost...")
o, r, r2 = train_model(
    xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                     reg_lambda=1.0, min_child_weight=5, tree_method='hist',
                     random_state=SEED), 'XGB')
oofs['xgb'] = o; msum['xgb'] = (r, r2); gc.collect()

# LightGBM GBDT
print("\n[6] LightGBM GBDT...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, min_child_samples=10, boosting_type='gbdt',
                      random_state=SEED, verbose=-1), 'LGB')
oofs['lgb'] = o; msum['lgb'] = (r, r2); gc.collect()

# CatBoost
print("\n[7] CatBoost...")
o, r, r2 = train_model(
    CatBoostRegressor(iterations=600, depth=7, learning_rate=0.03,
                      l2_leaf_reg=3.0, random_strength=1.0,
                      bagging_temperature=0.5, random_seed=SEED, verbose=0,
                      thread_count=1, allow_writing_files=False), 'CAT')
oofs['cat'] = o; msum['cat'] = (r, r2); gc.collect()

# ExtraTrees
print("\n[8] ExtraTrees...")
o, r, r2 = train_model(
    ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_split=5,
                        random_state=SEED, n_jobs=1), 'ET')
oofs['et'] = o; msum['et'] = (r, r2); gc.collect()

# LightGBM DART
print("\n[9] LightGBM DART...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, min_child_samples=10, boosting_type='dart',
                      random_state=SEED, verbose=-1, drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); gc.collect()

# ── Ensemble ──────────────────────────────────────────────────────────────────
print("\n[10] ENSEMBLE")

# Filter out models with NaN OOF predictions
valid_models = {k: v for k, v in oofs.items() if not np.all(np.isnan(v))}
if len(valid_models) < len(oofs):
    invalid = set(oofs.keys()) - set(valid_models.keys())
    print(f"  WARNING: Dropping {invalid} from ensemble (all NaN OOF predictions)")

ns = list(valid_models.keys())
mat = np.column_stack([valid_models[n] for n in ns])
vv = ~np.any(np.isnan(mat), axis=1)
mv = mat[vv]
yv = y.values[vv]
print(f"  Valid models: {ns}")
print(f"  Valid samples: {vv.sum()} / {len(y)}")

if len(yv) == 0 or len(ns) == 0:
    print("  ERROR: No valid models or samples for ensemble!")
    sys.exit(1)

# Convex blend
res = minimize(
    lambda w: np.sqrt(mean_squared_error(yv, mv @ w)),
    np.ones(len(ns)) / len(ns),
    method='SLSQP',
    bounds=[(0, 1)] * len(ns),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
cw = {n: round(float(w), 4) for n, w in zip(ns, res.x)}
cp = mv @ res.x
cr = res.fun
cr2 = r2_score(yv, cp)
print(f"  Convex: RMSE={cr:.6f} R2={cr2:.4f} w={cw}")

# 70/30 geometric + arithmetic hybrid
sn = sorted(cw.items(), key=lambda x: -x[1])
t1, t2 = sn[0][0], sn[1][0]
a, b = oofs[t1][vv], oofs[t2][vv]
hybrid70 = 0.7 * np.sqrt(np.abs(a * b)) * np.sign(a) + 0.3 * (0.6 * a + 0.4 * b)
h70_rmse = np.sqrt(mean_squared_error(yv, hybrid70))
h70_r2 = r2_score(yv, hybrid70)
print(f"  70/30 hybrid: RMSE={h70_rmse:.6f} R2={h70_r2:.4f}")

# Stacking
stack_X = mv.copy()
meta = Ridge(alpha=1.0)
meta.fit(stack_X, yv)
stack_pred = meta.predict(stack_X)
stack_rmse = np.sqrt(mean_squared_error(yv, stack_pred))
stack_r2 = r2_score(yv, stack_pred)
print(f"  Stacking: RMSE={stack_rmse:.6f} R2={stack_r2:.4f}")
print(f"  Stacking coefs: {dict(zip(ns, [round(c, 4) for c in meta.coef_]))}")

# Best ensemble
best_name, best_rmse, best_r2, best_pred = 'convex', cr, cr2, cp
if h70_rmse < best_rmse:
    best_name, best_rmse, best_r2, best_pred = 'hybrid70', h70_rmse, h70_r2, hybrid70
if stack_rmse < best_rmse:
    best_name, best_rmse, best_r2, best_pred = 'stacking', stack_rmse, stack_r2, stack_pred
print(f"\n  BEST: {best_name} RMSE={best_rmse:.6f} R2={best_r2:.4f}")

# ── Train final models on full data ──────────────────────────────────────────
print("\n[11] Training final models on full data...")

final_models = {}

# XGBoost final
print("  XGBoost final...")
m_xgb = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                          reg_lambda=1.0, min_child_weight=5, tree_method='hist',
                          random_state=SEED)
m_xgb.fit(X, y, verbose=False)
final_models['xgb'] = m_xgb

# LightGBM final
print("  LightGBM final...")
m_lgb = lgb.LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                           reg_lambda=1.0, min_child_samples=10, boosting_type='gbdt',
                           random_state=SEED, verbose=-1)
m_lgb.fit(X, y)
final_models['lgb'] = m_lgb

# CatBoost final
print("  CatBoost final...")
m_cat = CatBoostRegressor(iterations=600, depth=7, learning_rate=0.03,
                          l2_leaf_reg=3.0, random_strength=1.0,
                          bagging_temperature=0.5, random_seed=SEED, verbose=0,
                          thread_count=1, allow_writing_files=False)
m_cat.fit(X, y)
final_models['cat'] = m_cat

# ExtraTrees final
print("  ExtraTrees final...")
m_et = ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_split=5,
                           random_state=SEED, n_jobs=1)
m_et.fit(X, y)
final_models['et'] = m_et

# DART final
print("  LightGBM DART final...")
m_dart = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                            reg_lambda=1.0, min_child_samples=10, boosting_type='dart',
                            random_state=SEED, verbose=-1, drop_rate=0.1, max_drop=50)
m_dart.fit(X, y)
final_models['lgb_dart'] = m_dart

# ── Save models ───────────────────────────────────────────────────────────────
print("\n[12] Saving models...")
for name, model in final_models.items():
    model_path = MODELS_DIR / f"proxy_v1_{name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved {name} → {model_path}")

# Save meta-learner
meta_path = MODELS_DIR / "proxy_v1_stacking_meta.pkl"
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
print(f"  Saved stacking meta → {meta_path}")

# ── Generate submission ───────────────────────────────────────────────────────
print("\n[13] Generating submission...")

# Predict with all models
preds = {}
for name, model in final_models.items():
    preds[name] = model.predict(X)

# Ensemble prediction
pred_mat = np.column_stack([preds[n] for n in ns])
if best_name == 'convex':
    submission_pred = pred_mat @ res.x
elif best_name == 'hybrid70':
    a_sub, b_sub = preds[t1], preds[t2]
    submission_pred = 0.7 * np.sqrt(np.abs(a_sub * b_sub)) * np.sign(a_sub) + 0.3 * (0.6 * a_sub + 0.4 * b_sub)
elif best_name == 'stacking':
    submission_pred = meta.predict(pred_mat)
else:
    submission_pred = pred_mat @ res.x

# Save submission
submission = pd.DataFrame({
    'GEOID': geo.values,
    'coverage_gap_score': submission_pred,
})
submission.to_csv(OUT / "submission_proxy_v1.csv", index=False)
submission.to_csv(DL / "submission_proxy_v1.csv", index=False)
print(f"  Submission: {len(submission)} tracts")
print(f"  Pred stats: mean={submission_pred.mean():.4f}, std={submission_pred.std():.4f}, "
      f"min={submission_pred.min():.4f}, max={submission_pred.max():.4f}")

# ── Save pipeline state ──────────────────────────────────────────────────────
print("\n[14] Saving pipeline state...")

state = {
    'pipeline': 'proxy_v1_ensemble',
    'target': 'proxy_v1',
    'target_formula': '-mean(building_gap, road_gap, poi_facility_gap_corrected) - 2.0*svi_overall',
    'n_features': int(X.shape[1]),
    'n_tracts': int(X.shape[0]),
    'n_h3_blocks': int(blocks.nunique()),
    'cv_folds': NF,
    'models': {k: {'rmse': round(v[0], 6), 'r2': round(v[1], 4)} for k, v in msum.items()},
    'ensemble': {
        'convex': {'rmse': round(cr, 6), 'r2': round(cr2, 4), 'weights': cw},
        'hybrid70': {'rmse': round(h70_rmse, 6), 'r2': round(h70_r2, 4)},
        'stacking': {'rmse': round(stack_rmse, 6), 'r2': round(stack_r2, 4)},
    },
    'best_ensemble': best_name,
    'best_rmse': round(best_rmse, 6),
    'best_r2': round(best_r2, 4),
    'proxy_v1_stats': {
        'mean': round(float(proxy_v1.mean()), 4),
        'std': round(float(proxy_v1.std()), 4),
        'min': round(float(proxy_v1.min()), 4),
        'max': round(float(proxy_v1.max()), 4),
    },
}

with open(OUT / "pipeline_state_proxy_v1.json", 'w') as f:
    json.dump(state, f, indent=2)
print(f"  Saved to {OUT / 'pipeline_state_proxy_v1.json'}")

# Also save the features with proxy_v1
feat.to_parquet(OUT / "engineered_features_v3.parquet", index=False)
print(f"  Updated features with proxy_v1 column")

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {elapsed:.0f}s")
print(f"Best ensemble: {best_name} RMSE={best_rmse:.6f} R2={best_r2:.4f}")
print(f"{'=' * 72}")
