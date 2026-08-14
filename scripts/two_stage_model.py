#!/usr/bin/env python3
"""
TWO-STAGE MODEL: CLASSIFY -> REGRESS
Solves the 84.6% zero-gap problem.

  Stage 1 (Classifier): P(has_coverage_gap)
  Stage 2 (Regressor): E[gap_only | has_gap=1]
  Inference: gap_pred = P(has_gap) * E[gap|has_gap]
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import (mean_squared_error, r2_score, roc_auc_score,
                             f1_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
SUB_DIR = PROJ / "submissions"; SUB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = PROJ / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("TWO-STAGE MODEL: CLASSIFY -> REGRESS")
print("=" * 78)
t0 = time.time()

# ========================================================================
# 1. LOAD + MERGE + WEATHER INTERACTIONS
# ========================================================================
print("\n[1] Loading data...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Engineered features: {feat.shape}")

wf_df = pd.read_parquet(PROJ / "kaggle_dataset/weather_forecast_features.parquet")
wf_df['GEOID'] = wf_df['GEOID'].astype(str)
wf_keep = [c for c in wf_df.columns if c.startswith('wf_') and wf_df[c].dropna().std() > 1e-10]
print(f"  Weather features: {len(wf_keep)}")

feat = feat.merge(wf_df[['GEOID'] + wf_keep], on='GEOID', how='left')
del wf_df; gc.collect()

# Weather interactions
print("  Engineering weather interactions...")
wi = {}
bag = feat['building_area_gap'].fillna(0).values
rg = feat['road_gap'].fillna(0).values
bg = feat['building_gap'].fillna(0).values
cgs = np.abs(bg) + np.abs(rg)
rv = (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1).values
svi = feat['svi_overall'].fillna(0.5).values
trib = (feat['tribal_any'].fillna(0) > 0).astype(float).values

if 'wf_fire_weather_risk' in feat.columns:
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_fire_risk_x_bldg_area_gap'] = fwr * bag
    wi['wx_fire_risk_x_road_gap'] = fwr * rg
    wi['wx_fire_risk_x_svi'] = fwr * svi

heat_proxy = None
for c in ['wf_very_hot_days', 'wf_extreme_hot_days']:
    if c in feat.columns:
        heat_proxy = feat[c].fillna(0).values
        mx = heat_proxy.max()
        if mx > 0: heat_proxy = heat_proxy / mx
        break
if heat_proxy is not None:
    wi['wx_heat_alert_x_bldg_area_gap'] = heat_proxy * bag
    wi['wx_heat_alert_x_tribal'] = heat_proxy * trib

if 'wf_flood_risk' in feat.columns:
    flr = feat['wf_flood_risk'].fillna(0).values
    wi['wx_flood_risk_x_road_gap'] = flr * rg
    wi['wx_flood_risk_x_rural'] = flr * rv

if 'wf_storm_risk' in feat.columns:
    sr = feat['wf_storm_risk'].fillna(0).values
    if sr.std() > 1e-10:
        wi['wx_storm_risk_x_total_gap'] = sr * (bag + rg)

if 'wf_compound_hazard' in feat.columns:
    ch = feat['wf_compound_hazard'].fillna(0).values
    wi['wx_compound_hazard_x_coverage_gap'] = ch * cgs

if 'wf_fire_weather_risk' in feat.columns and 'wf_humidity_min' in feat.columns:
    fwr2 = feat['wf_fire_weather_risk'].fillna(0).values
    hum = feat['wf_humidity_min'].fillna(50).values
    enh = fwr2 * (1 - hum / 100).clip(0, 1)
    wi['wx_fire_enhanced_x_bldg_area_gap'] = enh * bag
    wi['wx_humidity_x_fire_risk'] = hum * fwr2

if 'wf_hot_days' in feat.columns:
    hd = feat['wf_hot_days'].fillna(0).values
    wi['wx_hot_days_x_bldg_area_gap'] = hd * bag

if 'wf_dry_days' in feat.columns and 'wf_fire_weather_risk' in feat.columns:
    dd = feat['wf_dry_days'].fillna(0).values
    fwr3 = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_dry_x_fire_risk'] = dd * fwr3

if 'wf_high_wind_days' in feat.columns:
    hwd = feat['wf_high_wind_days'].fillna(0).values
    wi['wx_high_wind_x_road_gap'] = hwd * rg

if 'wf_uv_max' in feat.columns:
    uv = feat['wf_uv_max'].fillna(0).values
    wi['wx_uv_x_bldg_area_gap'] = uv * bag

if 'wf_heavy_precip_days' in feat.columns and 'wf_flood_risk' in feat.columns:
    hpd = feat['wf_heavy_precip_days'].fillna(0).values
    flr2 = feat['wf_flood_risk'].fillna(0).values
    wi['wx_heavy_precip_x_flood_risk'] = hpd * flr2

if 'wf_vpd_max' in feat.columns and 'wf_fire_weather_risk' in feat.columns:
    vpd = feat['wf_vpd_max'].fillna(0).values
    fwr4 = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_vpd_x_fire_risk'] = vpd * fwr4

if wi:
    wd = pd.DataFrame(wi, index=feat.index).replace([np.inf, -np.inf], np.nan)
    feat = pd.concat([feat, wd], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    print(f"  +{len(wi)} weather interactions -> {feat.shape[1]} total")
del wi; gc.collect()

# ========================================================================
# 2. PREPARE TARGETS + FEATURE MATRIX
# ========================================================================
print("\n[2] Preparing targets and features...")
y_proxy = feat['proxy_merged'].copy()
geo = feat['GEOID'].astype(str).copy()
rural_penalty = (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1)
gap_only = y_proxy + rural_penalty
has_gap = (gap_only.abs() > 1e-10).astype(int)

print(f"  Total: {len(feat)}, Has gap: {has_gap.sum()} ({has_gap.mean()*100:.1f}%), No gap: {(1-has_gap).sum()}")

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
X_full = feat[fc].copy()

valid = y_proxy.notna()
X_full, y_proxy = X_full[valid], y_proxy[valid]
geo, rural_penalty = geo[valid], rural_penalty[valid]
gap_only, has_gap = gap_only[valid], has_gap[valid]

X_full = X_full.fillna(-999).replace([np.inf, -np.inf], -999)
stds = X_full.std()
X_full = X_full[stds[stds > 1e-10].index]
print(f"  Features: {X_full.shape[1]}, Tracts: {X_full.shape[0]}")
del feat; gc.collect()

# ========================================================================
# 3. FEATURE SELECTION
# ========================================================================
print("\n[3] Feature selection...")

def select_features(X, y, max_features=60):
    cs = X.corrwith(y).abs().fillna(0)
    top_k = min(max_features, len(cs))
    X_sel = X[cs.sort_values(ascending=False).head(top_k).index]
    cm = X_sel.corr().abs()
    up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    to_drop = [c for c in up.columns if any(up[c] > 0.98)]
    return X_sel.drop(columns=to_drop)

X_reg_sel = select_features(X_full, gap_only, max_features=60)
reg_features = list(X_reg_sel.columns)
print(f"  Regression: {len(reg_features)} features")

X_cls_sel = select_features(X_full, has_gap, max_features=60)
cls_features = list(set(X_cls_sel.columns) | set(X_reg_sel.columns))
if len(cls_features) > 80:
    corr_cls = X_full[cls_features].corrwith(has_gap).abs().fillna(0)
    cls_features = list(corr_cls.sort_values(ascending=False).head(80).index)
print(f"  Classification: {len(cls_features)} features")

# Build matrices before deleting X_full
X_cls_mat = X_full[cls_features].copy()
X_reg_mat = X_reg_sel.copy()
del X_full, X_reg_sel, X_cls_sel; gc.collect()

# ========================================================================
# 4. STAGE 1: CLASSIFIER
# ========================================================================
print("\n" + "=" * 78)
print("[4] STAGE 1: CLASSIFIER - P(has_coverage_gap)")
print("=" * 78)

X_cls_arr = X_cls_mat.values
y_cls = has_gap.values

n_neg, n_pos = (y_cls == 0).sum(), (y_cls == 1).sum()
scale_pos = n_neg / n_pos
print(f"  neg={n_neg} ({n_neg/len(y_cls)*100:.1f}%), pos={n_pos} ({n_pos/len(y_cls)*100:.1f}%)")
print(f"  scale_pos_weight={scale_pos:.2f}, features={X_cls_arr.shape[1]}")

# 3-fold CV (fast)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cls_oof = np.zeros(len(y_cls))
cls_aucs, cls_aps = [], []

for fold, (tr, va) in enumerate(skf.split(X_cls_arr, y_cls)):
    print(f"  Fold {fold+1}/3", end=" ")
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        scale_pos_weight=scale_pos,
        tree_method='hist', random_state=SEED, eval_metric='auc', verbosity=0
    )
    m.fit(X_cls_arr[tr], y_cls[tr], eval_set=[(X_cls_arr[va], y_cls[va])], verbose=False)
    p = m.predict_proba(X_cls_arr[va])[:, 1]
    cls_oof[va] = p
    auc = roc_auc_score(y_cls[va], p)
    ap = average_precision_score(y_cls[va], p)
    cls_aucs.append(auc); cls_aps.append(ap)
    print(f"AUC={auc:.4f} AP={ap:.4f}")
    gc.collect()

auc_m = np.mean(cls_aucs)
ap_m = np.mean(cls_aps)
print(f"  CV: AUC={auc_m:.4f} +/- {np.std(cls_aucs):.4f}, AP={ap_m:.4f}")

best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_cls, (cls_oof > t).astype(int))
    if f1 > best_f1: best_f1, best_thresh = f1, t
print(f"  Optimal threshold={best_thresh:.2f} F1={best_f1:.4f}")

print("  Retraining on ALL data...")
cls_final = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    scale_pos_weight=scale_pos,
    tree_method='hist', random_state=SEED, eval_metric='auc', verbosity=0
)
cls_final.fit(X_cls_arr, y_cls)
cls_proba_all = cls_final.predict_proba(X_cls_arr)[:, 1]
print(f"  Mean P(has_gap)={cls_proba_all.mean():.4f}")
gc.collect()

# ========================================================================
# 5. STAGE 2: REGRESSOR (non-zero gap only)
# ========================================================================
print("\n" + "=" * 78)
print("[5] STAGE 2: REGRESSOR - E[gap_only | has_gap=1]")
print("=" * 78)

nz_mask = has_gap == 1
X_reg = X_reg_mat[nz_mask].values
y_reg = gap_only[nz_mask].values

print(f"  Training on {len(y_reg)} non-zero-gap tracts")
print(f"  Target: mean={y_reg.mean():.4f} std={y_reg.std():.4f} range=[{y_reg.min():.4f}, {y_reg.max():.4f}]")

# 3 fast models for regressor
reg_cfgs = {
    'xgb': lambda: xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
        tree_method='hist', random_state=SEED),
    'lgb': lambda: lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
        boosting_type='gbdt', random_state=SEED, verbose=-1),
    'et': lambda: ExtraTreesRegressor(
        n_estimators=60, max_depth=10,
        min_samples_split=10, random_state=SEED, n_jobs=-1),
}

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
reg_oof_all, reg_rmses = {}, {}

for name, cfg in reg_cfgs.items():
    print(f"  [{name}]", end=" ")
    oof = np.zeros(len(y_reg))
    fr, fr2 = [], []
    for fold, (tr, va) in enumerate(kf.split(X_reg)):
        m = cfg()
        m.fit(X_reg[tr], y_reg[tr])
        oof[va] = m.predict(X_reg[va])
        fr.append(np.sqrt(mean_squared_error(y_reg[va], oof[va])))
        fr2.append(r2_score(y_reg[va], oof[va]))
    reg_oof_all[name] = oof
    reg_rmses[name] = np.mean(fr)
    print(f"RMSE={np.mean(fr):.6f} R2={np.mean(fr2):.4f}")
    gc.collect()

# Optimize weights
rn = list(reg_cfgs.keys())
rs = np.column_stack([reg_oof_all[n] for n in rn])
res = minimize(lambda w: np.sqrt(mean_squared_error(y_reg, rs @ w)),
               np.ones(len(rn))/len(rn), method='SLSQP',
               bounds=[(0,1)]*len(rn),
               constraints={'type':'eq','fun':lambda w:sum(w)-1})
rw = {n: round(float(w),4) for n,w in zip(rn, res.x)}
reg_oof_ens = rs @ res.x
reg_rmse = np.sqrt(mean_squared_error(y_reg, reg_oof_ens))
reg_r2 = r2_score(y_reg, reg_oof_ens)
print(f"  Weights: {rw}")
print(f"  OOF: RMSE={reg_rmse:.6f} R2={reg_r2:.4f}")

print("  Retraining on ALL non-zero data...")
reg_final = {}
for name, cfg in reg_cfgs.items():
    m = cfg(); m.fit(X_reg, y_reg); reg_final[name] = m
gc.collect()

# ========================================================================
# 6. SINGLE-STAGE BASELINE (for comparison)
# ========================================================================
print("\n" + "=" * 78)
print("[6] SINGLE-STAGE BASELINE")
print("=" * 78)

X_base = X_reg_mat.values
y_base = gap_only.values
print(f"  Training on ALL {len(y_base)} tracts")

base_cfgs = {
    'xgb': lambda: xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
        tree_method='hist', random_state=SEED),
    'lgb': lambda: lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
        boosting_type='gbdt', random_state=SEED, verbose=-1),
    'et': lambda: ExtraTreesRegressor(
        n_estimators=60, max_depth=10,
        min_samples_split=10, random_state=SEED, n_jobs=-1),
}

kf3 = KFold(n_splits=3, shuffle=True, random_state=SEED)
base_oof_all = {}

for name, cfg in base_cfgs.items():
    print(f"  [{name}]", end=" ")
    oof = np.zeros(len(y_base))
    fr = []
    for fold, (tr, va) in enumerate(kf3.split(X_base)):
        m = cfg(); m.fit(X_base[tr], y_base[tr])
        oof[va] = m.predict(X_base[va])
        fr.append(np.sqrt(mean_squared_error(y_base[va], oof[va])))
    base_oof_all[name] = oof
    print(f"RMSE={np.mean(fr):.6f}")
    gc.collect()

bn = list(base_cfgs.keys())
bs = np.column_stack([base_oof_all[n] for n in bn])
res_b = minimize(lambda w: np.sqrt(mean_squared_error(y_base, bs @ w)),
                 np.ones(len(bn))/len(bn), method='SLSQP',
                 bounds=[(0,1)]*len(bn),
                 constraints={'type':'eq','fun':lambda w:sum(w)-1})
bw = {n: round(float(w),4) for n,w in zip(bn, res_b.x)}
base_oof_ens = bs @ res_b.x
base_rmse = np.sqrt(mean_squared_error(y_base, base_oof_ens))
base_r2 = r2_score(y_base, base_oof_ens)
print(f"  Weights: {bw}")
print(f"  OOF: RMSE={base_rmse:.6f} R2={base_r2:.4f}")

# ========================================================================
# 7. COMPARISON
# ========================================================================
print("\n" + "=" * 78)
print("[7] COMPARISON: Two-Stage vs Single-Stage")
print("=" * 78)

rp_vals = rural_penalty.values
y_true = y_proxy.values

# Single-stage
base_cgs = np.clip(base_oof_ens - 1.0 * rp_vals, -3.0, 0.5)
base_rmse_f = np.sqrt(mean_squared_error(y_true, base_cgs))
base_r2_f = r2_score(y_true, base_cgs)

# Two-stage (OOF)
nz_idx = np.where(nz_mask.values)[0]
z_idx = np.where(~nz_mask.values)[0]
nz_mean_gap = y_reg.mean()

ts_gap = np.zeros(len(y_base))
for i, idx in enumerate(nz_idx):
    ts_gap[idx] = cls_oof[idx] * reg_oof_ens[i]
for idx in z_idx:
    if cls_oof[idx] > best_thresh:
        ts_gap[idx] = cls_oof[idx] * nz_mean_gap

ts_cgs = np.clip(ts_gap - 1.0 * rp_vals, -3.0, 0.5)
ts_rmse_f = np.sqrt(mean_squared_error(y_true, ts_cgs))
ts_r2_f = r2_score(y_true, ts_cgs)

print(f"\n  {'Metric':30s} {'Single-Stage':>14s} {'Two-Stage':>14s}")
print(f"  {'-'*30} {'-'*14} {'-'*14}")
print(f"  {'RMSE (proxy_merged)':30s} {base_rmse_f:14.6f} {ts_rmse_f:14.6f}")
print(f"  {'R2 (proxy_merged)':30s} {base_r2_f:14.4f} {ts_r2_f:14.4f}")

nz_b = np.abs(base_cgs[nz_mask.values] - y_true[nz_mask.values])
nz_t = np.abs(ts_cgs[nz_mask.values] - y_true[nz_mask.values])
z_b = np.abs(base_cgs[~nz_mask.values] - y_true[~nz_mask.values])
z_t = np.abs(ts_cgs[~nz_mask.values] - y_true[~nz_mask.values])

pct_nz = (nz_b.mean() - nz_t.mean()) / nz_b.mean() * 100
pct_z = (z_b.mean() - z_t.mean()) / z_b.mean() * 100

print(f"\n  Non-zero gap ({nz_mask.sum()} tracts - WHERE IT MATTERS):")
print(f"    Single MAE={nz_b.mean():.6f}  Two-stage MAE={nz_t.mean():.6f}  Improvement={pct_nz:+.1f}%")
print(f"\n  Zero gap ({(~nz_mask).sum()} tracts):")
print(f"    Single MAE={z_b.mean():.6f}  Two-stage MAE={z_t.mean():.6f}  Improvement={pct_z:+.1f}%")

# ========================================================================
# 8. FINAL PREDICTIONS
# ========================================================================
print("\n" + "=" * 78)
print("[8] FINAL TWO-STAGE PREDICTIONS")
print("=" * 78)

cls_proba = cls_final.predict_proba(X_cls_arr)[:, 1]

reg_preds = np.zeros((len(X_reg_mat), len(reg_final)))
for i, (name, model) in enumerate(reg_final.items()):
    reg_preds[:, i] = model.predict(X_reg_mat.values)

w_arr = np.array([rw[n] for n in rn])
reg_ens = reg_preds @ w_arr

ts_final = np.minimum(cls_proba * reg_ens, 0)

print(f"  gap_only: mean={ts_final.mean():.6f} std={ts_final.std():.6f}")
print(f"  gap_only: range=[{ts_final.min():.4f}, {ts_final.max():.4f}]")

ts_cgs_final = np.clip(ts_final - 1.0 * rp_vals, -3.0, 0.5)
print(f"  coverage_gap_score: mean={ts_cgs_final.mean():.4f} std={ts_cgs_final.std():.4f}")
print(f"  coverage_gap_score: range=[{ts_cgs_final.min():.4f}, {ts_cgs_final.max():.4f}]")

# ========================================================================
# 9. VALIDATE + SAVE
# ========================================================================
print("\n[9] Validating submission...")

submission = pd.DataFrame({
    'GEOID': geo.values,
    'coverage_gap_score': ts_cgs_final
})

assert len(submission) == 85396
assert submission['coverage_gap_score'].notna().all()
assert (submission['coverage_gap_score'] >= -3.0).all()
assert (submission['coverage_gap_score'] <= 0.5).all()
assert submission['GEOID'].nunique() == 85396
print("  VALIDATION PASSED")

sub_path = SUB_DIR / "submission_two_stage.csv"
submission.to_csv(sub_path, index=False)
dl_path = Path("/home/z/my-project/download/submission_two_stage.csv")
submission.to_csv(dl_path, index=False)
print(f"  Saved: {sub_path}")
print(f"  Saved: {dl_path}")

# ========================================================================
# 10. FEATURE IMPORTANCE + ANALYSIS
# ========================================================================
print("\n[10] Feature importance...")

cls_fi = pd.DataFrame({
    'feature': cls_features,
    'importance': cls_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 15 Classifier Features (P(has_gap)):")
for _, r in cls_fi.head(15).iterrows():
    print(f"    {r['feature']:45s} {r['importance']:.6f}")

reg_agg = np.zeros(len(reg_features))
tw = 0
for name, model in reg_final.items():
    try:
        imp = model.feature_importances_
        if len(imp) == len(reg_features):
            w = 1.0 / (reg_rmses[name] + 1e-10)
            reg_agg += w * imp; tw += w
    except: pass
if tw > 0: reg_agg /= tw

reg_fi = pd.DataFrame({
    'feature': reg_features,
    'importance': reg_agg
}).sort_values('importance', ascending=False)

print("\n  Top 15 Regressor Features (E[gap|has_gap]):")
for _, r in reg_fi.head(15).iterrows():
    print(f"    {r['feature']:45s} {r['importance']:.6f}")

cls_fi.to_csv(RESULTS / 'two_stage_classifier_feature_importance.csv', index=False)
reg_fi.to_csv(RESULTS / 'two_stage_regressor_feature_importance.csv', index=False)

# Calibration
print("\n  Probability calibration:")
for i in range(10):
    b0, b1 = i*0.1, (i+1)*0.1 if i < 9 else 1.01
    mask = (cls_proba >= b0) & (cls_proba < b1)
    n = mask.sum()
    actual = has_gap.values[mask].mean() if n > 0 else 0
    print(f"    P in [{b0:.1f},{b1:.1f}): n={n:>6d}, actual={actual:>6.3f}")

# Save results
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'two_stage_classify_regress',
    'data': {'n_total': int(len(y_base)), 'n_zero_gap': int((~nz_mask).sum()),
             'n_nonzero_gap': int(nz_mask.sum()), 'pct_zero_gap': float((~nz_mask).mean()*100)},
    'classifier': {'type': 'XGBoost', 'cv_auc': float(auc_m), 'cv_auc_std': float(np.std(cls_aucs)),
                   'cv_ap': float(ap_m), 'optimal_threshold': float(best_thresh),
                   'optimal_f1': float(best_f1), 'scale_pos_weight': float(scale_pos),
                   'n_features': int(len(cls_features))},
    'regressor': {'type': '3_model_ensemble', 'weights': rw, 'oof_rmse': float(reg_rmse),
                  'oof_r2': float(reg_r2), 'n_training': int(len(y_reg)),
                  'n_features': int(len(reg_features))},
    'baseline': {'type': '3_model_ensemble', 'weights': bw, 'oof_rmse': float(base_rmse),
                 'oof_r2': float(base_r2)},
    'comparison': {'single_rmse': float(base_rmse_f), 'single_r2': float(base_r2_f),
                   'two_stage_rmse': float(ts_rmse_f), 'two_stage_r2': float(ts_r2_f),
                   'nonzero_mae_single': float(nz_b.mean()), 'nonzero_mae_two_stage': float(nz_t.mean()),
                   'nonzero_improvement_pct': float(pct_nz),
                   'zero_mae_single': float(z_b.mean()), 'zero_mae_two_stage': float(z_t.mean()),
                   'zero_improvement_pct': float(pct_z)},
    'submission': {'mean': float(ts_cgs_final.mean()), 'std': float(ts_cgs_final.std()),
                   'min': float(ts_cgs_final.min()), 'max': float(ts_cgs_final.max()),
                   'median': float(np.median(ts_cgs_final))},
    'elapsed_sec': round(time.time() - t0, 1),
}

with open(RESULTS / 'two_stage_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved: two_stage_results.json")

el = time.time() - t0
print(f"\n{'='*78}")
print(f"DONE in {el:.0f}s")
print(f"Classifier: AUC={auc_m:.4f}, threshold={best_thresh:.2f}, F1={best_f1:.4f}")
print(f"Regressor:  RMSE={reg_rmse:.6f}, R2={reg_r2:.4f}")
print(f"Non-zero gap improvement: {pct_nz:+.1f}% MAE")
print(f"Submission: {len(submission)} tracts, mean={ts_cgs_final.mean():.4f}")
print(f"{'='*78}")
