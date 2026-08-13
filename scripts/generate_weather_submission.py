#!/usr/bin/env python3
"""
WEATHER-ENHANCED SUBMISSION GENERATOR
======================================
Produces the final submission CSV for the bias-bounty-mapping-equity challenge
using the weather-enhanced 5-model ensemble.

Pipeline:
  1. Load engineered features + weather features
  2. Engineer weather interaction features (same as pipeline_weather_enhanced.py)
  3. Select top 80 features via correlation + dedup
  4. Train 5-model ensemble on ALL data (XGBoost, LightGBM, CatBoost, ExtraTrees, LGBM-DART)
  5. Optimise convex ensemble weights via SLSQP on training residuals
  6. Predict for all 85,396 tracts
  7. Apply rural penalty: score = model.predict(X) - 1.0 * rural_penalty
  8. Clip to [-3.0, +0.5]
  9. Validate and save
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
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

print("=" * 78)
print("WEATHER-ENHANCED SUBMISSION GENERATOR")
print("=" * 78)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading data...")

# Load pre-engineered features from Phase 1
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Engineered features: {feat.shape}")

# Load weather features
wf_df = pd.read_parquet(PROJ / "kaggle_dataset/weather_forecast_features.parquet")
wf_df['GEOID'] = wf_df['GEOID'].astype(str)
print(f"  Weather features: {wf_df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MERGE WEATHER FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Merging weather features...")

# Only keep numeric weather columns with variance > 0
wf_numeric_cols = [c for c in wf_df.columns if c.startswith('wf_')]
wf_keep = []
for c in wf_numeric_cols:
    s = wf_df[c].dropna()
    if len(s) > 0 and s.std() > 1e-10:
        wf_keep.append(c)
    else:
        print(f"  Dropping zero-var: {c}")
print(f"  Keeping {len(wf_keep)} weather features with variance > 0")

before_cols = feat.shape[1]
feat = feat.merge(wf_df[['GEOID'] + wf_keep], on='GEOID', how='left')
print(f"  Merged weather: {before_cols} -> {feat.shape[1]} cols")
del wf_df; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENGINEER WEATHER INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Engineering weather interaction features...")

wi = {}

# Get gap values for interactions
building_area_gap_vals = feat['building_area_gap'].fillna(0).values if 'building_area_gap' in feat.columns else np.zeros(len(feat))
road_gap_vals = feat['road_gap'].fillna(0).values if 'road_gap' in feat.columns else np.zeros(len(feat))
building_gap_vals = feat['building_gap'].fillna(0).values if 'building_gap' in feat.columns else np.zeros(len(feat))

# Coverage gap score proxy
coverage_gap_score = np.abs(building_gap_vals) + np.abs(road_gap_vals)

# Vulnerability indicators
rural_vals = (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1).values if 'pct_urban' in feat.columns else np.zeros(len(feat))
svi_vals = feat['svi_overall'].fillna(0.5).values if 'svi_overall' in feat.columns else np.full(len(feat), 0.5)
tribal_vals = (feat['tribal_any'].fillna(0) > 0).astype(float).values if 'tribal_any' in feat.columns else np.zeros(len(feat))

# ── Weather × Coverage Gap Interactions ──
if 'wf_fire_weather_risk' in feat.columns:
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_fire_risk_x_bldg_area_gap'] = fwr * building_area_gap_vals
    wi['wx_fire_risk_x_road_gap'] = fwr * road_gap_vals
    wi['wx_fire_risk_x_svi'] = fwr * svi_vals

# wf_heat_alert proxy → wf_very_hot_days (normalized)
heat_proxy = None
if 'wf_very_hot_days' in feat.columns:
    heat_proxy = feat['wf_very_hot_days'].fillna(0).values
    hmax = heat_proxy.max()
    if hmax > 0: heat_proxy = heat_proxy / hmax
elif 'wf_extreme_hot_days' in feat.columns:
    heat_proxy = feat['wf_extreme_hot_days'].fillna(0).values
    hmax = heat_proxy.max()
    if hmax > 0: heat_proxy = heat_proxy / hmax

if heat_proxy is not None:
    wi['wx_heat_alert_x_bldg_area_gap'] = heat_proxy * building_area_gap_vals
    wi['wx_heat_alert_x_tribal'] = heat_proxy * tribal_vals

if 'wf_flood_risk' in feat.columns:
    flr = feat['wf_flood_risk'].fillna(0).values
    wi['wx_flood_risk_x_road_gap'] = flr * road_gap_vals
    wi['wx_flood_risk_x_rural'] = flr * rural_vals

if 'wf_storm_risk' in feat.columns:
    sr = feat['wf_storm_risk'].fillna(0).values
    if sr.std() > 1e-10:
        wi['wx_storm_risk_x_total_gap'] = sr * (building_area_gap_vals + road_gap_vals)

if 'wf_compound_hazard' in feat.columns:
    ch = feat['wf_compound_hazard'].fillna(0).values
    wi['wx_compound_hazard_x_coverage_gap'] = ch * coverage_gap_score

# wf_fire_weather_enhanced proxy: fire_weather_risk × (1 - humidity_min/100)
if 'wf_fire_weather_risk' in feat.columns and 'wf_humidity_min' in feat.columns:
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    hum = feat['wf_humidity_min'].fillna(50).values
    enhanced = fwr * (1 - hum / 100).clip(0, 1)
    wi['wx_fire_enhanced_x_bldg_area_gap'] = enhanced * building_area_gap_vals

if 'wf_humidity_min' in feat.columns and 'wf_fire_weather_risk' in feat.columns:
    hum = feat['wf_humidity_min'].fillna(50).values
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_humidity_x_fire_risk'] = hum * fwr

# Additional interactions
if 'wf_hot_days' in feat.columns:
    hd = feat['wf_hot_days'].fillna(0).values
    wi['wx_hot_days_x_bldg_area_gap'] = hd * building_area_gap_vals

if 'wf_dry_days' in feat.columns and 'wf_fire_weather_risk' in feat.columns:
    dd = feat['wf_dry_days'].fillna(0).values
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_dry_x_fire_risk'] = dd * fwr

if 'wf_high_wind_days' in feat.columns:
    hwd = feat['wf_high_wind_days'].fillna(0).values
    wi['wx_high_wind_x_road_gap'] = hwd * road_gap_vals

if 'wf_uv_max' in feat.columns:
    uv = feat['wf_uv_max'].fillna(0).values
    wi['wx_uv_x_bldg_area_gap'] = uv * building_area_gap_vals

if 'wf_heavy_precip_days' in feat.columns and 'wf_flood_risk' in feat.columns:
    hpd = feat['wf_heavy_precip_days'].fillna(0).values
    flr = feat['wf_flood_risk'].fillna(0).values
    wi['wx_heavy_precip_x_flood_risk'] = hpd * flr

if 'wf_vpd_max' in feat.columns and 'wf_fire_weather_risk' in feat.columns:
    vpd = feat['wf_vpd_max'].fillna(0).values
    fwr = feat['wf_fire_weather_risk'].fillna(0).values
    wi['wx_vpd_x_fire_risk'] = vpd * fwr

if wi:
    wd = pd.DataFrame(wi, index=feat.index)
    wd = wd.replace([np.inf, -np.inf], np.nan)
    feat = pd.concat([feat, wd], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    print(f"  +{len(wi)} weather interaction features -> {feat.shape[1]} total")
del wi; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 4. PREPARE FEATURES FOR TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Preparing features for training...")

y = feat['proxy_merged'].copy()
geo = feat['GEOID'].astype(str).copy()

# Store rural_penalty for inference
rural_penalty = feat['rural_penalty'].copy() if 'rural_penalty' in feat.columns else (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1)

# Tribal/rural/SVI flags for later analysis
tribal_flag = (feat['tribal_any'].fillna(0) > 0) if 'tribal_any' in feat.columns else pd.Series(False, index=feat.index)
rural_flag = rural_penalty > 0.5
svi_high = feat['svi_overall'].fillna(0.5) > 0.75 if 'svi_overall' in feat.columns else pd.Series(False, index=feat.index)

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

valid = y.notna()
X_full, y, geo = X_full[valid], y[valid], geo[valid]
rural_penalty = rural_penalty[valid]
tribal_flag = tribal_flag[valid]
rural_flag = rural_flag[valid]
svi_high = svi_high[valid]

X_full = X_full.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance
s = X_full.std()
X_full = X_full[s[s > 1e-10].index]

# Classify features
weather_cols = [c for c in X_full.columns if c.startswith('wf_')]
wx_cols = [c for c in X_full.columns if c.startswith('wx_')]
base_cols = [c for c in X_full.columns if not c.startswith('wf_') and not c.startswith('wx_')]

print(f"  Full: {X_full.shape[1]} features, {X_full.shape[0]} tracts")
print(f"  Base: {len(base_cols)}, Weather (wf_*): {len(weather_cols)}, Interactions (wx_*): {len(wx_cols)}")

del feat; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE SELECTION (correlation + dedup, max 80 features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Feature selection...")

def select_features(X, y, max_features=80):
    cs = X.corrwith(y).abs().fillna(0)
    top_k = min(max_features, len(cs))
    X_sel = X[cs.sort_values(ascending=False).head(top_k).index]
    cm = X_sel.corr().abs()
    up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    to_drop = [c for c in up.columns if any(up[c] > 0.98)]
    X_sel = X_sel.drop(columns=to_drop)
    return X_sel

X_sel = select_features(X_full, y, max_features=80)
print(f"  Selected {X_sel.shape[1]} features out of {X_full.shape[1]}")

# Show selected weather/interaction features
sel_wx = [c for c in X_sel.columns if c.startswith('wx_')]
sel_wf = [c for c in X_sel.columns if c.startswith('wf_')]
print(f"  Selected weather (wf_*): {len(sel_wf)}")
print(f"  Selected interactions (wx_*): {len(sel_wx)}")
if sel_wx:
    print(f"  wx features: {sel_wx}")
if sel_wf:
    print(f"  wf features: {sel_wf}")

del X_full; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN 5-MODEL ENSEMBLE ON ALL DATA (final training, no CV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Training 5-model ensemble on ALL data (final training)...")

X = X_sel
y_arr = y.values

# ── [1] XGBoost ──
print("\n  [1/5] XGBoost...")
m_xgb = xgb.XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
    tree_method='hist', random_state=SEED
)
m_xgb.fit(X, y_arr)
p_xgb = m_xgb.predict(X)
rmse_xgb = np.sqrt(mean_squared_error(y_arr, p_xgb))
r2_xgb = r2_score(y_arr, p_xgb)
print(f"    Train RMSE={rmse_xgb:.6f} R2={r2_xgb:.4f}")
gc.collect()

# ── [2] LightGBM GBDT ──
print("\n  [2/5] LightGBM GBDT...")
m_lgb = lgb.LGBMRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
    boosting_type='gbdt', random_state=SEED, verbose=-1
)
m_lgb.fit(X, y_arr)
p_lgb = m_lgb.predict(X)
rmse_lgb = np.sqrt(mean_squared_error(y_arr, p_lgb))
r2_lgb = r2_score(y_arr, p_lgb)
print(f"    Train RMSE={rmse_lgb:.6f} R2={r2_lgb:.4f}")
gc.collect()

# ── [3] CatBoost ──
print("\n  [3/5] CatBoost...")
m_cb = CatBoostRegressor(
    iterations=300, depth=5, learning_rate=0.05,
    l2_leaf_reg=3, random_seed=SEED, verbose=0
)
m_cb.fit(X, y_arr)
p_cb = m_cb.predict(X)
rmse_cb = np.sqrt(mean_squared_error(y_arr, p_cb))
r2_cb = r2_score(y_arr, p_cb)
print(f"    Train RMSE={rmse_cb:.6f} R2={r2_cb:.4f}")
gc.collect()

# ── [4] ExtraTrees ──
print("\n  [4/5] ExtraTrees...")
m_et = ExtraTreesRegressor(
    n_estimators=80, max_depth=10,
    min_samples_split=10, random_state=SEED, n_jobs=-1
)
m_et.fit(X, y_arr)
p_et = m_et.predict(X)
rmse_et = np.sqrt(mean_squared_error(y_arr, p_et))
r2_et = r2_score(y_arr, p_et)
print(f"    Train RMSE={rmse_et:.6f} R2={r2_et:.4f}")
gc.collect()

# ── [5] LightGBM DART ──
print("\n  [5/5] LightGBM DART...")
m_dart = lgb.LGBMRegressor(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
    boosting_type='dart', random_state=SEED, verbose=-1,
    drop_rate=0.1, max_drop=50
)
m_dart.fit(X, y_arr)
p_dart = m_dart.predict(X)
rmse_dart = np.sqrt(mean_squared_error(y_arr, p_dart))
r2_dart = r2_score(y_arr, p_dart)
print(f"    Train RMSE={rmse_dart:.6f} R2={r2_dart:.4f}")
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 7. OPTIMISE ENSEMBLE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Optimising ensemble weights...")

model_names = ['xgb', 'lgb', 'cb', 'et', 'lgb_dart']
model_preds_train = np.column_stack([p_xgb, p_lgb, p_cb, p_et, p_dart])
model_rmses = [rmse_xgb, rmse_lgb, rmse_cb, rmse_et, rmse_dart]

# Convex optimisation
res = minimize(
    lambda w: np.sqrt(mean_squared_error(y_arr, model_preds_train @ w)),
    np.ones(len(model_names)) / len(model_names), method='SLSQP',
    bounds=[(0, 1)] * len(model_names),
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
convex_weights = {n: round(float(w), 4) for n, w in zip(model_names, res.x)}
convex_rmse = res.fun
convex_r2 = r2_score(y_arr, model_preds_train @ res.x)

# Simple average
avg_pred = model_preds_train.mean(axis=1)
avg_rmse = np.sqrt(mean_squared_error(y_arr, avg_pred))
avg_r2 = r2_score(y_arr, avg_pred)

print(f"  Convex weights: {convex_weights}")
print(f"  Convex: RMSE={convex_rmse:.6f} R2={convex_r2:.4f}")
print(f"  Simple avg: RMSE={avg_rmse:.6f} R2={avg_r2:.4f}")

# Pick best
if convex_rmse <= avg_rmse:
    best_method = 'convex'
    best_w = res.x
else:
    best_method = 'simple_avg'
    best_w = np.ones(len(model_names)) / len(model_names)

print(f"  >>> Best ensemble: {best_method}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. GENERATE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Generating predictions for all tracts...")

# Predict with each model
pred_xgb = m_xgb.predict(X)
pred_lgb = m_lgb.predict(X)
pred_cb = m_cb.predict(X)
pred_et = m_et.predict(X)
pred_dart = m_dart.predict(X)

model_preds_all = np.column_stack([pred_xgb, pred_lgb, pred_cb, pred_et, pred_dart])

# Ensemble prediction (this is gap_only = proxy_merged without rural penalty)
gap_only_pred = model_preds_all @ best_w

print(f"  gap_only predictions: mean={gap_only_pred.mean():.4f} std={gap_only_pred.std():.4f}")
print(f"  gap_only range: [{gap_only_pred.min():.4f}, {gap_only_pred.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 9. APPLY RURAL PENALTY + CLIP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Applying rural penalty and clipping...")

# Inference formula: coverage_gap_score = model.predict(X) - 1.0 * rural_penalty
coverage_gap_score = gap_only_pred - 1.0 * rural_penalty.values

print(f"  Before clip: mean={coverage_gap_score.mean():.4f} std={coverage_gap_score.std():.4f}")
print(f"  Before clip: range=[{coverage_gap_score.min():.4f}, {coverage_gap_score.max():.4f}]")

# Clip to [-3.0, +0.5]
n_below = (coverage_gap_score < -3.0).sum()
n_above = (coverage_gap_score > 0.5).sum()
coverage_gap_score = np.clip(coverage_gap_score, -3.0, 0.5)
print(f"  Clipped {n_below} below -3.0, {n_above} above +0.5")
print(f"  After clip: mean={coverage_gap_score.mean():.4f} std={coverage_gap_score.std():.4f}")
print(f"  After clip: range=[{coverage_gap_score.min():.4f}, {coverage_gap_score.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 10. BUILD AND VALIDATE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Building and validating submission...")

submission = pd.DataFrame({
    'GEOID': geo.values,
    'coverage_gap_score': coverage_gap_score
})

# Validation checks
assert len(submission) == 85396, f"Expected 85,396 rows, got {len(submission)}"
assert submission.columns.tolist() == ['GEOID', 'coverage_gap_score'], f"Wrong columns: {submission.columns.tolist()}"
assert submission['coverage_gap_score'].notna().all(), "NaN in predictions!"
assert (submission['coverage_gap_score'] >= -3.0).all(), "Values below -3.0!"
assert (submission['coverage_gap_score'] <= 0.5).all(), "Values above 0.5!"
assert submission['GEOID'].nunique() == 85396, "Duplicate GEOIDs!"

print(f"  VALIDATION PASSED")
print(f"  Rows: {len(submission)}")
print(f"  Unique GEOIDs: {submission['GEOID'].nunique()}")
print(f"  NaN count: {submission['coverage_gap_score'].isna().sum()}")
print(f"  In [-3.0, 0.5]: {((submission['coverage_gap_score'] >= -3.0) & (submission['coverage_gap_score'] <= 0.5)).all()}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Saving submission...")

# Save to submissions directory
sub_path = SUB_DIR / "submissions_weather_enhanced.csv"
submission.to_csv(sub_path, index=False)
print(f"  Saved: {sub_path}")

# Save to download directory
dl_path = Path("/home/z/my-project/download/submission_weather_enhanced.csv")
submission.to_csv(dl_path, index=False)
print(f"  Saved: {dl_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12] Computing feature importance...")

models_dict = {
    'xgb': (m_xgb, rmse_xgb),
    'lgb': (m_lgb, rmse_lgb),
    'cb': (m_cb, rmse_cb),
    'et': (m_et, rmse_et),
    'lgb_dart': (m_dart, rmse_dart),
}

agg_imp = np.zeros(X.shape[1])
total_w = 0
for name, (model, rmse) in models_dict.items():
    try:
        imp = model.feature_importances_
        if len(imp) == X.shape[1]:
            w = 1.0 / (rmse + 1e-10)
            agg_imp += w * imp
            total_w += w
    except Exception as e:
        print(f"  Could not get importance from {name}: {e}")

if total_w > 0:
    agg_imp /= total_w

fi_df = pd.DataFrame({
    'feature': X.columns,
    'importance': agg_imp
}).sort_values('importance', ascending=False)

fi_df.to_csv(OUT / 'weather_submission_feature_importance.csv', index=False)

print(f"\n  Top 15 Features:")
for _, row in fi_df.head(15).iterrows():
    tag = ""
    if row['feature'].startswith('wf_'): tag = " [WEATHER]"
    elif row['feature'].startswith('wx_'): tag = " [WX-INTERACT]"
    print(f"    {row['feature']:45s} {row['importance']:.6f}{tag}")

wx_imp = fi_df[fi_df['feature'].str.startswith('wx_')]
if len(wx_imp) > 0:
    print(f"\n  Weather Interaction Features (wx_*):")
    for _, row in wx_imp.iterrows():
        print(f"    {row['feature']:45s} {row['importance']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13] Submission summary...")

scores = submission['coverage_gap_score']
print(f"\n  Distribution:")
print(f"    count:  {len(scores)}")
print(f"    mean:   {scores.mean():.6f}")
print(f"    std:    {scores.std():.6f}")
print(f"    min:    {scores.min():.6f}")
print(f"    25%:    {scores.quantile(0.25):.6f}")
print(f"    50%:    {scores.median():.6f}")
print(f"    75%:    {scores.quantile(0.75):.6f}")
print(f"    max:    {scores.max():.6f}")
print(f"    < -1.0: {(scores < -1.0).sum()} ({(scores < -1.0).mean()*100:.1f}%)")
print(f"    < -0.5: {(scores < -0.5).sum()} ({(scores < -0.5).mean()*100:.1f}%)")
print(f"    == 0:   {(scores == 0).sum()} ({(scores == 0).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 14. SAVE METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[14] Saving metadata...")

metadata = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'weather_enhanced_submission',
    'n_tracts': int(len(submission)),
    'n_features': int(X.shape[1]),
    'ensemble_method': best_method,
    'ensemble_weights': convex_weights if best_method == 'convex' else {n: 0.2 for n in model_names},
    'model_train_rmse': {n: float(r) for n, r in zip(model_names, model_rmses)},
    'submission_stats': {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max()),
        'median': float(scores.median()),
    },
    'clipping': {'lower': -3.0, 'upper': 0.5, 'n_clipped_below': int(n_below), 'n_clipped_above': int(n_above)},
    'rural_penalty_coefficient': 1.0,
    'selected_wx_features': sel_wx,
    'selected_wf_features': sel_wf,
    'elapsed_sec': round(time.time() - t0, 1),
}

with open(OUT / 'weather_submission_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"  Saved: weather_submission_metadata.json")

el = time.time() - t0
print(f"\n{'=' * 78}")
print(f"DONE in {el:.0f}s")
print(f"Submission: {len(submission)} tracts")
print(f"Ensemble: {best_method} with weights {convex_weights if best_method == 'convex' else 'simple_avg'}")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Score mean:  {scores.mean():.4f}")
print(f"Files: {sub_path}, {dl_path}")
print(f"{'=' * 78}")
