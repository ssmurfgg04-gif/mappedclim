#!/usr/bin/env python3
"""
WEATHER-ENHANCED PIPELINE — Tests whether weather forecast features improve
the bias-bounty-mapping-equity model.

Compares:
  A) Baseline: same features as merged pipeline (no weather)
  B) Weather:  baseline + wf_* features + weather×gap interactions + weather×vulnerability interactions

Trains 5-model ensemble: XGBoost, LightGBM, CatBoost, ExtraTrees, LightGBM-DART
Uses H3 spatial block CV (same as current pipeline)
Target: gap_only = proxy_merged (coverage gap score without rural penalty)
Inference adds: -1.0 * rural_penalty at prediction time
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
import h3

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
NF = 3  # number of CV folds

print("=" * 78)
print("WEATHER-ENHANCED PIPELINE: A/B test for weather features")
print("=" * 78)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PRE-ENGINEERED FEATURES (from Phase 1 output) + MERGE WEATHER
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading pre-engineered features + weather...")

# Load the existing engineered features from Phase 1
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
print(f"  Engineered features: {feat.shape}")

# Load weather features
wf_df = pd.read_parquet(PROJ / "kaggle_dataset/weather_forecast_features.parquet")
wf_df['GEOID'] = wf_df['GEOID'].astype(str)
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Weather features: {wf_df.shape}")

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

# Merge weather
before_cols = feat.shape[1]
feat = feat.merge(wf_df[['GEOID'] + wf_keep], on='GEOID', how='left')
print(f"  Merged weather: {before_cols} -> {feat.shape[1]} cols")
del wf_df; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 2. WEATHER INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Engineering weather interaction features...")

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
# 3. PREPARE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Preparing features for training...")

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
X_full = feat[fc].copy()

valid = y.notna()
X_full, y, geo = X_full[valid], y[valid], geo[valid]
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

# ══════════════════════════════════════════════════════════════════════════════
# 4. H3 SPATIAL CV SPLITS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Computing H3 spatial CV splits...")

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
# 5. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(X, y, max_features=80):
    cs = X.corrwith(y).abs().fillna(0)
    top_k = min(max_features, len(cs))
    X_sel = X[cs.sort_values(ascending=False).head(top_k).index]
    cm = X_sel.corr().abs()
    up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    to_drop = [c for c in up.columns if any(up[c] > 0.98)]
    X_sel = X_sel.drop(columns=to_drop)
    return X_sel

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING FUNCTION (5-model ensemble)
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(X, y, splits, label=""):
    print(f"\n  Training 5-model ensemble [{label}]...")
    print(f"  Features: {X.shape[1]}, Tracts: {X.shape[0]}")

    def train_model(model, name):
        oof = np.full(len(y), np.nan); scores = []
        fold_importances = []
        for fi, (ti, vi) in enumerate(splits):
            m = type(model)(**model.get_params())
            Xt, yt, Xv, yv = X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi]
            try:
                if isinstance(m, xgb.XGBRegressor):
                    m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
                elif isinstance(m, lgb.LGBMRegressor):
                    m.fit(Xt, yt, eval_set=[(Xv, yv)],
                          callbacks=[lgb.early_stopping(20, verbose=False)])
                elif isinstance(m, CatBoostRegressor):
                    m.fit(Xt, yt, eval_set=(Xv, yv), verbose=0, early_stopping_rounds=20)
                else:
                    m.fit(Xt, yt)
            except Exception as e:
                print(f"    {name} F{fi} err: {e}"); continue
            p = m.predict(Xv); oof[vi] = p
            rmse = np.sqrt(mean_squared_error(yv, p)); r2 = r2_score(yv, p)
            scores.append((rmse, r2))
            try:
                if hasattr(m, 'feature_importances_'):
                    fold_importances.append(m.feature_importances_)
            except:
                pass
            elapsed = time.time() - t0
            print(f"    {name} F{fi}: RMSE={rmse:.6f} R2={r2:.4f} ({elapsed:.0f}s)")
            del m; gc.collect()
        if scores:
            rmse_m = np.mean([s[0] for s in scores]); r2_m = np.mean([s[1] for s in scores])
            print(f"    {name} mean: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
            avg_imp = np.mean(fold_importances, axis=0) if fold_importances else None
            return oof, rmse_m, r2_m, avg_imp
        return oof, 999, 0, None

    oofs = {}; msum = {}; importances = {}

    # [1] XGBoost
    print("\n    [1] XGBoost...")
    o, r, r2, imp = train_model(
        xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7,
                         reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                         tree_method='hist', random_state=SEED), 'XGB')
    oofs['xgb'] = o; msum['xgb'] = (r, r2); importances['xgb'] = imp; gc.collect()

    # [2] LightGBM GBDT
    print("\n    [2] LightGBM GBDT...")
    o, r, r2, imp = train_model(
        lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7,
                          reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                          boosting_type='gbdt', random_state=SEED, verbose=-1), 'LGB')
    oofs['lgb'] = o; msum['lgb'] = (r, r2); importances['lgb'] = imp; gc.collect()

    # [3] CatBoost
    print("\n    [3] CatBoost...")
    o, r, r2, imp = train_model(
        CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05,
                          l2_leaf_reg=3, random_seed=SEED, verbose=0), 'CB')
    oofs['cb'] = o; msum['cb'] = (r, r2); importances['cb'] = imp; gc.collect()

    # [4] ExtraTrees
    print("\n    [4] ExtraTrees...")
    o, r, r2, imp = train_model(
        ExtraTreesRegressor(n_estimators=80, max_depth=10,
                            min_samples_split=10, random_state=SEED, n_jobs=-1), 'ET')
    oofs['et'] = o; msum['et'] = (r, r2); importances['et'] = imp; gc.collect()

    # [5] LightGBM DART
    print("\n    [5] LightGBM DART...")
    o, r, r2, imp = train_model(
        lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7,
                          reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                          boosting_type='dart', random_state=SEED, verbose=-1,
                          drop_rate=0.1, max_drop=50), 'LGB_DART')
    oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); importances['lgb_dart'] = imp; gc.collect()

    # ── ENSEMBLE ──
    print("\n    Ensembling...")
    ns = list(oofs.keys()); mat = np.column_stack([oofs[n] for n in ns])
    vv = ~np.any(np.isnan(mat), axis=1); mv = mat[vv]; yv = y.values[vv]

    res = minimize(
        lambda w: np.sqrt(mean_squared_error(yv, mv @ w)),
        np.ones(len(ns)) / len(ns), method='SLSQP',
        bounds=[(0, 1)] * len(ns),
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
    )
    cw = {n: round(float(w), 4) for n, w in zip(ns, res.x)}
    cp = mv @ res.x; cr_ = res.fun; cr2 = r2_score(yv, cp)
    print(f"    Convex: RMSE={cr_:.6f} R2={cr2:.4f} w={cw}")

    ap = mv.mean(axis=1)
    ar_ = np.sqrt(mean_squared_error(yv, ap)); ar2 = r2_score(yv, ap)
    print(f"    Simple avg: RMSE={ar_:.6f} R2={ar2:.4f}")

    best = min([('convex', cr_, cr2, cp), ('simple_avg', ar_, ar2, ap)], key=lambda x: x[1])
    bn, brm, br2_, bpred = best
    print(f"\n    >>> BEST: {bn} RMSE={brm:.6f} R2={br2_:.4f}")

    # Aggregate feature importances (weighted by inverse RMSE)
    agg_imp = np.zeros(X.shape[1])
    total_w = 0
    for name, imp in importances.items():
        if imp is not None and len(imp) == X.shape[1]:
            w = 1.0 / (msum[name][0] + 1e-10)
            agg_imp += w * imp
            total_w += w
    if total_w > 0:
        agg_imp /= total_w

    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': agg_imp
    }).sort_values('importance', ascending=False)

    return {
        'model_scores': msum,
        'best_name': bn,
        'best_rmse': brm,
        'best_r2': br2_,
        'convex_weights': cw,
        'feature_importance': fi_df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7. A/B TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("[A] BASELINE: Training without weather features")
print("=" * 78)

X_base = X_full[base_cols].copy()
X_base_sel = select_features(X_base, y, max_features=60)
print(f"  Selected {X_base_sel.shape[1]} base features")

baseline_results = train_ensemble(X_base_sel, y, splits, label="BASELINE")

print("\n" + "=" * 78)
print("[B] WEATHER-ENHANCED: Training with weather + interaction features")
print("=" * 78)

X_weather = X_full.copy()
X_weather_sel = select_features(X_weather, y, max_features=80)
print(f"  Selected {X_weather_sel.shape[1]} features (base + weather + interactions)")

weather_results = train_ensemble(X_weather_sel, y, splits, label="WEATHER-ENHANCED")

# ══════════════════════════════════════════════════════════════════════════════
# 8. ANALYSIS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("RESULTS SUMMARY")
print("=" * 78)

print(f"\n  BASELINE (no weather):")
print(f"    RMSE: {baseline_results['best_rmse']:.6f}")
print(f"    R2:   {baseline_results['best_r2']:.4f}")
print(f"    Best ensemble: {baseline_results['best_name']}")
print(f"    Weights: {baseline_results['convex_weights']}")

print(f"\n  WEATHER-ENHANCED:")
print(f"    RMSE: {weather_results['best_rmse']:.6f}")
print(f"    R2:   {weather_results['best_r2']:.4f}")
print(f"    Best ensemble: {weather_results['best_name']}")
print(f"    Weights: {weather_results['convex_weights']}")

delta_rmse = baseline_results['best_rmse'] - weather_results['best_rmse']
delta_r2 = weather_results['best_r2'] - baseline_results['best_r2']
pct_improve = (delta_rmse / baseline_results['best_rmse']) * 100

print(f"\n  DELTA:")
print(f"    RMSE change: {delta_rmse:+.6f} ({pct_improve:+.2f}%)")
print(f"    R2 change:   {delta_r2:+.4f}")
print(f"    Verdict: {'IMPROVED' if delta_rmse > 0 else 'NOT IMPROVED'}")

# Feature Importance: Weather features
fi = weather_results['feature_importance']

print(f"\n  Top 10 Weather Features (wf_*) by Importance:")
wf_imp = fi[fi['feature'].str.startswith('wf_')].head(10)
for _, row in wf_imp.iterrows():
    print(f"    {row['feature']:45s} {row['importance']:.6f}")

print(f"\n  Top 10 Weather Interaction Features (wx_*) by Importance:")
wx_imp = fi[fi['feature'].str.startswith('wx_')].head(10)
for _, row in wx_imp.iterrows():
    print(f"    {row['feature']:45s} {row['importance']:.6f}")

print(f"\n  Top 20 Overall Features (weather-enhanced model):")
for _, row in fi.head(20).iterrows():
    tag = ""
    if row['feature'].startswith('wf_'): tag = " [WEATHER]"
    elif row['feature'].startswith('wx_'): tag = " [WX-INTERACT]"
    print(f"    {row['feature']:45s} {row['importance']:.6f}{tag}")

print(f"\n  Per-Model Comparison:")
print(f"    {'Model':12s} {'Baseline RMSE':>14s} {'Weather RMSE':>14s} {'Delta':>10s}")
for model in baseline_results['model_scores']:
    b_rmse = baseline_results['model_scores'][model][0]
    w_rmse = weather_results['model_scores'][model][0]
    d = b_rmse - w_rmse
    print(f"    {model:12s} {b_rmse:14.6f} {w_rmse:14.6f} {d:+10.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Saving results...")

fi.to_csv(OUT / 'weather_feature_importance.csv', index=False)

comparison = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'weather_enhanced_ab_test',
    'baseline': {
        'rmse': float(baseline_results['best_rmse']),
        'r2': float(baseline_results['best_r2']),
        'ensemble': baseline_results['best_name'],
        'weights': baseline_results['convex_weights'],
        'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])}
                   for k, v in baseline_results['model_scores'].items()},
    },
    'weather_enhanced': {
        'rmse': float(weather_results['best_rmse']),
        'r2': float(weather_results['best_r2']),
        'ensemble': weather_results['best_name'],
        'weights': weather_results['convex_weights'],
        'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])}
                   for k, v in weather_results['model_scores'].items()},
    },
    'delta': {
        'rmse': float(delta_rmse),
        'r2': float(delta_r2),
        'pct_improvement': float(pct_improve),
        'weather_improves_model': bool(delta_rmse > 0),
    },
    'top_weather_features': wf_imp['feature'].tolist(),
    'top_weather_interactions': wx_imp['feature'].tolist(),
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(OUT / 'weather_enhanced_results.json', 'w') as f:
    json.dump(comparison, f, indent=2, default=str)

print(f"  Saved: weather_feature_importance.csv, weather_enhanced_results.json")

el = time.time() - t0
print(f"\n{'=' * 78}")
print(f"DONE in {el:.0f}s")
print(f"Baseline RMSE:      {baseline_results['best_rmse']:.6f}")
print(f"Weather RMSE:       {weather_results['best_rmse']:.6f}")
print(f"Delta RMSE:         {delta_rmse:+.6f} ({pct_improve:+.2f}%)")
print(f"Weather IMPROVES:   {'YES' if delta_rmse > 0 else 'NO'}")
print(f"{'=' * 78}")
