#!/usr/bin/env python3
"""
CORRECTED PIPELINE — Train on gap_only, apply rural penalty at inference
=========================================================================
Deterministic proved: adding rural penalty to TRAINING target destroys LORO R².
  - alpha=0.0 (gap_only):     LORO R² = +0.155 ✅
  - alpha=1.0 (proxy_merged): LORO R² = -72.87 ❌

Fix:
  y_train = gap_only = -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg))
  y_pred  = model.predict(X) - 1.0 * rural   # rural penalty at inference only
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import h3

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42
NF = 3

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = Path("/home/z/my-project/download"); DL.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("CORRECTED PIPELINE: Train on gap_only, rural penalty at inference")
print("=" * 72)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: LOAD + COMPUTE GAPS + gap_only TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading data + computing gap_only target...")

import pyarrow.parquet as pq
import pyarrow as pa

# Load minimal columns
nf_schema = pq.read_schema(PROJ / "data/features/national_tract_features.parquet")
all_nat_cols = nf_schema.names

must_have = ['GEOID', 'building_gap', 'road_gap', 'pct_urban', 'svi_overall',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'poi_cnt', 'poi_mean_confidence', 'poi_very_high_conf_fraction',
             'tribal_any', 'tribal_pct', 'pop_total',
             'bldg_total_sources', 'bldg_source_diversity',
             'source_coverage_fraction', 'source_diversity_entropy',
             'usgs_wildfire_ever', 'usgs_wildfire_burned_pct_area',
             'cvi_overall']

# Add _covered + KNN + ratio columns
for c in all_nat_cols:
    if '_covered' in c.lower() or '_knn' in c.lower() or '_county_' in c.lower():
        must_have.append(c)
    if c in ['building_ratio', 'road_ratio', 'building_count_ratio', 'road_length_ratio',
             'building_count_gap', 'road_count_gap', 'road_length_gap', 'road_count_ratio',
             'building_area_ratio']:
        must_have.append(c)

# Add numeric features (limited)
numeric_types = {pa.float32(), pa.float64(), pa.int32(), pa.int64(),
                 pa.uint8(), pa.bool_(), pa.int8(), pa.uint32()}
numeric_features = [c for c in all_nat_cols if c not in must_have and
                    nf_schema.field(c).type in numeric_types]
must_have.extend(numeric_features[:150])

load_cols = list(dict.fromkeys([c for c in must_have if c in all_nat_cols]))
feat = pd.read_parquet(PROJ / "data/features/national_tract_features.parquet", columns=load_cols)

# Strata
strata_schema = pq.read_schema(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
strata_need = ['GEOID', 'svi_overall', 'svi_socioeconomic', 'svi_household', 'svi_minority',
               'tribal_any', 'tribal_pct', 'pct_urban', 'pop_total',
               'cvi_overall', 'INTPTLAT', 'INTPTLON']
strata_load = [c for c in strata_need if c in strata_schema.names]
strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet", columns=strata_load)

feat['GEOID'] = feat['GEOID'].astype(str)
strata['GEOID'] = strata['GEOID'].astype(str)

new_cols = [c for c in strata_load if c not in feat.columns or c == 'GEOID']
if len(new_cols) > 1:
    feat = feat.merge(strata[new_cols], on='GEOID', how='left')
del strata; gc.collect()

# Centroid coords
if 'centroid_lat' not in feat.columns or feat['centroid_lat'].isna().all():
    if 'INTPTLAT' in feat.columns:
        feat['centroid_lat'] = pd.to_numeric(feat['INTPTLAT'], errors='coerce')
        feat['centroid_lon'] = pd.to_numeric(feat['INTPTLON'], errors='coerce')

print(f"  Loaded: {feat.shape}")

# ── Compute corrected gaps ──
bg = feat['building_gap'].fillna(0) if 'building_gap' in feat.columns else pd.Series(0, index=feat.index)
rg = feat['road_gap'].fillna(0) if 'road_gap' in feat.columns else pd.Series(0, index=feat.index)
pct_urban = feat['pct_urban'].fillna(0.5) if 'pct_urban' in feat.columns else pd.Series(0.5, index=feat.index)
rural = (1 - pct_urban).clip(0, 1)

# poi_facility_gap_corrected (same heuristic)
poi_total = feat['poi_cnt'].fillna(0) if 'poi_cnt' in feat.columns else pd.Series(0, index=feat.index)
if 'poi_very_high_conf_fraction' in feat.columns:
    corr_factor = feat['poi_very_high_conf_fraction'].fillna(0.1)
    if 'poi_mean_confidence' in feat.columns:
        corr_factor = corr_factor + (feat['poi_mean_confidence'].fillna(0.5) - 0.5).clip(0, 0.5) * 0.3
    corr_factor = corr_factor.clip(0.05, 0.5)
else:
    corr_factor = pd.Series(0.10, index=feat.index)
poi_corrected = poi_total * corr_factor
poi_q75 = np.log1p(poi_corrected.quantile(0.75)).clip(1, None)
poi_signal = -np.log1p(poi_corrected) / poi_q75
poi_gap_corr = 0.6 * bg + 0.4 * poi_signal

# building_area_gap (approximation)
building_area_gap = 1.3 * bg + 0.2 * bg * rural

# ── THE CORRECTED TARGET: gap_only (alpha=0) ──
gap_only = -np.mean([
    np.maximum(0, bg),
    2.0 * np.maximum(0, building_area_gap),
    np.maximum(0, rg),
    np.maximum(0, poi_gap_corr)
], axis=0)

# Store both for comparison
feat['gap_only'] = gap_only
feat['rural'] = rural

# Also compute proxy_merged (alpha=1) for comparison
proxy_merged = gap_only - 1.0 * rural

print(f"\n  gap_only (TRAINING target):  mean={gap_only.mean():.4f}, std={gap_only.std():.4f}, "
      f"range=[{gap_only.min():.4f}, {gap_only.max():.4f}]")
print(f"  proxy_merged (alpha=1):      mean={proxy_merged.mean():.4f}, std={proxy_merged.std():.4f}, "
      f"range=[{proxy_merged.min():.4f}, {proxy_merged.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (minimal — Deterministic proved source comp is a dud)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Feature engineering (minimal set)...")

nf = {}
F = lambda s, v=0: s.fillna(v) if s is not None else pd.Series(v, index=feat.index)

svi_col = feat.get('svi_overall'); tribal = feat.get('tribal_any')
pu = feat.get('pct_urban'); pop = feat.get('pop_total')
cvi = feat.get('cvi_overall'); wf = feat.get('usgs_wildfire_ever')

bv = F(bg).values; rv = F(rg).values; bav = F(building_area_gap).values

# Core gap features
nf['bldg_gap_clip'] = np.maximum(0, bv)
nf['area_gap_clip'] = np.maximum(0, bav)
nf['road_gap_clip'] = np.maximum(0, rv)
nf['bldg_gap_abs'] = np.abs(bv)
nf['road_gap_abs'] = np.abs(rv)
nf['bldg_x_area_gap'] = bv * bav
nf['bldg_road_diff'] = bv - rv

# Rural features (as INPUT features, NOT in target)
if pu is not None:
    puv = F(pu, 0.5).values; rur = (1 - puv).clip(0, 1)
    nf['rural_continuous'] = rur
    nf['rural_indicator'] = (puv < 0.5).astype(float)
    nf['rural_x_bldg_clip'] = rur * np.maximum(0, bv)
    nf['pct_urban_x_bldg'] = puv * bv

# Tribal
if tribal is not None:
    tf = (F(tribal).values > 0).astype(float)
    nf['tribal_x_bldg_clip'] = tf * np.maximum(0, bv)
    if pu is not None: nf['tribal_x_rural'] = tf * rur

# SVI (keep as feature — model can learn to ignore)
if svi_col is not None:
    nf['svi_x_bldg_clip'] = F(svi_col).values * np.maximum(0, bv)

# CVI
if cvi is not None: nf['cvi_x_bldg_clip'] = F(cvi).values * np.maximum(0, bv)

# Compound risk
comp = np.abs(bv) + np.abs(rv)
nf['compound_risk'] = comp
if pu is not None: nf['rural_x_risk'] = rur * comp

# Population
if pop is not None: nf['log_pop'] = np.log1p(F(pop).values)

# County LOO
county = feat['GEOID'].astype(str).str[:5]; bgv = F(bg)
cs = bgv.groupby(county).agg(['mean', 'count']); cs.columns = ['mean', 'count']
gm = bgv.mean(); sm = 10
cms = (cs['mean'] * cs['count'] + gm * sm) / (cs['count'] + sm)
nf['bldg_county_loo_smooth'] = cms[county].values

if nf:
    nd = pd.DataFrame(nf, index=feat.index)
    nd = nd.replace([np.inf, -np.inf], np.nan)
    feat = pd.concat([feat, nd], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    print(f"  +{len(nf)} features -> {feat.shape[1]} total")

gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# PREPARE MODELING DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Preparing modeling data...")

y = feat['gap_only'].copy()  # TRAIN ON gap_only!
geo = feat['GEOID'].astype(str).copy()
rural_for_inference = feat['rural'].copy()  # Save for inference-time penalty

drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
             'building_count_ratio', 'building_count_gap',
             'road_count_ratio', 'road_count_gap', 'road_length_ratio', 'road_length_gap',
             'poi_facility_gap', 'poi_to_facility_ratio',
             'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
             'gap_only', 'rural', 'proxy_merged',
             'building_area_gap', 'poi_facility_gap_corrected', 'poi_to_facility_ratio_corrected']

feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy()

valid = y.notna()
X, y, geo, rural_inf = X[valid], y[valid], geo[valid], rural_for_inference[valid]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

s = X.std(); X = X[s[s > 1e-10].index]
cs = X.corrwith(y).abs().fillna(0)
X = X[cs.sort_values(ascending=False).head(50).index]

cm = X.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X = X.drop(columns=to_drop)

print(f"  {X.shape[1]} features, {X.shape[0]} tracts")
print(f"  Target: gap_only, mean={y.mean():.4f}, std={y.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# H3 SPATIAL BLOCK CV
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] H3 spatial blocks...")

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
print(f"  {n_blocks} H3 blocks at resolution 4")

ub = list(blocks.unique()); np.random.shuffle(ub)
fa = {b: i % NF for i, b in enumerate(ub)}
sf = blocks.map(fa).values
splits = [(np.where(sf != f)[0], np.where(sf == f)[0]) for f in range(NF)]

del feat; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN 5-MODEL ENSEMBLE ON gap_only
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Training 5-model ensemble on gap_only...")

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
            elif isinstance(m, CatBoostRegressor):
                m.fit(Xt, yt, eval_set=(Xv, yv),
                      early_stopping_rounds=20, verbose=0)
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
        rmse_m = np.mean([s[0] for s in scores]); r2_m = np.mean([s[1] for s in scores])
        print(f"  {name} mean: RMSE={rmse_m:.6f} R2={r2_m:.4f}")
        return oof, rmse_m, r2_m
    return oof, 999, 0

oofs = {}; msum = {}

print("\n  [1] XGBoost...")
o, r, r2 = train_model(
    xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
                     tree_method='hist', random_state=SEED), 'XGB')
oofs['xgb'] = o; msum['xgb'] = (r, r2); gc.collect()

print("\n  [2] LightGBM...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='gbdt', random_state=SEED, verbose=-1), 'LGB')
oofs['lgb'] = o; msum['lgb'] = (r, r2); gc.collect()

print("\n  [3] CatBoost...")
o, r, r2 = train_model(
    CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                      l2_leaf_reg=3.0, random_strength=1.0,
                      bagging_temperature=0.5, random_seed=SEED,
                      verbose=0, thread_count=1, allow_writing_files=False), 'CAT')
oofs['cat'] = o; msum['cat'] = (r, r2); gc.collect()

print("\n  [4] ExtraTrees...")
o, r, r2 = train_model(
    ExtraTreesRegressor(n_estimators=80, max_depth=10,
                        min_samples_split=10, random_state=SEED, n_jobs=1), 'ET')
oofs['et'] = o; msum['et'] = (r, r2); gc.collect()

print("\n  [5] LightGBM DART...")
o, r, r2 = train_model(
    lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
                      boosting_type='dart', random_state=SEED, verbose=-1,
                      drop_rate=0.1, max_drop=50), 'DART')
oofs['lgb_dart'] = o; msum['lgb_dart'] = (r, r2); gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Ensembling...")

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

# Simple avg
ap = mv.mean(axis=1)
ar_ = np.sqrt(mean_squared_error(yv, ap)); ar2 = r2_score(yv, ap)
print(f"  Simple avg: RMSE={ar_:.6f} R2={ar2:.4f}")

best = min([('convex', cr_, cr2, cp), ('simple_avg', ar_, ar2, ap)], key=lambda x: x[1])
bn, brm, br2_, bpred_gap_only = best
print(f"\n  >>> BEST on gap_only: {bn} RMSE={brm:.6f} R2={br2_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# APPLY RURAL PENALTY AT INFERENCE TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Applying rural penalty at inference time...")

rural_vv = rural_inf.values[vv]

# Final score = model.predict(X) - alpha * rural
# Test alpha sweep at inference time
for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
    final_pred = bpred_gap_only - alpha * rural_vv
    final_rmse = np.sqrt(mean_squared_error(yv - alpha * rural_vv, final_pred))  # trivially 0 for gap_only
    # More useful: compare against proxy_merged (alpha=1)
    proxy_merged_vv = yv - 1.0 * rural_vv
    rmse_vs_merged = np.sqrt(mean_squared_error(proxy_merged_vv, final_pred))
    print(f"  alpha={alpha:.1f}: pred range=[{final_pred.min():.4f}, {final_pred.max():.4f}], "
          f"mean={final_pred.mean():.4f}")

# Use alpha=1.0 for final submission (matches the equity framing)
final_pred = bpred_gap_only - 1.0 * rural_vv
print(f"\n  Final (alpha=1.0 inference): mean={final_pred.mean():.4f}, std={final_pred.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# CIRCULARITY + CASE VALIDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Circularity test...")

svi_vals = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet",
                            columns=['svi_overall'])['svi_overall'].fillna(0.5).values
bg_vals = bg.values[:len(svi_vals)] if len(bg) <= len(svi_vals) else bg.values[:len(svi_vals)]
# Simple R²
lr = LinearRegression()
lr.fit(svi_vals[:len(bg_vals)].reshape(-1, 1), bg_vals)
svi_r2 = r2_score(bg_vals, lr.predict(svi_vals[:len(bg_vals)].reshape(-1, 1)))
print(f"  SVI → building_gap: R²={svi_r2:.4f} (RED HERRING)")

print("\n[9] Case validations (final score with alpha=1.0 inference)...")

geo_str = geo.values[vv]
tribal_check = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet",
                                columns=['GEOID', 'tribal_any'])
tribal_check['GEOID'] = tribal_check['GEOID'].astype(str)
tribal_map = dict(zip(tribal_check['GEOID'], tribal_check['tribal_any'].fillna(0)))

for name, mask in [
    ('Hidalgo TX', np.array([g.startswith('48215') for g in geo_str])),
    ('Maricopa AZ', np.array([g.startswith('04013') for g in geo_str])),
    ('Rural', rural_vv > 0.5),
    ('Urban', rural_vv <= 0.5),
    ('OK tribal', np.array([g.startswith('40') and tribal_map.get(g, 0) > 0 for g in geo_str])),
    ('OK non-tribal', np.array([g.startswith('40') and tribal_map.get(g, 0) == 0 for g in geo_str])),
]:
    if mask.sum() > 0:
        vals = final_pred[mask]
        print(f"  {name}: n={mask.sum()}, mean={vals.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# BIAS DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Bias discovery...")

resid = yv - bpred_gap_only  # residuals on gap_only training target
bias_findings = []

# Reload tribal/svi for bias analysis
feat_mini = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet",
                            columns=['GEOID', 'svi_overall', 'tribal_any', 'pct_urban'])
feat_mini['GEOID'] = feat_mini['GEOID'].astype(str)

for dim_name, col, method in [
    ('HighSVI vs LowSVI', 'svi_overall', 'quantile'),
    ('Tribal vs Non', 'tribal_any', 'binary'),
    ('Rural vs Urban', 'pct_urban', 'threshold'),
]:
    c = feat_mini[col] if col in feat_mini.columns else None
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

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Submission...")

# Use final_pred (gap_only prediction - 1.0 * rural) for submission
sub_df = pd.DataFrame({
    'GEOID': geo.values[vv],
    'coverage_gap_score': np.clip(final_pred, -3.0, 0.5)
})
sub_df.to_csv(OUT / 'submission_corrected.csv', index=False)
sub_df.to_csv(DL / 'submission_corrected.csv', index=False)
print(f"  {len(sub_df)} tracts in submission")
print(f"  Score stats: mean={sub_df['coverage_gap_score'].mean():.4f}, "
      f"std={sub_df['coverage_gap_score'].std():.4f}")

# Save results
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'corrected_gap_only_train_rural_inference',
    'training_target': 'gap_only',
    'inference_adjustment': 'final = predict(X) - 1.0 * rural',
    'gap_only_formula': 'gap_only = -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg))',
    'n_tracts': int(len(sub_df)), 'n_features': int(X.shape[1]),
    'cv_type': f'H3_spatial_block_{NF}fold', 'n_h3_blocks': int(n_blocks),
    'best_ensemble': bn, 'best_rmse_gap_only': float(brm), 'best_r2_gap_only': float(br2_),
    'convex_weights': cw,
    'models': {k: {'rmse': float(v[0]), 'r2': float(v[1])} for k, v in msum.items()},
    'circularity_svi_r2': float(svi_r2),
    'elapsed_sec': round(time.time() - t0, 1),
}
with open(OUT / 'pipeline_state_corrected.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

bdf.to_csv(OUT / 'bias_findings_corrected.csv', index=False)

oof_df = pd.DataFrame(oofs)
oof_df['GEOID'] = geo.values
oof_df['gap_only'] = y.values
oof_df['rural'] = rural_inf.values
oof_df.to_parquet(OUT / 'oof_predictions_corrected.parquet', index=False)

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Training: gap_only (alpha=0) — BEST {bn} RMSE={brm:.6f} R2={br2_:.4f}")
print(f"Inference: final = predict(X) - 1.0 * rural (alpha=1 at prediction time)")
print(f"Submission: {len(sub_df)} tracts")
print(f"Key insight: Train on gaps, apply equity penalty at inference only")
print(f"{'=' * 72}")
