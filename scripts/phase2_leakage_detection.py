#!/usr/bin/env python3
"""Run pipeline step by step - Phase 2: Feature selection + H3 blocks + leakage check"""
import numpy as np, pandas as pd, json, time, logging, warnings
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
import h3
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SEED = 42; np.random.seed(SEED)
PROJ = Path(__file__).resolve().parent.parent  # project root from scripts/
OUT = PROJ / "data/output"
t0 = time.time()

# Load engineered features
feat = pd.read_parquet(OUT / "engineered_features_v3.parquet")
log.info(f"Loaded engineered features: {feat.shape}")

# Target
tcol = 'building_gap'  # No coverage_gap_score in data yet

# Prepare features
drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
        'building_gap','road_gap','building_ratio','road_ratio','building_count_ratio',
        'building_count_gap','road_count_ratio','road_count_gap','road_length_ratio',
        'road_length_gap','poi_facility_gap','poi_to_facility_ratio',
        'coverage_gap_score','coverage_gap','gap_score','coverage_score']
feat = feat.loc[:, ~feat.columns.duplicated()]
fc = [c for c in feat.columns if c not in drop and pd.api.types.is_numeric_dtype(feat[c])]
X = feat[fc].copy(); y = feat[tcol].copy(); geo = feat['GEOID'].astype(str).copy()
v = y.notna(); X, y, geo = X[v], y[v], geo[v]
X = X.fillna(-999).replace([np.inf, -np.inf], -999)
s = X.std(); X = X[s[s > 1e-10].index]
nlp = X.isna().mean(); X = X[nlp[nlp < 0.95].index]
log.info(f"After cleanup: {X.shape[1]} feats, {X.shape[0]} tracts")

# Advanced feature selection
log.info("Advanced feature selection (corr+MI+var)...")
cs = X.corrwith(y).abs().fillna(0)
nmi = min(3000, len(X)); mi_idx = np.random.choice(len(X), nmi, replace=False)
try:
    mi = mutual_info_regression(X.iloc[mi_idx].fillna(-999), y.iloc[mi_idx], n_neighbors=5, random_state=SEED)
    ms = pd.Series(mi, index=X.columns); ms = ms/(ms.max()+1e-10)
except Exception as e:
    log.warning(f"MI failed: {e}"); ms = pd.Series(0, index=X.columns)
vs = X.var().fillna(0); vs = vs/(vs.max()+1e-10)
comb = 0.5*cs + 0.35*ms + 0.15*vs
sel = comb.sort_values(ascending=False).head(120).index.tolist()
X = X[sel]
log.info(f"Selected {len(sel)} features")

# Remove highly correlated
cm = X.corr().abs(); up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
td = [c for c in up.columns if any(up[c] > 0.98)]; X = X.drop(columns=td)
log.info(f"After dedup: {X.shape[1]} features, {X.shape[0]} tracts | mean={y.mean():.4f} std={y.std():.4f}")

# Compute H3 blocks
log.info("Computing H3 spatial blocks (res=4)...")
lats = feat.loc[v, 'centroid_lat']
lons = feat.loc[v, 'centroid_lon']
blocks = []
for la, lo in zip(lats.values, lons.values):
    try: blocks.append(h3.latlng_to_cell(float(la), float(lo), 4))
    except: blocks.append('unk')
blocks = pd.Series(blocks, index=geo.index)
nb = blocks.nunique()
log.info(f"{nb} H3 spatial blocks")

# Spatial split function
def spatial_split(X, y, blocks, nf=5):
    ub = list(blocks.unique()); np.random.seed(SEED); np.random.shuffle(ub)
    fa = {b: i%nf for i, b in enumerate(ub)}
    sf = blocks.map(fa).values
    return [(np.where(sf!=f)[0], np.where(sf==f)[0]) for f in range(nf)]

# Quick leakage check: Train XGBoost with County CV vs H3 CV
import xgboost as xgb
xm = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.02, subsample=0.8,
                       colsample_bytree=0.7, tree_method='hist', random_state=SEED)

# H3 CV
log.info("Training XGBoost with H3 Spatial Block CV...")
h3_splits = spatial_split(X, y, blocks, 5)
oof_h3 = np.full(len(y), np.nan)
for fi, (ti, vi) in enumerate(h3_splits):
    m = xgb.XGBRegressor(**xm.get_params())
    m.fit(X.iloc[ti], y.iloc[ti], eval_set=[(X.iloc[vi], y.iloc[vi])], verbose=False)
    p = m.predict(X.iloc[vi]); oof_h3[vi] = p
    rmse = np.sqrt(mean_squared_error(y.iloc[vi], p)); r2 = r2_score(y.iloc[vi], p)
    log.info(f"  H3 Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
vh3 = ~np.isnan(oof_h3)
h3_rmse = np.sqrt(mean_squared_error(y[vh3], oof_h3[vh3]))
h3_r2 = r2_score(y[vh3], oof_h3[vh3])
log.info(f"H3 CV: RMSE={h3_rmse:.6f} R2={h3_r2:.4f}")

# County CV
log.info("Training XGBoost with County GroupKFold...")
from sklearn.model_selection import GroupKFold
county_blocks = geo.str[:5]
gkf = GroupKFold(n_splits=5)
oof_county = np.full(len(y), np.nan)
for fi, (ti, vi) in enumerate(gkf.split(X, y, county_blocks)):
    m = xgb.XGBRegressor(**xm.get_params())
    m.fit(X.iloc[ti], y.iloc[ti], eval_set=[(X.iloc[vi], y.iloc[vi])], verbose=False)
    p = m.predict(X.iloc[vi]); oof_county[vi] = p
    rmse = np.sqrt(mean_squared_error(y.iloc[vi], p)); r2 = r2_score(y.iloc[vi], p)
    log.info(f"  County Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
vc = ~np.isnan(oof_county)
county_rmse = np.sqrt(mean_squared_error(y[vc], oof_county[vc]))
county_r2 = r2_score(y[vc], oof_county[vc])
log.info(f"County CV: RMSE={county_rmse:.6f} R2={county_r2:.4f}")

# Leakage analysis
ld = county_r2 - h3_r2
log.info(f"\n{'='*50}")
log.info(f"LEAKAGE ANALYSIS:")
log.info(f"  County CV R2: {county_r2:.4f}")
log.info(f"  H3 CV R2:     {h3_r2:.4f}")
log.info(f"  R2 Drop:       {ld:.4f}")
if ld > 0.05:
    log.warning(f"  *** SPATIAL LEAKAGE CONFIRMED! R2 drop = {ld:.4f} ***")
    log.warning(f"  County GroupKFold allows spatial autocorrelation to leak across county boundaries")
    log.warning(f"  H3 Spatial Block CV properly isolates geographically proximate tracts")
elif ld > 0.02:
    log.info(f"  MODERATE leakage detected (R2 drop = {ld:.4f})")
else:
    log.info(f"  Minimal leakage (R2 drop = {ld:.4f})")

# Save
res = {
    'county_cv_r2': float(county_r2), 'county_cv_rmse': float(county_rmse),
    'h3_cv_r2': float(h3_r2), 'h3_cv_rmse': float(h3_rmse),
    'r2_drop': float(ld), 'leakage_confirmed': bool(ld > 0.05),
    'n_h3_blocks': int(nb), 'n_features': int(X.shape[1]),
    'n_tracts': int(X.shape[0])
}
with open(OUT / 'leakage_analysis.json', 'w') as f: json.dump(res, f, indent=2)
log.info(f"Leakage analysis saved. Phase 2 done in {time.time()-t0:.1f}s")
