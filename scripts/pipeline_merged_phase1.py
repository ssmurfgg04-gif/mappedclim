#!/usr/bin/env python3
"""
MERGED PROXY PIPELINE — Phase 1: Data + Gaps + Proxy + Features
Memory-optimized: load only needed columns, process in chunks.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, time, gc, warnings, json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("PHASE 1: Data + Gaps + Proxy + Features (memory-optimized)")
print("=" * 72)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONLY NEEDED COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading data (minimal columns)...")

# First, scan what columns exist
import pyarrow.parquet as pq
nf_schema = pq.read_schema(PROJ / "data/features/national_tract_features.parquet")
all_nat_cols = nf_schema.names

# Columns we MUST have for gap computation + proxy
must_have = ['GEOID', 'building_gap', 'road_gap', 'pct_urban', 'svi_overall',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'poi_cnt', 'poi_mean_confidence', 'poi_very_high_conf_fraction',
             'tribal_any', 'tribal_pct', 'pop_total',
             'bldg_total_sources', 'bldg_source_diversity',
             'source_coverage_fraction', 'source_diversity_entropy',
             'usgs_wildfire_ever', 'usgs_wildfire_burned_pct_area',
             'cvi_overall', 'usfs_WHP_mean']

# Add all _covered columns
covered_cols_in_nat = [c for c in all_nat_cols if '_covered' in c.lower()]
must_have.extend(covered_cols_in_nat)

# Add KNN and county stats for building/road/poi
for prefix in ['building_gap', 'road_gap', 'poi_cnt']:
    for suffix in ['_knn5_mean', '_knn5_diff', '_knn10_mean', '_knn10_diff',
                   '_knn20_mean', '_knn20_diff', '_county_mean', '_county_std', '_county_dev']:
        col = prefix + suffix
        if col in all_nat_cols:
            must_have.append(col)

# Add ratio columns
for c in ['building_ratio', 'road_ratio', 'building_count_ratio', 'road_length_ratio',
          'building_count_gap', 'road_count_gap', 'road_length_gap', 'road_count_ratio']:
    if c in all_nat_cols:
        must_have.append(c)

# Add all other numeric features (KNN, confidence, etc.)
import pyarrow as pa
numeric_types = {pa.float32(), pa.float64(), pa.int32(), pa.int64(),
                 pa.uint8(), pa.bool_(), pa.int8(), pa.uint32()}
numeric_features = [c for c in all_nat_cols if c not in must_have and 
                    nf_schema.field(c).type in numeric_types]
# Limit to reasonable number
must_have.extend(numeric_features[:200])

# Deduplicate and filter to existing
load_cols = list(dict.fromkeys([c for c in must_have if c in all_nat_cols]))

feat = pd.read_parquet(PROJ / "data/features/national_tract_features.parquet", columns=load_cols)
print(f"  National features: {feat.shape}")

# Load strata — only columns we need that aren't already in feat
strata_cols_needed = ['GEOID', 'svi_overall', 'svi_socioeconomic', 'svi_household', 'svi_minority',
                      'svi_housing_transport', 'svi_pop', 'tribal_any', 'tribal_pct', 'tribal_legal',
                      'pct_urban', 'pop_rural', 'pop_urban', 'pop_total',
                      'cvi_overall', 'cvi_baseline', 'cvi_climate',
                      'INTPTLAT', 'INTPTLON',
                      'usgs_wildfire_ever', 'usgs_wildfire_burned_pct_area']
strata_schema = pq.read_schema(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
strata_load = [c for c in strata_cols_needed if c in strata_schema.names]
strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet", columns=strata_load)
print(f"  Strata: {strata.shape}")

feat['GEOID'] = feat['GEOID'].astype(str)
strata['GEOID'] = strata['GEOID'].astype(str)

# Merge only new columns
new_cols = [c for c in strata_load if c not in feat.columns or c == 'GEOID']
if len(new_cols) > 1:
    before = feat.shape[1]
    feat = feat.merge(strata[new_cols], on='GEOID', how='left')
    print(f"  Merged strata: {before} -> {feat.shape[1]} cols")

del strata; gc.collect()

# Centroid coords
if 'centroid_lat' not in feat.columns or feat['centroid_lat'].isna().all():
    if 'INTPTLAT' in feat.columns:
        feat['centroid_lat'] = pd.to_numeric(feat['INTPTLAT'], errors='coerce')
        feat['centroid_lon'] = pd.to_numeric(feat['INTPTLON'], errors='coerce')

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE GAPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Computing corrected gaps...")

# poi_facility_gap_corrected
poi_total = feat['poi_cnt'].fillna(0) if 'poi_cnt' in feat.columns else pd.Series(0, index=feat.index)
if 'poi_very_high_conf_fraction' in feat.columns:
    corr_factor = feat['poi_very_high_conf_fraction'].fillna(0.1)
    if 'poi_mean_confidence' in feat.columns:
        medium_weight = (feat['poi_mean_confidence'].fillna(0.5) - 0.5).clip(0, 0.5) * 0.3
        corr_factor = corr_factor + medium_weight
    corr_factor = corr_factor.clip(0.05, 0.5)
elif 'poi_mean_confidence' in feat.columns:
    corr_factor = (feat['poi_mean_confidence'].fillna(0.5) * 0.3).clip(0.05, 0.5)
else:
    corr_factor = pd.Series(0.10, index=feat.index)

poi_corrected = poi_total * corr_factor
bg = feat['building_gap'].fillna(0) if 'building_gap' in feat.columns else pd.Series(0, index=feat.index)
poi_q75 = np.log1p(poi_corrected.quantile(0.75)).clip(1, None)
poi_signal = -np.log1p(poi_corrected) / poi_q75
feat['poi_facility_gap_corrected'] = 0.6 * bg + 0.4 * poi_signal
print(f"  poi_facility_gap_corrected: mean={feat['poi_facility_gap_corrected'].mean():.4f}")

# building_area_gap
rural_for_area = (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1) if 'pct_urban' in feat.columns else pd.Series(0, index=feat.index)
feat['building_area_gap'] = 1.3 * bg + 0.2 * bg * rural_for_area
print(f"  building_area_gap: mean={feat['building_area_gap'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGED PROXY TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Computing merged proxy target...")

road_gap = feat['road_gap'].fillna(0) if 'road_gap' in feat.columns else pd.Series(0, index=feat.index)
building_area_gap = feat['building_area_gap'].fillna(0)
poi_gap_corr = feat['poi_facility_gap_corrected'].fillna(0)
pct_urban = feat['pct_urban'].fillna(0.5) if 'pct_urban' in feat.columns else pd.Series(0.5, index=feat.index)

# THE HONEST MERGED PROXY: No SVI, rural signal, clipped gaps
proxy_merged = -np.mean([
    np.maximum(0, bg),
    2.0 * np.maximum(0, building_area_gap),
    np.maximum(0, road_gap),
    np.maximum(0, poi_gap_corr)
], axis=0) - 1.0 * (1 - pct_urban).clip(0, 1)

feat['proxy_merged'] = proxy_merged

# Comparison proxies
svi = feat['svi_overall'].fillna(0.5) if 'svi_overall' in feat.columns else pd.Series(0.5, index=feat.index)
proxy_v1 = -np.mean([bg, road_gap, poi_gap_corr], axis=0) - 2.0 * svi

print(f"  proxy_merged: mean={proxy_merged.mean():.4f}, std={proxy_merged.std():.4f}, "
      f"range=[{proxy_merged.min():.4f}, {proxy_merged.max():.4f}]")
print(f"  proxy_v1:     mean={proxy_v1.mean():.4f}, std={proxy_v1.std():.4f}, "
      f"range=[{proxy_v1.min():.4f}, {proxy_v1.max():.4f}]")

del proxy_v1, svi, poi_total, poi_corrected, poi_signal; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (memory-efficient)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Feature engineering...")

nf = {}
F = lambda s, v=0: s.fillna(v) if s is not None else pd.Series(v, index=feat.index)

rg = feat.get('road_gap'); bag = feat.get('building_area_gap')
pfg = feat.get('poi_facility_gap_corrected')
pu = feat.get('pct_urban'); svi_col = feat.get('svi_overall')
tribal = feat.get('tribal_any'); tribal_pct = feat.get('tribal_pct')
cvi = feat.get('cvi_overall'); pop = feat.get('pop_total')
wf = feat.get('usgs_wildfire_ever'); wfa = feat.get('usgs_wildfire_burned_pct_area')

bv = F(bg).values
nf['bldg_gap_sq'] = bv**2; nf['bldg_gap_abs'] = np.abs(bv)
nf['bldg_gap_clip'] = np.maximum(0, bv)

if bag is not None:
    bav = F(bag).values
    nf['area_gap_abs'] = np.abs(bav); nf['area_gap_clip'] = np.maximum(0, bav)
    nf['bldg_x_area_gap'] = bv * bav; nf['bldg_minus_area_gap'] = bv - bav

if rg is not None:
    rv = F(rg).values
    nf['road_gap_abs'] = np.abs(rv); nf['road_gap_clip'] = np.maximum(0, rv)
    nf['bldg_road_diff'] = bv - rv; nf['bldg_road_product'] = bv * rv

if pfg is not None:
    pv = F(pfg).values; nf['poi_gap_clip'] = np.maximum(0, pv)

if pu is not None:
    puv = F(pu, 0.5).values; rur = (1 - puv).clip(0, 1)
    nf['rural_x_bldg'] = rur * bv; nf['rural_x_bldg_clip'] = rur * np.maximum(0, bv)
    nf['pct_urban_x_bldg'] = puv * bv
    if rg is not None: nf['rural_x_road'] = rur * F(rg).values
    if bag is not None: nf['rural_x_area_gap'] = rur * F(bag).values
    nf['rural_indicator'] = (puv < 0.5).astype(float)
    nf['rural_continuous'] = rur
else:
    rur = np.zeros(len(feat))

if tribal is not None:
    tf = (F(tribal).values > 0).astype(float)
    nf['tribal_x_bldg'] = tf * bv
    if pu is not None: nf['tribal_x_rural'] = tf * rur

if svi_col is not None:
    sv = F(svi_col).values
    nf['svi_x_bldg'] = sv * bv
    if pu is not None: nf['rural_x_svi_x_bldg'] = rur * sv * bv

if cvi is not None: nf['cvi_x_bldg'] = F(cvi).values * bv
if wf is not None: nf['wf_x_bldg'] = F(wf).values * bv

comp = np.abs(bv)
if rg is not None: comp += np.abs(F(rg).values)
nf['compound_risk'] = comp
if pu is not None: nf['rural_x_risk'] = rur * comp

if pop is not None:
    nf['log_pop'] = np.log1p(F(pop).values)

# County LOO
county = feat['GEOID'].astype(str).str[:5]; bgv = F(bg)
cs = bgv.groupby(county).agg(['mean', 'count']); cs.columns = ['mean', 'count']
gm = bgv.mean(); sm = 10
cms = (cs['mean'] * cs['count'] + gm * sm) / (cs['count'] + sm)
nf['bldg_county_loo_smooth'] = cms[county].values

# Source interactions
for sc in ['bldg_total_sources', 'bldg_source_diversity', 'source_coverage_fraction', 'source_diversity_entropy']:
    if sc in feat.columns:
        nf[f'{sc}_x_bldg'] = F(feat[sc], 0).values * bv

if nf:
    nd = pd.DataFrame(nf, index=feat.index)
    nd = nd.replace([np.inf, -np.inf], np.nan)
    feat = pd.concat([feat, nd], axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    print(f"  +{len(nf)} engineered features -> {feat.shape[1]} total")

del nf, nd; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# CIRCULARITY TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Circularity test...")

svi_vals = feat.get('svi_overall', pd.Series(0.5, index=feat.index)).fillna(0.5).values
bg_vals = feat.get('building_gap', pd.Series(0, index=feat.index)).fillna(0).values
rg_vals = feat.get('road_gap', pd.Series(0, index=feat.index)).fillna(0).values
rural_vals = (1 - feat.get('pct_urban', pd.Series(0.5, index=feat.index)).fillna(0.5)).clip(0, 1).values

lr = LinearRegression()
lr.fit(svi_vals.reshape(-1, 1), bg_vals); svi_r2 = r2_score(bg_vals, lr.predict(svi_vals.reshape(-1, 1)))
lr.fit(rural_vals.reshape(-1, 1), bg_vals); rural_r2 = r2_score(bg_vals, lr.predict(rural_vals.reshape(-1, 1)))

print(f"  SVI → building_gap: R²={svi_r2:.4f}")
print(f"  Rural → building_gap: R²={rural_r2:.4f}")
print(f"  Verdict: SVI is {'RED HERRING' if svi_r2 < 0.05 else 'useful'}, "
      f"Rural is {'GENUINE SIGNAL' if rural_r2 > 0.1 else 'weak'}")

# ══════════════════════════════════════════════════════════════════════════════
# CASE VALIDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Case validations...")

geo_str = feat['GEOID'].values
tribal_vals = feat.get('tribal_any', pd.Series(0, index=feat.index)).fillna(0).values

for name, mask in [
    ('Hidalgo TX (border)', np.array([g.startswith('48215') for g in geo_str])),
    ('Maricopa AZ (urban)', np.array([g.startswith('04013') for g in geo_str])),
    ('Rural tracts', rural_vals > 0.5),
    ('Urban tracts', rural_vals <= 0.5),
    ('OK tribal', np.array([g.startswith('40') for g in geo_str]) & (tribal_vals > 0)),
    ('OK non-tribal', np.array([g.startswith('40') for g in geo_str]) & (tribal_vals == 0)),
]:
    if mask.sum() > 0:
        vals = proxy_merged.values[mask]
        print(f"  {name}: n={mask.sum()}, proxy_mean={vals.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Saving...")

feat.to_parquet(OUT / "engineered_features_merged.parquet", index=False)
print(f"  Saved: {OUT / 'engineered_features_merged.parquet'} ({feat.shape})")

results = {
    'phase': 1,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'proxy_formula': 'proxy = -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg)) - 1.0*(1-pct_urban)',
    'n_tracts': int(feat.shape[0]),
    'n_features': int(feat.shape[1]),
    'proxy_merged_stats': {'mean': float(proxy_merged.mean()), 'std': float(proxy_merged.std()),
                           'min': float(proxy_merged.min()), 'max': float(proxy_merged.max())},
    'circularity': {'svi_to_bldg_r2': float(svi_r2), 'rural_to_bldg_r2': float(rural_r2)},
}
with open(OUT / 'phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

el = time.time() - t0
print(f"\nPhase 1 DONE in {el:.0f}s")
