#!/usr/bin/env python3
"""Run pipeline step by step - Phase 1: Data + Features"""
import numpy as np, pandas as pd, json, time, logging, warnings
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SEED = 42; np.random.seed(SEED)

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# Load data
log.info("Loading features...")
feat = pd.read_parquet(PROJ / "data/features/all_regions_enhanced_features.parquet")
log.info(f"Features: {feat.shape}")

log.info("Loading strata...")
strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
log.info(f"Strata: {strata.shape}")

# Find target
for tc in ['coverage_gap_score', 'coverage_gap', 'gap_score', 'mapping_gap_score']:
    if tc in feat.columns and feat[tc].notna().sum() > 0:
        log.info(f"TARGET FOUND: {tc}"); tcol = tc; break
else:
    log.warning("No actual target - using building_gap proxy"); tcol = 'building_gap'

# Merge strata
sw = ['GEOID', 'svi_overall', 'svi_socioeconomic', 'svi_household', 'svi_minority',
      'svi_housing_transport', 'svi_pop', 'tribal_any', 'tribal_pct', 'tribal_legal',
      'pct_urban', 'pop_rural', 'pop_urban', 'pop_total', 'usgs_wildfire_ever',
      'usgs_wildfire_burned_pct_area', 'cvi_overall', 'cvi_baseline', 'cvi_climate']
sw += [c for c in strata.columns if '_covered' in c.lower()]
sw += [c for c in strata.columns if any(h in c.lower() for h in ['wildfire','drought','usdm','heat','epht']) and '_covered' not in c.lower()][:15]
sw += [c for c in strata.columns if any(r in c.lower() for r in ['ruca','rucc','nchs'])][:5]
nc = [c for c in sw if c in strata.columns and c not in feat.columns]
nc = ['GEOID'] + [c for c in nc if c != 'GEOID']
if len(nc) > 1:
    ss = strata[nc].copy(); ss['GEOID'] = ss['GEOID'].astype(str); feat['GEOID'] = feat['GEOID'].astype(str)
    b = feat.shape[1]; feat = feat.merge(ss, on='GEOID', how='left')
    log.info(f"Strata merged: {b}->{feat.shape[1]}")

# Ensure numeric
for c in ['svi_overall','svi_socioeconomic','svi_household','svi_minority','svi_housing_transport',
          'svi_pop','tribal_pct','pct_urban','cvi_overall','cvi_baseline','cvi_climate',
          'usgs_wildfire_burned_pct_area','pop_total']:
    if c in feat.columns: feat[c] = pd.to_numeric(feat[c], errors='coerce')

# Feature engineering
log.info("Engineering features...")
nf = {}; F = lambda s, v=0: s.fillna(v) if s is not None else None
bg = feat.get('building_gap'); rg = feat.get('road_gap'); svi = feat.get('svi_overall')
svi_m = feat.get('svi_minority'); svi_s = feat.get('svi_socioeconomic'); svi_h = feat.get('svi_household')
svi_ht = feat.get('svi_housing_transport'); tribal = feat.get('tribal_any'); tribal_pct = feat.get('tribal_pct')
pu = feat.get('pct_urban'); wf = feat.get('usgs_wildfire_ever'); wfa = feat.get('usgs_wildfire_burned_pct_area')
cvi = feat.get('cvi_overall'); cvi_b = feat.get('cvi_baseline'); cvi_c = feat.get('cvi_climate')
pop = feat.get('pop_total'); br = feat.get('building_ratio'); rr = feat.get('road_ratio')

if bg is not None:
    bv = F(bg).values
    if svi is not None:
        sv = F(svi).values
        nf['svi_x_bldg'] = sv*bv; nf['svi_sq_x_bldg'] = sv**2*bv; nf['svi_abs_x_bldg_abs'] = np.abs(sv)*np.abs(bv)
        nf['svi_x_bldg_sq'] = sv*bv**2; nf['svi_cubed_x_bldg'] = sv**3*bv
    if svi_m is not None: nf['svi_min_x_bldg'] = F(svi_m).values*bv
    if svi_s is not None: nf['svi_soc_x_bldg'] = F(svi_s).values*bv
    if svi_h is not None: nf['svi_hh_x_bldg'] = F(svi_h).values*bv
    if svi_ht is not None: nf['svi_ht_x_bldg'] = F(svi_ht).values*bv
    if rg is not None:
        rv = F(rg).values
        if svi is not None: nf['svi_x_road'] = F(svi).values*rv; nf['svi_sq_x_road'] = F(svi).values**2*rv
        if svi_m is not None: nf['svi_min_x_road'] = F(svi_m).values*rv
    if tribal is not None:
        tf = (F(tribal).values>0).astype(float)
        nf['tribal_x_bldg'] = tf*bv; nf['tribal_pct_x_bldg'] = F(tribal_pct,0).values*bv
        if rg is not None: nf['tribal_x_road'] = tf*F(rg).values
        nf['tribal_x_bldg_sq'] = tf*bv**2
        if svi is not None: nf['tribal_x_svi_x_bldg'] = tf*F(svi).values*bv
        if cvi is not None: nf['tribal_x_cvi_x_bldg'] = tf*F(cvi).values*bv
    if pu is not None:
        puv = F(pu,0.5).values; rur = (1-puv).clip(0,1)
        nf['pct_urban_x_bldg'] = puv*bv; nf['rural_x_bldg'] = rur*bv; nf['rural_sq_x_bldg'] = rur**2*bv
        if rg is not None: nf['rural_x_road'] = rur*F(rg).values
        if svi is not None: nf['rural_x_svi_x_bldg'] = rur*F(svi).values*bv; nf['urban_x_svi_x_bldg'] = puv*F(svi).values*bv
    if wf is not None: nf['wf_x_bldg'] = F(wf).values*bv; nf['wf_flag_x_bldg'] = (F(wf).values>0).astype(float)*bv
    if wfa is not None: nf['wf_area_x_bldg'] = F(wfa).values*bv
    if wf is not None and svi is not None: nf['wf_x_svi_x_bldg'] = F(wf).values*F(svi).values*bv
    if cvi is not None:
        cv = F(cvi).values
        nf['cvi_x_bldg'] = cv*bv; nf['cvi_sq_x_bldg'] = cv**2*bv
        if svi is not None: nf['cvi_x_svi_x_bldg'] = cv*F(svi).values*bv
    if cvi_b is not None: nf['cvi_base_x_bldg'] = F(cvi_b).values*F(bg).values
    if cvi_c is not None: nf['cvi_clim_x_bldg'] = F(cvi_c).values*F(bg).values
    # Polynomial
    nf['bldg_gap_sq'] = bv**2; nf['bldg_gap_cu'] = bv**3; nf['bldg_gap_abs'] = np.abs(bv); nf['bldg_gap_log1p_abs'] = np.log1p(np.abs(bv))
    if rg is not None:
        rv = F(rg).values; nf['road_gap_sq'] = rv**2; nf['road_gap_abs'] = np.abs(rv)
        nf['bldg_road_ratio'] = bv/(np.abs(rv)+1e-8); nf['bldg_road_diff'] = bv-rv; nf['bldg_road_product'] = bv*rv
    if br is not None: nf['log_bldg_ratio'] = np.log1p(F(br).clip(lower=0).values); nf['bldg_ratio_sq'] = F(br).values**2
    if rr is not None: nf['log_road_ratio'] = np.log1p(F(rr).clip(lower=0).values)
    # Compound risk
    comp = np.abs(bv)
    if rg is not None: comp += np.abs(F(rg).values)
    if svi is not None: comp += np.clip(F(svi).values, 0, None)*0.1
    nf['compound_risk'] = comp; nf['compound_risk_sq'] = comp**2
    if tribal is not None: nf['tribal_x_risk'] = (F(tribal).values>0).astype(float)*comp
# Population
if pop is not None:
    lp = np.log1p(F(pop).values); nf['log_pop'] = lp
    if bg is not None: nf['log_pop_x_bldg'] = lp*F(bg).values
    if svi is not None: nf['log_pop_x_svi'] = lp*F(svi).values
# Coverage nulls
for cc in [c for c in feat.columns if '_covered' in c.lower()]: nf[f'{cc}_null'] = feat[cc].isna().astype(float).values
nulc = [k for k in nf if k.endswith('_null')]
if nulc: nf['total_nulls'] = np.sum([nf[k] for k in nulc], axis=0); nf['null_fraction'] = nf['total_nulls']/max(len(nulc),1)
# County LOO
if 'GEOID' in feat.columns and bg is not None:
    county = feat['GEOID'].astype(str).str[:5]; bgv = F(bg)
    cs = bgv.groupby(county).agg(['mean','count','std']); cs.columns = ['mean','count','std']
    gm = bgv.mean(); sm = 10
    cms = (cs['mean']*cs['count']+gm*sm)/(cs['count']+sm)
    nf['bldg_county_loo_smooth'] = cms[county].values; nf['bldg_county_count'] = cs['count'][county].values
    cm_ = cs['mean'][county].values; cc_ = cs['count'][county].values
    nf['bldg_county_loo'] = (cm_*cc_-bgv.values)/(cc_-1+1e-8)
# Region dummies
if 'region' in feat.columns:
    for r in feat['region'].unique(): nf[f'region_{r}'] = (feat['region']==r).astype(float).values
# Intersectional
if tribal is not None and svi is not None and pu is not None:
    t = (F(tribal).values>0).astype(float); sv = F(svi).values; puv = F(pu,0.5).values
    hs = (sv>np.nanquantile(sv,.75)).astype(float); ls = (sv<np.nanquantile(sv,.25)).astype(float)
    rur = (puv<.5).astype(float); urb = 1-rur
    nf['tribal_x_highsvi_x_rural'] = t*hs*rur; nf['tribal_x_lowsvi_x_rural'] = t*ls*rur
    nf['highsvi_x_rural'] = hs*rur; nf['tribal_x_rural'] = t*rur
    if bg is not None:
        bv = F(bg).values; nf['tribal_hsvi_rural_x_bldg'] = t*hs*rur*bv; nf['hsvi_rural_x_bldg'] = hs*rur*bv
    if wf is not None: nf['wf_x_rural_x_hsvi'] = F(wf).values*rur*hs; nf['wf_x_tribal'] = F(wf).values*t
    if cvi is not None:
        hc = (F(cvi).values>np.nanquantile(F(cvi).values,.75)).astype(float)
        nf['hcvi_x_hsvi_x_rural'] = hc*hs*rur; nf['hcvi_x_tribal'] = hc*t

if nf:
    nd = pd.DataFrame(nf, index=feat.index); nd = nd.replace([np.inf, -np.inf], np.nan)
    feat = pd.concat([feat, nd], axis=1); feat = feat.loc[:, ~feat.columns.duplicated()]
    log.info(f"+{len(nf)} features -> {feat.shape[1]} total")

# Save intermediate
feat.to_parquet(OUT / "engineered_features_v3.parquet", index=False)
log.info(f"Saved engineered features: {feat.shape}")
log.info(f"Phase 1 done in {time.time()-t0:.1f}s")
