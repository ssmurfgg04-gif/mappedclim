#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC AUDIT — Deep review of Maximizing luck pipeline
Checks: data leaks, feature sanity, gap_only consistency, model behavior, edge cases
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings, hashlib
from pathlib import Path
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"

print("=" * 72)
print("COMPREHENSIVE DIAGNOSTIC AUDIT")
print("=" * 72)
t0 = time.time()

diagnostics = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. SUBMISSION INTEGRITY — Verify rural penalty applied exactly once
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Submission integrity...")

sub = pd.read_csv(OUT / "submission_merged.csv")
sub['GEOID'] = sub['GEOID'].astype(str)

# Compute checksum
sub_hash = hashlib.md5(pd.util.hash_pandas_object(sub, index=True).values.tobytes()).hexdigest()
print(f"  Submission checksum: {sub_hash}")
diagnostics['submission_checksum'] = sub_hash

# Check that submission != proxy_merged (would mean old bug still present)
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet",
                       columns=['GEOID', 'gap_only', 'rural_penalty', 'proxy_merged', 'pct_urban'])
feat['GEOID'] = feat['GEOID'].astype(str)

merged = sub.merge(feat, on='GEOID', how='inner')

# If the old bug were still present, coverage_gap_score would equal proxy_merged
corr_with_proxy = np.corrcoef(merged['coverage_gap_score'], merged['proxy_merged'])[0, 1]
corr_with_gap_only = np.corrcoef(merged['coverage_gap_score'], merged['gap_only'])[0, 1]

# The CORRECT submission should be: model.predict(X) - 1.0*rural
# Since model predicts gap_only well (R²=0.978), coverage_gap_score ≈ gap_only - rural
# = proxy_merged. So correlation with proxy_merged is expected to be high.
# But we need to verify the inference formula is actually in the pipeline state.

with open(OUT / 'pipeline_state_merged.json') as f:
    state = json.load(f)

inference_formula = state.get('inference_formula', 'MISSING')
target = state.get('target', 'MISSING')
pipeline_name = state.get('pipeline', 'MISSING')

print(f"  Pipeline: {pipeline_name}")
print(f"  Target: {target}")
print(f"  Inference: {inference_formula}")
print(f"  Corr(submission, proxy_merged): {corr_with_proxy:.6f}")
print(f"  Corr(submission, gap_only): {corr_with_gap_only:.6f}")

diagnostics['deterministic_fix'] = {
    'pipeline': pipeline_name,
    'target': target,
    'inference_formula': inference_formula,
    'corr_submission_proxy_merged': float(corr_with_proxy),
    'corr_submission_gap_only': float(corr_with_gap_only),
    'target_is_gap_only': target == 'gap_only',
    'inference_has_rural_penalty': 'rural_penalty' in inference_formula,
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. GAP_ONLY CONSISTENCY — Phase 1 vs Phase 2
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] gap_only consistency check...")

feat_full = pd.read_parquet(OUT / "engineered_features_merged.parquet")

# Recompute gap_only from raw columns
bg = feat_full['building_gap'].fillna(0).values
road_gap = feat_full['road_gap'].fillna(0).values if 'road_gap' in feat_full.columns else np.zeros(len(feat_full))
building_area_gap = feat_full['building_area_gap'].fillna(0).values
poi_gap_corr = feat_full['poi_facility_gap_corrected'].fillna(0).values

gap_only_recomputed = -np.mean([
    np.maximum(0, bg),
    2.0 * np.maximum(0, building_area_gap),
    np.maximum(0, road_gap),
    np.maximum(0, poi_gap_corr)
], axis=0)

gap_only_stored = feat_full['gap_only'].values

max_diff = np.max(np.abs(gap_only_recomputed - gap_only_stored))
mean_diff = np.mean(np.abs(gap_only_recomputed - gap_only_stored))
allclose = np.allclose(gap_only_recomputed, gap_only_stored, atol=1e-10)

print(f"  Max |recomputed - stored|: {max_diff:.2e}")
print(f"  Mean |recomputed - stored|: {mean_diff:.2e}")
print(f"  All close (atol=1e-10): {allclose}")

diagnostics['gap_only_consistency'] = {
    'max_diff': float(max_diff),
    'mean_diff': float(mean_diff),
    'allclose': bool(allclose),
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. RURAL PENALTY — Verify exactly one application
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Rural penalty verification...")

pct_urban = feat_full['pct_urban'].fillna(0.5).values
rural_penalty_stored = feat_full['rural_penalty'].values
rural_penalty_recomputed = (1 - pct_urban).clip(0, 1)

max_rural_diff = np.max(np.abs(rural_penalty_stored - rural_penalty_recomputed))
print(f"  Max |rural_penalty_stored - recomputed|: {max_rural_diff:.2e}")

# Check that proxy_merged = gap_only - 1.0 * rural_penalty
proxy_merged_stored = feat_full['proxy_merged'].values
proxy_merged_recomputed = gap_only_stored - 1.0 * rural_penalty_stored
max_proxy_diff = np.max(np.abs(proxy_merged_stored - proxy_merged_recomputed))
print(f"  Max |proxy_merged_stored - (gap_only - rural)|: {max_proxy_diff:.2e}")

diagnostics['rural_penalty'] = {
    'max_rural_diff': float(max_rural_diff),
    'max_proxy_diff': float(max_proxy_diff),
    'rural_penalty_applied_once': max_proxy_diff < 1e-10,
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. OOF PREDICTIONS — Check for leakage and overfitting
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] OOF predictions analysis...")

oof = pd.read_parquet(OUT / "oof_predictions_merged.parquet")
model_cols = [c for c in oof.columns if c in ['xgb', 'lgb', 'et', 'cat', 'lgb_dart']]

# Check: are OOF predictions all present?
oof_complete = all(oof[c].notna().mean() > 0.99 for c in model_cols)
print(f"  OOF completeness: {oof_complete}")
for c in model_cols:
    na_rate = oof[c].isna().mean()
    print(f"    {c}: {na_rate*100:.2f}% NaN")

# Check: OOF R² against gap_only target
y = oof['gap_only'].values
oof_r2 = {}
for c in model_cols:
    valid = oof[c].notna()
    r2 = r2_score(y[valid], oof[c].values[valid])
    oof_r2[c] = float(r2)
    print(f"    {c} OOF R² (vs gap_only): {r2:.4f}")

# Check: prediction distribution matches target distribution?
y_mean, y_std = np.nanmean(y), np.nanstd(y)
for c in model_cols:
    pred_mean = oof[c].mean()
    pred_std = oof[c].std()
    print(f"    {c}: target mean={y_mean:.6f} std={y_std:.6f} | pred mean={pred_mean:.6f} std={pred_std:.6f}")

diagnostics['oof_analysis'] = {
    'completeness': bool(oof_complete),
    'r2_vs_gap_only': oof_r2,
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE SANITY — Ranges, NaN rates, impossible values
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Feature sanity checks...")

feat_sample = feat_full.sample(min(10000, len(feat_full)), random_state=42)

issues = []
for col in feat_sample.columns:
    if not pd.api.types.is_numeric_dtype(feat_sample[col]):
        continue
    vals = feat_sample[col].dropna()
    if len(vals) == 0:
        continue
    
    na_rate = feat_sample[col].isna().mean()
    inf_rate = np.isinf(feat_sample[col].fillna(0)).mean()
    
    # Check for impossible values
    if 'gap' in col.lower() and 'ratio' not in col.lower():
        # Gaps should be negative (shortfall) or zero
        pos_rate = (vals > 0.01).mean()
        if pos_rate > 0.5:
            issues.append(f"{col}: {pos_rate*100:.1f}% positive gap values (expected mostly ≤0)")
    
    if 'pct' in col.lower() or 'fraction' in col.lower():
        # Percentages should be 0-1
        out_range = ((vals < -0.01) | (vals > 1.01)).mean()
        if out_range > 0.01:
            issues.append(f"{col}: {out_range*100:.1f}% outside [0,1] range")
    
    if na_rate > 0.5:
        issues.append(f"{col}: {na_rate*100:.1f}% NaN")
    
    if inf_rate > 0:
        issues.append(f"{col}: {inf_rate*100:.1f}% inf")

print(f"  Found {len(issues)} feature issues:")
for issue in issues[:20]:
    print(f"    ⚠️ {issue}")
if len(issues) > 20:
    print(f"    ... and {len(issues)-20} more")

diagnostics['feature_issues'] = issues[:50]

# ══════════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES — Tracts with 0 buildings, 0 roads, etc.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Edge case analysis...")

edge_cases = {}

# Tracts with 0 building gap
zero_bldg = (feat_full['building_gap'].fillna(0).abs() < 1e-10).sum()
edge_cases['zero_building_gap'] = int(zero_bldg)
print(f"  Zero building_gap: {zero_bldg} tracts")

# Tracts with 0 road gap
if 'road_gap' in feat_full.columns:
    zero_road = (feat_full['road_gap'].fillna(0).abs() < 1e-10).sum()
    edge_cases['zero_road_gap'] = int(zero_road)
    print(f"  Zero road_gap: {zero_road} tracts")

# Tracts with zero gap_only (perfect coverage)
zero_gap = (feat_full['gap_only'].abs() < 1e-10).sum()
edge_cases['zero_gap_only'] = int(zero_gap)
print(f"  Zero gap_only: {zero_gap} tracts")

# Tribal tracts with 0 POI
if 'poi_cnt' in feat_full.columns:
    tribal_zero_poi = ((feat_full['tribal_any'].fillna(0) > 0) & (feat_full['poi_cnt'].fillna(0) == 0)).sum()
    edge_cases['tribal_zero_poi'] = int(tribal_zero_poi)
    print(f"  Tribal tracts with 0 POI: {tribal_zero_poi}")

# Tracts with NaN for all major gap columns
major_cols = ['building_gap', 'road_gap', 'poi_facility_gap_corrected']
all_nan = feat_full[major_cols].isna().all(axis=1).sum()
edge_cases['all_major_gaps_nan'] = int(all_nan)
print(f"  All major gaps NaN: {all_nan} tracts")

# Sample 100 random tracts for manual inspection
sample_tracts = feat_full[['GEOID', 'building_gap', 'road_gap', 'gap_only', 'rural_penalty', 'proxy_merged']].sample(100, random_state=42)
edge_cases['sample_100_geoids'] = sample_tracts['GEOID'].tolist()[:10]  # first 10

diagnostics['edge_cases'] = edge_cases

# ══════════════════════════════════════════════════════════════════════════════
# 7. STRATA COLUMN CORRELATIONS — Find unused predictive features
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Unused strata column correlations with gap_only...")

strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
strata['GEOID'] = strata['GEOID'].astype(str)

# Get gap_only per tract
gap_only_df = feat_full[['GEOID', 'gap_only']].copy()
gap_only_df['GEOID'] = gap_only_df['GEOID'].astype(str)

merged_strata = strata.merge(gap_only_df, on='GEOID', how='inner')
print(f"  Merged: {len(merged_strata)} tracts")

# Compute correlation with gap_only for all numeric strata columns
numeric_strata = [c for c in strata.columns if pd.api.types.is_numeric_dtype(strata[c]) and c != 'GEOID']
correlations = {}
for col in numeric_strata:
    if col in merged_strata.columns and merged_strata[col].notna().sum() > 1000:
        valid = merged_strata[[col, 'gap_only']].dropna()
        if len(valid) > 1000:
            corr = valid[col].corr(valid['gap_only'])
            correlations[col] = float(corr)

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Top 30 strata columns by |corr| with gap_only:")
for rank, (col, corr) in enumerate(sorted_corr[:30], 1):
    marker = " ← STRONG" if abs(corr) > 0.15 else ""
    print(f"    {rank:2d}. {col:45s} r={corr:+.4f}{marker}")

# Count how many have |r| > 0.1
strong_features = [(c, r) for c, r in sorted_corr if abs(r) > 0.1]
print(f"\n  Features with |r| > 0.1: {len(strong_features)}")
print(f"  Features with |r| > 0.15: {len([(c,r) for c,r in sorted_corr if abs(r) > 0.15])}")

# Identify which categories the strong features belong to
strong_names = set(c for c, r in strong_features)
categories_found = {
    'climate/carbon': len([c for c in strong_names if any(k in c.lower() for k in ['carbon', 'cvi', 'climate', 'pmdi', 'spi'])]),
    'heat/nasa': len([c for c in strong_names if any(k in c.lower() for k in ['heat', 'hwd', 'epht', 'gehe', 'uhe', 'wbgt'])]),
    'fire': len([c for c in strong_names if any(k in c.lower() for k in ['fire', 'whp', 'burn', 'fod', 'mtbs', 'nifc'])]),
    'drought': len([c for c in strong_names if any(k in c.lower() for k in ['drought', 'usdm', 'pmdi', 'spi'])]),
    'weather': len([c for c in strong_names if any(k in c.lower() for k in ['ghcn', 'noaa', 'weather'])]),
    'svi/vulnerability': len([c for c in strong_names if any(k in c.lower() for k in ['svi', 'vulnerab'])]),
    'rural/urban': len([c for c in strong_names if any(k in c.lower() for k in ['rural', 'urban', 'ruca', 'rucc', 'ur_class'])]),
    'tribal': len([c for c in strong_names if any(k in c.lower() for k in ['tribal', 'aiannh'])]),
    'geology': len([c for c in strong_names if any(k in c.lower() for k in ['usgs', 'geolog', 'landslide'])]),
}
print(f"\n  Strong feature categories:")
for cat, count in sorted(categories_found.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"    {cat}: {count} features with |r|>0.1")

diagnostics['strata_correlations'] = {
    'total_numeric_cols': len(numeric_strata),
    'strong_features_gt_0.1': len(strong_features),
    'strong_features_gt_0.15': len([(c,r) for c,r in sorted_corr if abs(r) > 0.15]),
    'top_30': [(c, r) for c, r in sorted_corr[:30]],
    'categories': categories_found,
}

# ══════════════════════════════════════════════════════════════════════════════
# 8. H3 SPATIAL LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] H3 spatial leakage check...")

import h3

lats = feat_full['centroid_lat'].values if 'centroid_lat' in feat_full.columns else None
lons = feat_full['centroid_lon'].values if 'centroid_lon' in feat_full.columns else None
if lats is None or pd.isna(lats).all():
    lats = pd.to_numeric(feat_full['INTPTLAT'], errors='coerce').values
    lons = pd.to_numeric(feat_full['INTPTLON'], errors='coerce').values

# Compute H3 cells
cells = []
for la, lo in zip(lats, lons):
    if not (np.isnan(la) or np.isnan(lo)):
        cells.append(h3.latlng_to_cell(float(la), float(lo), 4))
    else:
        cells.append('unk')

cells = pd.Series(cells)

# Check: any tracts in multiple blocks? (shouldn't happen with point-in-polygon)
unique_cells = cells.nunique()
total_tracts = len(cells)
print(f"  Unique H3 cells: {unique_cells}")
print(f"  Total tracts: {total_tracts}")
print(f"  Tracts per cell (mean): {total_tracts/unique_cells:.1f}")

# Check: neighbor overlap between folds
# H3 resolution 4 has ~22km edge length, so neighbors should be in different folds
# But some blocks might be assigned to same fold
ub = list(cells.unique()); np.random.shuffle(ub)
fa = {b: i % 3 for i, b in enumerate(ub)}
fold_assignments = cells.map(fa).values

# Count tracts at fold boundaries
boundary_tracts = 0
for i in range(0, len(cells), 1000):  # sample every 1000th
    cell = cells.iloc[i]
    if cell == 'unk':
        continue
    neighbors = h3.grid_disk(cell, 1) if hasattr(h3, 'grid_disk') else h3.k_ring(cell, 1)
    my_fold = fa.get(cell, -1)
    neighbor_folds = [fa.get(n, -1) for n in neighbors if n in fa]
    if len(set(neighbor_folds)) > 1:
        boundary_tracts += 1

print(f"  Tracts at fold boundaries (sample): {boundary_tracts}/~{total_tracts//1000}")

diagnostics['h3_leakage'] = {
    'unique_cells': int(unique_cells),
    'tracts_per_cell_mean': float(total_tracts/unique_cells),
}

# ══════════════════════════════════════════════════════════════════════════════
# 9. NATIONAL FEATURES CHECK — What columns exist that we're not using?
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] National features — unused columns analysis...")

nf = pd.read_parquet(PROJ / "data/features/national_tract_features.parquet")
print(f"  National features: {nf.shape[1]} columns")

# Check which columns in national features are NOT in engineered features
eng_cols = set(feat_full.columns)
nf_only = [c for c in nf.columns if c not in eng_cols]
print(f"  Columns in national but not in engineered: {len(nf_only)}")

# Check for key columns we might be missing
missing_important = []
for keyword in ['ms_bldg', 'bldg_ms', 'overture', 'tiger', 'source', 'coverage']:
    matching = [c for c in nf.columns if keyword in c.lower()]
    if matching:
        used = [c for c in matching if c in eng_cols]
        unused = [c for c in matching if c not in eng_cols]
        if unused:
            missing_important.extend(unused)
            print(f"  {keyword}: {len(used)} used, {len(unused)} unused")
            for u in unused[:3]:
                print(f"    - {u}")

diagnostics['unused_national_features'] = len(nf_only)
diagnostics['missing_important_features'] = missing_important[:20]

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DIAGNOSTIC SUMMARY")
print("=" * 72)

bugs_found = []
warnings_found = []

if not diagnostics['deterministic_fix']['target_is_gap_only']:
    bugs_found.append("CRITICAL: Training target is NOT gap_only!")
if not diagnostics['deterministic_fix']['inference_has_rural_penalty']:
    bugs_found.append("CRITICAL: Inference formula missing rural_penalty!")
if not diagnostics['gap_only_consistency']['allclose']:
    bugs_found.append("BUG: gap_only not consistent between Phase 1 and Phase 2!")
if not diagnostics['rural_penalty']['rural_penalty_applied_once']:
    bugs_found.append("BUG: proxy_merged ≠ gap_only - rural_penalty (double counting?)")
if diagnostics['oof_analysis']['r2_vs_gap_only'].get('xgb', 0) > 0.99:
    warnings_found.append("WARNING: OOF R² > 0.99 — possible overfitting or circularity")
if len(diagnostics['feature_issues']) > 0:
    warnings_found.append(f"WARNING: {len(diagnostics['feature_issues'])} feature sanity issues found")
if diagnostics['strata_correlations']['strong_features_gt_0.1'] > 20:
    warnings_found.append(f"INFO: {diagnostics['strata_correlations']['strong_features_gt_0.1']} unused strata columns with |r|>0.1")

print(f"\n  BUGS FOUND: {len(bugs_found)}")
for b in bugs_found:
    print(f"    🐛 {b}")

print(f"\n  WARNINGS: {len(warnings_found)}")
for w in warnings_found:
    print(f"    ⚠️ {w}")

if not bugs_found:
    print("\n  ✅ No critical bugs found!")

# Save
diagnostics['bugs_found'] = bugs_found
diagnostics['warnings_found'] = warnings_found
diagnostics['elapsed_sec'] = round(time.time() - t0, 1)

with open(OUT / 'diagnostic_report.json', 'w') as f:
    json.dump(diagnostics, f, indent=2, default=str)

print(f"\nDiagnostic report saved to {OUT / 'diagnostic_report.json'}")
print(f"Done in {time.time()-t0:.0f}s")
