#!/usr/bin/env python3
"""
STRATA FEATURE DISCOVERY — Find unused strata columns with signal,
rank by correlation with gap_only, identify top candidates for augmentation.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time
from pathlib import Path

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = Path("/home/z/my-project/download"); DL.mkdir(parents=True, exist_ok=True)
KAG = PROJ / "kaggle_dataset"

print("=" * 72)
print("STRATA FEATURE DISCOVERY")
print("=" * 72)
t0 = time.time()

# ── 1. Load engineered features (has gap_only target) ──
print("\n[1] Loading engineered features...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
print(f"  Base: {feat.shape}")
print(f"  Columns: {len(feat.columns)}")

# ── 2. Load strata table ──
print("\n[2] Loading strata table...")
strata = pd.read_parquet(KAG / "national-strata-tract-table.parquet")
strata['GEOID'] = strata['GEOID'].astype(str)
print(f"  Strata: {strata.shape}")
print(f"  Columns: {len(strata.columns)}")

# ── 3. Identify which strata columns are already in engineered features ──
print("\n[3] Identifying overlap...")
base_cols = set(feat.columns)
strata_cols = set(strata.columns)
already_used = base_cols & strata_cols
not_in_base = strata_cols - base_cols
print(f"  Strata columns already in engineered features: {len(already_used)}")
print(f"  Strata columns NOT in engineered features: {len(not_in_base)}")

# ── 4. Compute correlations with gap_only for all strata columns ──
print("\n[4] Computing correlations with gap_only...")
gap_only = feat['gap_only'].values
geoid_feat = feat['GEOID'].values

# Merge strata columns not in base
new_strata = strata[['GEOID'] + sorted(not_in_base)]
merged = feat[['GEOID', 'gap_only']].merge(new_strata, on='GEOID', how='left')

correlations = {}
for col in sorted(not_in_base):
    if col in merged.columns and pd.api.types.is_numeric_dtype(merged[col]):
        vals = merged[col].values
        valid_mask = ~np.isnan(vals) & ~np.isnan(gap_only)
        if valid_mask.sum() > 100:  # need at least 100 valid observations
            r = np.corrcoef(vals[valid_mask], gap_only[valid_mask])[0, 1]
            if not np.isnan(r):
                na_rate = np.isnan(vals).mean()
                correlations[col] = {'corr': round(r, 4), 'abs_corr': round(abs(r), 4), 'na_rate': round(na_rate, 4), 'n_valid': int(valid_mask.sum())}

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['abs_corr'], reverse=True)

print(f"\n  {len(sorted_corr)} numeric strata columns with valid correlations")
print(f"\n  Top 30 by |correlation| with gap_only:")
print(f"  {'Rank':<5} {'Feature':<45} {'Corr':>8} {'|Corr|':>8} {'NA%':>8} {'N':>8}")
print(f"  {'─'*5} {'─'*45} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for i, (col, info) in enumerate(sorted_corr[:30], 1):
    print(f"  {i:<5} {col:<45} {info['corr']:>8.4f} {info['abs_corr']:>8.4f} {info['na_rate']:>7.1%} {info['n_valid']:>8}")

# ── 5. Categorize features ──
print("\n[5] Categorizing by domain...")
categories = {
    'fire': ['fod', 'mtbs', 'usfs', 'nifc', 'usgs_wildfire', 'wildfire'],
    'climate_carbon': ['carbonplan', 'cvi_', 'pmdi', 'spi', 'ghcn', 'usdm', 'drought'],
    'weather': ['ghcn', 'noaa', 'temp_station', 'precip'],
    'heat': ['epht', 'hwd', 'gehe', 'uhe', 'heat', 'wbgt'],
    'rural_urban': ['rucc', 'ruca', 'nchs', 'ur_class'],
    'geology': ['usgs_', 'geology', 'soil', 'elevation'],
    'svi_health': ['svi', 'cdc', 'nchs', 'wonder', 'health'],
    'tribal': ['tribal', 'aiannh'],
    'population': ['pop_', 'housing', 'density'],
}

categorized = {cat: [] for cat in categories}
categorized['other'] = []

for col, info in sorted_corr:
    if info['abs_corr'] < 0.05:
        continue  # skip very weak
    assigned = False
    for cat, keywords in categories.items():
        if any(kw in col.lower() for kw in keywords):
            categorized[cat].append((col, info))
            assigned = True
            break
    if not assigned:
        categorized['other'].append((col, info))

for cat, items in categorized.items():
    if items:
        avg_abs_corr = np.mean([x[1]['abs_corr'] for x in items])
        print(f"  {cat:>20}: {len(items):>3} features, avg |r| = {avg_abs_corr:.4f}")

# ── 6. Also check currently-used features for comparison ──
print("\n[6] Current feature correlations (for comparison)...")
# Identify which base columns are used in the model
drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
             'building_count_ratio', 'building_count_gap',
             'road_count_ratio', 'road_count_gap', 'road_length_ratio', 'road_length_gap',
             'poi_facility_gap', 'poi_to_facility_ratio',
             'poi_facility_gap_corrected', 'poi_to_facility_ratio_corrected',
             'building_area_gap',
             'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
             'proxy_merged', 'gap_only', 'rural_penalty']

fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X_base = feat[fc].copy()
valid = feat['gap_only'].notna()
X_base = X_base[valid]
y_base = feat.loc[valid, 'gap_only']
X_base = X_base.fillna(-999).replace([np.inf, -np.inf], -999)

base_corrs = X_base.corrwith(y_base).abs().fillna(0).sort_values(ascending=False)
print(f"\n  Current top 15 features by |corr|:")
for i, (col, r) in enumerate(base_corrs.head(15).items(), 1):
    print(f"    {i:>2}. {col:<45} |r| = {r:.4f}")

# ── 7. Select top 20 NEW features for augmentation ──
print("\n[7] Selecting top 20 new features for augmentation...")

# Filter: |corr| > 0.05, NA rate < 50%, at least 1000 valid observations
candidates = [(col, info) for col, info in sorted_corr
              if info['abs_corr'] > 0.05 and info['na_rate'] < 0.5 and info['n_valid'] >= 1000]

# Check for collinearity among candidates
top_candidates = []
selected = []
for col, info in candidates:
    if len(selected) >= 20:
        break
    # Quick check: is this column too correlated with already selected?
    vals = merged[col].fillna(-999).values
    too_similar = False
    for sel_col in selected:
        sel_vals = merged[sel_col].fillna(-999).values
        r = np.corrcoef(vals, sel_vals)[0, 1]
        if abs(r) > 0.95:  # too collinear
            too_similar = True
            break
    if not too_similar:
        selected.append(col)
        top_candidates.append((col, info))

print(f"\n  Top 20 new features (after collinearity filter):")
print(f"  {'Rank':<5} {'Feature':<45} {'Corr':>8} {'Category':>20}")
print(f"  {'─'*5} {'─'*45} {'─'*8} {'─'*20}")
for i, (col, info) in enumerate(top_candidates, 1):
    # Find category
    cat_name = 'other'
    for cat, keywords in categories.items():
        if any(kw in col.lower() for kw in keywords):
            cat_name = cat
            break
    print(f"  {i:<5} {col:<45} {info['corr']:>8.4f} {cat_name:>20}")

# ── 8. Save results ──
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_strata_columns': int(len(strata.columns)),
    'already_in_engineered': int(len(already_used)),
    'not_in_engineered': int(len(not_in_base)),
    'numeric_with_valid_corr': int(len(sorted_corr)),
    'candidates_after_filter': int(len(candidates)),
    'top_20_features': [(col, info) for col, info in top_candidates],
    'category_counts': {cat: len(items) for cat, items in categorized.items()},
    'strong_features_abs_corr_gt_01': int(len([x for x in sorted_corr if x[1]['abs_corr'] > 0.1])),
    'strong_features_abs_corr_gt_02': int(len([x for x in sorted_corr if x[1]['abs_corr'] > 0.2])),
}

with open(OUT / 'strata_feature_discovery.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Also save as CSV for easy viewing
disc_df = pd.DataFrame([
    {'rank': i, 'feature': col, 'corr': info['corr'], 'abs_corr': info['abs_corr'],
     'na_rate': info['na_rate'], 'n_valid': info['n_valid'], 'selected': col in selected}
    for i, (col, info) in enumerate(sorted_corr, 1)
])
disc_df.to_csv(OUT / 'strata_feature_discovery.csv', index=False)
disc_df.to_csv(DL / 'strata_feature_discovery.csv', index=False)

print(f"\n  Results saved to {OUT / 'strata_feature_discovery.json'}")
print(f"  CSV saved to {DL / 'strata_feature_discovery.csv'}")

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"Top 20 features ready for augmentation pipeline")
print(f"{'=' * 72}")
