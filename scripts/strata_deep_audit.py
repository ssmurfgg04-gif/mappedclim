#!/usr/bin/env python3
"""
STRATA FEATURE DEEP AUDIT — Find which strata-origin features exist in 
engineered_features but are being DROPPED by the top-N correlation filter,
and identify the strongest ones to add back.
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
print("STRATA FEATURE DEEP AUDIT — What's in but unused?")
print("=" * 72)
t0 = time.time()

# ── 1. Load ──
print("\n[1] Loading data...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)
strata = pd.read_parquet(KAG / "national-strata-tract-table.parquet")
strata['GEOID'] = strata['GEOID'].astype(str)
print(f"  Engineered: {feat.shape}")
print(f"  Strata: {strata.shape}")

# ── 2. Identify strata-origin columns in engineered features ──
print("\n[2] Identifying strata-origin columns in engineered features...")
strata_col_set = set(strata.columns)
strata_in_eng = [c for c in feat.columns if c in strata_col_set]
print(f"  Strata columns in engineered features: {len(strata_in_eng)}")

# ── 3. Simulate Phase 2 feature selection pipeline ──
print("\n[3] Simulating Phase 2 feature selection (top 60 by |corr|)...")

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

y = feat['gap_only'].copy()
valid = y.notna()

fc = [c for c in feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feat[c])]
X_all = feat[fc].copy()
X_all = X_all[valid]
y_valid = y[valid]
X_all = X_all.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance
s = X_all.std()
X_all = X_all[s[s > 1e-10].index]

# Compute ALL correlations
all_corrs = X_all.corrwith(y_valid).abs().fillna(0).sort_values(ascending=False)

# ── 4. What makes it into top 60 (Phase 2 baseline)? ──
top60 = set(all_corrs.head(60).index)
# And top 80 (integrated 10x)?
top80 = set(all_corrs.head(80).index)

# Which strata-origin features are IN vs OUT?
strata_in_top60 = [c for c in strata_in_eng if c in top60]
strata_in_top80 = [c for c in strata_in_eng if c in top80]
strata_out_top60 = [c for c in strata_in_eng if c not in top60]
strata_out_top80 = [c for c in strata_in_eng if c not in top80]

print(f"\n  Total numeric features available: {len(all_corrs)}")
print(f"  Strata-origin in top 60: {len(strata_in_top60)}")
print(f"  Strata-origin in top 80 (but not top 60): {len(strata_in_top80) - len(strata_in_top60)}")
print(f"  Strata-origin OUT of top 80: {len(strata_out_top80)}")

# ── 5. Show ALL strata-origin features ranked by |corr| ──
print("\n[5] ALL strata-origin features ranked by |corr| with gap_only:")
strata_corrs = all_corrs[all_corrs.index.isin(strata_in_eng)].sort_values(ascending=False)

print(f"\n  {'Rank':<5} {'Feature':<50} {'|Corr|':>8} {'In Top60':>10} {'In Top80':>10}")
print(f"  {'─'*5} {'─'*50} {'─'*8} {'─'*10} {'─'*10}")
for i, (col, r) in enumerate(strata_corrs.items(), 1):
    in60 = '✓' if col in top60 else '✗'
    in80 = '✓' if col in top80 else '✗'
    marker = ' ← UNUSED' if col not in top80 else ''
    print(f"  {i:<5} {col:<50} {r:>8.4f} {in60:>10} {in80:>10}{marker}")

# ── 6. Category analysis of UNUSED strata features ──
print("\n[6] Category analysis of UNUSED strata features (|corr| > 0.05)...")

categories = {
    'fire': ['fod', 'mtbs', 'usfs', 'nifc', 'usgs_wildfire', 'wildfire', 'burn', 'flame', 'whp', 'bp_mean'],
    'climate_carbon': ['carbonplan', 'cvi_', 'pmdi', 'spi', 'crps', 'rps'],
    'weather': ['ghcn', 'noaa', 'temp_station', 'precip', 'prcp'],
    'heat': ['epht', 'hwd', 'gehe', 'uhe', 'heat', 'wbgt', 'cdw'],
    'rural_urban': ['rucc', 'ruca', 'nchs', 'ur_class'],
    'drought': ['usdm', 'drought', 'spi_', 'pmdi'],
    'geology': ['usgs_', 'geology', 'soil', 'elevation', 'mineral'],
    'tribal': ['tribal', 'aiannh'],
    'population': ['pop_', 'housing', 'density'],
    'svi': ['svi'],
    'infrastructure': ['rail', 'road', 'bridge', 'transit'],
}

unused_strata = [(col, r) for col, r in strata_corrs.items() if col not in top80 and r > 0.05]
cat_counts = {}
for col, r in unused_strata:
    cat_name = 'other'
    for cat, keywords in categories.items():
        if any(kw in col.lower() for kw in keywords):
            cat_name = cat
            break
    if cat_name not in cat_counts:
        cat_counts[cat_name] = {'count': 0, 'features': [], 'avg_abs_corr': []}
    cat_counts[cat_name]['count'] += 1
    cat_counts[cat_name]['features'].append(col)
    cat_counts[cat_name]['avg_abs_corr'].append(r)

for cat, info in sorted(cat_counts.items(), key=lambda x: -x[1]['count']):
    avg_r = np.mean(info['avg_abs_corr'])
    print(f"  {cat:>20}: {info['count']:>3} features, avg |r| = {avg_r:.4f}")
    for f in info['features'][:3]:  # show top 3
        corr_val = strata_corrs[f]
        print(f"    • {f:<45} |r| = {corr_val:.4f}")
    if len(info['features']) > 3:
        print(f"    ... and {len(info['features'])-3} more")

# ── 7. Top 20 NEW features to add (not in top80, strongest signal) ──
print("\n[7] Top 20 new features to add (not in current top 80, after collinearity filter)...")

# Get the candidate features (not in top 80, |corr| > 0.05)
candidates = [(col, r) for col, r in strata_corrs.items() if col not in top80 and r > 0.05]

# Collinearity check among candidates
selected = []
for col, r in candidates:
    if len(selected) >= 20:
        break
    # Check correlation with already-selected features
    vals = X_all[col].values
    too_similar = False
    for sel_col in selected:
        sel_vals = X_all[sel_col].values
        with np.errstate(all='ignore'):
            c = np.corrcoef(vals, sel_vals)[0, 1]
        if abs(c) > 0.95:
            too_similar = True
            break
    if not too_similar:
        selected.append(col)

print(f"\n  Selected {len(selected)} features:")
for i, col in enumerate(selected, 1):
    cat_name = 'other'
    for cat, keywords in categories.items():
        if any(kw in col.lower() for kw in keywords):
            cat_name = cat
            break
    r = strata_corrs[col]
    print(f"    {i:>2}. {col:<50} |r| = {r:.4f}  [{cat_name}]")

# ── 8. Also check NON-strata features that are unused ──
print("\n[8] Non-strata features with strong signal but unused (top 80 cutoff)...")

# Get engineered-only features (not from strata, not from drop_cols)
eng_only = [c for c in all_corrs.index if c not in strata_col_set and c not in top80]
eng_only_strong = [(c, all_corrs[c]) for c in eng_only if all_corrs[c] > 0.05]
eng_only_strong.sort(key=lambda x: -x[1])

if eng_only_strong:
    print(f"\n  {len(eng_only_strong)} non-strata features with |corr| > 0.05 outside top 80:")
    for i, (col, r) in enumerate(eng_only_strong[:15], 1):
        print(f"    {i:>2}. {col:<50} |r| = {r:.4f}")

# ── 9. Summary: The full picture ──
print("\n[9] SUMMARY — The full feature landscape:")
print(f"  Total numeric features available:    {len(all_corrs)}")
print(f"  Currently used (top 60, Phase 2):    {len(top60)}")
print(f"  Currently used (top 80, 10x):        {len(top80)}")
print(f"  Strata-origin total:                 {len(strata_in_eng)}")
print(f"  Strata-origin in top 60:             {len(strata_in_top60)}")
print(f"  Strata-origin in top 80:             {len(strata_in_top80)}")
print(f"  Strong unused strata (|r|>0.05):     {len(unused_strata)}")
print(f"  Very strong unused strata (|r|>0.1): {len([x for x in unused_strata if x[1] > 0.1])}")
print(f"  Top 20 candidates selected:           {len(selected)}")

# Potential R² gain estimate (rough: each feature ~ |r|² × variance_explained_fraction)
total_current_signal = sum(all_corrs.head(60).values ** 2)
total_with_20 = total_current_signal + sum(strata_corrs[col] ** 2 for col in selected)
pct_gain = (total_with_20 / total_current_signal - 1) * 100
print(f"\n  Rough signal estimate:")
print(f"    Current (top 60):  Σ|r|² = {total_current_signal:.4f}")
print(f"    + 20 new features: Σ|r|² = {total_with_20:.4f}")
print(f"    Potential gain:    +{pct_gain:.1f}% (upper bound, actual R² gain will be smaller)")

# ── 10. Save results ──
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_numeric_features': int(len(all_corrs)),
    'strata_origin_in_engineered': int(len(strata_in_eng)),
    'strata_in_top60': int(len(strata_in_top60)),
    'strata_in_top80': int(len(strata_in_top80)),
    'strong_unused_strata': int(len(unused_strata)),
    'very_strong_unused': int(len([x for x in unused_strata if x[1] > 0.1])),
    'top20_candidates': selected,
    'category_counts': {cat: info['count'] for cat, info in cat_counts.items()},
    'rough_signal_gain_pct': round(pct_gain, 1),
}

with open(OUT / 'strata_deep_audit.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save the full correlation table
corr_df = pd.DataFrame([
    {'feature': col, 'abs_corr': float(r), 'in_top60': col in top60, 'in_top80': col in top80,
     'is_strata': col in strata_col_set, 'selected_for_augmentation': col in selected}
    for col, r in all_corrs.items()
])
corr_df.to_csv(OUT / 'feature_correlation_full.csv', index=False)
corr_df.to_csv(DL / 'feature_correlation_full.csv', index=False)

print(f"\n  Results saved to {OUT / 'strata_deep_audit.json'}")
print(f"  Full correlation table: {DL / 'feature_correlation_full.csv'}")

el = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {el:.0f}s")
print(f"{'=' * 72}")
