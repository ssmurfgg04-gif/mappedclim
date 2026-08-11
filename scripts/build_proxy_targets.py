#!/usr/bin/env python3
"""
Build 4 proxy target formulations for Zindi ML competition and cross-validate.

Proxy 1 - Simple average
Proxy 2 - Vulnerability-weighted
Proxy 3 - Max gap (worst dimension drives score)
Proxy 4 - Population-weighted compound

Uses H3 spatial block 3-fold CV with XGBoost to evaluate each proxy.
Selects best proxy based on lowest std_rmse (most stable generalization).
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
import pandas as pd
import h3
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
FEATURE_PATH = "/home/z/my-project/bias-bounty-map/data/output/engineered_features_v3.parquet"
OUTPUT_COMPARISON = "/home/z/my-project/bias-bounty-map/data/output/proxy_target_comparison.json"

# ── Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("BUILDING PROXY TARGETS")
print("=" * 70)

print("\n[1] Loading engineered features...")
feat = pd.read_parquet(FEATURE_PATH)
print(f"    Shape: {feat.shape}")

# ── Check available columns ────────────────────────────────────────────────
print("\n[2] Checking available gap columns...")
has_road_length_gap = "road_length_gap" in feat.columns
has_poi_facility_gap = "poi_facility_gap" in feat.columns
has_svi_overall = "svi_overall" in feat.columns

print(f"    building_gap:       YES (non-null={feat['building_gap'].notna().sum()})")
print(f"    road_gap:           YES (non-null={feat['road_gap'].notna().sum()})")
print(f"    road_length_gap:    {'YES' if has_road_length_gap else 'NO (using road_gap as fallback)'}")
print(f"    poi_facility_gap:   {'YES' if has_poi_facility_gap else 'NO (using road_gap as fallback)'}")
print(f"    svi_overall:        {'YES' if has_svi_overall else 'NO (need to merge)'}")
print(f"    pop_total:          YES (non-null={feat['pop_total'].notna().sum()})")

# ── Prepare gap series ─────────────────────────────────────────────────────
building_gap = feat["building_gap"].fillna(0)
road_gap = feat["road_gap"].fillna(0)

# road_length_gap: use if exists, else fall back to road_gap
if has_road_length_gap:
    road_length_gap = feat["road_length_gap"].fillna(0)
else:
    road_length_gap = road_gap.copy()

# poi_facility_gap: use if exists, else fall back to road_gap
if has_poi_facility_gap:
    poi_facility_gap = feat["poi_facility_gap"].fillna(0)
else:
    poi_facility_gap = road_gap.copy()

# svi_overall
if has_svi_overall:
    svi_overall = feat["svi_overall"].copy()
else:
    # Load strata table and merge
    print("    Loading SVI from strata table...")
    strata_path = "/home/z/my-project/bias-bounty-map/kaggle_dataset/national-strata-tract-table.parquet"
    strata = pd.read_parquet(strata_path)
    svi_merge = strata[["GEOID", "svi_overall"]].set_index("GEOID")
    svi_overall = feat["GEOID"].map(svi_merge["svi_overall"])
    print(f"    SVI merged: non-null={svi_overall.notna().sum()}")

# pop_total
pop_total = feat["pop_total"].fillna(feat["pop_total"].median())

# ── Compute 4 proxy targets ────────────────────────────────────────────────
print("\n[3] Computing 4 proxy targets...")

# Proxy 1 - Simple average
proxy1 = 0.5 * building_gap + 0.3 * road_length_gap + 0.2 * poi_facility_gap
print(f"    Proxy 1 (Simple average):   mean={proxy1.mean():.4f}  std={proxy1.std():.4f}  range=[{proxy1.min():.4f}, {proxy1.max():.4f}]")

# Proxy 2 - Vulnerability-weighted
svi_weight = 1 + 2 * svi_overall.fillna(0.5)
proxy2 = svi_weight * (0.4 * building_gap + 0.4 * road_gap + 0.2 * poi_facility_gap) / svi_weight.mean()
print(f"    Proxy 2 (SVI-weighted):     mean={proxy2.mean():.4f}  std={proxy2.std():.4f}  range=[{proxy2.min():.4f}, {proxy2.max():.4f}]")

# Proxy 3 - Max gap (worst dimension drives score)
proxy3 = np.maximum(np.abs(building_gap), np.abs(road_gap))
if has_poi_facility_gap:
    proxy3 = np.maximum(proxy3, np.abs(poi_facility_gap))
# Preserve sign: make negative where both gaps are negative (underserved)
both_negative = (building_gap < 0) & (road_gap < 0)
proxy3 = proxy3 * np.where(both_negative, -1, 1)
print(f"    Proxy 3 (Max gap):          mean={proxy3.mean():.4f}  std={proxy3.std():.4f}  range=[{proxy3.min():.4f}, {proxy3.max():.4f}]")

# Proxy 4 - Population-weighted compound
pop = pop_total.fillna(pop_total.median())
proxy4 = (pop * building_gap + pop * road_gap) / (2 * pop)
proxy4 = proxy4.fillna(building_gap)  # fallback
print(f"    Proxy 4 (Pop-weighted):     mean={proxy4.mean():.4f}  std={proxy4.std():.4f}  range=[{proxy4.min():.4f}, {proxy4.max():.4f}]")

# ── Add proxy columns to feat ──────────────────────────────────────────────
feat["proxy_simple_avg"] = proxy1
feat["proxy_svi_weighted"] = proxy2
feat["proxy_max_gap"] = proxy3
feat["proxy_pop_weighted"] = proxy4

# ── H3 Spatial Block CV ───────────────────────────────────────────────────
print("\n[4] Computing H3 spatial blocks (resolution 4)...")

h3_resolution = 4
h3_cells = []
for lat, lon in zip(feat["centroid_lat"].values, feat["centroid_lon"].values):
    try:
        cell = h3.latlng_to_cell(lat, lon, h3_resolution)
    except Exception:
        cell = "unknown"
    h3_cells.append(cell)

feat["_h3_block"] = h3_cells
unique_blocks = feat["_h3_block"].nunique()
print(f"    H3 resolution {h3_resolution}: {unique_blocks} unique spatial blocks")

# ── Select feature columns for XGBoost ─────────────────────────────────────
# Exclude: GEOID, county_fips, gap targets, proxy targets, h3 block,
# centroid lat/lon (to avoid spatial leakage), and any string columns
exclude_patterns = [
    "GEOID", "county_fips", "_h3_block",
    "centroid_lat", "centroid_lon",
    "proxy_simple_avg", "proxy_svi_weighted", "proxy_max_gap", "proxy_pop_weighted",
]
# Also exclude raw gap columns that would leak target info
gap_leak_cols = [c for c in feat.columns if "gap" in c.lower()]

exclude_cols = set(exclude_patterns) | set(gap_leak_cols)

feature_cols = [
    c for c in feat.columns
    if c not in exclude_cols
    and feat[c].dtype in ["float64", "float32", "int64", "int32", "bool"]
    and feat[c].notna().sum() > 100  # need enough non-null
]

# Limit features to top 200 by variance to speed up CV
feat_variances = feat[feature_cols].var()
top_features = feat_variances.nlargest(200).index.tolist()

# Fill remaining NaN with 0 for modeling
X = feat[top_features].fillna(0).values
print(f"\n[5] Feature matrix: {X.shape[0]} rows x {X.shape[1]} features (top 200 by variance)")
print(f"    Excluded {len(exclude_cols)} columns (leakage / non-numeric), trimmed from {len(feature_cols)} to 200")

# ── XGBoost parameters ─────────────────────────────────────────────────────
xgb_params = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

# ── 3-fold H3 Spatial Block CV for each proxy ─────────────────────────────
proxy_targets = {
    "Proxy 1 - Simple avg": proxy1,
    "Proxy 2 - SVI-weighted": proxy2,
    "Proxy 3 - Max gap": proxy3,
    "Proxy 4 - Pop-weighted": proxy4,
}

results = {}
n_folds = 3

print(f"\n[6] Running {n_folds}-fold H3 Spatial Block CV for each proxy...")
print("-" * 70)

for name, y in proxy_targets.items():
    print(f"\n  >> Evaluating: {name}")
    
    # GroupKFold by H3 blocks
    groups = feat["_h3_block"].values
    gkf = GroupKFold(n_splits=n_folds)
    
    fold_rmses = []
    fold_r2s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y.values, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]
        
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        
        print(f"     Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}, "
              f"RMSE={rmse:.6f}, R2={r2:.6f}")
    
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    
    results[name] = {
        "fold_rmses": fold_rmses,
        "fold_r2s": fold_r2s,
        "mean_rmse": float(mean_rmse),
        "std_rmse": float(std_rmse),
        "mean_r2": float(mean_r2),
    }
    
    print(f"     >> Mean RMSE={mean_rmse:.6f}  Std RMSE={std_rmse:.6f}  Mean R2={mean_r2:.6f}")

# ── Select best proxy ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PROXY TARGET COMPARISON")
print("=" * 70)

best_proxy = min(results, key=lambda k: results[k]["std_rmse"])

# Print comparison table
header = f"{'Proxy':<30} {'Mean RMSE':>10} {'Std RMSE':>10} {'Mean R2':>10} {'Best':>6}"
print(header)
print("-" * len(header))

for name, r in results.items():
    is_best = "  ***" if name == best_proxy else ""
    print(f"{name:<30} {r['mean_rmse']:>10.6f} {r['std_rmse']:>10.6f} {r['mean_r2']:>10.6f}{is_best}")

print(f"\n  BEST PROXY (lowest std_rmse): {best_proxy}")
print(f"    Mean RMSE = {results[best_proxy]['mean_rmse']:.6f}")
print(f"    Std  RMSE = {results[best_proxy]['std_rmse']:.6f}")
print(f"    Mean R2   = {results[best_proxy]['mean_r2']:.6f}")

# ── Save results ───────────────────────────────────────────────────────────
print("\n[7] Saving results...")

# Save comparison JSON
comparison_out = {
    "best_proxy": best_proxy,
    "results": {
        k: {
            "mean_rmse": v["mean_rmse"],
            "std_rmse": v["std_rmse"],
            "mean_r2": v["mean_r2"],
            "fold_rmses": [float(x) for x in v["fold_rmses"]],
            "fold_r2s": [float(x) for x in v["fold_r2s"]],
        }
        for k, v in results.items()
    },
}
with open(OUTPUT_COMPARISON, "w") as f:
    json.dump(comparison_out, f, indent=2)
print(f"    Saved comparison to {OUTPUT_COMPARISON}")

# Drop temp h3 column before saving
feat.drop(columns=["_h3_block"], inplace=True)

# Save feature parquet with proxy columns added
feat.to_parquet(FEATURE_PATH, index=False)
print(f"    Saved features with proxy columns to {FEATURE_PATH}")
print(f"    New shape: {feat.shape}")
print(f"    New columns: proxy_simple_avg, proxy_svi_weighted, proxy_max_gap, proxy_pop_weighted")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
