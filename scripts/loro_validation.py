#!/usr/bin/env python3
"""
Leave-One-Region-Out (LORO) Validation for v3 Pipeline.

Key insight: If LORO RMSE is much higher than H3-CV RMSE, the model is
overfitting to regional patterns and won't generalize to new regions on
the private leaderboard.

For each region as holdout:
  - Train on the other 3 regions
  - Predict on the holdout region
  - Compute RMSE and R2 on holdout

Uses XGBRegressor with the same feature preparation as v3 pipeline.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Config ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "data/output"

FEATURE_PATH = OUT / "engineered_features_v3.parquet"
PIPELINE_STATE_PATH = OUT / "pipeline_state.json"
OUTPUT_PATH = OUT / "loro_validation.json"

# Columns to drop (non-numeric, identifiers, leakage)
DROP_COLS = [
    'GEOID', 'region', 'county_fips', 'state_fips',
    'centroid_lat', 'centroid_lon',
    'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
    'building_count_ratio', 'building_count_gap', 'road_count_ratio',
    'road_count_gap', 'road_length_ratio', 'road_length_gap',
    'poi_facility_gap', 'poi_to_facility_ratio',
    'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
]

# Target preference: proxy_simple_avg > building_gap
TARGET_CANDIDATES = ['proxy_simple_avg', 'building_gap']

# XGBoost parameters (as specified)
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method='hist',
    random_state=SEED,
)

# Feature selection parameters (matching v3 pipeline)
TOP_N_FEATURES = 80
HIGH_CORR_THRESHOLD = 0.98

# ── Start ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("LEAVE-ONE-REGION-OUT (LORO) VALIDATION")
print("=" * 72)
t0 = time.time()

# ── Load data ──────────────────────────────────────────────────────────────
print("\n[1] Loading engineered features...")
feat = pd.read_parquet(FEATURE_PATH)
print(f"    Shape: {feat.shape}")
print(f"    Regions: {feat['region'].unique().tolist()}")

# ── Select target ──────────────────────────────────────────────────────────
print("\n[2] Selecting target...")
target_col = None
for cand in TARGET_CANDIDATES:
    if cand in feat.columns:
        target_col = cand
        print(f"    Using target: {target_col}")
        break

if target_col is None:
    print("    ERROR: No valid target column found!")
    sys.exit(1)

y = feat[target_col].copy()
valid = y.notna()
print(f"    Non-null targets: {valid.sum()} / {len(y)}")

# ── Prepare features (same as v3 pipeline) ─────────────────────────────────
print("\n[3] Preparing features (v3 pipeline method)...")

# Remove duplicate columns
feat = feat.loc[:, ~feat.columns.duplicated()]

# Select numeric columns, excluding drop list
feature_cols = [
    c for c in feat.columns
    if c not in DROP_COLS
    and c not in TARGET_CANDIDATES  # also exclude proxy targets to avoid leakage
    and pd.api.types.is_numeric_dtype(feat[c])
]

X = feat[feature_cols].copy()
print(f"    After dropping non-numeric/leakage: {X.shape[1]} features")

# Filter to valid target rows
X = X[valid]
y = y[valid]
regions = feat.loc[valid, 'region'].copy()

# Fillna and inf
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

# Drop zero-variance columns
stds = X.std()
X = X[stds[stds > 1e-10].index]
print(f"    After dropping zero-variance: {X.shape[1]} features")

# Select top N by correlation with target
corr_with_y = X.corrwith(y).abs().fillna(0)
top_features = corr_with_y.sort_values(ascending=False).head(TOP_N_FEATURES).index
X = X[top_features]
print(f"    After top {TOP_N_FEATURES} by correlation: {X.shape[1]} features")

# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > HIGH_CORR_THRESHOLD)]
X = X.drop(columns=to_drop)
print(f"    After removing high-corr >{HIGH_CORR_THRESHOLD}: {X.shape[1]} features (dropped {len(to_drop)})")

print(f"\n    Final: {X.shape[0]} tracts x {X.shape[1]} features")

# ── LORO Validation ────────────────────────────────────────────────────────
unique_regions = sorted(regions.unique())
print(f"\n[4] Running LORO validation across {len(unique_regions)} regions...")
print(f"    Regions: {unique_regions}")
print("-" * 72)

loro_results = {}

for holdout_region in unique_regions:
    print(f"\n  >> Holdout: {holdout_region}")

    # Split by region
    train_mask = regions != holdout_region
    test_mask = regions == holdout_region

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    n_train = len(X_train)
    n_test = len(X_test)
    print(f"     Train: {n_train} tracts from {len(unique_regions) - 1} regions")
    print(f"     Test:  {n_test} tracts from {holdout_region}")

    # Train XGBoost
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Per-region breakdown within holdout (if we had sub-regions, but useful for diagnostics)
    mae = np.mean(np.abs(y_test - y_pred))
    mean_pred = y_pred.mean()
    mean_actual = y_test.mean()
    bias = mean_pred - mean_actual

    loro_results[holdout_region] = {
        'rmse': float(rmse),
        'r2': float(r2),
        'mae': float(mae),
        'n_train': int(n_train),
        'n_test': int(n_test),
        'mean_pred': float(mean_pred),
        'mean_actual': float(mean_actual),
        'bias': float(bias),
    }

    print(f"     RMSE = {rmse:.6f}")
    print(f"     R2   = {r2:.6f}")
    print(f"     MAE  = {mae:.6f}")
    print(f"     Bias = {bias:.6f} (pred_mean={mean_pred:.4f}, actual_mean={mean_actual:.4f})")

    # Feature importance (top 10)
    imp = model.feature_importances_
    fi = pd.Series(imp, index=X.columns).sort_values(ascending=False)
    print(f"     Top 5 features: {fi.head(5).index.tolist()}")

# ── Overall LORO metrics ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LORO RESULTS SUMMARY")
print("=" * 72)

rmses = [loro_results[r]['rmse'] for r in unique_regions]
r2s = [loro_results[r]['r2'] for r in unique_regions]
maes = [loro_results[r]['mae'] for r in unique_regions]

# Weighted average by test set size
total_test = sum(loro_results[r]['n_test'] for r in unique_regions)
weighted_rmse = np.sqrt(
    sum(loro_results[r]['rmse']**2 * loro_results[r]['n_test'] for r in unique_regions) / total_test
)
mean_rmse = np.mean(rmses)
std_rmse = np.std(rmses)
mean_r2 = np.mean(r2s)

print(f"\n{'Region':<20} {'RMSE':>10} {'R2':>10} {'MAE':>10} {'N_test':>8} {'Bias':>10}")
print("-" * 72)
for r in unique_regions:
    d = loro_results[r]
    print(f"{r:<20} {d['rmse']:>10.6f} {d['r2']:>10.6f} {d['mae']:>10.6f} {d['n_test']:>8d} {d['bias']:>10.6f}")

print("-" * 72)
print(f"{'Mean':<20} {mean_rmse:>10.6f} {mean_r2:>10.6f} {np.mean(maes):>10.6f}")
print(f"{'Std':<20} {std_rmse:>10.6f}")
print(f"{'Weighted RMSE':<20} {weighted_rmse:>10.6f}")
print(f"{'Max RMSE':<20} {max(rmses):>10.6f}")
print(f"{'Min RMSE':>20} {min(rmses):>10.6f}")

# ── Compare with H3-CV ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("COMPARISON: LORO vs H3-CV")
print("=" * 72)

h3_cv_info = {}
if PIPELINE_STATE_PATH.exists():
    pipeline_state = json.load(open(PIPELINE_STATE_PATH))
    h3_cv_info = pipeline_state.get('leakage_check', {})

    h3_rmse = h3_cv_info.get('h3_cv_rmse', None)
    h3_r2 = h3_cv_info.get('h3_cv_r2', None)

    print(f"\n  H3-CV (from pipeline_state.json):")
    print(f"    RMSE = {h3_rmse:.6f}" if h3_rmse else "    RMSE = N/A")
    print(f"    R2   = {h3_r2:.6f}" if h3_r2 else "    R2   = N/A")
    print(f"    N blocks = {h3_cv_info.get('n_h3_blocks', 'N/A')}")
    print(f"    N features = {h3_cv_info.get('n_features', 'N/A')}")

    print(f"\n  LORO (this run):")
    print(f"    Mean RMSE  = {mean_rmse:.6f}")
    print(f"    Weighted RMSE = {weighted_rmse:.6f}")
    print(f"    Mean R2    = {mean_r2:.6f}")

    if h3_rmse is not None:
        ratio = mean_rmse / h3_rmse
        rmse_increase = mean_rmse - h3_rmse
        print(f"\n  >>> LORO / H3-CV RMSE ratio = {ratio:.2f}x")
        print(f"  >>> LORO RMSE increase     = {rmse_increase:+.6f}")

        if ratio > 1.5:
            verdict = "SEVERE OVERFITTING"
            detail = "LORO RMSE is >1.5x H3-CV. Model relies heavily on regional patterns."
        elif ratio > 1.2:
            verdict = "MODERATE OVERFITTING"
            detail = "LORO RMSE is >1.2x H3-CV. Some regional overfitting detected."
        elif ratio > 1.05:
            verdict = "MILD OVERFITTING"
            detail = "LORO RMSE is slightly higher than H3-CV. Minor regional dependency."
        else:
            verdict = "GOOD GENERALIZATION"
            detail = "LORO RMSE is close to H3-CV. Model generalizes well across regions."

        print(f"\n  VERDICT: {verdict}")
        print(f"  {detail}")

        # Per-region comparison
        print(f"\n  Per-region vs H3-CV:")
        for r in unique_regions:
            r_rmse = loro_results[r]['rmse']
            r_ratio = r_rmse / h3_rmse
            flag = " <<<<" if r_ratio > 1.5 else (" <<!" if r_ratio > 1.2 else "")
            print(f"    {r:<20} RMSE={r_rmse:.6f}  ({r_ratio:.2f}x H3-CV){flag}")
else:
    print("\n  WARNING: pipeline_state.json not found. Cannot compare with H3-CV.")
    pipeline_state = {}

# ── Identify hardest regions ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("REGIONAL DIFFICULTY ANALYSIS")
print("=" * 72)

sorted_by_rmse = sorted(unique_regions, key=lambda r: loro_results[r]['rmse'], reverse=True)
print(f"\n  Hardest to predict (highest RMSE):")
for i, r in enumerate(sorted_by_rmse):
    d = loro_results[r]
    print(f"    {i+1}. {r:<20} RMSE={d['rmse']:.6f}  R2={d['r2']:.6f}  Bias={d['bias']:.6f}")

print(f"\n  Easiest to predict (lowest RMSE):")
for i, r in enumerate(sorted_by_rmse[::-1]):
    d = loro_results[r]
    print(f"    {i+1}. {r:<20} RMSE={d['rmse']:.6f}  R2={d['r2']:.6f}  Bias={d['bias']:.6f}")

# ── Cross-region feature importance stability ─────────────────────────────
print("\n" + "=" * 72)
print("CROSS-REGION FEATURE IMPORTANCE STABILITY")
print("=" * 72)

print("\n  Retraining to get per-region feature importances...")
region_importances = {}
for holdout_region in unique_regions:
    train_mask = regions != holdout_region
    test_mask = regions == holdout_region
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    region_importances[holdout_region] = imp

# Compute stability: how much does feature ranking vary across LORO folds?
all_imp = pd.DataFrame(region_importances)
mean_imp = all_imp.mean(axis=1).sort_values(ascending=False)
std_imp = all_imp.std(axis=1)
cv_imp = (std_imp / (mean_imp + 1e-10)).reindex(mean_imp.index)

print(f"\n  Top 15 features (mean importance across 4 LORO folds):")
print(f"  {'Feature':<35} {'Mean Imp':>10} {'Std Imp':>10} {'CV':>8}")
print("  " + "-" * 65)
for feat_name in mean_imp.head(15).index:
    mi = mean_imp[feat_name]
    si = std_imp[feat_name]
    cv = cv_imp[feat_name]
    print(f"  {feat_name:<35} {mi:>10.6f} {si:>10.6f} {cv:>8.2f}")

unstable_features = cv_imp.head(15)[cv_imp.head(15) > 1.0]
if len(unstable_features) > 0:
    print(f"\n  WARNING: {len(unstable_features)} top features have CV > 1.0 (unstable across regions)")
    print(f"  Unstable features: {unstable_features.index.tolist()}")

# ── Save results ───────────────────────────────────────────────────────────
print("\n[5] Saving results...")

output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'validation_type': 'LORO',
    'target': target_col,
    'n_features': int(X.shape[1]),
    'n_tracts': int(X.shape[0]),
    'xgb_params': XGB_PARAMS,
    'regions': unique_regions,
    'per_region': loro_results,
    'overall': {
        'mean_rmse': float(mean_rmse),
        'std_rmse': float(std_rmse),
        'weighted_rmse': float(weighted_rmse),
        'mean_r2': float(mean_r2),
        'max_rmse': float(max(rmses)),
        'min_rmse': float(min(rmses)),
        'rmse_range': float(max(rmses) - min(rmses)),
    },
    'comparison_with_h3cv': None,
}

if h3_cv_info:
    h3_rmse = h3_cv_info.get('h3_cv_rmse')
    h3_r2 = h3_cv_info.get('h3_cv_r2')
    output['comparison_with_h3cv'] = {
        'h3_cv_rmse': h3_rmse,
        'h3_cv_r2': h3_r2,
        'loro_mean_rmse': float(mean_rmse),
        'loro_weighted_rmse': float(weighted_rmse),
        'rmse_ratio': float(mean_rmse / h3_rmse) if h3_rmse else None,
        'rmse_increase': float(mean_rmse - h3_rmse) if h3_rmse else None,
        'verdict': verdict if 'verdict' in dir() else 'unknown',
    }

# Add feature importance stability
output['feature_stability'] = {
    'top_features': mean_imp.head(20).to_dict(),
    'unstable_features': unstable_features.to_dict() if len(unstable_features) > 0 else {},
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"    Saved to {OUTPUT_PATH}")

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {elapsed:.0f}s")
print(f"{'=' * 72}")

# Final verdict summary
if h3_cv_info and h3_cv_info.get('h3_cv_rmse'):
    ratio = mean_rmse / h3_cv_info['h3_cv_rmse']
    print(f"\n  LORO Mean RMSE:  {mean_rmse:.6f}")
    print(f"  H3-CV RMSE:      {h3_cv_info['h3_cv_rmse']:.6f}")
    print(f"  Ratio:           {ratio:.2f}x")
    if ratio > 1.2:
        print(f"\n  ACTION NEEDED: Model shows regional overfitting. Consider:")
        print(f"    - Adding region-invariant features")
        print(f"    - Using region as a feature (to learn cross-regional patterns)")
        print(f"    - Reducing model complexity (lower max_depth, more regularization)")
        print(f"    - Increasing training data diversity")
