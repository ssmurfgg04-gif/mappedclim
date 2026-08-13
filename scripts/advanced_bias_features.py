#!/usr/bin/env python3
"""
Advanced Bias Features: Temporal Decay Weighting + SHAP Tribal Bias Decomposition
=================================================================================
Combines Idea #6 (temporal decay weighting) and Idea #7 (SHAP-based tribal bias decomposition).

Part 1: Tests whether tracts with stale source data have higher predicted gaps.
Part 2: Decomposes the tribal bias into feature-level SHAP contributions.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import json
import time
import warnings
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT  = PROJ / "data/output"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ADVANCED BIAS FEATURES: TEMPORAL DECAY + SHAP TRIBAL BIAS DECOMPOSITION")
print("=" * 80)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[0] Loading data...")
feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
print(f"  Engineered features: {feat.shape}")

# Load OOF predictions for residual analysis
oof = pd.read_parquet(OUT / "oof_predictions_merged.parquet")
print(f"  OOF predictions:     {oof.shape}")

# Merge OOF into feat
feat['GEOID'] = feat['GEOID'].astype(str)
oof['GEOID']  = oof['GEOID'].astype(str)
df = feat.merge(oof[['GEOID', 'xgb', 'lgb', 'et']], on='GEOID', how='left')
print(f"  Merged:              {df.shape}")

# Derived flags
df['is_tribal'] = df['tribal_any'].astype(bool)
df['is_rural']  = df['rural_indicator'] > 0.5

n_tribal     = df['is_tribal'].sum()
n_non_tribal = (~df['is_tribal']).sum()
print(f"  Tribal tracts:       {n_tribal}")
print(f"  Non-tribal tracts:   {n_non_tribal}")

# Compute OOF ensemble prediction and residuals
df['oof_ensemble'] = (df['xgb'].fillna(0) + df['lgb'].fillna(0) + df['et'].fillna(0)) / 3
df['oof_residual'] = df['gap_only'] - df['oof_ensemble']

results = {}  # Will be saved to JSON

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: TEMPORAL DECAY WEIGHTING (Idea #6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 1: TEMPORAL DECAY WEIGHTING — Data Freshness Analysis")
print("=" * 80)
print("""
HYPOTHESIS: Tracts with stale source data (old OSM extracts, few contributors)
should have higher predicted gaps. The "freshness" of mapping data is a data
collection artifact that explains part of the missing 85% variance.
""")

# ── 1a. Identify and inspect freshness proxy features ──
print("\n[1a] Freshness proxy features:")
freshness_features = {
    'bldg_total_sources':        'Fewer sources → older/staler data',
    'bldg_source_diversity':     'Lower diversity → less community mapping effort',
    'source_coverage_fraction':  'Lower → more incomplete source coverage',
    'poi_mean_confidence':       'Lower → less reliable POI data',
    'poi_very_high_conf_fraction': 'Lower → less verified POI data',
}

for col, desc in freshness_features.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  ✓ {col:<35} mean={df[col].mean():>10.4f}  std={df[col].std():>10.4f}  non_null={non_null}  │ {desc}")
    else:
        print(f"  ✗ {col:<35} NOT FOUND  │ {desc}")

# ── 1b. Normalize each freshness metric to 0-1 ──
print("\n[1b] Normalizing freshness metrics to [0, 1] range...")

# For bldg_total_sources: higher = more fresh → already a "freshness" metric
# For bldg_source_diversity: higher = more fresh → already a "freshness" metric
# For source_coverage_fraction: higher = more fresh → already a "freshness" metric
# For poi_mean_confidence: higher = more fresh → already a "freshness" metric
# For poi_very_high_conf_fraction: higher = more fresh → already a "freshness" metric

normalized_freshness = pd.DataFrame(index=df.index)

for col in freshness_features:
    if col not in df.columns:
        continue
    vals = df[col].copy()
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-10:
        normalized_freshness[col + '_norm'] = 0.5
    else:
        normalized_freshness[col + '_norm'] = (vals - vmin) / (vmax - vmin)

norm_cols = [c for c in normalized_freshness.columns]
print(f"  Normalized columns: {norm_cols}")

# ── 1c. Create stale_source_score ──
# stale = 1 - mean(normalized_freshness_metrics)
# High stale = low freshness = data is old/incomplete
normalized_freshness['mean_freshness'] = normalized_freshness[norm_cols].mean(axis=1)
df['stale_source_score'] = 1.0 - normalized_freshness['mean_freshness']

print(f"\n[1c] stale_source_score created:")
print(f"  Mean:   {df['stale_source_score'].mean():.6f}")
print(f"  Median: {df['stale_source_score'].median():.6f}")
print(f"  Std:    {df['stale_source_score'].std():.6f}")
print(f"  Min:    {df['stale_source_score'].min():.6f}")
print(f"  Max:    {df['stale_source_score'].max():.6f}")
print(f"  Q25:    {df['stale_source_score'].quantile(0.25):.6f}")
print(f"  Q75:    {df['stale_source_score'].quantile(0.75):.6f}")

# ── 1d. Create interaction features ──
df['stale_x_building_gap'] = df['stale_source_score'] * df['building_gap']
df['stale_x_rural']        = df['stale_source_score'] * df['rural_continuous']

print(f"\n[1d] Interaction features created:")
print(f"  stale_x_building_gap: mean={df['stale_x_building_gap'].mean():.6f}, std={df['stale_x_building_gap'].std():.6f}")
print(f"  stale_x_rural:        mean={df['stale_x_rural'].mean():.6f}, std={df['stale_x_rural'].std():.6f}")

# ── 1e. Analysis: Correlation with gap_only ──
print("\n" + "-" * 80)
print("[1e] CORRELATION ANALYSIS: stale_source_score vs gap_only")
print("-" * 80)

valid_mask = df['gap_only'].notna() & df['stale_source_score'].notna()
pearson_r, pearson_p = pearsonr(df.loc[valid_mask, 'stale_source_score'], df.loc[valid_mask, 'gap_only'])
spearman_r, spearman_p = spearmanr(df.loc[valid_mask, 'stale_source_score'], df.loc[valid_mask, 'gap_only'])

print(f"  Pearson  r = {pearson_r:.6f}  (p = {pearson_p:.2e})")
print(f"  Spearman r = {spearman_r:.6f}  (p = {spearman_p:.2e})")
print(f"  → {'SIGNIFICANT' if pearson_p < 0.05 else 'not significant'}: Stale source data {'IS' if pearson_p < 0.05 else 'is NOT'} correlated with gap_only")

results['temporal_decay'] = {
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'spearman_r': float(spearman_r),
    'spearman_p': float(spearman_p),
}

# Also check correlations of individual freshness metrics
print(f"\n  Individual freshness metric correlations with gap_only:")
print(f"  {'Feature':<40} {'Pearson r':>10} {'p-value':>12} {'Direction':>10}")
print("  " + "-" * 75)
for col in freshness_features:
    if col not in df.columns:
        continue
    vm = df['gap_only'].notna() & df[col].notna()
    if vm.sum() < 100:
        continue
    pr, pp = pearsonr(df.loc[vm, col], df.loc[vm, 'gap_only'])
    direction = "↑ fresh→↑gap" if pr > 0 else "↑ fresh→↓gap"
    print(f"  {col:<40} {pr:>10.6f} {pp:>12.2e} {direction:>10}")

# ── 1f. Tribal vs Non-Tribal stale_source_score ──
print("\n" + "-" * 80)
print("[1f] TRIBAL vs NON-TRIBAL: Is tribal data staler?")
print("-" * 80)

tribal_stale     = df.loc[df['is_tribal'], 'stale_source_score']
non_tribal_stale = df.loc[~df['is_tribal'], 'stale_source_score']

print(f"  Tribal     stale: mean={tribal_stale.mean():.6f}, median={tribal_stale.median():.6f}, std={tribal_stale.std():.6f}")
print(f"  Non-tribal stale: mean={non_tribal_stale.mean():.6f}, median={non_tribal_stale.median():.6f}, std={non_tribal_stale.std():.6f}")
print(f"  Difference (tribal - non-tribal): {tribal_stale.mean() - non_tribal_stale.mean():.6f}")
print(f"  Ratio (tribal / non-tribal):      {tribal_stale.mean() / non_tribal_stale.mean():.4f}")

# t-test
from scipy.stats import ttest_ind
t_stat, t_p = ttest_ind(tribal_stale.dropna(), non_tribal_stale.dropna(), equal_var=False)
print(f"  Welch t-test: t={t_stat:.4f}, p={t_p:.2e}")
print(f"  → Tribal data IS {'significantly STALER' if (tribal_stale.mean() > non_tribal_stale.mean() and t_p < 0.05) else 'NOT significantly staler'} than non-tribal data")

results['temporal_decay']['tribal_stale_mean'] = float(tribal_stale.mean())
results['temporal_decay']['non_tribal_stale_mean'] = float(non_tribal_stale.mean())
results['temporal_decay']['tribal_vs_nontribal_t'] = float(t_stat)
results['temporal_decay']['tribal_vs_nontribal_p'] = float(t_p)

# Also check individual freshness metrics by tribal status
print(f"\n  Individual freshness metrics by tribal status:")
print(f"  {'Feature':<40} {'Tribal Mean':>12} {'NonTrib Mean':>13} {'Diff':>10} {'Ratio':>8}")
print("  " + "-" * 85)
for col in freshness_features:
    if col not in df.columns:
        continue
    t_mean = df.loc[df['is_tribal'], col].mean()
    nt_mean = df.loc[~df['is_tribal'], col].mean()
    ratio = t_mean / nt_mean if nt_mean != 0 else np.nan
    print(f"  {col:<40} {t_mean:>12.4f} {nt_mean:>13.4f} {t_mean-nt_mean:>10.4f} {ratio:>8.4f}")

# ── 1g. Rural vs Urban stale_source_score ──
print("\n" + "-" * 80)
print("[1g] RURAL vs URBAN: Is rural data staler?")
print("-" * 80)

rural_stale = df.loc[df['is_rural'], 'stale_source_score']
urban_stale = df.loc[~df['is_rural'], 'stale_source_score']

print(f"  Rural stale: mean={rural_stale.mean():.6f}, median={rural_stale.median():.6f}, std={rural_stale.std():.6f}")
print(f"  Urban stale: mean={urban_stale.mean():.6f}, median={urban_stale.median():.6f}, std={urban_stale.std():.6f}")
print(f"  Difference (rural - urban): {rural_stale.mean() - urban_stale.mean():.6f}")
print(f"  Ratio (rural / urban):      {rural_stale.mean() / urban_stale.mean():.4f}")

t_stat_ru, t_p_ru = ttest_ind(rural_stale.dropna(), urban_stale.dropna(), equal_var=False)
print(f"  Welch t-test: t={t_stat_ru:.4f}, p={t_p_ru:.2e}")
print(f"  → Rural data IS {'significantly STALER' if (rural_stale.mean() > urban_stale.mean() and t_p_ru < 0.05) else 'NOT significantly staler'} than urban data")

results['temporal_decay']['rural_stale_mean'] = float(rural_stale.mean())
results['temporal_decay']['urban_stale_mean'] = float(urban_stale.mean())
results['temporal_decay']['rural_vs_urban_t'] = float(t_stat_ru)
results['temporal_decay']['rural_vs_urban_p'] = float(t_p_ru)

# ── 1h. How much does stale_source_score explain of OOF residuals? ──
print("\n" + "-" * 80)
print("[1h] STALE SCORE vs OOF RESIDUALS: Does staleness explain unexplained variance?")
print("-" * 80)

resid_valid = df['oof_residual'].notna() & df['stale_source_score'].notna()
if resid_valid.sum() > 100:
    pr_resid, pp_resid = pearsonr(df.loc[resid_valid, 'stale_source_score'], df.loc[resid_valid, 'oof_residual'])
    sr_resid, sp_resid = spearmanr(df.loc[resid_valid, 'stale_source_score'], df.loc[resid_valid, 'oof_residual'])
    
    print(f"  OOF residual mean: {df['oof_residual'].mean():.6f}, std: {df['oof_residual'].std():.6f}")
    print(f"  Pearson  r = {pr_resid:.6f}  (p = {pp_resid:.2e})")
    print(f"  Spearman r = {sr_resid:.6f}  (p = {sp_resid:.2e})")
    print(f"  → stale_source_score explains {pr_resid**2*100:.4f}% of OOF residual variance")
    print(f"  → This is {'a meaningful' if pr_resid**2 > 0.001 else 'a small but detectable' if pr_resid**2 > 0.0001 else 'negligible'} amount of the unexplained variance")
    
    results['temporal_decay']['residual_pearson_r'] = float(pr_resid)
    results['temporal_decay']['residual_pearson_p'] = float(pp_resid)
    results['temporal_decay']['residual_r_squared'] = float(pr_resid**2)

    # Also check stale_x_building_gap and stale_x_rural correlations with residuals
    for interact_col in ['stale_x_building_gap', 'stale_x_rural']:
        vm2 = df['oof_residual'].notna() & df[interact_col].notna()
        if vm2.sum() > 100:
            pr2, pp2 = pearsonr(df.loc[vm2, interact_col], df.loc[vm2, 'oof_residual'])
            print(f"  {interact_col}: Pearson r = {pr2:.6f} (p = {pp2:.2e}), r² = {pr2**2*100:.4f}%")
else:
    print("  Not enough valid OOF residual data")

# ── 1i. Stale score quartile analysis ──
print("\n" + "-" * 80)
print("[1i] STALE SCORE QUARTILE ANALYSIS")
print("-" * 80)

# Use custom quantile-based binning to handle ties
stale_q25 = df['stale_source_score'].quantile(0.25)
stale_q50 = df['stale_source_score'].quantile(0.50)
stale_q75 = df['stale_source_score'].quantile(0.75)

def assign_stale_quartile(val):
    if val <= stale_q25:
        return 'Q1(Fresh)'
    elif val <= stale_q50:
        return 'Q2'
    elif val <= stale_q75:
        return 'Q3'
    else:
        return 'Q4(Stale)'

df['stale_quartile'] = df['stale_source_score'].apply(assign_stale_quartile)

print(f"\n{'Quartile':<14} {'N':>7} {'Mean Stale':>11} {'Mean Gap':>10} {'Mean Resid':>11} {'% Tribal':>9} {'% Rural':>8}")
print("-" * 65)

quartile_results = []
for q in ['Q1(Fresh)', 'Q2', 'Q3', 'Q4(Stale)']:
    qdf = df[df['stale_quartile'] == q]
    n = len(qdf)
    mean_stale = qdf['stale_source_score'].mean()
    mean_gap = qdf['gap_only'].mean()
    mean_resid = qdf['oof_residual'].mean() if qdf['oof_residual'].notna().sum() > 0 else np.nan
    pct_tribal = qdf['is_tribal'].mean() * 100
    pct_rural = qdf['is_rural'].mean() * 100
    print(f"{q:<14} {n:>7} {mean_stale:>11.6f} {mean_gap:>10.6f} {mean_resid:>11.6f} {pct_tribal:>8.2f}% {pct_rural:>7.2f}%")
    quartile_results.append({
        'quartile': q, 'n': int(n), 'mean_stale': float(mean_stale),
        'mean_gap': float(mean_gap), 'pct_tribal': float(pct_tribal), 'pct_rural': float(pct_rural)
    })

results['temporal_decay']['quartile_analysis'] = quartile_results

# Stale→Gap gradient
q1_gap = df.loc[df['stale_quartile'] == 'Q1(Fresh)', 'gap_only'].mean()
q4_gap = df.loc[df['stale_quartile'] == 'Q4(Stale)', 'gap_only'].mean()
print(f"\n  Gap gradient (Q4 stale - Q1 fresh): {q4_gap - q1_gap:.6f}")
print(f"  → Stale tracts have {'MORE negative' if (q4_gap - q1_gap) < 0 else 'LESS negative'} gaps ({'WORSE' if (q4_gap - q1_gap) < 0 else 'BETTER'} coverage)")

results['temporal_decay']['stale_gap_gradient'] = float(q4_gap - q1_gap)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: SHAP-BASED TRIBAL BIAS DECOMPOSITION (Idea #7)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2: SHAP-BASED TRIBAL BIAS DECOMPOSITION")
print("=" * 80)
print("""
HYPOTHESIS: We can decompose the tribal bias into which features contribute
most to the disparity. This is critical for the Bias Scoring API — we need
to show WHICH features cause tribal bias, not just that it exists.
""")

import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

# ── 2a. Prepare features (same approach as Phase 2) ──
print("\n[2a] Preparing features for XGBoost...")

drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
             'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON',
             'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
             'building_count_ratio', 'building_count_gap',
             'road_count_ratio', 'road_count_gap', 'road_length_ratio', 'road_length_gap',
             'poi_facility_gap', 'poi_to_facility_ratio',
             'poi_facility_gap_corrected', 'poi_to_facility_ratio_corrected',
             'building_area_gap',
             'coverage_gap_score', 'coverage_gap', 'gap_score', 'coverage_score',
             'proxy_merged', 'gap_only', 'rural_penalty',
             'xgb', 'lgb', 'et', 'oof_ensemble', 'oof_residual',
             'stale_source_score', 'stale_x_building_gap', 'stale_x_rural',
             'stale_quartile']

feat_clean = df.copy()
feat_clean = feat_clean.loc[:, ~feat_clean.columns.duplicated()]

# Select numeric columns, excluding drop_cols
fc = [c for c in feat_clean.columns 
      if c not in drop_cols 
      and pd.api.types.is_numeric_dtype(feat_clean[c])
      and not feat_clean[c].isna().all()
      and feat_clean[c].std() > 1e-10]

X_all = feat_clean[fc].copy()
y_all = df['gap_only'].copy()

# Handle missing/infinite
valid = y_all.notna()
X_all = X_all[valid]
y_all = y_all[valid]
X_all = X_all.fillna(-999).replace([np.inf, -np.inf], -999)

# Remove zero-variance
s = X_all.std()
X_all = X_all[s[s > 1e-10].index]

# Select top 60 features by correlation with target (same as Phase 2)
cs = X_all.corrwith(y_all).abs().fillna(0)
top_features = cs.sort_values(ascending=False).head(60).index.tolist()
X_all = X_all[top_features]

# Remove highly correlated features (>0.98)
cm = X_all.corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [c for c in up.columns if any(up[c] > 0.98)]
X_all = X_all.drop(columns=to_drop)

print(f"  Final feature set: {X_all.shape[1]} features, {X_all.shape[0]} tracts")
print(f"  Top 10 features by |corr| with gap_only:")
for i, (feat_name, corr_val) in enumerate(cs[top_features[:10]].items()):
    print(f"    {i+1}. {feat_name:<40} |r| = {corr_val:.4f}")

# Keep tribal/rural info aligned
tribal_mask_all = df.loc[valid, 'is_tribal'].values
rural_mask_all  = df.loc[valid, 'is_rural'].values

# ── 2b. Train XGBoost model ──
print("\n[2b] Training XGBoost model on gap_only...")

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train)

# Quick validation
from sklearn.metrics import r2_score, mean_squared_error
y_pred_val = model.predict(X_val)
val_r2 = r2_score(y_val, y_pred_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"  Validation R²:   {val_r2:.4f}")
print(f"  Validation RMSE: {val_rmse:.6f}")

# Retrain on ALL data for SHAP (more accurate explanations)
print("  Retraining on all data for SHAP analysis...")
model_full = xgb.XGBRegressor(**xgb_params)
model_full.fit(X_all, y_all)

y_pred_all = model_full.predict(X_all)
all_r2 = r2_score(y_all, y_pred_all)
print(f"  Full data R²:    {all_r2:.4f}")

results['shap_decomposition'] = {
    'model_val_r2': float(val_r2),
    'model_val_rmse': float(val_rmse),
    'model_full_r2': float(all_r2),
    'n_features': int(X_all.shape[1]),
}

# ── 2c. Compute SHAP values ──
print("\n[2c] Computing SHAP values with TreeExplainer...")

explainer = shap.TreeExplainer(model_full)
shap_values = explainer.shap_values(X_all)

print(f"  SHAP values shape: {shap_values.shape}")
print(f"  Base value (expected): {explainer.expected_value:.6f}")

# ── 2d. Overall SHAP feature importance ──
print("\n[2d] Overall SHAP feature importance (mean |SHAP|):")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X_all.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print(f"\n  {'Rank':<5} {'Feature':<45} {'Mean |SHAP|':>12}")
print("  " + "-" * 65)
for i, row in shap_importance.head(20).iterrows():
    rank = shap_importance.index.get_loc(i) + 1
    print(f"  {rank:<5} {row['feature']:<45} {row['mean_abs_shap']:>12.6f}")

# ── 2e. Tribal vs Non-Tribal SHAP decomposition ──
print("\n" + "-" * 80)
print("[2e] TRIBAL vs NON-TRIBAL SHAP DECOMPOSITION")
print("-" * 80)
print("""
For each feature, compute:
  - Mean SHAP value for tribal tracts
  - Mean SHAP value for non-tribal tracts
  - Difference = tribal - non-tribal (which features push tribal predictions more negative)
  - % of total bias explained by each feature
""")

# Mean SHAP per feature for each group (signed, not absolute)
tribal_shap_mean     = shap_values[tribal_mask_all].mean(axis=0)
non_tribal_shap_mean = shap_values[~tribal_mask_all].mean(axis=0)

# The total tribal bias in predictions
tribal_pred_mean     = y_pred_all[tribal_mask_all].mean()
non_tribal_pred_mean = y_pred_all[~tribal_mask_all].mean()
total_bias = tribal_pred_mean - non_tribal_pred_mean

print(f"  Tribal predicted mean:      {tribal_pred_mean:.6f}")
print(f"  Non-tribal predicted mean:  {non_tribal_pred_mean:.6f}")
print(f"  Total bias (tribal - nonT): {total_bias:.6f}")
print(f"  Tribal bias ratio:          {tribal_pred_mean / non_tribal_pred_mean:.4f}")

# SHAP decomposition: difference in mean SHAP per feature
shap_diff = tribal_shap_mean - non_tribal_shap_mean

# The sum of SHAP differences should approximate the total bias
shap_diff_sum = shap_diff.sum()
print(f"  Sum of SHAP differences:    {shap_diff_sum:.6f}")
print(f"  (Should ≈ total bias of {total_bias:.6f})")

# Create decomposition table
decomposition = pd.DataFrame({
    'feature': X_all.columns,
    'tribal_mean_shap': tribal_shap_mean,
    'non_tribal_mean_shap': non_tribal_shap_mean,
    'shap_difference': shap_diff,
}).sort_values('shap_difference', ascending=True)  # Most negative first = features that make tribal worse

# % of total bias (use absolute for attribution)
decomposition['pct_of_total_bias'] = (decomposition['shap_difference'] / total_bias * 100)

# Also compute mean |SHAP| per group for effect magnitude
decomposition['tribal_mean_abs_shap'] = np.abs(shap_values[tribal_mask_all]).mean(axis=0)
decomposition['non_tribal_mean_abs_shap'] = np.abs(shap_values[~tribal_mask_all]).mean(axis=0)

print(f"\n  SHAP BIAS DECOMPOSITION TABLE:")
print(f"  {'Feature':<40} {'Trib SHAP':>10} {'NonT SHAP':>10} {'Diff':>10} {'% Bias':>8} {'Trib |SHAP|':>11} {'NonT |SHAP|':>11}")
print("  " + "-" * 105)

# Show top 25 features contributing to tribal bias (most negative difference = making tribal worse)
top_bias_features = decomposition.head(25)
for _, row in top_bias_features.iterrows():
    print(f"  {row['feature']:<40} {row['tribal_mean_shap']:>10.6f} {row['non_tribal_mean_shap']:>10.6f} {row['shap_difference']:>10.6f} {row['pct_of_total_bias']:>7.2f}% {row['tribal_mean_abs_shap']:>11.6f} {row['non_tribal_mean_abs_shap']:>11.6f}")

# ── 2f. Focus on specific features ──
print("\n" + "-" * 80)
print("[2f] FOCUS: Key bias-related features")
print("-" * 80)

focus_features = ['rural_continuous', 'rural_indicator', 'tribal_x_bldg', 'svi_x_bldg', 
                  'compound_risk', 'compound_risk_sq', 'cvi_x_bldg', 'pct_urban_x_bldg',
                  'rural_x_bldg', 'rural_x_bldg_clip', 'rural_penalty', 'tribal_x_rural',
                  'rural_x_risk', 'svi_overall', 'svi_socioeconomic', 'svi_minority']

print(f"\n  {'Feature':<40} {'Trib SHAP':>10} {'NonT SHAP':>10} {'Diff':>10} {'% Bias':>8} {'In Model?':>10}")
print("  " + "-" * 90)

focus_results = []
for feat_name in focus_features:
    if feat_name in decomposition['feature'].values:
        row = decomposition[decomposition['feature'] == feat_name].iloc[0]
        print(f"  {feat_name:<40} {row['tribal_mean_shap']:>10.6f} {row['non_tribal_mean_shap']:>10.6f} {row['shap_difference']:>10.6f} {row['pct_of_total_bias']:>7.2f}% {'YES':>10}")
        focus_results.append({
            'feature': feat_name,
            'tribal_mean_shap': float(row['tribal_mean_shap']),
            'non_tribal_mean_shap': float(row['non_tribal_mean_shap']),
            'shap_difference': float(row['shap_difference']),
            'pct_of_total_bias': float(row['pct_of_total_bias']),
        })
    else:
        print(f"  {feat_name:<40} {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'NO':>10}")

results['shap_decomposition']['focus_features'] = focus_results

# ── 2g. Category-level SHAP decomposition ──
print("\n" + "-" * 80)
print("[2g] CATEGORY-LEVEL SHAP DECOMPOSITION")
print("-" * 80)

# Group features by category
categories = {
    'Rural/Geographic':  [c for c in X_all.columns if any(k in c.lower() for k in ['rural', 'pct_urban', 'urban'])],
    'Building/Gap':      [c for c in X_all.columns if any(k in c.lower() for k in ['bldg', 'building', 'gap'])],
    'SVI/Vulnerability': [c for c in X_all.columns if any(k in c.lower() for k in ['svi', 'cvi', 'compound_risk'])],
    'Tribal/Intersection': [c for c in X_all.columns if any(k in c.lower() for k in ['tribal'])],
    'Road/Network':      [c for c in X_all.columns if any(k in c.lower() for k in ['road', 'tiger'])],
    'Source/DataQuality': [c for c in X_all.columns if any(k in c.lower() for k in ['source', 'coverage', 'diversity'])],
    'POI/Infrastructure': [c for c in X_all.columns if any(k in c.lower() for k in ['poi', 'facil'])],
    'Climate/Risk':      [c for c in X_all.columns if any(k in c.lower() for k in ['wildfire', 'fire', 'drought', 'heat', 'climate', 'carbonplan', 'fod', 'mtbs', 'nifc', 'usfs', 'usgs', 'spi', 'pmdi', 'usdm', 'wf_'])],
    'Spatial/KNN':       [c for c in X_all.columns if 'knn' in c.lower()],
    'County/Regional':   [c for c in X_all.columns if 'county' in c.lower()],
}

print(f"\n  {'Category':<25} {'N Feats':>8} {'Trib Σ SHAP':>12} {'NonT Σ SHAP':>12} {'Diff':>10} {'% Bias':>8}")
print("  " + "-" * 80)

category_results = []
for cat_name, cat_features in categories.items():
    if not cat_features:
        continue
    # Get indices of features in this category
    feat_indices = [list(X_all.columns).index(f) for f in cat_features if f in X_all.columns]
    if not feat_indices:
        continue
    
    # Sum of mean SHAP values for this category
    tribal_cat_shap  = tribal_shap_mean[feat_indices].sum()
    nontrib_cat_shap = non_tribal_shap_mean[feat_indices].sum()
    cat_diff = tribal_cat_shap - nontrib_cat_shap
    cat_pct = cat_diff / total_bias * 100 if total_bias != 0 else 0
    
    print(f"  {cat_name:<25} {len(feat_indices):>8} {tribal_cat_shap:>12.6f} {nontrib_cat_shap:>12.6f} {cat_diff:>10.6f} {cat_pct:>7.2f}%")
    category_results.append({
        'category': cat_name,
        'n_features': len(feat_indices),
        'tribal_sum_shap': float(tribal_cat_shap),
        'non_tribal_sum_shap': float(nontrib_cat_shap),
        'difference': float(cat_diff),
        'pct_of_bias': float(cat_pct),
    })

results['shap_decomposition']['category_decomposition'] = category_results

# ── 2h. SHAP value distribution comparison for top features ──
print("\n" + "-" * 80)
print("[2h] SHAP DISTRIBUTION COMPARISON: Top 5 bias-driving features")
print("-" * 80)

top5_features = decomposition.head(5)['feature'].values
for feat_name in top5_features:
    feat_idx = list(X_all.columns).index(feat_name)
    tribal_shap_vals  = shap_values[tribal_mask_all, feat_idx]
    nontrib_shap_vals = shap_values[~tribal_mask_all, feat_idx]
    
    print(f"\n  {feat_name}:")
    print(f"    Tribal     SHAP: mean={tribal_shap_vals.mean():.6f}, median={np.median(tribal_shap_vals):.6f}, std={tribal_shap_vals.std():.6f}")
    print(f"    Non-tribal SHAP: mean={nontrib_shap_vals.mean():.6f}, median={np.median(nontrib_shap_vals):.6f}, std={nontrib_shap_vals.std():.6f}")
    print(f"    → This feature makes tribal predictions {abs(tribal_shap_vals.mean() - nontrib_shap_vals.mean()):.6f} more negative")

# ── 2i. Cumulative SHAP bias explanation ──
print("\n" + "-" * 80)
print("[2i] CUMULATIVE BIAS EXPLANATION: How many features explain 90% of bias?")
print("-" * 80)

# Sort by absolute contribution to bias
decomposition_sorted = decomposition.reindex(
    decomposition['shap_difference'].abs().sort_values(ascending=False).index
)

cumulative_pct = 0
n_for_90 = 0
cumulative_list = []
for i, (_, row) in enumerate(decomposition_sorted.iterrows()):
    abs_pct = abs(row['pct_of_total_bias'])
    cumulative_pct += abs_pct
    cumulative_list.append({
        'feature': row['feature'],
        'abs_pct_bias': abs_pct,
        'cumulative_pct': cumulative_pct,
    })
    if cumulative_pct >= 90 and n_for_90 == 0:
        n_for_90 = i + 1

print(f"\n  {'Rank':<5} {'Feature':<45} {'| % Bias |':>10} {'Cumulative %':>13}")
print("  " + "-" * 75)
for i, item in enumerate(cumulative_list[:15]):
    print(f"  {i+1:<5} {item['feature']:<45} {item['abs_pct_bias']:>9.2f}% {item['cumulative_pct']:>12.2f}%")

print(f"\n  → {n_for_90} features explain ≥90% of the tribal bias (out of {len(decomposition)} total)")
results['shap_decomposition']['n_features_for_90pct_bias'] = n_for_90

# ── 2j. SHAP-based bias decomposition table (for API) ──
print("\n" + "-" * 80)
print("[2j] FINAL SHAP BIAS DECOMPOSITION TABLE (for Bias Scoring API)")
print("-" * 80)

api_table = decomposition_sorted.head(15)[['feature', 'tribal_mean_shap', 'non_tribal_mean_shap', 
                                             'shap_difference', 'pct_of_total_bias',
                                             'tribal_mean_abs_shap', 'non_tribal_mean_abs_shap']].copy()
api_table = api_table.round(6)

print(f"\n  Feature name | Tribal mean SHAP | Non-tribal mean SHAP | Difference | % of total bias")
print("  " + "-" * 95)
for _, row in api_table.iterrows():
    flag = " ⚠️" if abs(row['pct_of_total_bias']) > 5 else ""
    print(f"  {row['feature']:<40} {row['tribal_mean_shap']:>10.6f} {row['non_tribal_mean_shap']:>10.6f} {row['shap_difference']:>10.6f} {row['pct_of_total_bias']:>8.2f}%{flag}")

# Convert for JSON
api_table_json = []
for _, row in api_table.iterrows():
    api_table_json.append({
        'feature': row['feature'],
        'tribal_mean_shap': float(row['tribal_mean_shap']),
        'non_tribal_mean_shap': float(row['non_tribal_mean_shap']),
        'shap_difference': float(row['shap_difference']),
        'pct_of_total_bias': float(row['pct_of_total_bias']),
    })

results['shap_decomposition']['api_table'] = api_table_json

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED INSIGHT: Stale Score × Tribal Interaction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMBINED INSIGHT: Temporal Decay × Tribal Bias Interaction")
print("=" * 80)

# Does stale data disproportionately affect tribal tracts?
# If tribal tracts have both staler data AND higher gaps, part of the tribal
# bias may be a data collection artifact, not real infrastructure disparity.

tribal_stale_mean = df.loc[df['is_tribal'], 'stale_source_score'].mean()
nontrib_stale_mean = df.loc[~df['is_tribal'], 'stale_source_score'].mean()
tribal_gap_mean = df.loc[df['is_tribal'], 'gap_only'].mean()
nontrib_gap_mean = df.loc[~df['is_tribal'], 'gap_only'].mean()

print(f"\n  Tribal tracts:     stale={tribal_stale_mean:.4f}, gap_only={tribal_gap_mean:.6f}")
print(f"  Non-tribal tracts: stale={nontrib_stale_mean:.4f}, gap_only={nontrib_gap_mean:.6f}")
print(f"  Stale ratio (T/NT): {tribal_stale_mean/nontrib_stale_mean:.4f}")
print(f"  Gap ratio (T/NT):   {tribal_gap_mean/nontrib_gap_mean:.4f}")

# Compute: how much of the tribal gap disparity is attributable to stale data?
# Simple mediation estimate: if stale score explains X% of gap_only variance,
# and tribal tracts are Y% staler, then approximately X*Y% of the tribal gap
# disparity could be a data artifact
stale_gap_r2 = pearson_r**2  # from Part 1
stale_disparity_ratio = (tribal_stale_mean - nontrib_stale_mean) / nontrib_stale_mean
gap_disparity = tribal_gap_mean - nontrib_gap_mean
mediation_estimate = stale_gap_r2 * abs(stale_disparity_ratio) * 100

print(f"\n  Stale→Gap r²:                     {stale_gap_r2:.6f} ({stale_gap_r2*100:.4f}%)")
print(f"  Stale disparity ratio (T-NT)/NT:  {stale_disparity_ratio:.4f} ({stale_disparity_ratio*100:.2f}%)")
print(f"  Gap disparity (T-NT):             {gap_disparity:.6f}")
print(f"  Estimated data-artifact component: ~{mediation_estimate:.4f}% of tribal gap disparity")
print(f"  → {'SIGNIFICANT data artifact' if mediation_estimate > 1 else 'Modest data artifact' if mediation_estimate > 0.1 else 'Small data artifact'}: part of tribal bias may stem from stale source data")

results['combined_insight'] = {
    'stale_gap_r_squared': float(stale_gap_r2),
    'stale_disparity_ratio': float(stale_disparity_ratio),
    'gap_disparity': float(gap_disparity),
    'mediation_estimate_pct': float(mediation_estimate),
    'tribal_stale_mean': float(tribal_stale_mean),
    'non_tribal_stale_mean': float(nontrib_stale_mean),
    'tribal_gap_mean': float(tribal_gap_mean),
    'non_tribal_gap_mean': float(nontrib_gap_mean),
}

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save full decomposition table as CSV
decomposition_output = decomposition_sorted.copy()
decomposition_output.to_csv(OUT / 'shap_bias_decomposition.csv', index=False)
print(f"  Saved: shap_bias_decomposition.csv ({len(decomposition_output)} rows)")

# Save JSON results
with open(OUT / 'advanced_bias_features_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: advanced_bias_features_results.json")

elapsed = time.time() - t0
print(f"\n  Elapsed: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ PART 1: TEMPORAL DECAY WEIGHTING                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ stale_source_score ↔ gap_only:                                             │
│   Pearson  r = {pearson_r:.6f}  (p = {pearson_p:.2e})                                  │
│   Spearman r = {spearman_r:.6f}  (p = {spearman_p:.2e})                                  │
│   → Stale data {'IS' if pearson_p < 0.05 else 'is NOT'} significantly correlated with gaps              │
│                                                                             │
│ Tribal stale score:  {tribal_stale.mean():.6f}  vs  Non-tribal: {non_tribal_stale.mean():.6f}              │
│   → Tribal data is {'STALER' if tribal_stale.mean() > non_tribal_stale.mean() else 'FRESHER'} (ratio: {tribal_stale.mean()/non_tribal_stale.mean():.4f})                      │
│                                                                             │
│ Rural stale score:  {rural_stale.mean():.6f}  vs  Urban:    {urban_stale.mean():.6f}              │
│   → Rural data is {'STALER' if rural_stale.mean() > urban_stale.mean() else 'FRESHER'} (ratio: {rural_stale.mean()/urban_stale.mean():.4f})                       │
│                                                                             │
│ Stale→Residual: r = {pr_resid:.6f}, r² = {pr_resid**2*100:.4f}% of OOF residual variance           │
│ Stale gap gradient (Q4-Q1): {q4_gap - q1_gap:.6f}                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PART 2: SHAP TRIBAL BIAS DECOMPOSITION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Total tribal bias (pred diff): {total_bias:.6f}                                  │
│ Tribal bias ratio: {tribal_pred_mean/non_tribal_pred_mean:.4f}                                          │
│ {n_for_90} features explain ≥90% of tribal bias                             │
│                                                                             │
│ Top 5 features driving tribal bias (most negative SHAP diff):""")

for i, (_, row) in enumerate(decomposition.head(5).iterrows()):
    print(f"│   {i+1}. {row['feature']:<38} diff={row['shap_difference']:>8.6f} ({row['pct_of_total_bias']:>6.2f}%)")

print(f"""│                                                                             │
│ Key insight: The tribal bias decomposes into feature-level                │
│ contributions. The Bias Scoring API should flag features with             │
│ |pct_of_total_bias| > 5% as bias-driving.                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ COMBINED: Data artifact component ≈ {mediation_estimate:.4f}% of tribal gap disparity        │
│ → Part of observed tribal bias may stem from stale/mapping data           │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("ANALYSIS COMPLETE.")
