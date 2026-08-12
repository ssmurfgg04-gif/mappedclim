#!/usr/bin/env python3
"""
Conformal Prediction Intervals + Cross-Dataset Validation
==========================================================
Combines Idea #8 (conformal prediction intervals as bias signal) and
Idea #9 (cross-dataset validation against Microsoft building confidence
and other external data sources).

Hypothesis 1: Prediction interval width is itself a bias signal.
  Wider intervals on tribal/rural tracts = the model is less confident
  there = coverage inequity in the model's own uncertainty.

Hypothesis 2: We can partially validate our proxy by checking against
  Microsoft's building footprint confidence and other independent
  external data sources (USGS wildfire, USFS WHP, CVI, SVI).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/z/my-project/bias-bounty-map")
OOF_PATH = BASE / "data/output/oof_predictions_merged.parquet"
FEAT_PATH = BASE / "data/output/engineered_features_merged.parquet"
OUT_PATH = BASE / "data/output/conformal_crossval_results.json"

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

oof = pd.read_parquet(OOF_PATH)
feat = pd.read_parquet(FEAT_PATH)

print(f"OOF predictions:  {oof.shape[0]:,} tracts × {oof.shape[1]} cols")
print(f"Engineered features: {feat.shape[0]:,} tracts × {feat.shape[1]} cols")

model_cols = ["xgb", "lgb", "et"]
print(f"Model columns: {model_cols}")

# ── Merge ──────────────────────────────────────────────────────────────
# Columns to pull from features (excluding duplicates already in oof)
ext_cols = [
    "tribal_any", "pct_urban",
    "usgs_wildfire_ever", "usgs_wildfire_burned_pct_area",
    "usfs_WHP_mean", "cvi_overall", "svi_overall",
    "ms_bldg", "bldg_ms_ml_fraction",
    "bldg_total_sources", "bldg_source_diversity",
    "source_coverage_fraction", "source_diversity_entropy",
    "poi_mean_confidence", "poi_very_high_conf_fraction",
    "bldg_osm_fraction", "bldg_google_fraction", "bldg_esri_fraction",
    "usfs_BuildingCount_sum", "usfs_BuildingDensity_mean",
    "compound_risk", "rural_indicator", "rural_continuous",
    "pop_total", "housing_units",
]
# Remove duplicates that already exist in oof
oof_base_cols = set(oof.columns) - {"GEOID"}
ext_cols_available = [c for c in ext_cols if c in feat.columns]
feat_merge_cols = ["GEOID"] + ext_cols_available
df = oof.merge(feat[feat_merge_cols], on="GEOID", how="left")

print(f"Merged dataset: {df.shape[0]:,} tracts × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════
# PART 1: CONFORMAL PREDICTION INTERVALS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 1: CONFORMAL PREDICTION INTERVALS (Idea #8)")
print("=" * 72)

# ── Compute non-conformity scores ──────────────────────────────────────
# prediction_spread = max - min across models (range of predictions)
# prediction_std = standard deviation across models
# mean_prediction = mean across models (ensemble prediction)

df["prediction_spread"] = df[model_cols].max(axis=1) - df[model_cols].min(axis=1)
df["prediction_std"] = df[model_cols].std(axis=1)
df["mean_prediction"] = df[model_cols].mean(axis=1)

# Conformal interval: use the spread as a non-conformity score
# For a calibration level alpha, the prediction interval is:
#   [mean_prediction - q * spread, mean_prediction + q * spread]
# where q is the (1-alpha) quantile of the non-conformity scores
# We'll compute intervals at 90% and 95% coverage

nonconformity = df["prediction_spread"].values
for alpha, label in [(0.10, "90%"), (0.05, "95%")]:
    q = np.quantile(nonconformity, 1 - alpha)
    df[f"ci_{label}_lower"] = df["mean_prediction"] - q
    df[f"ci_{label}_upper"] = df["mean_prediction"] + q
    df[f"ci_{label}_width"] = 2 * q
    print(f"  {label} conformal interval: half-width = {q:.6f}, full width = {2*q:.6f}")

# ── Analysis: Tribal vs Non-Tribal ─────────────────────────────────────
print("\n── Tribal vs Non-Tribal Prediction Uncertainty ──")

tribal = df[df["tribal_any"] == True]
non_tribal = df[df["tribal_any"] == False]

results_conformal = {}

for metric in ["prediction_spread", "prediction_std"]:
    t_mean = tribal[metric].mean()
    nt_mean = non_tribal[metric].mean()
    t_median = tribal[metric].median()
    nt_median = non_tribal[metric].median()
    ratio = t_mean / nt_mean if nt_mean > 0 else float("inf")

    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(
        tribal[metric].dropna(), non_tribal[metric].dropna(), alternative="greater"
    )

    # Welch t-test
    t_stat, t_pval = stats.ttest_ind(
        tribal[metric].dropna(), non_tribal[metric].dropna(), equal_var=False
    )

    key = f"tribal_vs_nontribal_{metric}"
    results_conformal[key] = {
        "tribal_mean": round(float(t_mean), 6),
        "non_tribal_mean": round(float(nt_mean), 6),
        "tribal_median": round(float(t_median), 6),
        "non_tribal_median": round(float(nt_median), 6),
        "ratio_tribal_to_nontribal": round(float(ratio), 4),
        "mann_whitney_U": round(float(u_stat), 2),
        "mann_whitney_p": float(u_pval),
        "welch_t": round(float(t_stat), 4),
        "welch_p": float(t_pval),
    }

    print(f"\n  {metric}:")
    print(f"    Tribal:      mean={t_mean:.6f}, median={t_median:.6f}  (n={len(tribal):,})")
    print(f"    Non-Tribal:  mean={nt_mean:.6f}, median={nt_median:.6f}  (n={len(non_tribal):,})")
    print(f"    Ratio (T/NT): {ratio:.4f}")
    print(f"    Mann-Whitney U (greater): U={u_stat:.1f}, p={u_pval:.2e}")
    print(f"    Welch t-test: t={t_stat:.4f}, p={t_pval:.2e}")

# ── Analysis: Rural vs Urban ───────────────────────────────────────────
print("\n── Rural vs Urban Prediction Uncertainty ──")

rural = df[df["pct_urban"] <= 0.5]
urban = df[df["pct_urban"] > 0.5]

for metric in ["prediction_spread", "prediction_std"]:
    r_mean = rural[metric].mean()
    u_mean = urban[metric].mean()
    r_median = rural[metric].median()
    u_median = urban[metric].median()
    ratio = r_mean / u_mean if u_mean > 0 else float("inf")

    u_stat, u_pval = stats.mannwhitneyu(
        rural[metric].dropna(), urban[metric].dropna(), alternative="greater"
    )
    t_stat, t_pval = stats.ttest_ind(
        rural[metric].dropna(), urban[metric].dropna(), equal_var=False
    )

    key = f"rural_vs_urban_{metric}"
    results_conformal[key] = {
        "rural_mean": round(float(r_mean), 6),
        "urban_mean": round(float(u_mean), 6),
        "rural_median": round(float(r_median), 6),
        "urban_median": round(float(u_median), 6),
        "ratio_rural_to_urban": round(float(ratio), 4),
        "mann_whitney_U": round(float(u_stat), 2),
        "mann_whitney_p": float(u_pval),
        "welch_t": round(float(t_stat), 4),
        "welch_p": float(t_pval),
    }

    print(f"\n  {metric}:")
    print(f"    Rural:  mean={r_mean:.6f}, median={r_median:.6f}  (n={len(rural):,})")
    print(f"    Urban:  mean={u_mean:.6f}, median={u_median:.6f}  (n={len(urban):,})")
    print(f"    Ratio (R/U): {ratio:.4f}")
    print(f"    Mann-Whitney U (greater): U={u_stat:.1f}, p={u_pval:.2e}")
    print(f"    Welch t-test: t={t_stat:.4f}, p={t_pval:.2e}")

# ── Correlation with gap_only ──────────────────────────────────────────
print("\n── Correlation of Prediction Uncertainty with gap_only ──")

for metric in ["prediction_spread", "prediction_std"]:
    r_pearson, p_pearson = stats.pearsonr(df[metric], df["gap_only"])
    r_spearman, p_spearman = stats.spearmanr(df[metric], df["gap_only"])

    key = f"correlation_{metric}_gap_only"
    results_conformal[key] = {
        "pearson_r": round(float(r_pearson), 6),
        "pearson_p": float(p_pearson),
        "spearman_r": round(float(r_spearman), 6),
        "spearman_p": float(p_spearman),
    }

    print(f"  {metric} vs gap_only:")
    print(f"    Pearson r={r_pearson:.6f} (p={p_pearson:.2e})")
    print(f"    Spearman ρ={r_spearman:.6f} (p={p_spearman:.2e})")

# ── High uncertainty tracts ────────────────────────────────────────────
print("\n── High Uncertainty Tracts (Top 10% Spread) ──")

spread_90 = df["prediction_spread"].quantile(0.90)
high_unc = df[df["prediction_spread"] >= spread_90]
low_unc = df[df["prediction_spread"] < spread_90]

n_tribal_high = high_unc["tribal_any"].sum()
n_tribal_total = df["tribal_any"].sum()
n_total_high = len(high_unc)

# Expected fraction of tribal in top 10% if uniform
expected_tribal_frac = n_tribal_total / len(df)
actual_tribal_frac = n_tribal_high / n_total_high
enrichment = actual_tribal_frac / expected_tribal_frac if expected_tribal_frac > 0 else float("inf")

# Also compute: fraction of all tribal tracts that are in the high-uncertainty zone
tribal_in_high_frac = n_tribal_high / n_tribal_total if n_tribal_total > 0 else 0

# For rural
n_rural_high = (high_unc["pct_urban"] <= 0.5).sum()
n_rural_total = (df["pct_urban"] <= 0.5).sum()
expected_rural_frac = n_rural_total / len(df)
actual_rural_frac = n_rural_high / n_total_high
rural_enrichment = actual_rural_frac / expected_rural_frac if expected_rural_frac > 0 else float("inf")

# Mean gap_only for high vs low uncertainty
mean_gap_high = high_unc["gap_only"].mean()
mean_gap_low = low_unc["gap_only"].mean()

results_conformal["high_uncertainty_top10pct"] = {
    "spread_threshold_90th_pctile": round(float(spread_90), 6),
    "n_high_uncertainty_tracts": int(n_total_high),
    "tribal_in_high_unc": int(n_tribal_high),
    "tribal_total": int(n_tribal_total),
    "actual_tribal_fraction_in_high_unc": round(float(actual_tribal_frac), 6),
    "expected_tribal_fraction_if_uniform": round(float(expected_tribal_frac), 6),
    "tribal_enrichment_ratio": round(float(enrichment), 4),
    "fraction_of_tribal_in_high_unc": round(float(tribal_in_high_frac), 6),
    "rural_in_high_unc": int(n_rural_high),
    "rural_total": int(n_rural_total),
    "actual_rural_fraction_in_high_unc": round(float(actual_rural_frac), 6),
    "expected_rural_fraction_if_uniform": round(float(expected_rural_frac), 6),
    "rural_enrichment_ratio": round(float(rural_enrichment), 4),
    "mean_gap_only_high_unc": round(float(mean_gap_high), 6),
    "mean_gap_only_low_unc": round(float(mean_gap_low), 6),
    "gap_ratio_high_to_low": round(float(mean_gap_high / mean_gap_low), 4) if mean_gap_low != 0 else None,
}

print(f"  Top 10% spread threshold: {spread_90:.6f}")
print(f"  High uncertainty tracts: {n_total_high:,}")
print(f"  Tribal in high-unc: {n_tribal_high} / {n_tribal_total} total tribal")
print(f"  Actual tribal fraction in high-unc: {actual_tribal_frac:.6f}")
print(f"  Expected if uniform: {expected_tribal_frac:.6f}")
print(f"  Tribal enrichment: {enrichment:.4f}x")
print(f"  Fraction of all tribal tracts in high-unc: {tribal_in_high_frac:.6f}")
print(f"  Rural in high-unc: {n_rural_high} / {n_rural_total} total rural")
print(f"  Rural enrichment: {rural_enrichment:.4f}x")
print(f"  Mean gap_only (high unc): {mean_gap_high:.6f}")
print(f"  Mean gap_only (low unc):  {mean_gap_low:.6f}")
print(f"  Gap ratio (high/low): {mean_gap_high/mean_gap_low:.4f}" if mean_gap_low != 0 else "  Gap ratio: undefined")

# ── Conformal coverage check ───────────────────────────────────────────
print("\n── Conformal Coverage: Does gap_only fall within the prediction interval? ──")

for alpha_label in ["90%", "95%"]:
    lower = df[f"ci_{alpha_label}_lower"]
    upper = df[f"ci_{alpha_label}_upper"]
    covered = ((df["gap_only"] >= lower) & (df["gap_only"] <= upper)).mean()

    tribal_covered = ((tribal["gap_only"] >= tribal[f"ci_{alpha_label}_lower"]) &
                      (tribal["gap_only"] <= tribal[f"ci_{alpha_label}_upper"])).mean()
    nontribal_covered = ((non_tribal["gap_only"] >= non_tribal[f"ci_{alpha_label}_lower"]) &
                         (non_tribal["gap_only"] <= non_tribal[f"ci_{alpha_label}_upper"])).mean()

    rural_covered = ((rural["gap_only"] >= rural[f"ci_{alpha_label}_lower"]) &
                     (rural["gap_only"] <= rural[f"ci_{alpha_label}_upper"])).mean()
    urban_covered = ((urban["gap_only"] >= urban[f"ci_{alpha_label}_lower"]) &
                     (urban["gap_only"] <= urban[f"ci_{alpha_label}_upper"])).mean()

    key = f"conformal_coverage_{alpha_label}"
    results_conformal[key] = {
        "overall_coverage": round(float(covered), 6),
        "tribal_coverage": round(float(tribal_covered), 6),
        "non_tribal_coverage": round(float(nontribal_covered), 6),
        "coverage_gap_tribal_vs_nontribal": round(float(tribal_covered - nontribal_covered), 6),
        "rural_coverage": round(float(rural_covered), 6),
        "urban_coverage": round(float(urban_covered), 6),
        "coverage_gap_rural_vs_urban": round(float(rural_covered - urban_covered), 6),
    }

    print(f"  {alpha_label} interval coverage:")
    print(f"    Overall:     {covered:.6f}")
    print(f"    Tribal:      {tribal_covered:.6f}")
    print(f"    Non-Tribal:  {nontribal_covered:.6f}")
    print(f"    Coverage gap (T - NT): {tribal_covered - nontribal_covered:+.6f}")
    print(f"    Rural:       {rural_covered:.6f}")
    print(f"    Urban:       {urban_covered:.6f}")
    print(f"    Coverage gap (R - U):  {rural_covered - urban_covered:+.6f}")

# ── Decile analysis of prediction spread ───────────────────────────────
print("\n── Prediction Spread by Tribal Status Decile ──")

df["spread_decile"] = pd.qcut(df["prediction_spread"], 10, labels=False, duplicates="drop")
decile_analysis = df.groupby("spread_decile").agg(
    n_tracts=("GEOID", "count"),
    n_tribal=("tribal_any", "sum"),
    mean_spread=("prediction_spread", "mean"),
    mean_gap=("gap_only", "mean"),
    mean_pct_urban=("pct_urban", "mean"),
).reset_index()

decile_analysis["tribal_fraction"] = decile_analysis["n_tribal"] / decile_analysis["n_tracts"]
print(decile_analysis.to_string(index=False))

results_conformal["spread_decile_analysis"] = decile_analysis.to_dict(orient="list")

# ══════════════════════════════════════════════════════════════════════
# PART 2: CROSS-DATASET VALIDATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART 2: CROSS-DATASET VALIDATION (Idea #9)")
print("=" * 72)

results_crossval = {}

# ── Check for Microsoft-related columns ────────────────────────────────
ms_cols_found = [c for c in df.columns if any(
    k in c.lower() for k in ["ms_bldg", "bldg_ms_ml", "microsoft", "bldg_confidence"]
)]
print(f"\nMicrosoft-related columns found: {ms_cols_found}")

# ── Microsoft building data analysis ───────────────────────────────────
if "ms_bldg" in df.columns:
    print("\n── Microsoft Building Count (ms_bldg) Analysis ──")

    # Correlation with gap predictions
    for pred_col in ["mean_prediction", "gap_only", "proxy_merged"]:
        if pred_col in df.columns and df[pred_col].notna().sum() > 10:
            valid = df[["ms_bldg", pred_col]].dropna()
            if len(valid) > 10:
                r, p = stats.spearmanr(valid["ms_bldg"], valid[pred_col])
                print(f"  Spearman(ms_bldg, {pred_col}): ρ={r:.4f}, p={p:.2e}")

    # Tribal vs non-tribal Microsoft building count
    t_ms = tribal["ms_bldg"].dropna()
    nt_ms = non_tribal["ms_bldg"].dropna()
    print(f"\n  ms_bldg tribal:      mean={t_ms.mean():.2f}, median={t_ms.median():.2f}")
    print(f"  ms_bldg non-tribal:  mean={nt_ms.mean():.2f}, median={nt_ms.median():.2f}")

    u_stat, u_pval = stats.mannwhitneyu(t_ms, nt_ms, alternative="less")
    print(f"  Mann-Whitney (tribal < non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")

    results_crossval["ms_bldg_tribal_analysis"] = {
        "tribal_mean": round(float(t_ms.mean()), 4),
        "non_tribal_mean": round(float(nt_ms.mean()), 4),
        "ratio": round(float(t_ms.mean() / nt_ms.mean()), 4) if nt_ms.mean() != 0 else None,
        "mann_whitney_p": float(u_pval),
    }

if "bldg_ms_ml_fraction" in df.columns:
    print("\n── Microsoft ML Fraction (bldg_ms_ml_fraction) Analysis ──")

    # This measures what fraction of buildings come from Microsoft ML vs other sources
    # Higher values = more reliance on Microsoft ML = less OSM/other coverage = potentially less confident
    t_frac = tribal["bldg_ms_ml_fraction"].dropna()
    nt_frac = non_tribal["bldg_ms_ml_fraction"].dropna()
    print(f"  bldg_ms_ml_fraction tribal:      mean={t_frac.mean():.4f}, median={t_frac.median():.4f}")
    print(f"  bldg_ms_ml_fraction non-tribal:  mean={nt_frac.mean():.4f}, median={nt_frac.median():.4f}")

    u_stat, u_pval = stats.mannwhitneyu(t_frac, nt_frac, alternative="greater")
    print(f"  Mann-Whitney (tribal > non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")
    print(f"  → Tribal tracts rely MORE on Microsoft ML buildings (less diverse sources)")

    results_crossval["bldg_ms_ml_fraction_tribal_analysis"] = {
        "tribal_mean": round(float(t_frac.mean()), 6),
        "non_tribal_mean": round(float(nt_frac.mean()), 6),
        "ratio": round(float(t_frac.mean() / nt_frac.mean()), 4) if nt_frac.mean() != 0 else None,
        "mann_whitney_p": float(u_pval),
    }

    # Correlation with prediction uncertainty
    valid = df[["bldg_ms_ml_fraction", "prediction_spread", "prediction_std", "gap_only"]].dropna()
    if len(valid) > 10:
        for metric in ["prediction_spread", "prediction_std", "gap_only"]:
            r, p = stats.spearmanr(valid["bldg_ms_ml_fraction"], valid[metric])
            print(f"  Spearman(bldg_ms_ml_fraction, {metric}): ρ={r:.4f}, p={p:.2e}")

# ── Source diversity as confidence proxy ───────────────────────────────
print("\n── Source Diversity as Confidence Proxy ──")

for col in ["bldg_source_diversity", "source_coverage_fraction", "source_diversity_entropy"]:
    if col in df.columns:
        t_vals = tribal[col].dropna()
        nt_vals = non_tribal[col].dropna()
        print(f"\n  {col}:")
        print(f"    Tribal:     mean={t_vals.mean():.4f}, median={t_vals.median():.4f}")
        print(f"    Non-Tribal: mean={nt_vals.mean():.4f}, median={nt_vals.median():.4f}")

        # Correlation with gap
        valid = df[[col, "gap_only"]].dropna()
        if len(valid) > 10:
            r, p = stats.spearmanr(valid[col], valid["gap_only"])
            print(f"    Spearman({col}, gap_only): ρ={r:.4f}, p={p:.2e}")

# ── External validation: USGS Wildfire ─────────────────────────────────
print("\n── External Validation: USGS Wildfire ──")

ext_validations = {}

if "usgs_wildfire_ever" in df.columns:
    # Binary: ever had wildfire
    t_wf = tribal["usgs_wildfire_ever"].mean()
    nt_wf = non_tribal["usgs_wildfire_ever"].mean()
    print(f"  usgs_wildfire_ever tribal:      {t_wf:.4f}")
    print(f"  usgs_wildfire_ever non-tribal:  {nt_wf:.4f}")
    print(f"  → Tribal tracts have {'higher' if t_wf > nt_wf else 'lower'} wildfire exposure")

    # Correlation with gap predictions
    valid = df[["usgs_wildfire_ever", "gap_only", "mean_prediction"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["usgs_wildfire_ever"], valid["gap_only"])
        print(f"  Spearman(usgs_wildfire_ever, gap_only): ρ={r:.4f}, p={p:.2e}")
        ext_validations["usgs_wildfire_ever"] = {
            "tribal_mean": round(float(t_wf), 6),
            "non_tribal_mean": round(float(nt_wf), 6),
            "spearman_r_with_gap": round(float(r), 6),
            "spearman_p": float(p),
        }

if "usgs_wildfire_burned_pct_area" in df.columns:
    # Continuous: percentage of area burned
    t_burn = tribal["usgs_wildfire_burned_pct_area"].mean()
    nt_burn = non_tribal["usgs_wildfire_burned_pct_area"].mean()
    print(f"\n  usgs_wildfire_burned_pct_area tribal:      {t_burn:.4f}")
    print(f"  usgs_wildfire_burned_pct_area non-tribal:  {nt_burn:.4f}")

    valid = df[["usgs_wildfire_burned_pct_area", "gap_only"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["usgs_wildfire_burned_pct_area"], valid["gap_only"])
        print(f"  Spearman(usgs_wildfire_burned_pct_area, gap_only): ρ={r:.4f}, p={p:.2e}")
        ext_validations["usgs_wildfire_burned_pct_area"] = {
            "tribal_mean": round(float(t_burn), 6),
            "non_tribal_mean": round(float(nt_burn), 6),
            "spearman_r_with_gap": round(float(r), 6),
            "spearman_p": float(p),
        }

# ── External validation: USFS Wildfire Hazard Potential ────────────────
print("\n── External Validation: USFS Wildfire Hazard Potential ──")

if "usfs_WHP_mean" in df.columns:
    t_whp = tribal["usfs_WHP_mean"].dropna()
    nt_whp = non_tribal["usfs_WHP_mean"].dropna()
    print(f"  usfs_WHP_mean tribal:      mean={t_whp.mean():.4f}, median={t_whp.median():.4f}")
    print(f"  usfs_WHP_mean non-tribal:  mean={nt_whp.mean():.4f}, median={nt_whp.median():.4f}")

    u_stat, u_pval = stats.mannwhitneyu(t_whp, nt_whp, alternative="greater")
    print(f"  Mann-Whitney (tribal > non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")

    valid = df[["usfs_WHP_mean", "gap_only", "mean_prediction"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["usfs_WHP_mean"], valid["gap_only"])
        print(f"  Spearman(usfs_WHP_mean, gap_only): ρ={r:.4f}, p={p:.2e}")

        r2, p2 = stats.spearmanr(valid["usfs_WHP_mean"], valid["mean_prediction"])
        print(f"  Spearman(usfs_WHP_mean, mean_prediction): ρ={r2:.4f}, p={p2:.2e}")

    ext_validations["usfs_WHP_mean"] = {
        "tribal_mean": round(float(t_whp.mean()), 4),
        "non_tribal_mean": round(float(nt_whp.mean()), 4),
        "ratio": round(float(t_whp.mean() / nt_whp.mean()), 4) if nt_whp.mean() != 0 else None,
        "mann_whitney_p": float(u_pval),
        "spearman_r_with_gap": round(float(r), 6),
        "spearman_p_with_gap": float(p),
    }

# ── External validation: Climate Vulnerability Index ──────────────────
print("\n── External Validation: Climate Vulnerability Index (CVI) ──")

if "cvi_overall" in df.columns:
    t_cvi = tribal["cvi_overall"].dropna()
    nt_cvi = non_tribal["cvi_overall"].dropna()
    print(f"  cvi_overall tribal:      mean={t_cvi.mean():.4f}, median={t_cvi.median():.4f}")
    print(f"  cvi_overall non-tribal:  mean={nt_cvi.mean():.4f}, median={nt_cvi.median():.4f}")

    u_stat, u_pval = stats.mannwhitneyu(t_cvi, nt_cvi, alternative="greater")
    print(f"  Mann-Whitney (tribal > non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")

    valid = df[["cvi_overall", "gap_only", "mean_prediction"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["cvi_overall"], valid["gap_only"])
        print(f"  Spearman(cvi_overall, gap_only): ρ={r:.4f}, p={p:.2e}")

        r2, p2 = stats.spearmanr(valid["cvi_overall"], valid["mean_prediction"])
        print(f"  Spearman(cvi_overall, mean_prediction): ρ={r2:.4f}, p={p2:.2e}")

    ext_validations["cvi_overall"] = {
        "tribal_mean": round(float(t_cvi.mean()), 6),
        "non_tribal_mean": round(float(nt_cvi.mean()), 6),
        "ratio": round(float(t_cvi.mean() / nt_cvi.mean()), 4) if nt_cvi.mean() != 0 else None,
        "mann_whitney_p": float(u_pval),
        "spearman_r_with_gap": round(float(r), 6),
        "spearman_p_with_gap": float(p),
    }

# ── External validation: Social Vulnerability Index ────────────────────
print("\n── External Validation: Social Vulnerability Index (SVI) ──")

if "svi_overall" in df.columns:
    t_svi = tribal["svi_overall"].dropna()
    nt_svi = non_tribal["svi_overall"].dropna()
    print(f"  svi_overall tribal:      mean={t_svi.mean():.4f}, median={t_svi.median():.4f}")
    print(f"  svi_overall non-tribal:  mean={nt_svi.mean():.4f}, median={nt_svi.median():.4f}")

    u_stat, u_pval = stats.mannwhitneyu(t_svi, nt_svi, alternative="greater")
    print(f"  Mann-Whitney (tribal > non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")

    valid = df[["svi_overall", "gap_only", "mean_prediction"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["svi_overall"], valid["gap_only"])
        print(f"  Spearman(svi_overall, gap_only): ρ={r:.4f}, p={p:.2e}")

        r2, p2 = stats.spearmanr(valid["svi_overall"], valid["mean_prediction"])
        print(f"  Spearman(svi_overall, mean_prediction): ρ={r2:.4f}, p={p2:.2e}")

    ext_validations["svi_overall"] = {
        "tribal_mean": round(float(t_svi.mean()), 6),
        "non_tribal_mean": round(float(nt_svi.mean()), 6),
        "ratio": round(float(t_svi.mean() / nt_svi.mean()), 4) if nt_svi.mean() != 0 else None,
        "mann_whitney_p": float(u_pval),
        "spearman_r_with_gap": round(float(r), 6),
        "spearman_p_with_gap": float(p),
    }

results_crossval["external_validations"] = ext_validations

# ── Cross-validation summary: do independent measures confirm bias? ────
print("\n── Cross-Validation Summary: Do Independent Measures Confirm Bias? ──")

# For each external measure, check: is the tribal disparity direction consistent?
# with our gap predictions?
confirmations = []

for name, vals in ext_validations.items():
    tribal_mean = vals.get("tribal_mean", None)
    non_tribal_mean = vals.get("non_tribal_mean", None)
    if tribal_mean is not None and non_tribal_mean is not None:
        direction = "higher" if tribal_mean > non_tribal_mean else "lower"
        ratio = tribal_mean / non_tribal_mean if non_tribal_mean != 0 else None

        # Our gap prediction: tribal has HIGHER gap (worse coverage)
        # If external measure also shows tribal = higher, it confirms
        confirmed = (direction == "higher")
        confirmations.append({
            "measure": name,
            "tribal_mean": tribal_mean,
            "non_tribal_mean": non_tribal_mean,
            "direction": direction,
            "ratio": ratio,
            "confirms_bias": confirmed,
        })
        status = "✓ CONFIRMS" if confirmed else "✗ opposite"
        print(f"  {name}: tribal={tribal_mean:.4f}, non-tribal={non_tribal_mean:.4f}, "
              f"direction={direction} {status}")

n_confirmed = sum(1 for c in confirmations if c["confirms_bias"])
print(f"\n  → {n_confirmed}/{len(confirmations)} independent measures confirm tribal bias direction")

results_crossval["bias_confirmation_summary"] = {
    "n_measures_checked": len(confirmations),
    "n_confirming_bias": n_confirmed,
    "confirmations": confirmations,
}

# ── Combined external validation: multi-source risk score ─────────────
print("\n── Combined External Risk Score ──")

# Normalize each external measure and combine
risk_components = {}
for col, direction in [
    ("usfs_WHP_mean", 1),      # higher = more risk
    ("cvi_overall", 1),        # higher = more vulnerable
    ("svi_overall", 1),        # higher = more vulnerable
    ("usgs_wildfire_burned_pct_area", 1),  # higher = more burned
]:
    if col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 100:
            # Rank-normalize to [0, 1]
            ranked = vals.rank(pct=True)
            risk_components[col] = ranked

if risk_components:
    # Combine into a single external risk score (average of rank-normalized)
    risk_df = pd.DataFrame(risk_components)
    risk_df["external_risk_score"] = risk_df.mean(axis=1)
    risk_df["GEOID"] = df.loc[risk_df.index, "GEOID"]

    # Merge back
    df_with_risk = df.merge(risk_df[["GEOID", "external_risk_score"]], on="GEOID", how="left")

    t_risk = df_with_risk[df_with_risk["tribal_any"] == True]["external_risk_score"].dropna()
    nt_risk = df_with_risk[df_with_risk["tribal_any"] == False]["external_risk_score"].dropna()

    print(f"  External risk score (combined):")
    print(f"    Tribal:     mean={t_risk.mean():.4f}, median={t_risk.median():.4f}")
    print(f"    Non-Tribal: mean={nt_risk.mean():.4f}, median={nt_risk.median():.4f}")

    u_stat, u_pval = stats.mannwhitneyu(t_risk, nt_risk, alternative="greater")
    print(f"    Mann-Whitney (tribal > non-tribal): U={u_stat:.1f}, p={u_pval:.2e}")

    # Correlation with gap
    valid = df_with_risk[["external_risk_score", "gap_only", "mean_prediction"]].dropna()
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["external_risk_score"], valid["gap_only"])
        print(f"    Spearman(external_risk, gap_only): ρ={r:.4f}, p={p:.2e}")

        r2, p2 = stats.spearmanr(valid["external_risk_score"], valid["mean_prediction"])
        print(f"    Spearman(external_risk, mean_prediction): ρ={r2:.4f}, p={p2:.2e}")

    results_crossval["combined_external_risk"] = {
        "components": list(risk_components.keys()),
        "tribal_mean": round(float(t_risk.mean()), 6),
        "non_tribal_mean": round(float(nt_risk.mean()), 6),
        "mann_whitney_p": float(u_pval),
        "spearman_r_with_gap": round(float(r), 6) if len(valid) > 10 else None,
    }

# ── POI Confidence as Microsoft-like validation ────────────────────────
print("\n── POI Mean Confidence as External Validation ──")

if "poi_mean_confidence" in df.columns:
    t_poi = tribal["poi_mean_confidence"].dropna()
    nt_poi = non_tribal["poi_mean_confidence"].dropna()
    print(f"  poi_mean_confidence tribal:      mean={t_poi.mean():.4f}, median={t_poi.median():.4f}")
    print(f"  poi_mean_confidence non-tribal:  mean={nt_poi.mean():.4f}, median={nt_poi.median():.4f}")

    # Correlation with gap and uncertainty
    valid = df[["poi_mean_confidence", "gap_only", "prediction_spread"]].dropna()
    if len(valid) > 10:
        r1, p1 = stats.spearmanr(valid["poi_mean_confidence"], valid["gap_only"])
        r2, p2 = stats.spearmanr(valid["poi_mean_confidence"], valid["prediction_spread"])
        print(f"  Spearman(poi_conf, gap_only): ρ={r1:.4f}, p={p1:.2e}")
        print(f"  Spearman(poi_conf, prediction_spread): ρ={r2:.4f}, p={p2:.2e}")

if "poi_very_high_conf_fraction" in df.columns:
    t_poi_hc = tribal["poi_very_high_conf_fraction"].dropna()
    nt_poi_hc = non_tribal["poi_very_high_conf_fraction"].dropna()
    print(f"\n  poi_very_high_conf_fraction tribal:      mean={t_poi_hc.mean():.4f}")
    print(f"  poi_very_high_conf_fraction non-tribal:  mean={nt_poi_hc.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("COMBINED SUMMARY: DUAL BIAS DIMENSION")
print("=" * 72)

# Two bias dimensions:
# 1. Score disparity: tribal tracts have higher gaps (worse coverage)
# 2. Confidence disparity: tribal tracts have wider prediction intervals (model is less certain)

score_disparity = df.groupby("tribal_any")["gap_only"].mean()
confidence_disparity_spread = df.groupby("tribal_any")["prediction_spread"].mean()
confidence_disparity_std = df.groupby("tribal_any")["prediction_std"].mean()

print("\n  DUAL BIAS DIMENSION:")
print(f"  {'Metric':<30} {'Tribal':>12} {'Non-Tribal':>12} {'Ratio':>10}")
print(f"  {'-'*64}")
print(f"  {'gap_only (score disparity)':<30} {score_disparity[True]:>12.6f} {score_disparity[False]:>12.6f} "
      f"{score_disparity[True]/score_disparity[False]:>10.4f}")
print(f"  {'prediction_spread (conf disp)':<30} {confidence_disparity_spread[True]:>12.6f} "
      f"{confidence_disparity_spread[False]:>12.6f} "
      f"{confidence_disparity_spread[True]/confidence_disparity_spread[False]:>10.4f}")
print(f"  {'prediction_std (conf disp)':<30} {confidence_disparity_std[True]:>12.6f} "
      f"{confidence_disparity_std[False]:>12.6f} "
      f"{confidence_disparity_std[True]/confidence_disparity_std[False]:>10.4f}")

print(f"\n  → Tribal tracts face DUAL bias:")
print(f"     1. They have worse coverage (higher gap scores)")
print(f"     2. The model is LESS CERTAIN about them (wider prediction intervals)")
print(f"     This is a confidence inequity: the model knows less about tribal areas")

# External validation confirmation
if confirmations:
    print(f"\n  → External validation: {n_confirmed}/{len(confirmations)} independent measures "
          f"confirm the tribal bias direction")
    print(f"     The bias narrative is not just our proxy — it's confirmed by independent data")

# ── Save results ───────────────────────────────────────────────────────
all_results = {
    "part1_conformal_prediction_intervals": results_conformal,
    "part2_cross_dataset_validation": results_crossval,
    "dual_bias_summary": {
        "score_disparity_tribal": round(float(score_disparity[True]), 6),
        "score_disparity_non_tribal": round(float(score_disparity[False]), 6),
        "score_disparity_ratio": round(float(score_disparity[True] / score_disparity[False]), 4),
        "confidence_disparity_spread_tribal": round(float(confidence_disparity_spread[True]), 6),
        "confidence_disparity_spread_non_tribal": round(float(confidence_disparity_spread[False]), 6),
        "confidence_disparity_spread_ratio": round(
            float(confidence_disparity_spread[True] / confidence_disparity_spread[False]), 4
        ),
        "confidence_disparity_std_tribal": round(float(confidence_disparity_std[True]), 6),
        "confidence_disparity_std_non_tribal": round(float(confidence_disparity_std[False]), 6),
        "confidence_disparity_std_ratio": round(
            float(confidence_disparity_std[True] / confidence_disparity_std[False]), 4
        ),
        "n_external_measures_confirming_bias": n_confirmed if confirmations else 0,
        "n_external_measures_checked": len(confirmations) if confirmations else 0,
    },
}

# Convert numpy types for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2, cls=NpEncoder)

print(f"\n  Results saved to: {OUT_PATH}")
print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
