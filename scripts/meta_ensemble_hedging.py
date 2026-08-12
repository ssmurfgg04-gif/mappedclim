#!/usr/bin/env python3
"""
Meta-Ensemble Hedging: Blend predictions from two machines (Deterministic vs Maximizing-luck)
to reduce variance without changing bias.

Machine 1 (Deterministic): 3-model ensemble (XGB + LGB + ET) → submission_merged.csv
Machine 2 (Maximizing-luck): 5-model ensemble (XGB + LGB + ET + CatBoost + DART) → submission_merged_v2.csv

Key insight: If the two ensembles disagree on a tract, that's uncertainty.
Meta-ensembling hedges that uncertainty away.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/home/z/my-project/bias-bounty-map/data/output"
DOWNLOAD = "/home/z/my-project/download"

SUB_V1 = os.path.join(BASE, "submission_merged.csv")
SUB_V2 = os.path.join(BASE, "submission_merged_v2.csv")
OOF_V1 = os.path.join(BASE, "oof_predictions_merged.parquet")
OOF_V2 = os.path.join(BASE, "oof_predictions_merged_v2.parquet")
FEATURES = os.path.join(BASE, "engineered_features_merged.parquet")

OUT_META = os.path.join(BASE, "submission_meta_ensemble.csv")
OUT_DOWNLOAD = os.path.join(DOWNLOAD, "submission_meta_ensemble.csv")


def load_data():
    """Load both submissions, OOF predictions, and feature metadata."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Submissions
    sub_v1 = pd.read_csv(SUB_V1)
    sub_v2 = pd.read_csv(SUB_V2)

    # Ensure GEOID is string for consistent joins
    sub_v1["GEOID"] = sub_v1["GEOID"].astype(str)
    sub_v2["GEOID"] = sub_v2["GEOID"].astype(str)

    print(f"  V1 submission: {sub_v1.shape[0]:,} tracts")
    print(f"  V2 submission: {sub_v2.shape[0]:,} tracts")

    # Verify alignment
    assert set(sub_v1["GEOID"]) == set(sub_v2["GEOID"]), "GEOID mismatch between submissions!"
    print(f"  GEOID sets match ✓")

    # Merge on GEOID
    merged = sub_v1.merge(sub_v2, on="GEOID", suffixes=("_v1", "_v2"))
    merged = merged.sort_values("GEOID").reset_index(drop=True)

    # OOF predictions
    oof_v1 = pd.read_parquet(OOF_V1)
    oof_v2 = pd.read_parquet(OOF_V2)
    oof_v1["GEOID"] = oof_v1["GEOID"].astype(str)
    oof_v2["GEOID"] = oof_v2["GEOID"].astype(str)
    print(f"  V1 OOF: {oof_v1.shape[0]:,} rows, columns: {list(oof_v1.columns)}")
    print(f"  V2 OOF: {oof_v2.shape[0]:,} rows, columns: {list(oof_v2.columns)}")

    # Engineered features (for tribal/rural info)
    feat = pd.read_parquet(FEATURES)
    feat["GEOID"] = feat["GEOID"].astype(str)
    print(f"  Features: {feat.shape[0]:,} rows")

    # Extract relevant columns
    meta_info = feat[["GEOID", "tribal_any", "rural_indicator"]].copy()
    meta_info = meta_info.sort_values("GEOID").reset_index(drop=True)

    return merged, oof_v1, oof_v2, meta_info


def compute_blend_strategies(merged):
    """Compute four blending strategies."""
    print("\n" + "=" * 80)
    print("BLENDING STRATEGIES")
    print("=" * 80)

    p1 = merged["coverage_gap_score_v1"].values
    p2 = merged["coverage_gap_score_v2"].values

    blends = {}

    # ── 1. Equal Weight Blend (50/50) ─────────────────────────────────────────
    blends["equal_50_50"] = 0.5 * p1 + 0.5 * p2
    print(f"\n  1. Equal Weight Blend (50/50)")

    # ── 2. Optimized Blend (minimize variance of the blend) ──────────────────
    # The optimal weight that minimizes variance of the blend:
    # w* = Var(p2) / (Var(p1) + Var(p2))  — inverse-variance weighting
    var1 = np.var(p1)
    var2 = np.var(p2)
    w_opt = var2 / (var1 + var2)  # weight for p1
    blends["optimized"] = w_opt * p1 + (1 - w_opt) * p2
    print(f"\n  2. Optimized Blend (inverse-variance weighting)")
    print(f"     Var(v1)={var1:.6f}, Var(v2)={var2:.6f}")
    print(f"     Optimal weight: w_v1={w_opt:.4f}, w_v2={1-w_opt:.4f}")

    # ── 3. Rank-Averaged Blend ────────────────────────────────────────────────
    # Average the ranks instead of raw scores, then map back to score space
    from scipy.stats import rankdata
    ranks1 = rankdata(p1)
    ranks2 = rankdata(p2)
    avg_ranks = 0.5 * ranks1 + 0.5 * ranks2
    # Map average ranks back to score quantiles using the average distribution
    # Use the sorted average of both prediction arrays as the reference distribution
    all_preds = np.concatenate([p1, p2])
    # Create the blended score by mapping avg_ranks to quantiles of the combined distribution
    sorted_combined = np.sort(all_preds)
    n = len(p1)
    rank_positions = ((avg_ranks - 1) / (2 * n - 1) * (len(sorted_combined) - 1)).astype(int)
    rank_positions = np.clip(rank_positions, 0, len(sorted_combined) - 1)
    blends["rank_averaged"] = sorted_combined[rank_positions]
    print(f"\n  3. Rank-Averaged Blend")
    print(f"     Avg rank correlation between p1 & p2: {stats.spearmanr(p1, p2).correlation:.6f}")

    # ── 4. Conservative Blend (max of the two scores = min penalty) ──────────
    # Scores are negative (penalties), so max = least penalty = conservative
    blends["conservative"] = np.maximum(p1, p2)
    print(f"\n  4. Conservative Blend (max = min penalty)")
    n_conservative_v1 = np.sum(p1 >= p2)
    n_conservative_v2 = np.sum(p2 > p1)
    print(f"     V1 dominates: {n_conservative_v1:,} tracts, V2 dominates: {n_conservative_v2:,} tracts")

    return blends, w_opt


def analyze_blends(blends, merged, meta_info):
    """For each blend, compute score distribution, rural/urban, tribal/non-tribal."""
    print("\n" + "=" * 80)
    print("ANALYSIS OF BLENDS")
    print("=" * 80)

    p1 = merged["coverage_gap_score_v1"].values
    p2 = merged["coverage_gap_score_v2"].values
    disagreement = np.abs(p1 - p2)

    # Correlation between the two source predictions
    pearson_r = np.corrcoef(p1, p2)[0, 1]
    spearman_r = stats.spearmanr(p1, p2).correlation
    print(f"\n  ── Source Prediction Correlation ──")
    print(f"     Pearson:  {pearson_r:.6f}")
    print(f"     Spearman: {spearman_r:.6f}")
    print(f"     Mean |disagreement|: {np.mean(disagreement):.6f}")
    print(f"     Max  |disagreement|: {np.max(disagreement):.6f}")
    print(f"     Std  |disagreement|: {np.std(disagreement):.6f}")

    # Disagreement quantiles
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"     Disagreement P{int(q*100)}: {np.quantile(disagreement, q):.6f}")

    # Rural/tribal flags aligned with merged
    is_rural = meta_info["rural_indicator"].values == 1.0
    is_tribal = meta_info["tribal_any"].values.astype(bool)

    results = {}

    for name, scores in blends.items():
        print(f"\n  ── {name} ──")

        # Score distribution
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        median_s = np.median(scores)
        skew_s = stats.skew(scores)
        kurt_s = stats.kurtosis(scores)

        print(f"     Mean:    {mean_s:.6f}")
        print(f"     Std:     {std_s:.6f}")
        print(f"     Min:     {min_s:.6f}")
        print(f"     Max:     {max_s:.6f}")
        print(f"     Median:  {median_s:.6f}")
        print(f"     Skew:    {skew_s:.4f}")
        print(f"     Kurtosis:{kurt_s:.4f}")

        # Rural vs Urban
        rural_scores = scores[is_rural]
        urban_scores = scores[~is_rural]
        print(f"     ── Rural vs Urban ──")
        print(f"       Rural  ({is_rural.sum():,} tracts): mean={np.mean(rural_scores):.6f}, std={np.std(rural_scores):.6f}")
        print(f"       Urban  ({(~is_rural).sum():,} tracts): mean={np.mean(urban_scores):.6f}, std={np.std(urban_scores):.6f}")
        print(f"       Rural-Urban gap: {np.mean(rural_scores) - np.mean(urban_scores):.6f}")

        # Tribal vs Non-tribal
        tribal_scores = scores[is_tribal]
        nontribal_scores = scores[~is_tribal]
        print(f"     ── Tribal vs Non-Tribal ──")
        print(f"       Tribal     ({is_tribal.sum():,} tracts): mean={np.mean(tribal_scores):.6f}, std={np.std(tribal_scores):.6f}")
        print(f"       Non-tribal ({(~is_tribal).sum():,} tracts): mean={np.mean(nontribal_scores):.6f}, std={np.std(nontribal_scores):.6f}")
        print(f"       Tribal-NonTribal gap: {np.mean(tribal_scores) - np.mean(nontribal_scores):.6f}")

        # Correlation with each source
        corr_v1 = np.corrcoef(scores, p1)[0, 1]
        corr_v2 = np.corrcoef(scores, p2)[0, 1]
        print(f"     ── Correlation with sources ──")
        print(f"       Corr with V1: {corr_v1:.6f}")
        print(f"       Corr with V2: {corr_v2:.6f}")

        # Variance reduction vs individual sources
        var_v1 = np.var(p1)
        var_v2 = np.var(p2)
        var_blend = np.var(scores)
        print(f"     ── Variance reduction ──")
        print(f"       Var(V1):     {var_v1:.6f}")
        print(f"       Var(V2):     {var_v2:.6f}")
        print(f"       Var(blend):  {var_blend:.6f}")
        print(f"       Reduction vs V1: {(1 - var_blend/var_v1)*100:.2f}%")
        print(f"       Reduction vs V2: {(1 - var_blend/var_v2)*100:.2f}%")

        results[name] = {
            "mean": mean_s, "std": std_s, "min": min_s, "max": max_s,
            "median": median_s, "skew": skew_s, "kurtosis": kurt_s,
            "rural_mean": np.mean(rural_scores), "urban_mean": np.mean(urban_scores),
            "rural_urban_gap": np.mean(rural_scores) - np.mean(urban_scores),
            "tribal_mean": np.mean(tribal_scores), "nontribal_mean": np.mean(nontribal_scores),
            "tribal_nontribal_gap": np.mean(tribal_scores) - np.mean(nontribal_scores),
            "corr_v1": corr_v1, "corr_v2": corr_v2,
            "var": var_blend, "var_reduction_vs_v1": (1 - var_blend/var_v1)*100,
            "var_reduction_vs_v2": (1 - var_blend/var_v2)*100,
        }

    return results, pearson_r, spearman_r


def select_best_blend(blends, results, merged):
    """Select the best blend based on a composite criterion.

    We want:
    - Lower variance (hedge quality)
    - Reasonable score distribution (not too extreme)
    - Balanced correlation with both sources (true hedging, not domination)

    Criterion: minimize (var_norm + imbalance_norm)
    where var_norm = var / min(var_v1, var_v2)
    and imbalance_norm = |corr_v1 - corr_v2|
    """
    print("\n" + "=" * 80)
    print("SELECTING BEST BLEND")
    print("=" * 80)

    var_v1 = np.var(merged["coverage_gap_score_v1"].values)
    var_v2 = np.var(merged["coverage_gap_score_v2"].values)
    min_var = min(var_v1, var_v2)

    best_name = None
    best_score = np.inf
    best_details = None

    print(f"\n  Reference min variance: {min_var:.6f}")
    print(f"\n  {'Blend':<20} {'VarRatio':>10} {'Imbalance':>10} {'Composite':>10} {'Verdict':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, r in results.items():
        var_ratio = r["var"] / min_var
        imbalance = abs(r["corr_v1"] - r["corr_v2"])
        # Composite: lower is better
        # Weight variance reduction more (0.7) vs balance (0.3)
        composite = 0.7 * var_ratio + 0.3 * imbalance

        verdict = ""
        if composite < best_score:
            best_score = composite
            best_name = name
            verdict = "★ BEST"
            best_details = (var_ratio, imbalance, composite)

        print(f"  {name:<20} {var_ratio:>10.4f} {imbalance:>10.4f} {composite:>10.4f} {verdict:>10}")

    print(f"\n  >>> Best blend: {best_name} (composite={best_score:.4f})")
    if best_details:
        print(f"      Var ratio: {best_details[0]:.4f}, Imbalance: {best_details[1]:.4f}")

    return best_name, blends[best_name]


def analyze_disagreement(merged, meta_info):
    """Deep dive into where the two machines disagree."""
    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS (Uncertainty Map)")
    print("=" * 80)

    p1 = merged["coverage_gap_score_v1"].values
    p2 = merged["coverage_gap_score_v2"].values
    disagreement = np.abs(p1 - p2)

    is_rural = meta_info["rural_indicator"].values == 1.0
    is_tribal = meta_info["tribal_any"].values.astype(bool)

    # Top-K most disagreed tracts
    top_k = 20
    top_idx = np.argsort(disagreement)[::-1][:top_k]
    print(f"\n  Top-{top_k} most disagreed tracts:")
    print(f"  {'GEOID':<14} {'V1':>10} {'V2':>10} {'|Diff|':>10} {'Rural':>6} {'Tribal':>7}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*7}")
    for idx in top_idx:
        geoid = merged.iloc[idx]["GEOID"]
        r = "Y" if is_rural[idx] else "N"
        t = "Y" if is_tribal[idx] else "N"
        print(f"  {geoid:<14} {p1[idx]:>10.6f} {p2[idx]:>10.6f} {disagreement[idx]:>10.6f} {r:>6} {t:>7}")

    # Disagreement by rural/tribal
    print(f"\n  Disagreement by segment:")
    for label, mask in [("Rural", is_rural), ("Urban", ~is_rural),
                         ("Tribal", is_tribal), ("Non-tribal", ~is_tribal),
                         ("Rural+Tribal", is_rural & is_tribal),
                         ("Urban+Non-tribal", ~is_rural & ~is_tribal)]:
        if mask.sum() > 0:
            d = disagreement[mask]
            print(f"    {label:<20}: mean={np.mean(d):.6f}, std={np.std(d):.6f}, "
                  f"max={np.max(d):.6f}, n={mask.sum():,}")

    # Agreement direction: which machine is more punitive?
    v1_more_punitive = np.sum(p1 < p2)  # v1 score lower (more negative)
    v2_more_punitive = np.sum(p2 < p1)
    print(f"\n  Direction of disagreement:")
    print(f"    V1 more punitive (lower score): {v1_more_punitive:,} tracts")
    print(f"    V2 more punitive (lower score): {v2_more_punitive:,} tracts")

    # Sign test by rural/tribal
    for label, mask in [("Rural", is_rural), ("Urban", ~is_rural),
                         ("Tribal", is_tribal), ("Non-tribal", ~is_tribal)]:
        if mask.sum() > 0:
            v1_wins = np.sum((p1 < p2) & mask)
            v2_wins = np.sum((p2 < p1) & mask)
            print(f"    {label:<20}: V1 more punitive={v1_wins:,}, V2 more punitive={v2_wins:,}")


def save_best_blend(merged, best_scores, best_name):
    """Save the best blend to output and download directories."""
    print("\n" + "=" * 80)
    print("SAVING BEST BLEND")
    print("=" * 80)

    submission = pd.DataFrame({
        "GEOID": merged["GEOID"].values,
        "coverage_gap_score": best_scores
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_META), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DOWNLOAD), exist_ok=True)

    submission.to_csv(OUT_META, index=False)
    submission.to_csv(OUT_DOWNLOAD, index=False)

    print(f"  Saved {len(submission):,} tracts to:")
    print(f"    {OUT_META}")
    print(f"    {OUT_DOWNLOAD}")
    print(f"  Blend strategy: {best_name}")
    print(f"  Score range: [{best_scores.min():.6f}, {best_scores.max():.6f}]")
    print(f"  Score mean:  {best_scores.mean():.6f}")
    print(f"  Score std:   {best_scores.std():.6f}")


def oof_correlation_analysis(oof_v1, oof_v2, merged):
    """Analyze correlation between OOF predictions of the two ensembles."""
    print("\n" + "=" * 80)
    print("OOF PREDICTION CORRELATION ANALYSIS")
    print("=" * 80)

    # Align on GEOID
    oof_v1 = oof_v1.sort_values("GEOID").reset_index(drop=True)
    oof_v2 = oof_v2.sort_values("GEOID").reset_index(drop=True)

    # V1 models
    v1_models = [c for c in oof_v1.columns if c not in ("GEOID", "gap_only", "rural_penalty", "proxy_merged")]
    v2_models = [c for c in oof_v2.columns if c not in ("GEOID", "proxy_merged")]

    print(f"\n  V1 models: {v1_models}")
    print(f"  V2 models: {v2_models}")

    # Cross-correlation matrix between all models
    all_preds = {}
    for m in v1_models:
        all_preds[f"v1_{m}"] = oof_v1[m].values
    for m in v2_models:
        all_preds[f"v2_{m}"] = oof_v2[m].values

    pred_df = pd.DataFrame(all_preds)
    corr_matrix = pred_df.corr()

    print(f"\n  Cross-ensemble correlations (V1 vs V2 models):")
    v1_cols = [f"v1_{m}" for m in v1_models]
    v2_cols = [f"v2_{m}" for m in v2_models]

    print(f"  {'':>12}", end="")
    for c in v2_cols:
        print(f" {c:>12}", end="")
    print()
    for r in v1_cols:
        print(f"  {r:>12}", end="")
        for c in v2_cols:
            print(f" {corr_matrix.loc[r, c]:>12.4f}", end="")
        print()

    # Mean cross-ensemble correlation
    cross_corrs = []
    for r in v1_cols:
        for c in v2_cols:
            cross_corrs.append(corr_matrix.loc[r, c])
    print(f"\n  Mean cross-ensemble correlation: {np.mean(cross_corrs):.4f}")
    print(f"  Min  cross-ensemble correlation: {np.min(cross_corrs):.4f}")
    print(f"  Max  cross-ensemble correlation: {np.max(cross_corrs):.4f}")

    # Within-ensemble correlations
    print(f"\n  Within V1 ensemble correlations:")
    for i in range(len(v1_cols)):
        for j in range(i+1, len(v1_cols)):
            c = corr_matrix.loc[v1_cols[i], v1_cols[j]]
            print(f"    {v1_cols[i]} vs {v1_cols[j]}: {c:.4f}")

    print(f"\n  Within V2 ensemble correlations:")
    for i in range(len(v2_cols)):
        for j in range(i+1, len(v2_cols)):
            c = corr_matrix.loc[v2_cols[i], v2_cols[j]]
            print(f"    {v2_cols[i]} vs {v2_cols[j]}: {c:.4f}")


def main():
    print("╔" + "═"*78 + "╗")
    print("║" + " META-ENSEMBLE HEDGING: Reducing Variance Between Two Machines ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Load data
    merged, oof_v1, oof_v2, meta_info = load_data()

    # Compute blend strategies
    blends, w_opt = compute_blend_strategies(merged)

    # Analyze all blends
    results, pearson_r, spearman_r = analyze_blends(blends, merged, meta_info)

    # Select best blend
    best_name, best_scores = select_best_blend(blends, results, merged)

    # Disagreement analysis
    analyze_disagreement(merged, meta_info)

    # OOF correlation analysis
    oof_correlation_analysis(oof_v1, oof_v2, merged)

    # Save best blend
    save_best_blend(merged, best_scores, best_name)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " SUMMARY ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\n  Source correlation: Pearson={pearson_r:.6f}, Spearman={spearman_r:.6f}")
    print(f"  Best blend strategy: {best_name}")
    print(f"  Optimized weight: w_v1={w_opt:.4f}, w_v2={1-w_opt:.4f}")
    print(f"\n  All blend strategies ranked by composite score:")
    var_v1 = np.var(merged["coverage_gap_score_v1"].values)
    var_v2 = np.var(merged["coverage_gap_score_v2"].values)
    min_var = min(var_v1, var_v2)

    rankings = []
    for name, r in results.items():
        var_ratio = r["var"] / min_var
        imbalance = abs(r["corr_v1"] - r["corr_v2"])
        composite = 0.7 * var_ratio + 0.3 * imbalance
        rankings.append((name, composite, r))

    rankings.sort(key=lambda x: x[1])
    for rank, (name, composite, r) in enumerate(rankings, 1):
        marker = " ★" if name == best_name else ""
        print(f"    {rank}. {name:<20} composite={composite:.4f} "
              f"var_red_v1={r['var_reduction_vs_v1']:.1f}% "
              f"var_red_v2={r['var_reduction_vs_v2']:.1f}% "
              f"rural_urban_gap={r['rural_urban_gap']:.6f} "
              f"tribal_gap={r['tribal_nontribal_gap']:.6f}{marker}")

    print(f"\n  Key insight: Meta-ensembling reduces prediction variance by hedging")
    print(f"  between two differently-constructed ensembles. When they agree, we're")
    print(f"  confident. When they disagree, the blend takes the middle ground,")
    print(f"  reducing the risk of over- or under-estimation on any single tract.")
    print()


if __name__ == "__main__":
    main()
