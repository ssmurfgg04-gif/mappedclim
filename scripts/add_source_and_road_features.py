#!/usr/bin/env python3
"""
Add source composition features and road class breakdown features
to the national tract features for the Zindi competition.

Part 1: Source Composition Features  — from _covered columns in strata table
Part 2: Road Class Breakdown         — from road columns in national features + RUCA/NCHS
Part 3: Merge and Save               — update national + regional feature files
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATA_PATH = PROJECT_ROOT / "kaggle_dataset" / "national-strata-tract-table.parquet"
NATIONAL_FEAT_PATH = PROJECT_ROOT / "data" / "features" / "national_tract_features.parquet"
REGIONAL_FEAT_PATH = PROJECT_ROOT / "data" / "output" / "engineered_features_v3.parquet"


# ═══════════════════════════════════════════════════════════════════════
# Part 1 — Source Composition Features
# ═══════════════════════════════════════════════════════════════════════
def compute_source_composition(strata: pd.DataFrame) -> pd.DataFrame:
    """
    From the _covered columns in the strata table, compute per-tract
    source composition features.
    """
    t0 = time.time()
    print("=" * 60)
    print("PART 1: Source Composition Features")
    print("=" * 60)

    # Identify _covered columns
    covered_cols = [c for c in strata.columns if "_covered" in c.lower()]
    print(f"  Found {len(covered_cols)} _covered columns: {covered_cols}")

    # Also search for overture / osm columns (not necessarily _covered)
    overture_cols = [c for c in strata.columns if "overture" in c.lower()]
    osm_cols = [c for c in strata.columns if "osm" in c.lower()]
    print(f"  Overture columns in strata: {overture_cols}")
    print(f"  OSM columns in strata: {osm_cols}")

    # ── per-tract computation ────────────────────────────────────────────
    covered_matrix = strata[covered_cols].copy()

    # Treat boolean True / 1 as covered, False / 0 / NaN as not
    # Coerce to numeric then binarize: 1 if truthy, 0 otherwise
    for c in covered_cols:
        if covered_matrix[c].dtype == bool:
            covered_matrix[c] = covered_matrix[c].astype(int)
        else:
            covered_matrix[c] = covered_matrix[c].fillna(0).astype(float)
            covered_matrix[c] = (covered_matrix[c] > 0).astype(int)

    # 1. source_coverage_count — number of _covered cols that are non-null
    #    (all are non-null here, but handle generically)
    source_coverage_count = strata[covered_cols].notna().sum(axis=1).astype(int)

    # 2. source_coverage_true_count — number of _covered cols that are True/1
    source_coverage_true_count = covered_matrix.sum(axis=1).astype(int)

    # 3. source_coverage_fraction = true_count / count
    source_coverage_fraction = np.where(
        source_coverage_count > 0,
        source_coverage_true_count / source_coverage_count,
        0.0,
    )

    # 4. Shannon entropy of source distribution
    #    p_i = 1/n_covered for each covered source, 0 otherwise
    #    H = -sum(p_i * log(p_i)) for p_i > 0
    n_covered = source_coverage_true_count.values.astype(float)
    n_total = len(covered_cols)
    # When n_covered > 0: p_i = 1/n_covered for each covered source
    # H = n_covered * (1/n_covered) * log(n_covered) = log(n_covered)
    source_diversity_entropy = np.where(
        n_covered > 0, np.log(n_covered), 0.0
    )

    # 5. all_sources_covered — binary, 1 if ALL _covered cols are True
    all_sources_covered = (source_coverage_true_count == len(covered_cols)).astype(int)

    # 6. overture_source_count — columns containing 'overture' that are True/1
    if overture_cols:
        overture_source_count = (
            strata[overture_cols].fillna(0).astype(float).gt(0).sum(axis=1).astype(int)
        )
    else:
        # No overture _covered columns in strata — check national features later
        overture_source_count = pd.Series(0, index=strata.index, dtype=int)

    # 7. osm_source_count — columns containing 'osm' that are True/1
    if osm_cols:
        osm_source_count = (
            strata[osm_cols].fillna(0).astype(float).gt(0).sum(axis=1).astype(int)
        )
    else:
        osm_source_count = pd.Series(0, index=strata.index, dtype=int)

    # ── assemble ─────────────────────────────────────────────────────────
    result = pd.DataFrame({"GEOID": strata["GEOID"].values})
    result["source_coverage_count"] = source_coverage_count.values
    result["source_coverage_true_count"] = source_coverage_true_count.values
    result["source_coverage_fraction"] = source_coverage_fraction
    result["source_diversity_entropy"] = source_diversity_entropy
    result["all_sources_covered"] = all_sources_covered.values
    result["overture_source_count"] = overture_source_count.values
    result["osm_source_count"] = osm_source_count.values

    elapsed = time.time() - t0
    print(f"  Source composition features computed in {elapsed:.2f}s")
    print(f"  Rows: {len(result)}")
    new_cols = [c for c in result.columns if c != "GEOID"]
    print(f"  New columns: {new_cols}")

    # Summary
    print("\n  ── Source Composition Summary ──")
    for c in result.columns:
        if c == "GEOID":
            continue
        s = result[c]
        if s.dtype in [np.float64, np.float32]:
            print(f"    {c:35s}  mean={s.mean():.4f}  std={s.std():.4f}  "
                  f"min={s.min():.4f}  max={s.max():.4f}")
        else:
            vc = s.value_counts().sort_index()
            print(f"    {c:35s}  {dict(vc)}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — Road Class Breakdown Features
# ═══════════════════════════════════════════════════════════════════════
def compute_road_class_features(
    strata: pd.DataFrame, national_feat: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute road class breakdown features. Since the strata table has no
    direct road class columns, we derive them from:
      - ov_road / tiger_road counts (Overture vs TIGER road sources)
      - RUCA primary code (urban-rural commuting area classification)
      - NCHS 2013 code (urban-rural county classification)
      - ur_class (Urban / Rural)
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PART 2: Road Class Breakdown Features")
    print("=" * 60)

    # Merge relevant columns from strata onto national features
    merge_cols = ["GEOID"]
    for c in ["ruca_primary", "ruca_covered", "nchs_2013", "nchs_covered", "ur_class"]:
        if c in strata.columns:
            merge_cols.append(c)

    nf = national_feat[["GEOID"]].copy()
    for c in merge_cols[1:]:
        nf[c] = strata.set_index("GEOID").loc[nf["GEOID"], c].values

    # Also pull road columns from national features
    road_cols_in_nf = [c for c in national_feat.columns
                       if c in ("ov_road", "tiger_road", "road_ratio", "road_gap")]
    for c in road_cols_in_nf:
        nf[c] = national_feat.set_index("GEOID").loc[nf["GEOID"], c].values

    print(f"  Road columns in national features: {road_cols_in_nf}")

    # ── road_class_diversity ─────────────────────────────────────────────
    # Number of distinct road data sources with non-zero coverage
    # ov_road > 0 → Overture source present
    # tiger_road > 0 → TIGER source present
    ov_present = np.zeros(len(nf), dtype=int)
    tiger_present = np.zeros(len(nf), dtype=int)
    if "ov_road" in nf.columns:
        ov_present = (nf["ov_road"].fillna(0) > 0).astype(int).values
    if "tiger_road" in nf.columns:
        tiger_present = (nf["tiger_road"].fillna(0) > 0).astype(int).values
    road_class_diversity = ov_present + tiger_present

    # ── residential_road_coverage ────────────────────────────────────────
    # Proxy: RUCA codes >= 4 indicate non-metropolitan areas with
    # predominantly residential/local roads
    residential_road_coverage = np.zeros(len(nf), dtype=float)
    if "ruca_primary" in nf.columns:
        ruca = nf["ruca_primary"].fillna(99)
        # RUCA 4-10 are non-metro — higher residential road share
        # Scale: 0 (metro core, RUCA 1) → 1 (most rural, RUCA 10)
        residential_road_coverage = np.where(
            (ruca >= 4) & (ruca <= 10),
            (ruca - 3) / 7.0,  # ranges from 1/7 to 1.0
            np.where(ruca <= 3, 0.1, 0.0),  # metro areas: low residential
        )
        # If ov_road is 0, set to 0 (no roads at all)
        if "ov_road" in nf.columns:
            residential_road_coverage = np.where(
                nf["ov_road"].fillna(0) > 0, residential_road_coverage, 0.0
            )

    # ── primary_road_coverage ────────────────────────────────────────────
    # Proxy: RUCA codes 1-3 are metropolitan core — primary/motorway roads
    primary_road_coverage = np.zeros(len(nf), dtype=float)
    if "ruca_primary" in nf.columns:
        ruca = nf["ruca_primary"].fillna(99)
        # RUCA 1 = metro core, 2-3 = metro fringe → more primary roads
        primary_road_coverage = np.where(
            ruca == 1, 1.0,
            np.where(ruca == 2, 0.8,
                     np.where(ruca == 3, 0.6,
                              np.where((ruca >= 4) & (ruca <= 10), 0.2, 0.0))))
        if "ov_road" in nf.columns:
            primary_road_coverage = np.where(
                nf["ov_road"].fillna(0) > 0, primary_road_coverage, 0.0
            )

    # ── secondary_road_coverage ──────────────────────────────────────────
    # Proxy: RUCA 2-5 → secondary road mix
    secondary_road_coverage = np.zeros(len(nf), dtype=float)
    if "ruca_primary" in nf.columns:
        ruca = nf["ruca_primary"].fillna(99)
        secondary_road_coverage = np.where(
            (ruca >= 2) & (ruca <= 3), 0.5,
            np.where((ruca >= 4) & (ruca <= 5), 0.7,
                     np.where((ruca >= 6) & (ruca <= 7), 0.4,
                              np.where(ruca == 1, 0.3, 0.1))))
        if "ov_road" in nf.columns:
            secondary_road_coverage = np.where(
                nf["ov_road"].fillna(0) > 0, secondary_road_coverage, 0.0
            )

    # ── road_class_entropy ───────────────────────────────────────────────
    # Entropy of the road-type distribution for each tract
    # p_primary, p_secondary, p_residential  (normalized to sum=1)
    p_arr = np.column_stack([
        primary_road_coverage,
        secondary_road_coverage,
        residential_road_coverage,
    ])
    row_sums = p_arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p_norm = p_arr / row_sums

    # Shannon entropy: H = -sum(p_i * log(p_i)) for p_i > 0
    # Use safe log to avoid RuntimeWarning on log(0)
    p_safe = np.where(p_norm > 0, p_norm, 1.0)  # dummy 1.0 for log(0) cases
    log_p = np.where(p_norm > 0, np.log(p_safe), 0.0)
    road_class_entropy = -np.sum(p_norm * log_p, axis=1)
    # If only one road type, entropy = 0; if all equal, entropy = log(3)

    # ── major_road_fraction ──────────────────────────────────────────────
    # Fraction of roads that are major (primary + motorway)
    major_road_fraction = np.where(
        row_sums.ravel() > 0,
        primary_road_coverage / row_sums.ravel(),
        0.0,
    )

    # ── minor_road_fraction ──────────────────────────────────────────────
    # Fraction of roads that are minor (residential + secondary)
    minor_road_fraction = np.where(
        row_sums.ravel() > 0,
        (residential_road_coverage + secondary_road_coverage) / row_sums.ravel(),
        0.0,
    )

    # ── additional: Overture/TIGER road source features ─────────────────
    # ov_road and tiger_road as presence indicators (useful for model)
    ov_road_present = ov_present.astype(int)
    tiger_road_present = tiger_present.astype(int)
    # Road source diversity (0-2: how many road sources)
    # Already computed as road_class_diversity, but add explicit name
    road_source_diversity = road_class_diversity.copy()

    # ── assemble ─────────────────────────────────────────────────────────
    result = pd.DataFrame({"GEOID": nf["GEOID"].values})
    result["road_class_diversity"] = road_class_diversity
    result["residential_road_coverage"] = residential_road_coverage
    result["primary_road_coverage"] = primary_road_coverage
    result["secondary_road_coverage"] = secondary_road_coverage
    result["road_class_entropy"] = road_class_entropy
    result["major_road_fraction"] = major_road_fraction
    result["minor_road_fraction"] = minor_road_fraction
    result["ov_road_present"] = ov_road_present
    result["tiger_road_present"] = tiger_road_present
    result["road_source_diversity"] = road_source_diversity

    elapsed = time.time() - t0
    print(f"  Road class features computed in {elapsed:.2f}s")
    print(f"  Rows: {len(result)}")
    print(f"  New columns: {[c for c in result.columns if c != 'GEOID']}")

    # Summary
    print("\n  ── Road Class Summary ──")
    for c in result.columns:
        if c == "GEOID":
            continue
        s = result[c]
        if s.dtype in [np.float64, np.float32]:
            print(f"    {c:35s}  mean={s.mean():.4f}  std={s.std():.4f}  "
                  f"min={s.min():.4f}  max={s.max():.4f}")
        else:
            vc = s.value_counts().sort_index()
            print(f"    {c:35s}  {dict(vc)}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 3 — Merge and Save
# ═══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 60)
    print("ADD SOURCE AND ROAD FEATURES")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading data...")
    strata = pd.read_parquet(STRATA_PATH)
    print(f"  Strata table: {strata.shape}")

    national_feat = pd.read_parquet(NATIONAL_FEAT_PATH)
    print(f"  National features: {national_feat.shape}")

    # ── Part 1: Source Composition ───────────────────────────────────────
    source_features = compute_source_composition(strata)

    # ── Part 2: Road Class Breakdown ─────────────────────────────────────
    road_features = compute_road_class_features(strata, national_feat)

    # ── Combine new features ─────────────────────────────────────────────
    new_features = source_features.merge(road_features, on="GEOID", how="inner")
    new_feature_cols = [c for c in new_features.columns if c != "GEOID"]
    print(f"\n  Total new feature columns: {len(new_feature_cols)}")
    print(f"  New features: {new_feature_cols}")

    # ── Check for existing columns (avoid duplicates) ───────────────────
    existing_new_cols = [c for c in new_feature_cols if c in national_feat.columns]
    if existing_new_cols:
        print(f"\n  WARNING: Dropping existing columns from national features: "
              f"{existing_new_cols}")
        national_feat = national_feat.drop(columns=existing_new_cols)

    # ── Merge into national features ─────────────────────────────────────
    print("\nMerging into national features...")
    national_feat = national_feat.merge(new_features, on="GEOID", how="left")
    print(f"  National features after merge: {national_feat.shape}")

    # Fill NaN for tracts not in strata (shouldn't happen, but safety)
    for c in new_feature_cols:
        if national_feat[c].isna().any():
            n_na = national_feat[c].isna().sum()
            print(f"  Filling {n_na} NaN values in {c}")
            if national_feat[c].dtype in [np.float64, np.float32]:
                national_feat[c] = national_feat[c].fillna(0.0)
            else:
                national_feat[c] = national_feat[c].fillna(0)

    # ── Save national features ───────────────────────────────────────────
    print(f"\nSaving national features to {NATIONAL_FEAT_PATH}...")
    national_feat.to_parquet(NATIONAL_FEAT_PATH, index=False)
    print(f"  Saved: {national_feat.shape}")

    # ── Update regional engineered_features_v3.parquet ───────────────────
    print(f"\nUpdating regional features at {REGIONAL_FEAT_PATH}...")
    if os.path.exists(REGIONAL_FEAT_PATH):
        regional_feat = pd.read_parquet(REGIONAL_FEAT_PATH)
        print(f"  Regional features before: {regional_feat.shape}")

        # Drop existing new columns if present
        existing_regional = [c for c in new_feature_cols if c in regional_feat.columns]
        if existing_regional:
            print(f"  Dropping existing columns: {existing_regional}")
            regional_feat = regional_feat.drop(columns=existing_regional)

        # Merge — only for GEOIDs present in regional set
        regional_new = new_features[new_features["GEOID"].isin(regional_feat["GEOID"])]
        regional_feat = regional_feat.merge(regional_new, on="GEOID", how="left")

        # Fill NaN
        for c in new_feature_cols:
            if c in regional_feat.columns and regional_feat[c].isna().any():
                n_na = regional_feat[c].isna().sum()
                print(f"  Filling {n_na} NaN values in {c}")
                if regional_feat[c].dtype in [np.float64, np.float32]:
                    regional_feat[c] = regional_feat[c].fillna(0.0)
                else:
                    regional_feat[c] = regional_feat[c].fillna(0)

        regional_feat.to_parquet(REGIONAL_FEAT_PATH, index=False)
        print(f"  Regional features after: {regional_feat.shape}")
        print(f"  Regional tracts matched: {len(regional_new)}")
    else:
        print(f"  WARNING: Regional features file not found at {REGIONAL_FEAT_PATH}")

    # ── Final Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total new feature columns added: {len(new_feature_cols)}")
    print(f"  Columns: {new_feature_cols}")
    print(f"  National features final shape: {national_feat.shape}")
    if os.path.exists(REGIONAL_FEAT_PATH):
        regional_check = pd.read_parquet(REGIONAL_FEAT_PATH)
        print(f"  Regional features final shape: {regional_check.shape}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
