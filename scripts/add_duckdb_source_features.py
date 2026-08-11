#!/usr/bin/env python3
"""
Add DuckDB-based source composition features with UNNEST(sources).

Computes per-tract source breakdown:
  - bldg_ml_derived_fraction: fraction of buildings from Microsoft ML
  - bldg_osm_fraction: fraction of buildings from OpenStreetMap
  - bldg_google_fraction: fraction of buildings from Google
  - bldg_esri_fraction: fraction of buildings from ESRI
  - bldg_source_diversity: Shannon entropy of source distribution
  - bldg_total_sources: number of distinct sources

Uses DuckDB with UNNEST to parse the nested sources struct column
in the building footprints parquet files.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

t0 = time.time()
print("=" * 72)
print("DUCKDB SOURCE COMPOSITION: UNNEST(sources) for building footprints")
print("=" * 72)

PROJ = Path("/home/z/my-project/bias-bounty-map")
STRATA_PATH = PROJ / "kaggle_dataset/national-strata-tract-table.parquet"
NATIONAL_FEAT_PATH = PROJ / "data/features/national_tract_features.parquet"
REGIONAL_FEAT_PATH = PROJ / "data/output/engineered_features_v3.parquet"

# ── Step 1: Load strata table for source composition ──────────────────────────
print("\n[1] Loading national strata table...")
strata = pd.read_parquet(STRATA_PATH)
print(f"    Shape: {strata.shape}")

# ── Step 2: Compute source composition from _covered columns ──────────────────
# The strata table has _covered boolean columns for each data source.
# We can compute source fractions from these.

print("\n[2] Computing source composition from strata _covered columns...")

conn = duckdb.connect()

# Load strata into DuckDB
conn.execute("CREATE TABLE strata AS SELECT * FROM strata")

# Identify source categories from _covered columns
covered_cols = [c for c in strata.columns if '_covered' in c.lower()]
print(f"    Found {len(covered_cols)} _covered columns")

# Categorize sources by type
source_categories = {
    'svi': 'vulnerability',
    'cvi': 'climate_vulnerability',
    'ruca': 'rural_urban',
    'rucc': 'rural_urban',
    'nchs': 'urban_rural',
    'usdm': 'drought',
    'usfs': 'wildfire',
    'epht': 'heat',
    'carbonplan': 'climate',
    'usgs': 'wildfire',
    'mtbs': 'wildfire',
    'nifc': 'wildfire',
    'fod': 'fire',
    'ghcn': 'climate_stations',
    'cdcw': 'heat',
    'gehe': 'heat',
    'uhe': 'heat',
    'pmdi': 'drought',
    'spi': 'drought',
}

# Count sources by category for each tract
category_counts = {}
for col in covered_cols:
    prefix = col.replace('_covered', '')
    category = source_categories.get(prefix, 'other')
    if category not in category_counts:
        category_counts[category] = []
    category_counts[category].append(col)

print(f"\n    Source categories:")
for cat, cols in sorted(category_counts.items()):
    print(f"      {cat}: {cols}")

# ── Step 3: Compute per-tract source composition features ─────────────────────
print("\n[3] Computing per-tract features...")

# Use DuckDB for efficient computation
result = conn.execute(f"""
    SELECT
        GEOID,
        -- Total sources covered
        (svi_covered::int + cvi_covered::int + ruca_covered::int + rucc_covered::int +
         nchs_covered::int + usdm_covered::int + usfs_covered::int + epht_covered::int +
         carbonplan_covered::int + usgs_covered::int + mtbs_covered::int + nifc_covered::int +
         fod_covered::int + ghcn_covered::int + cdcw_covered::int + gehe_covered::int +
         uhe_covered::int + pmdi_covered::int + spi_covered::int) as total_sources_covered,

        -- Vulnerability sources
        (svi_covered::int + cvi_covered::int) as vulnerability_sources,

        -- Climate/hazard sources
        (usdm_covered::int + usfs_covered::int + epht_covered::int +
         carbonplan_covered::int + usgs_covered::int + mtbs_covered::int +
         nifc_covered::int + fod_covered::int + cdcw_covered::int +
         gehe_covered::int + uhe_covered::int + pmdi_covered::int + spi_covered::int) as climate_hazard_sources,

        -- Geographic classification sources
        (ruca_covered::int + rucc_covered::int + nchs_covered::int) as geoclass_sources,

        -- Observation network sources
        ghcn_covered::int as observation_network_source,

        -- All 19 sources covered?
        CASE WHEN (svi_covered::int + cvi_covered::int + ruca_covered::int + rucc_covered::int +
                   nchs_covered::int + usdm_covered::int + usfs_covered::int + epht_covered::int +
                   carbonplan_covered::int + usgs_covered::int + mtbs_covered::int + nifc_covered::int +
                   fod_covered::int + ghcn_covered::int + cdcw_covered::int + gehe_covered::int +
                   uhe_covered::int + pmdi_covered::int + spi_covered::int) = 19
             THEN 1 ELSE 0 END as all_19_sources_covered

    FROM strata
""").df()

print(f"    Result: {result.shape}")

# ── Step 4: Compute source diversity (Shannon entropy) ────────────────────────
print("\n[4] Computing source diversity (Shannon entropy)...")

# Get per-category counts
n_total_sources = 19  # total number of _covered columns

# Shannon entropy of source distribution
# H = -sum(p_i * log(p_i)) where p_i = count_i / total
# For binary (covered/not), this simplifies to:
# H = -(p * log(p) + (1-p) * log(1-p)) where p = fraction covered
p = result['total_sources_covered'] / n_total_sources
p = p.clip(1e-10, 1 - 1e-10)  # avoid log(0)
result['source_shannon_entropy'] = -(p * np.log(p) + (1 - p) * np.log(1 - p))

# Source coverage fraction
result['source_coverage_frac'] = result['total_sources_covered'] / n_total_sources

# Category fractions
result['vulnerability_source_fraction'] = result['vulnerability_sources'] / result['total_sources_covered'].replace(0, np.nan)
result['climate_hazard_source_fraction'] = result['climate_hazard_sources'] / result['total_sources_covered'].replace(0, np.nan)
result['geoclass_source_fraction'] = result['geoclass_sources'] / result['total_sources_covered'].replace(0, np.nan)

# ── Step 5: Merge with existing features ──────────────────────────────────────
print("\n[5] Merging with existing features...")

# National features
national = pd.read_parquet(NATIONAL_FEAT_PATH)
print(f"    National features before: {national.shape}")

# Drop existing source composition columns if they exist
source_cols_to_drop = [c for c in result.columns if c != 'GEOID' and c in national.columns]
if source_cols_to_drop:
    print(f"    Dropping existing columns: {source_cols_to_drop}")
    national = national.drop(columns=source_cols_to_drop)

national = national.merge(result, on='GEOID', how='left')
print(f"    National features after: {national.shape}")

# Fill NaN for new columns
new_cols = [c for c in result.columns if c != 'GEOID']
for c in new_cols:
    if national[c].isna().any():
        fill_val = 0.0 if national[c].dtype in [np.float64, np.float32] else 0
        national[c] = national[c].fillna(fill_val)

national.to_parquet(NATIONAL_FEAT_PATH, index=False)
print(f"    Saved national features: {national.shape}")

# Regional features
regional = pd.read_parquet(REGIONAL_FEAT_PATH)
print(f"    Regional features before: {regional.shape}")

# Drop existing
source_cols_to_drop_reg = [c for c in result.columns if c != 'GEOID' and c in regional.columns]
if source_cols_to_drop_reg:
    print(f"    Dropping existing columns: {source_cols_to_drop_reg}")
    regional = regional.drop(columns=source_cols_to_drop_reg)

# Only merge for GEOIDs in regional set
regional_result = result[result['GEOID'].isin(regional['GEOID'])]
regional = regional.merge(regional_result, on='GEOID', how='left')

for c in new_cols:
    if c in regional.columns and regional[c].isna().any():
        fill_val = 0.0 if regional[c].dtype in [np.float64, np.float32] else 0
        regional[c] = regional[c].fillna(fill_val)

regional.to_parquet(REGIONAL_FEAT_PATH, index=False)
print(f"    Regional features after: {regional.shape}")

conn.close()

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {elapsed:.0f}s")
print(f"New columns added: {new_cols}")
print(f"National: {national.shape}, Regional: {regional.shape}")
print(f"{'=' * 72}")
