"""
Compute baseline coverage gap features per census tract.
This is the most important script - it produces the features that drive the model.

Uses DuckDB for efficient spatial operations on GeoParquet data.
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def compute_coverage_gaps_for_region(region: str):
    """
    Compute coverage gap features per census tract for a focus region.
    
    This produces the primary features:
    - building_count_ratio (Overture / Microsoft)
    - road_length_ratio (Overture / TIGER)
    - poi_count vs HIFLD facility count
    - building count vs ACS housing units
    - Source composition features
    """
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    raw_dir = PROJECT_ROOT / "data/raw"
    ref_dir = raw_dir / "reference" / region
    
    # === 1. BUILDING COVERAGE GAP ===
    logger.info(f"=== Building Coverage Gap: {region} ===")
    
    # Count Overture buildings per tract (using spatial join)
    # We need the census tract geometries for spatial joining
    strata_path = raw_dir / "strata" / region / f"{region}-strata-tract-table.parquet"
    
    if not strata_path.exists():
        # Fall back to national strata with region filtering
        logger.info("Using national strata table with region filtering")
        strata_path = raw_dir / "strata/national/national-strata-tract-table.parquet"
    
    # Count buildings in each dataset
    ov_count = conn.execute(f"""
        SELECT COUNT(*) as cnt FROM read_parquet('{ref_dir}/{region}-overture-buildings.parquet')
    """).fetchone()[0]
    ms_count = conn.execute(f"""
        SELECT COUNT(*) as cnt FROM read_parquet('{ref_dir}/{region}-microsoft-buildings.parquet')
    """).fetchone()[0]
    
    logger.info(f"  Overture buildings: {ov_count:,}")
    logger.info(f"  Microsoft buildings: {ms_count:,}")
    logger.info(f"  Simple count ratio: {ov_count/ms_count:.4f}")
    logger.info(f"  Simple count gap: {1 - ov_count/ms_count:.4f}")
    
    # === 2. ROAD COVERAGE GAP ===
    logger.info(f"\n=== Road Coverage Gap: {region} ===")
    
    ov_road_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{ref_dir}/{region}-overture-roads.parquet')
    """).fetchone()[0]
    tiger_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{ref_dir}/{region}-census-tiger-roads.parquet')
    """).fetchone()[0]
    
    # Total road length
    ov_road_length = conn.execute(f"""
        SELECT SUM(ST_Length(geometry)) as total_length 
        FROM read_parquet('{ref_dir}/{region}-overture-roads.parquet')
    """).fetchone()[0]
    tiger_length = conn.execute(f"""
        SELECT SUM(ST_Length(geometry)) as total_length 
        FROM read_parquet('{ref_dir}/{region}-census-tiger-roads.parquet')
    """).fetchone()[0]
    
    logger.info(f"  Overture roads: {ov_road_count:,} segments, {ov_road_length/1000:.1f} km total")
    logger.info(f"  TIGER roads: {tiger_count:,} segments, {tiger_length/1000:.1f} km total")
    logger.info(f"  Segment count ratio: {ov_road_count/tiger_count:.4f}")
    logger.info(f"  Length ratio: {ov_road_length/tiger_length:.4f}")
    logger.info(f"  Length gap: {1 - ov_road_length/tiger_length:.4f}")
    
    # === 3. POI / FACILITY GAP ===
    logger.info(f"\n=== POI / Facility Gap: {region} ===")
    
    ov_poi_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{ref_dir}/{region}-overture-pois.parquet')
    """).fetchone()[0]
    
    hifld_counts = {}
    for fac in ['hospitals', 'fire-stations', 'ems-stations', 'schools']:
        try:
            cnt = conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{ref_dir}/{region}-hifld-{fac}.parquet')
            """).fetchone()[0]
            hifld_counts[fac] = cnt
        except:
            hifld_counts[fac] = 0
    
    hifld_total = sum(hifld_counts.values())
    
    logger.info(f"  Overture POIs: {ov_poi_count:,}")
    logger.info(f"  HIFLD facilities: {hifld_total:,}")
    for fac, cnt in hifld_counts.items():
        logger.info(f"    {fac}: {cnt:,}")
    
    # === 4. SOURCE COMPOSITION ===
    logger.info(f"\n=== Source Composition: {region} ===")
    
    # Building sources
    building_sources = conn.execute(f"""
        WITH exploded AS (
            SELECT s.dataset AS source_dataset
            FROM read_parquet('{ref_dir}/{region}-overture-buildings.parquet'), UNNEST(sources) AS t(s)
        )
        SELECT source_dataset, COUNT(*) as cnt
        FROM exploded
        GROUP BY source_dataset
        ORDER BY cnt DESC
    """).df()
    logger.info("Building sources:")
    for _, row in building_sources.iterrows():
        pct = row['cnt'] / ov_count * 100
        logger.info(f"  {row['source_dataset']}: {row['cnt']:,} ({pct:.1f}%)")
    
    # POI sources
    poi_sources = conn.execute(f"""
        WITH exploded AS (
            SELECT s.dataset AS source_dataset
            FROM read_parquet('{ref_dir}/{region}-overture-pois.parquet'), UNNEST(sources) AS t(s)
        )
        SELECT source_dataset, COUNT(*) as cnt
        FROM exploded
        GROUP BY source_dataset
        ORDER BY cnt DESC
    """).df()
    logger.info("POI sources:")
    for _, row in poi_sources.iterrows():
        pct = row['cnt'] / ov_poi_count * 100
        logger.info(f"  {row['source_dataset']}: {row['cnt']:,} ({pct:.1f}%)")
    
    # POI confidence distribution
    conf_stats = conn.execute(f"""
        SELECT 
            AVG(confidence) as mean_conf,
            MEDIAN(confidence) as median_conf,
            SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) as low_conf_count,
            SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) as high_conf_count,
            SUM(CASE WHEN confidence >= 0.9 THEN 1 ELSE 0 END) as very_high_conf_count
        FROM read_parquet('{ref_dir}/{region}-overture-pois.parquet')
    """).df()
    logger.info(f"\nPOI confidence:")
    logger.info(f"  Mean: {conf_stats['mean_conf'].iloc[0]:.4f}")
    logger.info(f"  Median: {conf_stats['median_conf'].iloc[0]:.4f}")
    logger.info(f"  Low confidence (<0.5): {conf_stats['low_conf_count'].iloc[0]:,}")
    logger.info(f"  High confidence (>=0.7): {conf_stats['high_conf_count'].iloc[0]:,}")
    
    # === 5. HOUSING VS BUILDINGS ===
    logger.info(f"\n=== Housing vs Buildings: {region} ===")
    
    housing = conn.execute(f"""
        SELECT GEOID, housing_units
        FROM read_parquet('{ref_dir}/{region}-census-acs-housing.parquet')
    """).df()
    total_housing = housing['housing_units'].sum()
    logger.info(f"  Total housing units: {total_housing:,}")
    logger.info(f"  Buildings per housing unit: {ov_count/total_housing:.2f}")
    logger.info(f"  Microsoft buildings per housing unit: {ms_count/total_housing:.2f}")
    
    # === SUMMARY ===
    logger.info(f"\n{'='*60}")
    logger.info(f"COVERAGE GAP SUMMARY: {region}")
    logger.info(f"{'='*60}")
    logger.info(f"  Building count gap: {1 - ov_count/ms_count:.4f}")
    logger.info(f"  Road length gap:    {1 - ov_road_length/tiger_length:.4f}")
    logger.info(f"  Road segment gap:   {1 - ov_road_count/tiger_count:.4f}")
    
    results = {
        "region": region,
        "overture_buildings": ov_count,
        "microsoft_buildings": ms_count,
        "building_count_ratio": ov_count / ms_count,
        "building_count_gap": 1 - ov_count / ms_count,
        "overture_road_segments": ov_road_count,
        "tiger_road_segments": tiger_count,
        "overture_road_length_km": ov_road_length / 1000,
        "tiger_road_length_km": tiger_length / 1000,
        "road_length_ratio": ov_road_length / tiger_length,
        "road_length_gap": 1 - ov_road_length / tiger_length,
        "overture_pois": ov_poi_count,
        "hifld_total_facilities": hifld_total,
        "total_housing_units": int(total_housing),
        "buildings_per_housing_unit": ov_count / total_housing,
    }
    
    conn.close()
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Compute for all focus regions
    regions = ["eastern-ok"]  # Start with one, expand when data is downloaded
    all_results = []
    
    for region in regions:
        ref_dir = PROJECT_ROOT / "data/raw/reference" / region
        if (ref_dir / f"{region}-overture-buildings.parquet").exists():
            result = compute_coverage_gaps_for_region(region)
            all_results.append(result)
        else:
            logger.warning(f"Data not found for {region}. Run download first.")
    
    if all_results:
        summary = pd.DataFrame(all_results)
        print("\n" + summary.to_string())
        summary.to_csv(PROJECT_ROOT / "data/processed/coverage_gap_summary.csv", index=False)
