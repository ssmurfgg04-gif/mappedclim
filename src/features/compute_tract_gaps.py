"""
Compute per-tract coverage gap features for ALL 4 focus regions.
Uses DuckDB for efficient spatial operations.

This produces the primary features for each census tract:
- building_count_ratio (Overture / Microsoft)
- road_length_ratio (Overture / TIGER)  
- poi_count vs HIFLD facility count
- building count vs ACS housing units
- Source composition features per tract
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def compute_tract_level_gaps(region: str) -> pd.DataFrame:
    """
    Compute per-tract coverage gap features using DuckDB spatial joins.
    """
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    raw_dir = PROJECT_ROOT / "data/raw"
    ref_dir = raw_dir / "reference" / region
    strata_dir = raw_dir / "strata" / region
    
    # === Load census tract polygons ===
    tracts_path = strata_dir / f"{region}-census-tracts.parquet"
    if not tracts_path.exists():
        logger.warning(f"No census tracts file for {region}")
        conn.close()
        return pd.DataFrame()
    
    # Get tract GEOIDs from strata table
    strata_path = strata_dir / f"{region}-strata-tract-table.parquet"
    if strata_path.exists():
        tract_geoids = conn.execute(f"""
            SELECT GEOID FROM read_parquet('{strata_path}')
        """).df()
    else:
        logger.warning(f"No strata table for {region}")
        conn.close()
        return pd.DataFrame()
    
    logger.info(f"Processing {region}: {len(tract_geoids):,} tracts")
    
    # === BUILDING COVERAGE PER TRACT ===
    logger.info(f"  Computing building coverage per tract...")
    
    # Count Overture buildings per tract (spatial join using tract polygons)
    ov_build_path = ref_dir / f"{region}-overture-buildings.parquet"
    ms_build_path = ref_dir / f"{region}-microsoft-buildings.parquet"
    
    if ov_build_path.exists() and ms_build_path.exists():
        # Use bbox-based spatial join for efficiency
        ov_per_tract = conn.execute(f"""
            WITH tracts AS (
                SELECT GEOID, geometry as tract_geom, 
                       bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                       bbox.xmax as t_xmax, bbox.ymax as t_ymax
                FROM read_parquet('{tracts_path}')
            ),
            buildings AS (
                SELECT bbox.xmin as b_xmin, bbox.ymin as b_ymin,
                       bbox.xmax as b_xmax, bbox.ymax as b_ymax,
                       geometry as b_geom
                FROM read_parquet('{ov_build_path}')
            )
            SELECT t.GEOID, COUNT(*) as overture_building_count
            FROM tracts t
            LEFT JOIN buildings b ON 
                b.b_xmax >= t.t_xmin AND b.b_xmin <= t.t_xmax AND
                b.b_ymax >= t.t_ymin AND b.b_ymin <= t.t_ymax AND
                ST_Contains(t.tract_geom, ST_Centroid(b.b_geom))
            GROUP BY t.GEOID
        """).df()
        
        ms_per_tract = conn.execute(f"""
            WITH tracts AS (
                SELECT GEOID, geometry as tract_geom,
                       bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                       bbox.xmax as t_xmax, bbox.ymax as t_ymax
                FROM read_parquet('{tracts_path}')
            ),
            buildings AS (
                SELECT bbox.xmin as b_xmin, bbox.ymin as b_ymin,
                       bbox.xmax as b_xmax, bbox.ymax as b_ymax,
                       geometry as b_geom
                FROM read_parquet('{ms_build_path}')
            )
            SELECT t.GEOID, COUNT(*) as ms_building_count
            FROM tracts t
            LEFT JOIN buildings b ON 
                b.b_xmax >= t.t_xmin AND b.b_xmin <= t.t_xmax AND
                b.b_ymax >= t.t_ymin AND b.b_ymin <= t.t_ymax AND
                ST_Contains(t.tract_geom, ST_Centroid(b.b_geom))
            GROUP BY t.GEOID
        """).df()
        
        logger.info(f"    Overture buildings joined: {ov_per_tract['overture_building_count'].sum():,}")
        logger.info(f"    Microsoft buildings joined: {ms_per_tract['ms_building_count'].sum():,}")
    else:
        ov_per_tract = pd.DataFrame(columns=["GEOID", "overture_building_count"])
        ms_per_tract = pd.DataFrame(columns=["GEOID", "ms_building_count"])
    
    # === ROAD COVERAGE PER TRACT ===
    logger.info(f"  Computing road coverage per tract...")
    
    ov_road_path = ref_dir / f"{region}-overture-roads.parquet"
    tiger_path = ref_dir / f"{region}-census-tiger-roads.parquet"
    
    if ov_road_path.exists() and tiger_path.exists():
        # Count road segments and compute length per tract
        ov_road_per_tract = conn.execute(f"""
            WITH tracts AS (
                SELECT GEOID, geometry as tract_geom,
                       bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                       bbox.xmax as t_xmax, bbox.ymax as t_ymax
                FROM read_parquet('{tracts_path}')
            ),
            roads AS (
                SELECT bbox.xmin as r_xmin, bbox.ymin as r_ymin,
                       bbox.xmax as r_xmax, bbox.ymax as r_ymax,
                       geometry as r_geom
                FROM read_parquet('{ov_road_path}')
            )
            SELECT t.GEOID, 
                   COUNT(*) as overture_road_count,
                   SUM(ST_Length(r.r_geom)) as overture_road_length
            FROM tracts t
            LEFT JOIN roads r ON 
                r.r_xmax >= t.t_xmin AND r.r_xmin <= t.t_xmax AND
                r.r_ymax >= t.t_ymin AND r.r_ymin <= t.t_ymax AND
                ST_Intersects(t.tract_geom, r.r_geom)
            GROUP BY t.GEOID
        """).df()
        
        tiger_per_tract = conn.execute(f"""
            WITH tracts AS (
                SELECT GEOID, geometry as tract_geom,
                       bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                       bbox.xmax as t_xmax, bbox.ymax as t_ymax
                FROM read_parquet('{tracts_path}')
            ),
            roads AS (
                SELECT bbox.xmin as r_xmin, bbox.ymin as r_ymin,
                       bbox.xmax as r_xmax, bbox.ymax as r_ymax,
                       geometry as r_geom
                FROM read_parquet('{tiger_path}')
            )
            SELECT t.GEOID,
                   COUNT(*) as tiger_road_count,
                   SUM(ST_Length(r.r_geom)) as tiger_road_length
            FROM tracts t
            LEFT JOIN roads r ON 
                r.r_xmax >= t.t_xmin AND r.r_xmin <= t.t_xmax AND
                r.r_ymax >= t.t_ymin AND r.r_ymin <= t.t_ymax AND
                ST_Intersects(t.tract_geom, r.r_geom)
            GROUP BY t.GEOID
        """).df()
    else:
        ov_road_per_tract = pd.DataFrame(columns=["GEOID", "overture_road_count", "overture_road_length"])
        tiger_per_tract = pd.DataFrame(columns=["GEOID", "tiger_road_count", "tiger_road_length"])
    
    # === POI COUNT PER TRACT ===
    logger.info(f"  Computing POI coverage per tract...")
    
    poi_path = ref_dir / f"{region}-overture-pois.parquet"
    if poi_path.exists():
        poi_per_tract = conn.execute(f"""
            WITH tracts AS (
                SELECT GEOID, geometry as tract_geom,
                       bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                       bbox.xmax as t_xmax, bbox.ymax as t_ymax
                FROM read_parquet('{tracts_path}')
            ),
            pois AS (
                SELECT bbox.xmin as p_xmin, bbox.ymin as p_ymin,
                       bbox.xmax as p_xmax, bbox.ymax as p_ymax,
                       geometry as p_geom, confidence
                FROM read_parquet('{poi_path}')
            )
            SELECT t.GEOID,
                   COUNT(*) as overture_poi_count,
                   AVG(p.confidence) as mean_poi_confidence,
                   SUM(CASE WHEN p.confidence < 0.5 THEN 1 ELSE 0 END) as low_conf_poi_count,
                   SUM(CASE WHEN p.confidence >= 0.7 THEN 1 ELSE 0 END) as high_conf_poi_count
            FROM tracts t
            LEFT JOIN pois p ON 
                p.p_xmax >= t.t_xmin AND p.p_xmin <= t.t_xmax AND
                p.p_ymax >= t.t_ymin AND p.p_ymin <= t.t_ymax AND
                ST_Contains(t.tract_geom, p.p_geom)
            GROUP BY t.GEOID
        """).df()
    else:
        poi_per_tract = pd.DataFrame(columns=["GEOID", "overture_poi_count", "mean_poi_confidence", "low_conf_poi_count", "high_conf_poi_count"])
    
    # === HIFLD FACILITIES PER TRACT ===
    logger.info(f"  Computing HIFLD facilities per tract...")
    
    facility_counts = {}
    for fac in ['hospitals', 'fire-stations', 'ems-stations', 'schools']:
        fac_path = ref_dir / f"{region}-hifld-{fac}.parquet"
        if fac_path.exists():
            try:
                fac_count = conn.execute(f"""
                    WITH tracts AS (
                        SELECT GEOID, geometry as tract_geom,
                               bbox.xmin as t_xmin, bbox.ymin as t_ymin,
                               bbox.xmax as t_xmax, bbox.ymax as t_ymax
                        FROM read_parquet('{tracts_path}')
                    ),
                    fac AS (
                        SELECT bbox.xmin as f_xmin, bbox.ymin as f_ymin,
                               bbox.xmax as f_xmax, bbox.ymax as f_ymax,
                               geometry as f_geom
                        FROM read_parquet('{fac_path}')
                    )
                    SELECT t.GEOID, COUNT(*) as cnt
                    FROM tracts t
                    LEFT JOIN fac f ON 
                        f.f_xmax >= t.t_xmin AND f.f_xmin <= t.t_xmax AND
                        f.f_ymax >= t.t_ymin AND f.f_ymin <= t.t_ymax AND
                        ST_Contains(t.tract_geom, f.f_geom)
                    GROUP BY t.GEOID
                """).df()
                facility_counts[fac] = fac_count.rename(columns={"cnt": f"hifld_{fac}_count"})
            except:
                pass
    
    # === HOUSING UNITS PER TRACT ===
    logger.info(f"  Computing housing units per tract...")
    
    housing_path = ref_dir / f"{region}-census-acs-housing.parquet"
    if housing_path.exists():
        housing = conn.execute(f"""
            SELECT GEOID, housing_units
            FROM read_parquet('{housing_path}')
        """).df()
    else:
        housing = pd.DataFrame(columns=["GEOID", "housing_units"])
    
    # === MERGE ALL FEATURES ===
    logger.info(f"  Merging features...")
    
    # Start with all tract GEOIDs
    features = tract_geoids.copy()
    
    # Left join all per-tract counts
    for df in [ov_per_tract, ms_per_tract, ov_road_per_tract, tiger_per_tract, 
               poi_per_tract, housing]:
        if not df.empty and 'GEOID' in df.columns:
            features = features.merge(df, on='GEOID', how='left')
    
    for fac_name, fac_df in facility_counts.items():
        if not fac_df.empty:
            features = features.merge(fac_df, on='GEOID', how='left')
    
    # Fill NaN with 0 for count columns
    count_cols = [c for c in features.columns if c != 'GEOID' and c != 'mean_poi_confidence']
    for col in count_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    
    # === COMPUTE DERIVED FEATURES ===
    logger.info(f"  Computing derived features...")
    
    # Building ratio and gap
    features['building_count_ratio'] = features['overture_building_count'] / features['ms_building_count'].replace(0, np.nan)
    features['building_count_gap'] = 1 - features['building_count_ratio']
    
    # Road ratio and gap
    features['road_count_ratio'] = features['overture_road_count'] / features['tiger_road_count'].replace(0, np.nan)
    features['road_count_gap'] = 1 - features['road_count_ratio']
    features['road_length_ratio'] = features['overture_road_length'] / features['tiger_road_length'].replace(0, np.nan)
    features['road_length_gap'] = 1 - features['road_length_ratio']
    
    # POI features
    hifld_cols = [c for c in features.columns if c.startswith('hifld_') and c.endswith('_count')]
    features['hifld_total_facility_count'] = features[hifld_cols].sum(axis=1)
    features['poi_to_facility_ratio'] = features['overture_poi_count'] / features['hifld_total_facility_count'].replace(0, np.nan)
    features['poi_facility_gap'] = 1 - features['poi_to_facility_ratio']
    
    # Low confidence fraction
    features['low_confidence_fraction'] = features['low_conf_poi_count'] / features['overture_poi_count'].replace(0, np.nan)
    
    # Housing features
    features['buildings_per_housing_unit'] = features['overture_building_count'] / features['housing_units'].replace(0, np.nan)
    
    # Clip ratios
    for col in ['building_count_ratio', 'road_count_ratio', 'road_length_ratio']:
        features[col] = features[col].clip(0, 5)
    for col in ['building_count_gap', 'road_count_gap', 'road_length_gap']:
        features[col] = features[col].clip(-4, 1)
    
    # Region indicator
    features['region'] = region
    
    logger.info(f"  {region}: {features.shape[0]:,} tracts, {features.shape[1]} features")
    
    conn.close()
    return features


def compute_all_regions():
    """Compute per-tract features for all 4 focus regions."""
    regions = ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']
    all_features = []
    
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {region}")
        logger.info(f"{'='*60}")
        
        try:
            features = compute_tract_level_gaps(region)
            if not features.empty:
                all_features.append(features)
                
                # Save per-region
                out_path = PROJECT_ROOT / f"data/features/{region}_tract_features.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                features.to_parquet(out_path)
                logger.info(f"Saved to {out_path}")
        except Exception as e:
            logger.error(f"Failed for {region}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_features:
        # Combine all regions
        combined = pd.concat(all_features, ignore_index=True)
        out_path = PROJECT_ROOT / "data/features/all_regions_tract_features.parquet"
        combined.to_parquet(out_path)
        logger.info(f"\nCombined features: {combined.shape[0]:,} tracts, {combined.shape[1]} columns")
        logger.info(f"Saved to {out_path}")
        
        # Summary stats
        for col in ['building_count_gap', 'road_length_gap', 'poi_facility_gap']:
            if col in combined.columns:
                logger.info(f"  {col}: mean={combined[col].mean():.4f}, std={combined[col].std():.4f}")
        
        return combined
    
    return pd.DataFrame()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    compute_all_regions()
