"""
Enhanced feature engineering pipeline for Bias Bounty Mapping Equity Challenge.

Merges:
1. Per-tract coverage gap features (buildings, roads, POIs, facilities, housing)
2. Strata table features (SVI, CVI, tribal, rural/urban, hazard indicators) 
3. Source composition features (ML fraction, OSM fraction, diversity, staleness)
4. Null-as-signal features (*_covered flags, data coverage depth)
5. Spatial lag features (k-nearest neighbor aggregates)
6. Interaction terms (svi_x_rural, tribal_x_wildfire, compound_risk)

This produces 100+ features per tract for the final model.
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.neighbors import BallTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_strata_features(region: str) -> pd.DataFrame:
    """Load and preprocess strata table features for a region."""
    strata_path = PROJECT_ROOT / f"data/raw/strata/{region}/{region}-strata-tract-table.parquet"
    if not strata_path.exists():
        logger.warning(f"No strata table for {region}")
        return pd.DataFrame()
    
    df = pd.read_parquet(strata_path)
    logger.info(f"  Strata table: {df.shape[0]} tracts, {df.shape[1]} columns")
    
    # Identify key feature columns
    feature_cols = []
    
    # SVI columns (Social Vulnerability Index)
    svi_cols = [c for c in df.columns if 'svi' in c.lower() and 'covered' not in c.lower()]
    feature_cols.extend(svi_cols)
    
    # CVI columns (Climate Vulnerability Index)
    cvi_cols = [c for c in df.columns if 'cvi' in c.lower() and 'covered' not in c.lower()]
    feature_cols.extend(cvi_cols)
    
    # Rural/Urban indicators
    ruca_cols = [c for c in df.columns if 'ruca' in c.lower() or 'rural' in c.lower() or 'urban' in c.lower()]
    feature_cols.extend(ruca_cols)
    
    # Tribal indicators
    tribal_cols = [c for c in df.columns if 'tribal' in c.lower() and 'covered' not in c.lower()]
    feature_cols.extend(tribal_cols)
    
    # Hazard indicators
    hazard_cols = [c for c in df.columns if any(h in c.lower() for h in ['wildfire', 'flood', 'drought', 'hurricane', 'earthquake', 'hazard', 'usdm', 'usfs']) and 'covered' not in c.lower()]
    feature_cols.extend(hazard_cols)
    
    # Covered flags (null-as-signal)
    covered_cols = [c for c in df.columns if c.endswith('_covered')]
    feature_cols.extend(covered_cols)
    
    # ACS demographic columns
    acs_cols = [c for c in df.columns if any(p in c.lower() for p in ['population', 'median_income', 'poverty', 'uninsur', 'age_', 'hispanic', 'black', 'native', 'asian', 'white', 'minority'])]
    feature_cols.extend(acs_cols)
    
    # Housing columns
    housing_cols = [c for c in df.columns if any(p in c.lower() for p in ['housing', 'occupancy', 'rent', 'mobile', 'vacant'])]
    feature_cols.extend(housing_cols)
    
    # Deduplicate while preserving order
    seen = set()
    unique_cols = []
    for c in feature_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            unique_cols.append(c)
    
    # Always include GEOID
    if 'GEOID' not in unique_cols:
        unique_cols = ['GEOID'] + unique_cols
    
    result = df[unique_cols].copy()
    
    # Convert covered flags to numeric: 1=covered, 0=not covered, -1=null (data doesn't reach)
    for col in covered_cols:
        if col in result.columns:
            result[col] = result[col].map({True: 1, False: 0, None: -1, pd.NA: -1})
            result[col] = result[col].fillna(-1).astype(int)
    
    # Compute data coverage depth (how many data layers reach this tract)
    if covered_cols:
        result['data_coverage_depth'] = sum(
            (result[c] != -1).astype(int) for c in covered_cols if c in result.columns
        )
        result['data_coverage_fraction'] = result['data_coverage_depth'] / len(covered_cols)
    
    logger.info(f"  Extracted {len(unique_cols)} strata features ({len(covered_cols)} covered flags)")
    
    return result


def compute_source_composition_features(region: str) -> pd.DataFrame:
    """
    Compute source composition features per tract from Overture data.
    Uses DuckDB to parse nested sources[] structs.
    """
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    ref_dir = PROJECT_ROOT / f"data/raw/reference/{region}"
    strata_dir = PROJECT_ROOT / f"data/raw/strata/{region}"
    tracts_path = strata_dir / f"{region}-census-tracts.parquet"
    
    results = {}
    
    # === Building source composition ===
    ov_build_path = ref_dir / f"{region}-overture-buildings.parquet"
    if ov_build_path.exists():
        try:
            # Per-tract source counts with bbox join
            src_comp = conn.execute(f"""
                WITH tracts AS (
                    SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1
                    FROM read_parquet('{tracts_path}')
                ),
                buildings AS (
                    SELECT bbox.xmin bx0, bbox.ymin by0, bbox.xmax bx1, bbox.ymax by1,
                           sources
                    FROM read_parquet('{ov_build_path}')
                ),
                joined AS (
                    SELECT t.GEOID, b.sources
                    FROM tracts t JOIN buildings b ON 
                        b.bx1>=t.x0 AND b.bx0<=t.x1 AND b.by1>=t.y0 AND b.by0<=t.y1
                ),
                exploded AS (
                    SELECT GEOID, s.dataset AS source_dataset
                    FROM joined, UNNEST(sources) AS t(s)
                )
                SELECT 
                    GEOID,
                    COUNT(*) AS bldg_total_sources,
                    COUNT(DISTINCT source_dataset) AS bldg_source_diversity,
                    CAST(SUM(CASE WHEN source_dataset = 'OpenStreetMap' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS bldg_osm_fraction,
                    CAST(SUM(CASE WHEN source_dataset LIKE '%Microsoft%' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS bldg_ms_ml_fraction,
                    CAST(SUM(CASE WHEN source_dataset LIKE '%Google%' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS bldg_google_fraction,
                    CAST(SUM(CASE WHEN source_dataset LIKE '%Esri%' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS bldg_esri_fraction
                FROM exploded
                GROUP BY GEOID
            """).df()
            results['building_sources'] = src_comp
            logger.info(f"  Building source composition: {len(src_comp)} tracts")
        except Exception as e:
            logger.warning(f"  Building source composition failed: {e}")
    
    # === POI confidence features ===
    poi_path = ref_dir / f"{region}-overture-pois.parquet"
    if poi_path.exists():
        try:
            poi_comp = conn.execute(f"""
                WITH tracts AS (
                    SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1
                    FROM read_parquet('{tracts_path}')
                ),
                pois AS (
                    SELECT bbox.xmin px0, bbox.ymin py0, bbox.xmax px1, bbox.ymax py1,
                           confidence
                    FROM read_parquet('{poi_path}')
                ),
                joined AS (
                    SELECT t.GEOID, p.confidence
                    FROM tracts t JOIN pois p ON 
                        p.px1>=t.x0 AND p.px0<=t.x1 AND p.py1>=t.y0 AND p.py0<=t.y1
                )
                SELECT 
                    GEOID,
                    COUNT(*) AS poi_count_by_conf,
                    AVG(confidence) AS poi_mean_confidence,
                    CAST(SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS poi_low_conf_fraction,
                    CAST(SUM(CASE WHEN confidence >= 0.9 THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS poi_very_high_conf_fraction
                FROM joined
                GROUP BY GEOID
            """).df()
            results['poi_confidence'] = poi_comp
            logger.info(f"  POI confidence features: {len(poi_comp)} tracts")
        except Exception as e:
            logger.warning(f"  POI confidence features failed: {e}")
    
    conn.close()
    
    # Merge source composition results
    if not results:
        return pd.DataFrame()
    
    merged = None
    for key, df in results.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='GEOID', how='outer')
    
    return merged


def compute_spatial_lag_features(features: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Compute spatial lag features using k-nearest neighbors.
    These capture spatial autocorrelation and neighborhood context.
    """
    strata_dir = PROJECT_ROOT / f"data/raw/strata/{region}"
    tracts_path = strata_dir / f"{region}-census-tracts.parquet"
    
    if not tracts_path.exists():
        logger.warning(f"  No census tracts file for spatial lags")
        return features
    
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    # Get tract centroids
    centroids = conn.execute(f"""
        SELECT GEOID, 
               ST_X(ST_Centroid(geometry)) AS centroid_lon,
               ST_Y(ST_Centroid(geometry)) AS centroid_lat
        FROM read_parquet('{tracts_path}')
    """).df()
    conn.close()
    
    # Merge centroids with features
    features = features.merge(centroids, on='GEOID', how='left')
    
    # Build BallTree for k-nearest neighbor queries
    valid_mask = features['centroid_lat'].notna() & features['centroid_lon'].notna()
    if valid_mask.sum() < 10:
        logger.warning(f"  Too few valid centroids for spatial lags")
        return features
    
    coords = features.loc[valid_mask, ['centroid_lat', 'centroid_lon']].values
    coords_rad = np.deg2rad(coords)
    tree = BallTree(coords_rad, metric='haversine')
    
    # Key numeric columns to compute spatial lags for
    lag_cols = [c for c in features.columns 
                if features[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                and c not in ['GEOID', 'centroid_lat', 'centroid_lon']
                and features[c].notna().sum() > len(features) * 0.5]  # At least 50% non-null
    
    # Limit to most important features to avoid explosion
    lag_cols = lag_cols[:20]
    
    for k in [5, 10, 20]:
        distances, indices = tree.query(coords_rad, k=min(k + 1, len(coords_rad)))
        
        # Remove self from neighbors
        indices = indices[:, 1:]
        
        for col in lag_cols:
            if col not in features.columns:
                continue
            
            values = features.loc[valid_mask, col].values
            
            # Compute neighbor mean
            neighbor_means = np.array([
                np.nanmean(values[indices[i]]) if len(indices[i]) > 0 else np.nan
                for i in range(len(indices))
            ])
            
            lag_col_name = f"{col}_knn{k}_mean"
            features.loc[valid_mask, lag_col_name] = neighbor_means
            
            # Spatial lag: difference from neighbors (local spatial outlier)
            diff_col_name = f"{col}_knn{k}_diff"
            features.loc[valid_mask, diff_col_name] = values - neighbor_means
    
    logger.info(f"  Spatial lag features: {len([c for c in features.columns if '_knn' in c])} columns")
    
    return features


def _is_numeric(series: pd.Series) -> bool:
    """Check if a pandas Series is numeric (not string/object)."""
    return series.dtype in [np.float64, np.float32, np.int64, np.int32, float, int] or \
           (series.dtype == object and pd.to_numeric(series, errors='coerce').notna().mean() > 0.5)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Safely convert to numeric, returning NaN for non-numeric."""
    if series.dtype in [np.float64, np.float32, np.int64, np.int32]:
        return series
    return pd.to_numeric(series, errors='coerce')


def compute_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction features between vulnerability and coverage gap dimensions.
    These capture compounding effects that linear models miss.
    """
    # Filter to only numeric candidates for interactions
    svi_candidates = [c for c in features.columns if 'svi' in c.lower() and 'covered' not in c.lower() and _is_numeric(features[c])]
    rural_candidates = [c for c in features.columns if ('ruca' in c.lower() or 'rural' in c.lower()) and _is_numeric(features[c])]
    tribal_candidates = [c for c in features.columns if 'tribal' in c.lower() and 'covered' not in c.lower() and _is_numeric(features[c])]
    hazard_candidates = [c for c in features.columns if any(h in c.lower() for h in ['wildfire', 'flood', 'drought', 'usfs', 'usdm']) and 'covered' not in c.lower() and _is_numeric(features[c])]
    gap_candidates = [c for c in features.columns if ('gap' in c.lower() or 'ratio' in c.lower()) and _is_numeric(features[c])]
    
    # SVI x Rural interaction
    if svi_candidates and rural_candidates:
        svi_col = svi_candidates[0]
        rural_col = rural_candidates[0]
        features['svi_x_rural'] = _to_numeric(features[svi_col]).fillna(0) * _to_numeric(features[rural_col]).fillna(0)
    
    # SVI x Building gap interaction
    if svi_candidates and 'building_gap' in features.columns and _is_numeric(features['building_gap']):
        svi_col = svi_candidates[0]
        features['svi_x_building_gap'] = _to_numeric(features[svi_col]).fillna(0) * features['building_gap'].fillna(0)
    
    # SVI x Road gap interaction
    if svi_candidates and 'road_gap' in features.columns and _is_numeric(features['road_gap']):
        svi_col = svi_candidates[0]
        features['svi_x_road_gap'] = _to_numeric(features[svi_col]).fillna(0) * features['road_gap'].fillna(0)
    
    # Tribal x Hazard interaction
    if tribal_candidates and hazard_candidates:
        tribal_col = tribal_candidates[0]
        hazard_col = hazard_candidates[0]
        features['tribal_x_hazard'] = _to_numeric(features[tribal_col]).fillna(0) * _to_numeric(features[hazard_col]).fillna(0)
    
    # Tribal x Building gap
    if tribal_candidates and 'building_gap' in features.columns and _is_numeric(features['building_gap']):
        tribal_col = tribal_candidates[0]
        features['tribal_x_building_gap'] = _to_numeric(features[tribal_col]).fillna(0) * features['building_gap'].fillna(0)
    
    # Compound risk score: building gap + road gap + low data coverage
    gap_cols = [c for c in ['building_gap', 'road_gap'] if c in features.columns and _is_numeric(features[c])]
    if gap_cols:
        compound = sum(features[c].fillna(0) for c in gap_cols)
        if 'data_coverage_fraction' in features.columns and _is_numeric(features['data_coverage_fraction']):
            compound = compound + (1 - features['data_coverage_fraction'].fillna(1))
        features['compound_risk_score'] = compound
    
    # Coverage gap x ML-derived fraction (models less reliable for ML-derived data)
    if 'building_gap' in features.columns and 'bldg_ms_ml_fraction' in features.columns:
        if _is_numeric(features['building_gap']) and _is_numeric(features['bldg_ms_ml_fraction']):
            features['gap_x_ml_fraction'] = features['building_gap'].fillna(0) * features['bldg_ms_ml_fraction'].fillna(0)
    
    # Log transforms of key features
    for col in ['building_ratio', 'road_ratio', 'bldg_per_housing']:
        if col in features.columns and _is_numeric(features[col]):
            features[f'log_{col}'] = np.log1p(features[col].clip(lower=0).fillna(0))
    
    # Squared terms for nonlinear relationships
    for col in ['building_gap', 'road_gap']:
        if col in features.columns and _is_numeric(features[col]):
            features[f'{col}_sq'] = features[col].fillna(0) ** 2
    
    logger.info(f"  Interaction features: {len([c for c in features.columns if any(p in c for p in ['_x_', 'compound_', 'log_', '_sq'])])} columns")
    
    return features


def compute_county_aggregate_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute county-level aggregate features (between-tract variation).
    """
    # Derive county FIPS from GEOID (first 5 digits of 11-digit GEOID)
    features['county_fips'] = features['GEOID'].str[:5]
    features['state_fips'] = features['GEOID'].str[:2]
    
    # Numeric columns for aggregation
    numeric_cols = [c for c in features.columns 
                   if features[c].dtype in [np.float64, np.float32] 
                   and c not in ['GEOID', 'centroid_lat', 'centroid_lon']
                   and features[c].notna().sum() > len(features) * 0.5]
    numeric_cols = numeric_cols[:15]  # Limit to avoid explosion
    
    if not numeric_cols or 'county_fips' not in features.columns:
        return features
    
    # County means
    county_means = features.groupby('county_fips')[numeric_cols].mean()
    county_means.columns = [f'{c}_county_mean' for c in county_means.columns]
    features = features.merge(county_means, left_on='county_fips', right_index=True, how='left')
    
    # County standard deviations (measures within-county inequality)
    county_stds = features.groupby('county_fips')[numeric_cols].std()
    county_stds.columns = [f'{c}_county_std' for c in county_stds.columns]
    features = features.merge(county_stds, left_on='county_fips', right_index=True, how='left')
    
    # Deviation from county mean (how unusual is this tract in its county?)
    for col in numeric_cols[:10]:
        mean_col = f'{col}_county_mean'
        if mean_col in features.columns:
            features[f'{col}_county_dev'] = features[col].fillna(0) - features[mean_col].fillna(0)
    
    logger.info(f"  County aggregate features: {len([c for c in features.columns if '_county_' in c])} columns")
    
    return features


def build_enhanced_features(region: str) -> pd.DataFrame:
    """
    Build the complete enhanced feature set for a focus region.
    
    Pipeline:
    1. Load base coverage gap features
    2. Merge strata table features (SVI, CVI, tribal, rural, hazard)
    3. Add source composition features (ML fraction, OSM fraction, diversity)
    4. Add null-as-signal features (*_covered flags, data coverage depth)
    5. Add spatial lag features (k-NN aggregates)
    6. Add interaction features (svi_x_rural, tribal_x_hazard, compound_risk)
    7. Add county aggregate features (between-tract variation)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Building enhanced features: {region}")
    logger.info(f"{'='*60}")
    
    # 1. Load base coverage gap features
    base_path = PROJECT_ROOT / f"data/features/{region}_tract_features.parquet"
    if not base_path.exists():
        logger.error(f"  No base features for {region}")
        return pd.DataFrame()
    
    features = pd.read_parquet(base_path)
    logger.info(f"  Base features: {features.shape[0]} tracts, {features.shape[1]} columns")
    
    # 2. Merge strata table features
    strata_features = load_strata_features(region)
    if not strata_features.empty:
        features = features.merge(strata_features, on='GEOID', how='left')
        logger.info(f"  After strata merge: {features.shape[1]} columns")
    
    # 3. Add source composition features
    source_features = compute_source_composition_features(region)
    if not source_features.empty:
        features = features.merge(source_features, on='GEOID', how='left')
        logger.info(f"  After source composition: {features.shape[1]} columns")
    
    # 4. Spatial lag features (adds k-NN neighbor averages)
    features = compute_spatial_lag_features(features, region)
    logger.info(f"  After spatial lags: {features.shape[1]} columns")
    
    # 5. Interaction features
    features = compute_interaction_features(features)
    logger.info(f"  After interactions: {features.shape[1]} columns")
    
    # 6. County aggregate features
    features = compute_county_aggregate_features(features)
    logger.info(f"  After county aggregates: {features.shape[1]} columns")
    
    # Final cleanup
    # Replace inf with nan
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN in numeric columns with 0 (for count-type features)
    # or leave as NaN for ratio-type features (tree models handle NaN)
    
    # Add region indicator
    features['region'] = region
    
    # Summary
    total_features = features.shape[1] - 1  # exclude GEOID
    numeric_features = features.select_dtypes(include=[np.number]).shape[1]
    logger.info(f"\n  FINAL: {features.shape[0]} tracts, {total_features} total features, {numeric_features} numeric")
    
    return features


def build_national_enhanced_features() -> pd.DataFrame:
    """Build enhanced features for national strata table."""
    logger.info("\nBuilding national enhanced features from strata table...")
    
    strata_path = PROJECT_ROOT / "data/raw/strata/national/national-strata-tract-table.parquet"
    if not strata_path.exists():
        logger.error("National strata table not found")
        return pd.DataFrame()
    
    df = pd.read_parquet(strata_path)
    logger.info(f"  National strata: {df.shape[0]} tracts, {df.shape[1]} columns")
    
    # Process covered flags
    covered_cols = [c for c in df.columns if c.endswith('_covered')]
    
    for col in covered_cols:
        df[col] = df[col].map({True: 1, False: 0, None: -1, pd.NA: -1})
        df[col] = df[col].fillna(-1).astype(int)
    
    # Data coverage depth
    if covered_cols:
        df['data_coverage_depth'] = sum(
            (df[c] != -1).astype(int) for c in covered_cols if c in df.columns
        )
        df['data_coverage_fraction'] = df['data_coverage_depth'] / len(covered_cols)
    
    # Derive county and state
    df['county_fips'] = df['GEOID'].str[:5]
    df['state_fips'] = df['GEOID'].str[:2]
    
    # CONUS indicator
    conus_states = ['01','04','05','06','08','09','10','11','12','13','15','16','17','18','19',
                    '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34',
                    '35','36','37','38','39','40','41','42','44','45','46','47','48','49','50',
                    '51','53','54','55','56']
    df['is_conus'] = df['state_fips'].isin(conus_states).astype(int)
    
    logger.info(f"  National enhanced: {df.shape[0]} tracts, {df.shape[1]} columns")
    
    return df


def run_pipeline():
    """Run the full enhanced feature engineering pipeline."""
    regions = ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']
    
    all_features = []
    
    for region in regions:
        features = build_enhanced_features(region)
        if not features.empty:
            all_features.append(features)
            
            # Save enhanced features
            out_path = PROJECT_ROOT / f"data/features/{region}_enhanced_features.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(out_path)
            logger.info(f"  Saved: {out_path}")
    
    # Combine all regions
    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        
        # Handle duplicate columns from merge
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        out_path = PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet"
        combined.to_parquet(out_path)
        logger.info(f"\nCombined enhanced features: {combined.shape[0]} tracts, {combined.shape[1]} columns")
        logger.info(f"Saved: {out_path}")
        
        # Print feature summary
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  Numeric features: {len(numeric_cols)}")
        
        # Key feature statistics
        for col in ['building_gap', 'road_gap', 'compound_risk_score']:
            if col in combined.columns:
                logger.info(f"  {col}: mean={combined[col].mean():.4f}, "
                          f"std={combined[col].std():.4f}, "
                          f"null%={combined[col].isna().mean()*100:.1f}%")
    
    # Build national features
    national = build_national_enhanced_features()
    if not national.empty:
        out_path = PROJECT_ROOT / "data/features/national_enhanced_features.parquet"
        national.to_parquet(out_path)
        logger.info(f"  Saved national: {out_path}")
    
    return combined if all_features else pd.DataFrame()


if __name__ == "__main__":
    run_pipeline()
