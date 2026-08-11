"""
Feature engineering for Bias Bounty Mapping Equity Challenge.

Computes coverage gap features, source composition features, null flags,
structural/topological features, and spatial lag features per census tract.

This is the 80% of the work that determines 80% of the score.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CoverageGapFeatures:
    """
    Compute coverage gap features by comparing Overture Maps against
    reference datasets (TIGER/Line, Microsoft Buildings, HIFLD).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def compute_building_gap_features(
        self,
        overture_buildings: gpd.GeoDataFrame,
        microsoft_buildings: gpd.GeoDataFrame,
        tracts: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Compute building coverage gap features per census tract.
        
        Key insight: Compare both COUNT and AREA ratios, plus
        spatial distribution metrics (clustering, sparsity).
        """
        logger.info("Computing building coverage gap features...")

        # Spatial join: count buildings per tract
        overture_in_tracts = gpd.sjoin(overture_buildings, tracts, how="left", predicate="within")
        ms_in_tracts = gpd.sjoin(microsoft_buildings, tracts, how="left", predicate="within")

        # Count per tract
        ov_counts = overture_in_tracts.groupby("GEOID_right").size().rename("overture_building_count")
        ms_counts = ms_in_tracts.groupby("GEOID_right").size().rename("ms_building_count")

        # Area per tract
        ov_areas = overture_in_tracts.groupby("GEOID_right")["geometry"].apply(
            lambda g: g.area.sum()
        ).rename("overture_building_area")
        ms_areas = ms_in_tracts.groupby("GEOID_right")["geometry"].apply(
            lambda g: g.area.sum()
        ).rename("ms_building_area")

        # Merge
        features = pd.DataFrame(index=tracts["GEOID"])
        features = features.join(ov_counts).join(ms_counts).join(ov_areas).join(ms_areas)

        # Fill NaN with 0 (no buildings in tract)
        for col in ["overture_building_count", "ms_building_count",
                     "overture_building_area", "ms_building_area"]:
            features[col] = features[col].fillna(0)

        # === RATIO FEATURES (most important) ===
        features["building_count_ratio"] = (
            features["overture_building_count"] / features["ms_building_count"].replace(0, np.nan)
        )
        features["building_area_ratio"] = (
            features["overture_building_area"] / features["ms_building_area"].replace(0, np.nan)
        )

        # === GAP FEATURES (what we're predicting) ===
        features["building_count_gap"] = 1 - features["building_count_ratio"]
        features["building_area_gap"] = 1 - features["building_area_ratio"]

        # === DIFFERENCE FEATURES ===
        features["building_count_diff"] = (
            features["ms_building_count"] - features["overture_building_count"]
        )
        features["building_area_diff"] = (
            features["ms_building_area"] - features["overture_building_area"]
        )

        # === DENSITY FEATURES ===
        tract_area = tracts.set_index("GEOID")["ALAND"] / 1e6  # sq km
        features["tract_area_sqkm"] = tract_area
        features["overture_building_density"] = (
            features["overture_building_count"] / features["tract_area_sqkm"].replace(0, np.nan)
        )
        features["ms_building_density"] = (
            features["ms_building_count"] / features["tract_area_sqkm"].replace(0, np.nan)
        )
        features["building_density_gap"] = (
            features["ms_building_density"] - features["overture_building_density"]
        )

        # === LOG TRANSFORMS (for skewed distributions) ===
        features["log_overture_building_count"] = np.log1p(features["overture_building_count"])
        features["log_ms_building_count"] = np.log1p(features["ms_building_count"])

        # Clip ratios to reasonable range
        for col in ["building_count_ratio", "building_area_ratio"]:
            features[col] = features[col].clip(0, 5)
        for col in ["building_count_gap", "building_area_gap"]:
            features[col] = features[col].clip(-4, 1)

        logger.info(f"Building gap features: {features.shape}")
        return features

    def compute_road_gap_features(
        self,
        overture_roads: gpd.GeoDataFrame,
        tiger_roads: gpd.GeoDataFrame,
        tracts: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Compute road network coverage gap features per census tract.
        Goes beyond simple length ratios to include topology metrics.
        """
        logger.info("Computing road coverage gap features...")

        # Spatial join
        ov_roads_in_tracts = gpd.sjoin(overture_roads, tracts, how="left", predicate="within")
        tiger_in_tracts = gpd.sjoin(tiger_roads, tracts, how="left", predicate="within")

        # Total road length per tract
        ov_lengths = ov_roads_in_tracts.groupby("GEOID_right")["geometry"].apply(
            lambda g: g.length.sum()
        ).rename("overture_road_length")
        tiger_lengths = tiger_in_tracts.groupby("GEOID_right")["geometry"].apply(
            lambda g: g.length.sum()
        ).rename("tiger_road_length")

        # Road segment count per tract
        ov_seg_counts = ov_roads_in_tracts.groupby("GEOID_right").size().rename("overture_road_count")
        tiger_seg_counts = tiger_in_tracts.groupby("GEOID_right").size().rename("tiger_road_count")

        features = pd.DataFrame(index=tracts["GEOID"])
        features = features.join(ov_lengths).join(tiger_lengths).join(ov_seg_counts).join(tiger_seg_counts)

        for col in features.columns:
            features[col] = features[col].fillna(0)

        # === LENGTH RATIO (primary feature) ===
        features["road_length_ratio"] = (
            features["overture_road_length"] / features["tiger_road_length"].replace(0, np.nan)
        )
        features["road_length_gap"] = 1 - features["road_length_ratio"]

        # === SEGMENT COUNT RATIO ===
        features["road_count_ratio"] = (
            features["overture_road_count"] / features["tiger_road_count"].replace(0, np.nan)
        )
        features["road_count_gap"] = 1 - features["road_count_ratio"]

        # === ROAD DENSITY ===
        features["overture_road_density"] = (
            features["overture_road_length"] / features.index.map(
                lambda g: tracts.loc[tracts["GEOID"] == g, "ALAND"].values
            ).astype(float) / 1e6
        )

        # === BY ROAD CLASS (if available) ===
        if "road_class" in overture_roads.columns or "subtype" in overture_roads.columns:
            class_col = "subtype" if "subtype" in overture_roads.columns else "road_class"
            for cls in ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]:
                cls_roads = ov_roads_in_tracts[ov_roads_in_tracts[class_col] == cls]
                cls_length = cls_roads.groupby("GEOID_right")["geometry"].apply(
                    lambda g: g.length.sum() if len(g) > 0 else 0
                ).rename(f"overture_{cls}_length")
                features = features.join(cls_length)
                features[f"overture_{cls}_length"] = features[f"overture_{cls}_length"].fillna(0)
                features[f"overture_{cls}_fraction"] = (
                    features[f"overture_{cls}_length"] / 
                    features["overture_road_length"].replace(0, np.nan)
                )

        # Clip
        for col in ["road_length_ratio", "road_count_ratio"]:
            features[col] = features[col].clip(0, 5)
        for col in ["road_length_gap", "road_count_gap"]:
            features[col] = features[col].clip(-4, 1)

        logger.info(f"Road gap features: {features.shape}")
        return features

    def compute_poi_gap_features(
        self,
        overture_pois: gpd.GeoDataFrame,
        hifld_facilities: Dict[str, gpd.GeoDataFrame],
        tracts: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Compute POI / critical facility coverage gap features.
        Uses HIFLD hospitals, fire stations, EMS, schools as reference.
        """
        logger.info("Computing POI/critical facility gap features...")

        features = pd.DataFrame(index=tracts["GEOID"])

        # Count Overture POIs per tract
        ov_pois_in_tracts = gpd.sjoin(overture_pois, tracts, how="left", predicate="within")
        poi_counts = ov_pois_in_tracts.groupby("GEOID_right").size().rename("overture_poi_count")
        features = features.join(poi_counts)
        features["overture_poi_count"] = features["overture_poi_count"].fillna(0)

        # Count each HIFLD facility type per tract
        for facility_type, gdf in hifld_facilities.items():
            facilities_in_tracts = gpd.sjoin(gdf, tracts, how="left", predicate="within")
            fac_counts = facilities_in_tracts.groupby("GEOID_right").size().rename(
                f"hifld_{facility_type}_count"
            )
            features = features.join(fac_counts)
            features[f"hifld_{facility_type}_count"] = features[f"hifld_{facility_type}_count"].fillna(0)

        # Total HIFLD facility count
        hifld_cols = [c for c in features.columns if c.startswith("hifld_") and c.endswith("_count")]
        features["hifld_total_facility_count"] = features[hifld_cols].sum(axis=1)

        # === POI / FACILITY RATIO ===
        features["poi_to_facility_ratio"] = (
            features["overture_poi_count"] / features["hifld_total_facility_count"].replace(0, np.nan)
        )
        features["poi_facility_gap"] = 1 - features["poi_to_facility_ratio"]

        # === PER-CAPITA FEATURES (if population available) ===
        # POI density per 1000 population
        if "total_population" in tracts.columns:
            pop = tracts.set_index("GEOID")["total_population"]
            features["poi_per_1000_pop"] = (
                features["overture_poi_count"] / pop.replace(0, np.nan) * 1000
            )

        # === EMERGENCY ACCESS FEATURES ===
        # Minimum distance from tract centroid to nearest facility
        tract_centroids = tracts.geometry.centroid
        for facility_type, gdf in hifld_facilities.items():
            if len(gdf) > 0:
                min_dists = []
                for centroid in tract_centroids:
                    dists = gdf.geometry.distance(centroid)
                    min_dists.append(dists.min())
                features[f"min_dist_to_{facility_type}"] = min_dists
            else:
                features[f"min_dist_to_{facility_type}"] = np.nan

        # === CATEGORY COMPLETENESS ===
        # Check if specific critical facility types are present/absent
        for facility_type in hifld_facilities:
            features[f"has_{facility_type}"] = (
                features[f"hifld_{facility_type}_count"] > 0
            ).astype(int)

        # Clip
        features["poi_to_facility_ratio"] = features["poi_to_facility_ratio"].clip(0, 10)
        features["poi_facility_gap"] = features["poi_facility_gap"].clip(-9, 1)

        logger.info(f"POI gap features: {features.shape}")
        return features

    def compute_housing_gap_features(
        self,
        overture_buildings: gpd.GeoDataFrame,
        acs_housing: pd.DataFrame,
        tracts: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Compare building counts to ACS housing unit counts.
        Low buildings-per-housing-unit ratio indicates under-mapping.
        """
        logger.info("Computing housing coverage gap features...")

        features = pd.DataFrame(index=tracts["GEOID"])

        # Overture building count per tract (already computed in building features)
        ov_in_tracts = gpd.sjoin(overture_buildings, tracts, how="left", predicate="within")
        ov_counts = ov_in_tracts.groupby("GEOID_right").size().rename("overture_building_count")

        # Merge with ACS housing
        if "GEOID" in acs_housing.columns:
            housing = acs_housing.set_index("GEOID")["housing_units"]
        else:
            housing = acs_housing["housing_units"]

        features = features.join(ov_counts).join(housing)
        features["overture_building_count"] = features["overture_building_count"].fillna(0)
        features["housing_units"] = features["housing_units"].fillna(0)

        # Buildings per housing unit
        features["buildings_per_housing_unit"] = (
            features["overture_building_count"] / features["housing_units"].replace(0, np.nan)
        )
        # Low ratio = under-mapped
        features["housing_mapping_gap"] = 1 - features["buildings_per_housing_unit"].clip(0, 2)

        logger.info(f"Housing gap features: {features.shape}")
        return features


class SourceCompositionFeatures:
    """
    Parse the Overture sources[] column for provenance-based features.
    THIS IS THE BIGGEST COMPETITIVE ADVANTAGE.
    """

    def compute_building_source_features(self, overture_buildings: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute source composition features for buildings per tract."""
        logger.info("Computing building source composition features...")

        # Explode sources array
        if "sources" not in overture_buildings.columns:
            logger.warning("No sources column found")
            return pd.DataFrame()

        records = []
        for _, row in overture_buildings.iterrows():
            geoid = row.get("tract_geoid", row.get("GEOID_right", None))
            if geoid is None:
                continue
            for src in row["sources"]:
                records.append({
                    "GEOID": geoid,
                    "source_dataset": src.get("dataset", "unknown"),
                    "source_update_time": src.get("update_time", None),
                    "source_confidence": src.get("confidence", None),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Aggregate per tract
        features = df.groupby("GEOID").agg(
            total_buildings=("GEOID", "count"),
            osm_buildings=("source_dataset", lambda x: (x == "OpenStreetMap").sum()),
            ms_ml_buildings=("source_dataset", lambda x: x.str.contains("Microsoft", na=False).sum()),
            google_ml_buildings=("source_dataset", lambda x: x.str.contains("Google", na=False).sum()),
            esri_buildings=("source_dataset", lambda x: x.str.contains("Esri", na=False).sum()),
            source_diversity=("source_dataset", "nunique"),
        )

        # Derived features
        features["ml_derived_fraction"] = (
            (features["total_buildings"] - features["osm_buildings"] - features["esri_buildings"])
            / features["total_buildings"].replace(0, np.nan)
        )
        features["osm_fraction"] = features["osm_buildings"] / features["total_buildings"].replace(0, np.nan)

        # Staleness (days since last OSM update)
        df["update_date"] = pd.to_datetime(df["source_update_time"], errors="coerce")
        osm_only = df[df["source_dataset"] == "OpenStreetMap"]
        if len(osm_only) > 0:
            staleness = osm_only.groupby("GEOID")["update_date"].agg(
                mean_osm_staleness_days=lambda x: (pd.Timestamp.now() - x).dt.days.mean(),
                max_osm_staleness_days=lambda x: (pd.Timestamp.now() - x).dt.days.max(),
                min_osm_staleness_days=lambda x: (pd.Timestamp.now() - x).dt.days.min(),
            )
            features = features.join(staleness)

        logger.info(f"Source composition features: {features.shape}")
        return features

    def compute_poi_confidence_features(self, overture_pois: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute confidence-based features from Overture Places theme."""
        logger.info("Computing POI confidence features...")

        if "confidence" not in overture_pois.columns:
            logger.warning("No confidence column found")
            return pd.DataFrame()

        geoid_col = "tract_geoid" if "tract_geoid" in overture_pois.columns else "GEOID_right"

        features = overture_pois.groupby(geoid_col).agg(
            total_pois=("confidence", "count"),
            mean_poi_confidence=("confidence", "mean"),
            median_poi_confidence=("confidence", "median"),
            std_poi_confidence=("confidence", "std"),
            low_conf_poi_count=("confidence", lambda x: (x < 0.5).sum()),
            high_conf_poi_count=("confidence", lambda x: (x >= 0.7).sum()),
            very_high_conf_poi_count=("confidence", lambda x: (x >= 0.9).sum()),
        )

        features["low_confidence_fraction"] = (
            features["low_conf_poi_count"] / features["total_pois"].replace(0, np.nan)
        )
        features["high_confidence_fraction"] = (
            features["high_conf_poi_count"] / features["total_pois"].replace(0, np.nan)
        )

        # Source diversity for POIs
        if "sources" in overture_pois.columns:
            source_div = overture_pois.groupby(geoid_col)["sources"].apply(
                lambda x: len(set(
                    ds for srcs in x for ds in [s.get("dataset", "") for s in srcs]
                ))
            ).rename("poi_source_diversity")
            features = features.join(source_div)

        logger.info(f"POI confidence features: {features.shape}")
        return features


class NullFlagFeatures:
    """
    Create features from *_covered flags.
    NULL IS SIGNAL, NOT MISSING DATA.
    """

    def compute_coverage_flags(self, strata_df: pd.DataFrame) -> pd.DataFrame:
        """Create binary and ordinal features from covered flags."""
        logger.info("Computing null flag features...")

        features = pd.DataFrame(index=strata_df.index)

        # Find all _covered columns
        covered_cols = [c for c in strata_df.columns if c.endswith("_covered")]
        logger.info(f"Found {len(covered_cols)} _covered columns")

        for col in covered_cols:
            safe_name = col.replace("-", "_").replace(".", "_")
            # -1 = data doesn't reach this tract (NULL)
            # 0 = explicitly not covered
            # 1 = covered
            features[f"{safe_name}_flag"] = (
                strata_df[col].map({True: 1, False: 0}).fillna(-1).astype(int)
            )

        # CONUS indicator
        conus_cols = [c for c in covered_cols if any(
            k in c for k in ["usfs", "cdc_epht", "cdc_wonder", "carbonplan"]
        )]
        if conus_cols:
            features["is_conus"] = (
                strata_df[conus_cols].notna().all(axis=1).astype(int)
            )

        # Data coverage depth (how many layers reach this tract)
        features["data_coverage_depth"] = (
            strata_df[covered_cols].notna().sum(axis=1)
        )

        # Data coverage fraction
        features["data_coverage_fraction"] = (
            features["data_coverage_depth"] / max(len(covered_cols), 1)
        )

        # Specific coverage indicators
        if "usfs_wildfire_covered" in strata_df.columns:
            features["has_wildfire_data"] = strata_df["usfs_wildfire_covered"].fillna(False).astype(int)
        if "cdc_epht_covered" in strata_df.columns:
            features["has_heat_data"] = strata_df["cdc_epht_covered"].fillna(False).astype(int)
        if "usdm_covered" in strata_df.columns:
            features["has_drought_data"] = strata_df["usdm_covered"].fillna(False).astype(int)

        logger.info(f"Null flag features: {features.shape}")
        return features


class SpatialLagFeatures:
    """
    Compute spatial lag features (neighbor mean/max/min gaps).
    Coverage gaps are spatially autocorrelated - if your neighbors
    are under-mapped, you probably are too.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compute_spatial_lags(
        self,
        features: pd.DataFrame,
        tracts: gpd.GeoDataFrame,
        columns: List[str],
        k_neighbors: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """Compute spatial lag (neighbor aggregate) features."""
        from sklearn.neighbors import BallTree

        logger.info(f"Computing spatial lag features for {len(columns)} columns...")

        # Compute centroids in radians for BallTree (haversine)
        centroids = tracts.geometry.centroid
        coords = np.column_stack([
            np.radians(centroids.y),  # lat
            np.radians(centroids.x),  # lon
        ])

        tree = BallTree(coords, metric="haversine")

        lag_features = pd.DataFrame(index=features.index)

        for k in k_neighbors:
            _, indices = tree.query(coords, k=min(k + 1, len(coords)))

            for col in columns:
                if col not in features.columns:
                    continue

                vals = features[col].values

                # Neighbor mean (excluding self)
                neighbor_vals = vals[indices[:, 1:]]
                mask = ~np.isnan(neighbor_vals)

                mean_vals = np.where(
                    mask.any(axis=1),
                    np.nanmean(neighbor_vals, axis=1),
                    np.nan
                )
                lag_features[f"spatial_lag_k{k}_mean_{col}"] = mean_vals

                # Neighbor max
                max_vals = np.where(
                    mask.any(axis=1),
                    np.nanmax(np.where(mask, neighbor_vals, -np.inf), axis=1),
                    np.nan
                )
                lag_features[f"spatial_lag_k{k}_max_{col}"] = max_vals

                # Neighbor std (spatial heterogeneity)
                std_vals = np.where(
                    mask.any(axis=1),
                    np.nanstd(neighbor_vals, axis=1),
                    np.nan
                )
                lag_features[f"spatial_lag_k{k}_std_{col}"] = std_vals

        logger.info(f"Spatial lag features: {lag_features.shape}")
        return lag_features

    def compute_county_aggregates(
        self,
        features: pd.DataFrame,
        tracts: gpd.GeoDataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """Compute county-level aggregates and join back to tracts."""
        logger.info("Computing county-level aggregate features...")

        # Get county FIPS from GEOID (first 5 digits)
        county_fips = features.index.str[:5]
        county_features = pd.DataFrame(index=features.index)
        county_features["county_fips"] = county_fips

        for col in columns:
            if col not in features.columns:
                continue

            # County mean
            county_mean = features.groupby(county_fips)[col].transform("mean")
            county_features[f"county_mean_{col}"] = county_mean

            # Tract deviation from county mean
            county_features[f"county_dev_{col}"] = features[col] - county_mean

            # County std
            county_std = features.groupby(county_fips)[col].transform("std")
            county_features[f"county_std_{col}"] = county_std

            # Z-score within county
            county_features[f"county_zscore_{col}"] = (
                county_features[f"county_dev_{col}"] / county_std.replace(0, np.nan)
            )

        logger.info(f"County aggregate features: {county_features.shape}")
        return county_features

    def compute_tribal_proximity(
        self,
        tracts: gpd.GeoDataFrame,
        tribal_lands: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Compute distance to nearest tribal land boundary."""
        logger.info("Computing tribal land proximity features...")

        features = pd.DataFrame(index=tracts["GEOID"])

        # Binary: is tract tribal?
        tribal_tracts = gpd.sjoin(tracts, tribal_lands, how="left", predicate="intersects")
        features["is_tribal"] = tribal_tracts.index.duplicated(keep="first").astype(int)

        # Distance to nearest tribal boundary
        tract_centroids = tracts.geometry.centroid
        tribal_boundary = tribal_lands.geometry.unary_union.boundary

        min_dists = tract_centroids.distance(tribal_boundary)
        features["dist_to_tribal_boundary"] = min_dists.values

        # Overlap area with tribal lands
        overlaps = []
        for tract_geom in tracts.geometry:
            intersection = tract_geom.intersection(tribal_lands.geometry.unary_union)
            overlaps.append(intersection.area / tract_geom.area if tract_geom.area > 0 else 0)
        features["tribal_overlap_fraction"] = overlaps

        logger.info(f"Tribal proximity features: {features.shape}")
        return features


class VulnerabilityFeatures:
    """
    Engineer features from SVI, CVI, tribal, and hazard data.
    These are both features and the evaluation dimensions.
    """

    def compute_vulnerability_features(self, strata_df: pd.DataFrame) -> pd.DataFrame:
        """Create vulnerability-related features from the strata table."""
        logger.info("Computing vulnerability features...")

        features = pd.DataFrame(index=strata_df.index)

        # SVI features
        svi_cols = [c for c in strata_df.columns if "svi" in c.lower()]
        for col in svi_cols:
            features[col] = strata_df[col]

        # CVI features
        cvi_cols = [c for c in strata_df.columns if "cvi" in c.lower()]
        for col in cvi_cols:
            features[col] = strata_df[col]

        # Hazard features
        hazard_cols = [c for c in strata_df.columns if any(
            k in c.lower() for k in ["heat", "wildfire", "drought", "fire"]
        )]
        for col in hazard_cols:
            features[col] = strata_df[col]

        # === INTERACTION FEATURES (critical for Bias Discovery) ===
        if "svi_overall" in strata_df.columns:
            if "rural_urban_code" in strata_df.columns:
                features["svi_x_rural"] = strata_df["svi_overall"] * (strata_df["rural_urban_code"] >= 4).astype(int)

            for hazard in ["heat_risk", "wildfire_risk", "drought_risk"]:
                if hazard in strata_df.columns:
                    features[f"svi_x_{hazard}"] = strata_df["svi_overall"] * strata_df[hazard]

        # === COMPOUND RISK SCORE ===
        risk_cols = []
        for col in ["heat_risk", "wildfire_risk", "drought_risk"]:
            if col in strata_df.columns:
                risk_cols.append(col)
        if risk_cols:
            features["compound_risk_score"] = strata_df[risk_cols].sum(axis=1)
            features["max_single_risk"] = strata_df[risk_cols].max(axis=1)
            features["risk_diversity"] = (strata_df[risk_cols] > 0).sum(axis=1)

        logger.info(f"Vulnerability features: {features.shape}")
        return features


def build_all_features(
    strata_df: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    overture_buildings: Optional[gpd.GeoDataFrame] = None,
    microsoft_buildings: Optional[gpd.GeoDataFrame] = None,
    overture_roads: Optional[gpd.GeoDataFrame] = None,
    tiger_roads: Optional[gpd.GeoDataFrame] = None,
    overture_pois: Optional[gpd.GeoDataFrame] = None,
    hifld_facilities: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    tribal_lands: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Combines all feature classes into a single feature matrix.
    """
    logger.info("=" * 60)
    logger.info("BUILDING ALL FEATURES")
    logger.info("=" * 60)

    all_features = [pd.DataFrame(index=tracts["GEOID"])]

    # 1. Coverage gap features
    if overture_buildings is not None and microsoft_buildings is not None:
        gap_engineer = CoverageGapFeatures()
        building_features = gap_engineer.compute_building_gap_features(
            overture_buildings, microsoft_buildings, tracts
        )
        all_features.append(building_features)

    if overture_roads is not None and tiger_roads is not None:
        road_features = gap_engineer.compute_road_gap_features(
            overture_roads, tiger_roads, tracts
        )
        all_features.append(road_features)

    if overture_pois is not None and hifld_facilities is not None:
        poi_features = gap_engineer.compute_poi_gap_features(
            overture_pois, hifld_facilities, tracts
        )
        all_features.append(poi_features)

    # 2. Source composition features
    if overture_buildings is not None:
        source_engineer = SourceCompositionFeatures()
        source_features = source_engineer.compute_building_source_features(overture_buildings)
        if not source_features.empty:
            all_features.append(source_features)

    if overture_pois is not None:
        poi_conf_features = source_engineer.compute_poi_confidence_features(overture_pois)
        if not poi_conf_features.empty:
            all_features.append(poi_conf_features)

    # 3. Null flag features
    flag_engineer = NullFlagFeatures()
    flag_features = flag_engineer.compute_coverage_flags(strata_df)
    all_features.append(flag_features)

    # 4. Vulnerability features
    vuln_engineer = VulnerabilityFeatures()
    vuln_features = vuln_engineer.compute_vulnerability_features(strata_df)
    all_features.append(vuln_features)

    # 5. Spatial lag features
    lag_engineer = SpatialLagFeatures()

    # Get lag columns from coverage gap features
    lag_cols = [c for c in all_features[0].columns if "gap" in c or "ratio" in c]
    if lag_cols:
        merged = pd.concat(all_features, axis=1)
        spatial_lags = lag_engineer.compute_spatial_lags(merged, tracts, lag_cols)
        all_features.append(spatial_lags)

    # County aggregates
    county_features = lag_engineer.compute_county_aggregates(
        pd.concat(all_features, axis=1), tracts, lag_cols
    )
    all_features.append(county_features)

    # 6. Tribal proximity
    if tribal_lands is not None:
        tribal_features = lag_engineer.compute_tribal_proximity(tracts, tribal_lands)
        all_features.append(tribal_features)

    # Combine all features
    final_features = pd.concat(all_features, axis=1)

    # Remove duplicate columns
    final_features = final_features.loc[:, ~final_features.columns.duplicated()]

    logger.info(f"Final feature matrix: {final_features.shape}")
    return final_features
