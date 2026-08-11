"""
Data access layer for Bias Bounty Mapping Equity Challenge.
Handles S3 access via DuckDB, bbox pushdown, and nested struct parsing.
"""

import duckdb
import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataAccess:
    """Manages DuckDB connections and data loading from Source Cooperative S3."""

    def __init__(self, config_path: str = "config/paths.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.s3_base = self.config["s3_base"]
        self.s3_region = self.config["s3_region"]
        self.conn = None

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection with spatial + httpfs extensions."""
        if self.conn is None:
            self.conn = duckdb.connect()
            self.conn.execute("INSTALL spatial; LOAD spatial;")
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")
            self.conn.execute(f"SET s3_region='{self.s3_region}';")
            # Don't need S3 credentials for public Source Cooperative data
            self.conn.execute("SET s3_access_key_id='';")
            self.conn.execute("SET s3_secret_access_key='';")
            # Try to install overture community extension
            try:
                self.conn.execute("INSTALL overture FROM community; LOAD overture;")
                logger.info("Overture community extension loaded")
            except Exception:
                logger.warning("Overture community extension not available, using raw SQL")
        return self.conn

    def list_available_files(self, prefix: str = "") -> List[str]:
        """List available Parquet files in the S3 bucket."""
        conn = self._get_conn()
        path = f"{self.s3_base}/{prefix}*" if prefix else f"{self.s3_base}/*"
        try:
            result = conn.execute(f"SELECT * FROM glob('{path}')").fetchall()
            return [r[0] for r in result]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def load_national_strata(self, local_cache: Optional[str] = None) -> str:
        """
        Load the national strata table (85,396 census tracts, 232 columns).
        Returns a DuckDB table name for subsequent queries.
        """
        conn = self._get_conn()
        strata_file = self.config["national_strata"]

        if local_cache and Path(local_cache).exists():
            logger.info(f"Loading national strata from local cache: {local_cache}")
            conn.execute(
                f"CREATE OR REPLACE TABLE national_strata AS "
                f"SELECT * FROM read_parquet('{local_cache}')"
            )
        else:
            s3_path = f"{self.s3_base}/strata/{strata_file}"
            logger.info(f"Loading national strata from S3: {s3_path}")
            conn.execute(
                f"CREATE OR REPLACE TABLE national_strata AS "
                f"SELECT * FROM read_parquet('{s3_path}')"
            )

        count = conn.execute("SELECT COUNT(*) FROM national_strata").fetchone()[0]
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='national_strata'"
        ).fetchall()
        logger.info(f"National strata: {count} tracts, {len(cols)} columns")
        return "national_strata"

    def load_region_data(self, region: str, data_type: str) -> str:
        """
        Load data for a specific focus region.
        
        Args:
            region: One of 'maricopa_az', 'northern_ca', 'eastern_ok', 'south_central_tx'
            data_type: Type of data (e.g., 'overture-buildings', 'microsoft-buildings', 
                       'census-tiger-roads', 'hifld-hospitals', etc.)
        
        Returns:
            DuckDB table name
        """
        conn = self._get_conn()
        s3_path = f"{self.s3_base}/geoparquet/{region}/{region}-{data_type}.parquet"
        table_name = f"{region}_{data_type.replace('-', '_')}"

        logger.info(f"Loading {region}/{data_type} from S3")
        conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_parquet('{s3_path}')"
        )
        return table_name

    def query_with_bbox(
        self,
        s3_path: str,
        bbox: Tuple[float, float, float, float],
        additional_where: str = "",
        select_cols: str = "*",
    ) -> str:
        """
        Query Parquet with bbox pushdown for efficient spatial filtering.
        Uses the bbox column for row-group skipping (much faster than ST_Intersects).
        
        Args:
            s3_path: Full S3 path to Parquet file(s)
            bbox: (xmin, ymin, xmax, ymax) bounding box
            additional_where: Additional WHERE clause conditions
            select_cols: Columns to select
        
        Returns:
            DuckDB table name with results
        """
        conn = self._get_conn()
        xmin, ymin, xmax, ymax = bbox

        where_parts = [
            f"bbox.xmin <= {xmax}",
            f"bbox.xmax >= {xmin}",
            f"bbox.ymin <= {ymax}",
            f"bbox.ymax >= {ymin}",
        ]
        if additional_where:
            where_parts.append(additional_where)

        where_clause = " AND ".join(where_parts)
        table_name = f"bbox_query_{abs(hash(s3_path)) % 10000}"

        conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT {select_cols} FROM read_parquet('{s3_path}') "
            f"WHERE {where_clause}"
        )
        return table_name

    def compute_source_composition(self, table_name: str, theme: str = "buildings") -> str:
        """
        Parse the nested sources[] column to compute source composition features.
        This is a CRITICAL feature that most competitors will skip.
        
        Returns a table with per-tract source composition metrics.
        """
        conn = self._get_conn()
        result_table = f"{table_name}_source_comp"

        if theme == "buildings":
            conn.execute(f"""
                CREATE OR REPLACE TABLE {result_table} AS
                WITH exploded AS (
                    SELECT 
                        tract_geoid,
                        s.dataset AS source_dataset,
                        s.update_time AS source_update_time,
                        s.confidence AS source_confidence
                    FROM {table_name}, UNNEST(sources) AS t(s)
                )
                SELECT 
                    tract_geoid,
                    COUNT(*) AS total_features,
                    SUM(CASE WHEN source_dataset = 'OpenStreetMap' THEN 1 ELSE 0 END) AS osm_count,
                    SUM(CASE WHEN source_dataset LIKE '%Microsoft%' THEN 1 ELSE 0 END) AS ms_ml_count,
                    SUM(CASE WHEN source_dataset LIKE '%Google%' THEN 1 ELSE 0 END) AS google_ml_count,
                    SUM(CASE WHEN source_dataset LIKE '%Esri%' THEN 1 ELSE 0 END) AS esri_count,
                    -- ML-derived fraction (no human verification)
                    CAST(SUM(CASE WHEN source_dataset NOT IN ('OpenStreetMap', 'Esri Community Maps') 
                        THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0) AS ml_derived_fraction,
                    -- OSM fraction
                    CAST(SUM(CASE WHEN source_dataset = 'OpenStreetMap' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS osm_fraction,
                    -- Source diversity (more sources = better conflated coverage)
                    COUNT(DISTINCT source_dataset) AS source_diversity,
                    -- Mean days since last OSM update (stale = potential gaps)
                    AVG(CASE WHEN source_dataset = 'OpenStreetMap' 
                        THEN CURRENT_DATE - CAST(source_update_time AS DATE) 
                        ELSE NULL END) AS mean_osm_staleness_days,
                    -- Max update recency
                    MAX(CASE WHEN source_dataset = 'OpenStreetMap' 
                        THEN CURRENT_DATE - CAST(source_update_time AS DATE) 
                        ELSE NULL END) AS max_osm_staleness_days
                FROM exploded
                GROUP BY tract_geoid
            """)
        elif theme == "places":
            conn.execute(f"""
                CREATE OR REPLACE TABLE {result_table} AS
                WITH exploded AS (
                    SELECT 
                        tract_geoid,
                        s.dataset AS source_dataset,
                        confidence AS poi_confidence
                    FROM {table_name}, UNNEST(sources) AS t(s)
                )
                SELECT 
                    tract_geoid,
                    COUNT(*) AS total_pois,
                    AVG(poi_confidence) AS mean_poi_confidence,
                    CAST(SUM(CASE WHEN poi_confidence < 0.5 THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS low_confidence_fraction,
                    SUM(CASE WHEN poi_confidence >= 0.7 THEN 1 ELSE 0 END) AS high_conf_poi_count,
                    SUM(CASE WHEN poi_confidence >= 0.9 THEN 1 ELSE 0 END) AS very_high_conf_poi_count,
                    -- Source composition for POIs
                    COUNT(DISTINCT source_dataset) AS poi_source_diversity,
                    CAST(SUM(CASE WHEN source_dataset = 'meta' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS meta_fraction,
                    CAST(SUM(CASE WHEN source_dataset = 'Foursquare' THEN 1 ELSE 0 END) AS DOUBLE) 
                        / NULLIF(COUNT(*), 0) AS foursquare_fraction
                FROM exploded
                GROUP BY tract_geoid
            """)

        return result_table

    def compute_coverage_flags(self, strata_table: str = "national_strata") -> str:
        """
        Create binary features from *_covered flags.
        NULL is SIGNAL, not missing data.
        
        Key insight: A tract with usfs_covered = False tells you
        the wildfire risk data doesn't exist, which correlates with
        being outside CONUS or in a data desert.
        """
        conn = self._get_conn()
        
        # Get all _covered columns
        covered_cols = conn.execute(f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = '{strata_table}' 
            AND column_name LIKE '%_covered'
        """).fetchall()
        
        select_parts = ["*"]
        for (col,) in covered_cols:
            safe_col = col.replace("-", "_").replace(".", "_")
            # Binary: 1 if covered, 0 if not covered, -1 if null (data doesn't reach)
            select_parts.append(
                f"CASE WHEN {col} IS NULL THEN -1 "
                f"WHEN {col} = true THEN 1 "
                f"ELSE 0 END AS {safe_col}_flag"
            )
        
        # CONUS indicator (derived from which covered flags are non-null)
        select_parts.append(
            "CASE WHEN usfs_covered IS NOT NULL AND cdc_epht_covered IS NOT NULL "
            "THEN 1 ELSE 0 END AS is_conus"
        )
        
        # Data coverage depth (how many data layers reach this tract)
        covered_sum = " + ".join(
            f"CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END"
            for (col,) in covered_cols
        )
        select_parts.append(f"({covered_sum}) AS data_coverage_depth")
        
        result_table = f"{strata_table}_with_flags"
        conn.execute(
            f"CREATE OR REPLACE TABLE {result_table} AS "
            f"SELECT {', '.join(select_parts)} FROM {strata_table}"
        )
        
        return result_table

    def export_to_pandas(self, table_name: str) -> "pd.DataFrame":
        """Export DuckDB table to pandas DataFrame."""
        conn = self._get_conn()
        return conn.execute(f"SELECT * FROM {table_name}").df()

    def export_to_parquet(self, table_name: str, output_path: str):
        """Export DuckDB table to Parquet file."""
        conn = self._get_conn()
        conn.execute(f"COPY (SELECT * FROM {table_name}) TO '{output_path}' (FORMAT PARQUET)")
        logger.info(f"Exported {table_name} to {output_path}")

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
