"""
Download and cache data from Source Cooperative via HTTPS.
Updated to use direct HTTPS access since DuckDB S3 has SSL issues.
"""

import os
import sys
import logging
import yaml
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"


def download_file(url: str, local_path: Path, force: bool = False) -> bool:
    """Download a single file with progress logging."""
    if local_path.exists() and not force:
        logger.info(f"[SKIP] {local_path.name} already exists")
        return True

    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"[OK] {local_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"[FAIL] {url}: {e}")
        return False


def download_data():
    """Download all required data files from Source Cooperative."""
    with open(PROJECT_ROOT / "config" / "paths.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = PROJECT_ROOT / config["local"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # === 1. National Strata Table (CRITICAL) ===
    logger.info("=" * 60)
    logger.info("Downloading National Strata Tables")
    logger.info("=" * 60)
    
    national_files = [
        "strata/national/national-strata-tract-table.parquet",
        "strata/national/national-census-tract-table.parquet",
        "strata/national/national-census-tracts.parquet",
        "strata/national/national-svi-tract-table.parquet",
        "strata/national/national-cvi-tract-table.parquet",
        "strata/national/national-tribal-tract-table.parquet",
        "strata/national/national-ruca-tract-table.parquet",
        "strata/national/national-usfs-wildfire-tract-table.parquet",
        "strata/national/national-usdm-drought-tract-table.parquet",
        "strata/national/national-epht-heat-tract-table.parquet",
        "strata/national/national-census-aiannh.parquet",
        "strata/national/national-census-tribal-tracts.parquet",
    ]
    
    for f in national_files:
        url = f"{BASE_URL}/{f}"
        local = raw_dir / f
        download_file(url, local)

    # === 2. Focus Region Reference Data ===
    regions = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]
    reference_layers = [
        "overture-buildings",
        "overture-roads",
        "overture-rail",
        "overture-infrastructure",
        "overture-pois",
        "microsoft-buildings",
        "census-tiger-roads",
        "census-acs-housing",
        "census-cbp",
        "hifld-hospitals",
        "hifld-fire-stations",
        "hifld-ems-stations",
        "hifld-schools",
    ]
    
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading Reference Data: {region}")
        logger.info(f"{'='*60}")
        
        for layer in reference_layers:
            fname = f"{region}-{layer}.parquet"
            url = f"{BASE_URL}/reference/{region}/{fname}"
            local = raw_dir / "reference" / region / fname
            download_file(url, local)

    # === 3. Focus Region Strata Data ===
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading Strata Data: {region}")
        logger.info(f"{'='*60}")
        
        strata_layers = [
            "strata-tract-table",
            "svi-tract-table",
            "cvi-tract-table",
            "ruca-tract-table",
            "tribal-tract-table",
            "usdm-drought-tract-table",
            "usfs-wildfire-tract-table",
            "census-tracts",
            "census-aiannh",
            "census-tribal-tracts",
        ]
        
        for layer in strata_layers:
            fname = f"{region}-{layer}.parquet"
            url = f"{BASE_URL}/strata/{region}/{fname}"
            local = raw_dir / "strata" / region / fname
            download_file(url, local)

    # === 4. Boundary Files ===
    logger.info(f"\n{'='*60}")
    logger.info("Downloading Boundary Files")
    logger.info(f"{'='*60}")
    
    boundary_files = [
        "boundaries/all-aois.geojson",
        "boundaries/maricopa-az-aoi.geojson",
        "boundaries/northern-ca-aoi.geojson",
        "boundaries/eastern-ok-aoi.geojson",
        "boundaries/south-central-tx-aoi.geojson",
    ]
    
    for f in boundary_files:
        url = f"{BASE_URL}/{f}"
        local = raw_dir / f
        download_file(url, local)

    # === 5. README ===
    logger.info(f"\n{'='*60}")
    logger.info("Downloading Data README")
    logger.info(f"{'='*60}")
    
    url = f"{BASE_URL}/README.md"
    local = raw_dir / "DATA_README.md"
    download_file(url, local)

    logger.info(f"\n{'='*60}")
    logger.info("DATA DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    download_data()
