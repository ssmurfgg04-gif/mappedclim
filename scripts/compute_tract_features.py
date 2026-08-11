"""
Fast per-tract feature computation for a single region.
Uses bbox-only joins (no ST_Contains) for speed.
"""

import duckdb
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def compute_region(region):
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    raw = f'data/raw'
    ref = f'{raw}/reference/{region}'
    strata_dir = f'{raw}/strata/{region}'
    tracts_path = f'{strata_dir}/{region}-census-tracts.parquet'
    strata_path = f'{strata_dir}/{region}-strata-tract-table.parquet'
    
    strata = conn.execute(f"SELECT GEOID FROM read_parquet('{strata_path}')").df()
    print(f'{region}: {len(strata):,} tracts')
    
    # Building counts (bbox join)
    ov_b = conn.execute(f"""
        WITH t AS (SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1 FROM read_parquet('{tracts_path}')),
             b AS (SELECT bbox.xmin bx0, bbox.ymin by0, bbox.xmax bx1, bbox.ymax by1 FROM read_parquet('{ref}/{region}-overture-buildings.parquet'))
        SELECT t.GEOID, COUNT(*) ov_bldg FROM t JOIN b ON b.bx1>=t.x0 AND b.bx0<=t.x1 AND b.by1>=t.y0 AND b.by0<=t.y1 GROUP BY t.GEOID
    """).df()
    
    ms_b = conn.execute(f"""
        WITH t AS (SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1 FROM read_parquet('{tracts_path}')),
             b AS (SELECT bbox.xmin bx0, bbox.ymin by0, bbox.xmax bx1, bbox.ymax by1 FROM read_parquet('{ref}/{region}-microsoft-buildings.parquet'))
        SELECT t.GEOID, COUNT(*) ms_bldg FROM t JOIN b ON b.bx1>=t.x0 AND b.bx0<=t.x1 AND b.by1>=t.y0 AND b.by0<=t.y1 GROUP BY t.GEOID
    """).df()
    
    # Road counts
    ov_r = conn.execute(f"""
        WITH t AS (SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1 FROM read_parquet('{tracts_path}')),
             r AS (SELECT bbox.xmin rx0, bbox.ymin ry0, bbox.xmax rx1, bbox.ymax ry1 FROM read_parquet('{ref}/{region}-overture-roads.parquet'))
        SELECT t.GEOID, COUNT(*) ov_road FROM t JOIN r ON r.rx1>=t.x0 AND r.rx0<=t.x1 AND r.ry1>=t.y0 AND r.ry0<=t.y1 GROUP BY t.GEOID
    """).df()
    
    ti_r = conn.execute(f"""
        WITH t AS (SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1 FROM read_parquet('{tracts_path}')),
             r AS (SELECT bbox.xmin rx0, bbox.ymin ry0, bbox.xmax rx1, bbox.ymax ry1 FROM read_parquet('{ref}/{region}-census-tiger-roads.parquet'))
        SELECT t.GEOID, COUNT(*) tiger_road FROM t JOIN r ON r.rx1>=t.x0 AND r.rx0<=t.x1 AND r.ry1>=t.y0 AND r.ry0<=t.y1 GROUP BY t.GEOID
    """).df()
    
    # POI counts
    pois = conn.execute(f"""
        WITH t AS (SELECT GEOID, bbox.xmin x0, bbox.ymin y0, bbox.xmax x1, bbox.ymax y1 FROM read_parquet('{tracts_path}')),
             p AS (SELECT bbox.xmin px0, bbox.ymin py0, bbox.xmax px1, bbox.ymax py1 FROM read_parquet('{ref}/{region}-overture-pois.parquet'))
        SELECT t.GEOID, COUNT(*) poi_cnt FROM t JOIN p ON p.px1>=t.x0 AND p.px0<=t.x1 AND p.py1>=t.y0 AND p.py0<=t.y1 GROUP BY t.GEOID
    """).df()
    
    # Housing
    housing = conn.execute(f"SELECT GEOID, housing_units FROM read_parquet('{ref}/{region}-census-acs-housing.parquet')").df()
    
    # Merge
    m = strata.merge(ov_b, on='GEOID', how='left').merge(ms_b, on='GEOID', how='left')
    m = m.merge(ov_r, on='GEOID', how='left').merge(ti_r, on='GEOID', how='left')
    m = m.merge(pois, on='GEOID', how='left').merge(housing, on='GEOID', how='left')
    
    for c in ['ov_bldg','ms_bldg','ov_road','tiger_road','poi_cnt','housing_units']:
        m[c] = m[c].fillna(0)
    
    m['building_ratio'] = m['ov_bldg'] / m['ms_bldg'].replace(0, np.nan)
    m['building_gap'] = (1 - m['building_ratio']).clip(-4, 1)
    m['road_ratio'] = m['ov_road'] / m['tiger_road'].replace(0, np.nan)
    m['road_gap'] = (1 - m['road_ratio']).clip(-4, 1)
    m['bldg_per_housing'] = m['ov_bldg'] / m['housing_units'].replace(0, np.nan)
    m['region'] = region
    
    # Save
    out = f'data/features/{region}_tract_features.parquet'
    Path('data/features').mkdir(parents=True, exist_ok=True)
    m.to_parquet(out)
    
    print(f'  building_gap: mean={m["building_gap"].mean():.4f}, median={m["building_gap"].median():.4f}')
    print(f'  road_gap: mean={m["road_gap"].mean():.4f}, median={m["road_gap"].median():.4f}')
    print(f'  Saved: {out}')
    
    conn.close()
    return m

if __name__ == "__main__":
    for region in ['maricopa-az', 'northern-ca', 'south-central-tx']:
        try:
            compute_region(region)
        except Exception as e:
            print(f'FAILED {region}: {e}')
