"""FAST grid-partitioned building counting v4 — bbox-overlap filtered candidates.

Pass 1 (light): grid cells + bbox-overlap filter -> DISTINCT (bid, gid) candidates.
  Only bbox + file_row_number are read (no geometry) - cheap.
Pass 2: join geometries, apply exact predicate (intersects / centroid / within).
"""
import duckdb
import pandas as pd
import numpy as np
import sys
import time
import os

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("SET memory_limit='2.8GB'; SET threads=2; SET preserve_insertion_order=false;")
con.execute("SET max_temp_directory_size='3GB'; SET temp_directory=os.path.join(DATA, 'tmp');")
os.makedirs(os.path.join(DATA, "tmp"), exist_ok=True)

D = DATA
ss = pd.read_csv(SS_PATH, dtype={"GEOID": str})

REGIONS = sys.argv[1].split(",") if len(sys.argv) > 1 else ['south-central-tx']
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["intersects"]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 50

PRED = {
    "intersects": "ST_Intersects(b.geometry, t.geometry)",
    "centroid":   "ST_Contains(t.geometry, ST_Centroid(b.geometry))",
    "within":     "ST_Covers(t.geometry, b.geometry)",
}

def count_buildings(path, tracts, variant):
    # Pass 1: candidate pairs via grid cells + bbox overlap (integers only)
    con.execute(f"""
        CREATE OR REPLACE TABLE pairs AS
        WITH bcells AS (
            SELECT b.file_row_number AS bid, b.xmin, b.xmax, b.ymin, b.ymax,
                   gsx.cx AS cx, gsy.cy AS cy
            FROM (SELECT file_row_number,
                         bbox.xmin AS xmin, bbox.xmax AS xmax,
                         bbox.ymin AS ymin, bbox.ymax AS ymax
                  FROM read_parquet('{path}', file_row_number=true)) b,
                 generate_series(floor(b.xmin * {G})::INT, ceil(b.xmax * {G})::INT) AS gsx(cx),
                 generate_series(floor(b.ymin * {G})::INT, ceil(b.ymax * {G})::INT) AS gsy(cy)
        ), tcells AS (
            SELECT t.GEOID AS gid, t.xmin, t.xmax, t.ymin, t.ymax,
                   gsx.cx AS cx, gsy.cy AS cy
            FROM (SELECT GEOID,
                         bbox.xmin AS xmin, bbox.xmax AS xmax,
                         bbox.ymin AS ymin, bbox.ymax AS ymax
                  FROM read_parquet('{tracts}')) t,
                 generate_series(floor(t.xmin * {G})::INT, ceil(t.xmax * {G})::INT) AS gsx(cx),
                 generate_series(floor(t.ymin * {G})::INT, ceil(t.ymax * {G})::INT) AS gsy(cy)
        )
        SELECT DISTINCT bc.bid, tc.gid
        FROM bcells bc JOIN tcells tc ON bc.cx = tc.cx AND bc.cy = tc.cy
        WHERE bc.xmin <= tc.xmax AND bc.xmax >= tc.xmin
          AND bc.ymin <= tc.ymax AND bc.ymax >= tc.ymin
    """)
    npairs = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    # Pass 2: exact predicate
    q = f"""
        SELECT p.gid AS GEOID, COUNT(*) AS cnt
        FROM pairs p
        JOIN (SELECT file_row_number, geometry FROM read_parquet('{path}', file_row_number=true)) b
          ON b.file_row_number = p.bid
        JOIN (SELECT GEOID, geometry FROM read_parquet('{tracts}')) t
          ON t.GEOID = p.gid
        WHERE {PRED[variant]}
        GROUP BY p.gid
    """
    df = con.execute(q).df()
    df["GEOID"] = df["GEOID"].astype(str)
    con.execute("DROP TABLE IF EXISTS pairs")
    return df, npairs

for region in REGIONS:
    tracts = f"{D}/tracts/{region}-census-tracts.parquet"
    sub = ss[ss['region'] == region][['GEOID', 'building_defined']]
    for variant in VARIANTS:
        t0 = time.time()
        ov, np1 = count_buildings(f"{D}/buildings/{region}-overture-buildings.parquet", tracts, variant)
        ov = ov.rename(columns={"cnt": "ov_cnt"})
        t1 = time.time()
        ms, np2 = count_buildings(f"{D}/buildings/{region}-microsoft-buildings.parquet", tracts, variant)
        ms = ms.rename(columns={"cnt": "ms_cnt"})
        res = sub.merge(ov, on="GEOID", how="left").merge(ms, on="GEOID", how="left")
        res[['ov_cnt', 'ms_cnt']] = res[['ov_cnt', 'ms_cnt']].fillna(0)
        defined_calc = res['ms_cnt'] > 0
        flag_match = (defined_calc == res['building_defined']).mean()
        res['gap'] = np.where(res['ms_cnt'] > 0,
                              1 - np.minimum(1.0, res['ov_cnt'] / res['ms_cnt'].replace(0, np.nan)), 0.0)
        res['variant'] = variant
        res.to_parquet(f"{D}/building_gaps_{variant}_{region}.parquet")
        print(f"[{region}] {variant:11s} flags={flag_match:.6f} ({defined_calc.sum()}/{res['building_defined'].sum()}) "
              f"ov={res['ov_cnt'].mean():.1f} ms={res['ms_cnt'].mean():.1f} gap_sum={res['gap'].sum():.4f} "
              f"[ov {t1-t0:.0f}s ({np1:,} pairs) + ms {time.time()-t1:.0f}s ({np2:,} pairs)]", flush=True)
