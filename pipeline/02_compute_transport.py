"""Compute transport gaps per tract: Overture highway-class road lengths vs TIGER S1100/S1200.

Methodology:
  - Filter Overture roads: class IN (motorway, trunk, primary, secondary)
  - Filter TIGER roads: MTFCC IN (S1100, S1200)
  - Project both roads and tract polygons to EPSG:5070 (meters) with always_xy=true
  - Clip road geometry to tract (ST_Intersection), measure length
  - transport_defined := tiger_len > 0
  - transport_gap := 1 - min(1, ov_len / tiger_len)

Validates defined flags against the leaked sample-submission flags and README counts.
"""
import duckdb
import pandas as pd
import numpy as np
import sys
import time

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")

D = DATA
ss = pd.read_csv(SS_PATH, dtype={"GEOID": str})

REGIONS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    'eastern-ok', 'maricopa-az', 'northern-ca', 'south-central-tx']

all_results = []
for region in REGIONS:
    t0 = time.time()
    tracts = f"{D}/tracts/{region}-census-tracts.parquet"
    ov = f"{D}/roads/{region}-overture-roads.parquet"
    tg = f"{D}/roads/{region}-census-tiger-roads.parquet"

    # TIGER highway length per tract
    tiger = con.execute(f"""
        WITH t AS (
            SELECT GEOID, bbox,
                   ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS g
            FROM read_parquet('{tracts}')
        ), r AS (
            SELECT bbox,
                   ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS g
            FROM read_parquet('{tg}')
            WHERE MTFCC IN ('S1100', 'S1200')
        )
        SELECT t.GEOID, SUM(ST_Length(ST_Intersection(r.g, t.g))) AS tiger_len
        FROM t JOIN r
          ON r.bbox.xmin <= t.bbox.xmax AND r.bbox.xmax >= t.bbox.xmin
         AND r.bbox.ymin <= t.bbox.ymax AND r.bbox.ymax >= t.bbox.ymin
         AND ST_Intersects(r.g, t.g)
        GROUP BY t.GEOID
    """).df()
    tiger["GEOID"] = tiger["GEOID"].astype(str)

    # Overture highway length per tract
    overt = con.execute(f"""
        WITH t AS (
            SELECT GEOID, bbox,
                   ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS g
            FROM read_parquet('{tracts}')
        ), r AS (
            SELECT bbox,
                   ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS g
            FROM read_parquet('{ov}')
            WHERE class IN ('motorway', 'trunk', 'primary', 'secondary')
        )
        SELECT t.GEOID, SUM(ST_Length(ST_Intersection(r.g, t.g))) AS ov_len
        FROM t JOIN r
          ON r.bbox.xmin <= t.bbox.xmax AND r.bbox.xmax >= t.bbox.xmin
         AND r.bbox.ymin <= t.bbox.ymax AND r.bbox.ymax >= t.bbox.ymin
         AND ST_Intersects(r.g, t.g)
        GROUP BY t.GEOID
    """).df()
    overt["GEOID"] = overt["GEOID"].astype(str)

    res = ss[ss['region'] == region][['GEOID', 'transport_defined']].merge(
        tiger, on='GEOID', how='left').merge(overt, on='GEOID', how='left')
    res['tiger_len'] = res['tiger_len'].fillna(0.0)
    res['ov_len'] = res['ov_len'].fillna(0.0)

    # Validation 1: defined flags
    calc_def = res['tiger_len'] > 0
    match = (calc_def == res['transport_defined']).mean()
    print(f"[{region}] {len(res)} tracts | tiger_defined match: {match:.6f} "
          f"(calc {calc_def.sum()} vs leaked {res['transport_defined'].sum()}) | {time.time()-t0:.1f}s")

    # Compute gap
    res['transport_gap_calc'] = np.where(
        res['tiger_len'] > 0,
        1 - np.minimum(1.0, res['ov_len'] / res['tiger_len'].replace(0, np.nan)),
        0.0)
    res['region'] = region
    all_results.append(res)
    print(f"  tiger_len mean={res['tiger_len'].mean():.1f}m | ov_len mean={res['ov_len'].mean():.1f}m | "
          f"gap sum={res['transport_gap_calc'].sum():.4f} | gap mean (defined)="
          f"{res.loc[res['transport_defined'], 'transport_gap_calc'].mean():.6f}")

out = pd.concat(all_results, ignore_index=True)
out.to_parquet(f"{D}/transport_gaps_{'-'.join(REGIONS)}.parquet")
print(f"\nSaved {len(out)} rows")
print(f"\nGLOBAL: transport gap sum = {out['transport_gap_calc'].sum():.4f}")
print(f"Placeholder target: 0.111771 * 9379 = {0.111771 * 9379:.4f}")
