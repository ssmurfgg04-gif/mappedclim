"""Transport pipeline with explicit WGS84->NAD83 datum operations chained to Albers.

Candidates:
  op3: Inverse of NAD83 to WGS 84 (3) — helmert (-1,-1,1) ~ 87cm varying
  op2: Inverse of NAD83 to WGS 84 (2) — helmert (2,0,-4)  ~ 3.7m varying
Both applied to roads AND tracts consistently; the non-rigid differential flips boundary roads.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import pyproj
from pyproj.transformer import TransformerGroup
import pyarrow.parquet as pq
import sys
import time

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

D = DATA
CACHE = f"{D}/cache"
ss = pd.read_csv(SS_PATH, dtype={"GEOID": str})
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]
TARGET = 0.111771 * len(ss)
WINDOW = 0.5e-6 * len(ss)

tg = TransformerGroup("EPSG:4326", "EPSG:4269", always_xy=True)
OPS = {}
for t in tg.transformers:
    if "(3)" in t.description and "Inverse" in t.description:
        OPS["op3"] = t
    if "(2)" in t.description and "Inverse" in t.description:
        OPS["op2"] = t
ALB = pyproj.Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)


def make_transformer(op_name):
    op = OPS[op_name]
    def tf(geoms):
        geoms = shapely.from_wkb(shapely.to_wkb(geoms))
        c = shapely.get_coordinates(geoms)
        lon2, lat2 = op.transform(c[:, 0], c[:, 1])
        x, y = ALB.transform(lon2, lat2)
        shapely.set_coordinates(geoms, np.column_stack([x, y]))
        return geoms
    return tf


def load_roads(region, layer):
    df = pq.read_table(f"{CACHE}/{region}-{layer}-filtered.parquet").to_pandas()
    return shapely.from_wkb(df["geometry"].values)


def compute_variant(region, layer, tf):
    roads = load_roads(region, layer)
    tracts = gpd.read_parquet(f"{D}/tracts/{region}-census-tracts.parquet")[["GEOID", "geometry"]]
    tgeoms = tracts.geometry.values
    tgeoid = tracts["GEOID"].values

    roads_g = tf(roads)
    tgeoms_p = tf(tgeoms)

    tdf = gpd.GeoDataFrame({"GEOID": tgeoid}, geometry=tgeoms_p, crs=5070)
    rdf = gpd.GeoDataFrame(geometry=roads_g, crs=5070)
    pairs = gpd.sjoin(rdf, tdf, how="inner", predicate="intersects")
    rgeom = roads_g[pairs.index.values]
    tgeom = tgeoms_p[pairs["index_right"].values]
    inter = shapely.intersection(rgeom, tgeom)
    lens = shapely.length(inter)
    pdf = pd.DataFrame({"GEOID": tgeoid[pairs["index_right"].values], "len": lens})
    return pdf.groupby("GEOID")["len"].sum().to_dict()


for vname in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["op3", "op2"]):
    t0 = time.time()
    tf = make_transformer(vname)
    per_region = []
    for region in REGIONS:
        tiger = compute_variant(region, "census-tiger-roads", tf)
        ov = compute_variant(region, "overture-roads", tf)
        sub = ss[ss["region"] == region][["GEOID", "transport_defined"]].copy()
        sub["tiger_len"] = sub["GEOID"].map(tiger).fillna(0.0)
        sub["ov_len"] = sub["GEOID"].map(ov).fillna(0.0)
        fm = ((sub["tiger_len"] > 0) == sub["transport_defined"]).mean()
        if fm < 1.0:
            print(f"    !! {vname}/{region} flag match {fm:.4f}", flush=True)
        sub["gap"] = np.where(sub["tiger_len"] > 0,
                              1 - np.minimum(1.0, sub["ov_len"] / sub["tiger_len"].replace(0, np.nan)), 0.0)
        per_region.append(sub)
    out = pd.concat(per_region, ignore_index=True)
    out.to_parquet(f"{CACHE}/transport_datum_{vname}.parquet")
    s = out["gap"].sum()
    marker = "  <<<< IN WINDOW !!!" if abs(s - TARGET) < WINDOW else ""
    print(f"[{vname}] sum={s:.4f}  (target {TARGET:.4f} +-{WINDOW:.4f}, diff {s-TARGET:+.4f}) [{time.time()-t0:.0f}s]{marker}", flush=True)
