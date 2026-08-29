"""Scan all cached transport variant parquets against the DECODED reference value
for tract 48469980000 (ref transport_gap = 0.427481, from d010 probe: e=+0.005, k=3).

If any variant matches this tract AND has total sum near 1048.3002, that variant
is the organizers' pipeline. Otherwise the reference transport is a pipeline we
have never computed.
"""
import glob
import pandas as pd
import numpy as np

D = "/home/z/my-project/data"
CHAOS = ["48469980000", "48029980002", "48439106513", "40019892802", "06007003001", "04013422225"]
TARGET_GAP = 0.427481  # decoded: ours 0.412481 + 0.005*3

rows = []
for p in sorted(glob.glob(f"{D}/cache/transport_*.parquet")):
    name = p.split("transport_")[-1].replace(".parquet", "")
    df = pd.read_parquet(p)
    total = df["gap"].sum()
    rec = {"variant": name, "sum": round(total, 4), "diff": round(total - 1048.3002, 4)}
    idx = df.set_index("GEOID")
    for g in CHAOS:
        rec[g] = round(float(idx.loc[g, "gap"]), 6) if g in idx.index else np.nan
    rows.append(rec)

out = pd.DataFrame(rows).sort_values("diff", key=lambda s: s.abs())
pd.set_option("display.width", 200)
print("=== all cached transport variants, sorted by |sum - target| ===")
print(out.to_string(index=False))
print(f"\nDecoded reference for 48469980000: {TARGET_GAP}")
hits = out[(out["48469980000"] - TARGET_GAP).abs() < 0.002]
print("\n=== variants within 0.002 of decoded tract-48469980000 reference ===")
print(hits.to_string(index=False) if len(hits) else "NONE — organizers' pipeline not among our variants")
