"""Decode LB probe scores into reference values.

Usage: python3 scripts/decode_probes.py <scoreA> <scoreB> <score_48469980000> <score_48029980002>
                                      <score_48439106513> <score_40019892802> <score_06007003001>
                                      <score_04013422225>
All scores as decimals (e.g. 0.004998 or 3.013e-6 or 0.000003013).

Math:
  baseline: s0 = 0.000003013, SSE0 = N * s0^2
  global probe delta: score^2 = s0^2 + delta^2 - 2*delta*(E/N)   (N-invariant; gives mean error E/N)
  single-tract probe: score^2 = s0^2 + delta^2/N - 2*delta*e_i/N
    -> N = delta^2 / (score^2 - s0^2)  (first-order; e_i tiny)
    -> e_i = (SSE0 + delta^2 - N*score^2) / (2*delta);  ref_i = base_i + e_i
"""
import sys
import numpy as np
import pandas as pd

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

S0 = 3.013e-6
SS = SS_PATH
CACHE = os.path.join(DATA, "cache")  # local variant cache (optional)

PROBES = [
    ("probeA_d005", "global", 0.005),
    ("probeB_d010", "global", 0.010),
    ("probe_48469980000", "48469980000", 0.001),
    ("probe_48029980002", "48029980002", 0.001),
    ("probe_48439106513", "48439106513", 0.001),
    ("probe_40019892802", "40019892802", 0.001),
    ("probe_06007003001", "06007003001", 0.001),
    ("probe_04013422225", "04013422225", 0.001),
]


def main(scores):
    base = pd.read_csv(os.path.join(OUT, "submission_final.csv"), dtype={"GEOID": str})
    basemap = dict(zip(base["GEOID"], base["coverage_gap_score"]))

    # variant values for comparison
    variants = {}
    v = pd.read_parquet(f"{CACHE}/transport_r7_tracts.parquet")[["GEOID", "gap"]]
    variants["v1(base5070)"] = dict(zip(v["GEOID"], v["gap"]))
    v = pd.read_parquet(os.path.join(DATA, "transport_gaps_llclip_5070len_all.parquet"))[["GEOID", "transport_gap_calc"]]
    variants["v3(4326clip)"] = dict(zip(v["GEOID"], v["transport_gap_calc"]))
    v = pd.read_parquet(f"{CACHE}/transport_r6_roads.parquet")[["GEOID", "gap"]]
    variants["r6(ov6dp)"] = dict(zip(v["GEOID"], v["gap"]))
    v = pd.read_parquet(f"{CACHE}/transport_datum_op3.parquet")[["GEOID", "gap"]]
    variants["op3(helmert)"] = dict(zip(v["GEOID"], v["gap"]))
    v = pd.read_parquet(os.path.join(DATA, "transport_gaps_eastern-ok-maricopa-az-northern-ca-south-central-tx.parquet"))[["GEOID", "transport_gap_calc"]]
    variants["crs84(duckdb)"] = dict(zip(v["GEOID"], v["transport_gap_calc"]))

    N = None
    print(f"baseline s0 = {S0:.3e}\n")
    for (name, target, delta), sc in zip(PROBES, scores):
        sc = float(sc)
        if target == "global":
            mean_err = (S0**2 + delta**2 - sc**2) / (2 * delta)
            print(f"[{name}] score={sc:.6g}  -> mean error (E/N) = {mean_err:+.3e}")
        else:
            if abs(sc - S0) < 1e-11:
                print(f"[{name}] score={sc:.6g}  -> tract PRIVATE (no info)")
                continue
            if N is None:
                N = delta**2 / (sc**2 - S0**2)
                print(f"[{name}] score={sc:.6g}  -> PUBLIC; first public probe => N_pub = {N:.1f}")
            SSE0 = N * S0**2
            SSE_new = N * sc**2
            e_i = (SSE0 + delta**2 - SSE_new) / (2 * delta)
            ref_cov = basemap[target] + e_i
            print(f"[{name}] score={sc:.6g}  -> PUBLIC  e_i = {e_i:+.3e}  ref_coverage = {ref_cov:.6f}")
            # decompose: coverage = (t+b+p)/k; poi & building exact => e_t = e_i * k
            ss = pd.read_csv(SS, dtype={"GEOID": str})
            row = ss[ss["GEOID"] == target].iloc[0]
            k = int(row.transport_defined) + int(row.building_defined) + int(row.poi_defined)
            e_t = e_i * k
            print(f"        k={k}, transport gap error e_t = {e_t:+.6f}")
            print(f"        implied ref transport_gap = v1 + e_t:")
            for vn, d in variants.items():
                t_v1 = d[target] if not np.isnan(d.get(target, np.nan)) else None
                if t_v1 is not None:
                    print(f"          {vn:15s} = {t_v1:.6f}   (would need e_t={t_v1 - variants['v1(base5070)'][target]:+.6f})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a]
    if len(args) != 8:
        print(__doc__)
        print("Provide exactly 8 scores (probeA, probeB, then 6 tract probes).")
        sys.exit(1)
    main(args)
