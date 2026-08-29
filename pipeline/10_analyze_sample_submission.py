"""Analyze the uploaded SampleSubmission (2).csv from the Zindi Bias Bounty challenge."""
import pandas as pd
import numpy as np

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)

SS = SS_PATH
df = pd.read_csv(SS, dtype={"GEOID": str})

print("=== BASIC INFO ===")
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
print(f"\nRegion counts:\n{df['region'].value_counts()}")
print(f"\nGEOID lengths: {df['GEOID'].str.len().value_counts().to_dict()}")
print(f"Duplicated GEOIDs: {df['GEOID'].duplicated().sum()}")
print(f"Nulls per column:\n{df.isnull().sum()[lambda s: s > 0]}")

print("\n=== coverage_gap_score DISTRIBUTION ===")
print(df['coverage_gap_score'].describe())
print(f"Unique values: {df['coverage_gap_score'].nunique()}")
print(f"Zero count: {(df['coverage_gap_score'] == 0).sum()}")

print("\n=== COMPONENT GAPS DISTRIBUTION ===")
for c in ['transport_gap', 'building_gap', 'poi_gap',
          'poi_gap_fire', 'poi_gap_ems', 'poi_gap_schools', 'poi_gap_cbp']:
    print(f"{c:18s} min={df[c].min():.6f} mean={df[c].mean():.6f} max={df[c].max():.6f} "
          f"unique={df[c].nunique()}")

print("\n=== DEFINED FLAGS (True fraction) ===")
for c in ['transport_defined', 'building_defined', 'poi_defined',
          'poi_defined_fire', 'poi_defined_ems', 'poi_defined_schools', 'poi_defined_cbp']:
    print(f"{c:20s} {df[c].mean():.4f}")

print("\n=== FORMULA CONSISTENCY CHECK ===")
# Competition formula: coverage_gap = mean of defined components
#   transport_gap (if defined), building_gap (if defined),
#   poi_gap (if defined) where poi_gap = mean(facilities_half, establishments_half)
#   facilities_half = mean of defined fire/ems/schools gaps
#   establishments_half = cbp gap
# Check A: direct mean of the three main components where defined
comp_sum = np.zeros(len(df)); comp_cnt = np.zeros(len(df))
for g, d in [('transport_gap', 'transport_defined'), ('building_gap', 'building_defined'), ('poi_gap', 'poi_defined')]:
    comp_sum += np.where(df[d], df[g], 0.0)
    comp_cnt += df[d].astype(int)
calc_a = np.where(comp_cnt > 0, comp_sum / np.maximum(comp_cnt, 1), np.nan)
err_a = calc_a - df['coverage_gap_score']
print(f"A) mean of 3 main components: median|err|={np.nanmedian(np.abs(err_a)):.2e}, "
      f"max|err|={np.nanmax(np.abs(err_a)):.2e}, exact(<1e-6): {(np.abs(err_a) < 1e-6).mean():.4f}")

# Check B: recompute poi_gap from sub-components per competition description
fac_sum = np.zeros(len(df)); fac_cnt = np.zeros(len(df))
for g, d in [('poi_gap_fire', 'poi_defined_fire'), ('poi_gap_ems', 'poi_defined_ems'),
             ('poi_gap_schools', 'poi_defined_schools')]:
    fac_sum += np.where(df[d], df[g], 0.0)
    fac_cnt += df[d].astype(int)
fac_half = np.where(fac_cnt > 0, fac_sum / np.maximum(fac_cnt, 1), np.nan)
est_half = df['poi_gap_cbp']
poi_halves = []
for fh, eh, pd_, pg in zip(fac_half, est_half, df['poi_defined'], df['poi_gap']):
    vals = [v for v, ok in [(fh, True), (eh, True)] if pd.isna(v) is False and ok] if pd_ else []
    poi_halves.append(np.mean(vals) if len(vals) else np.nan)
poi_halves = np.array(poi_halves, dtype=float)
err_poi = poi_halves - df['poi_gap']
print(f"B) poi_gap from halves:        median|err|={np.nanmedian(np.abs(err_poi)):.2e}, "
      f"exact(<1e-6): {(np.abs(err_poi) < 1e-6).mean():.4f}")

print("\n=== SAMPLE ROWS ===")
print(df.head(8).to_string())

print("\n=== ANY IDENTICAL ROWS PATTERN (first 20 rows of key cols) ===")
print(df[['GEOID', 'coverage_gap_score', 'transport_gap', 'building_gap', 'poi_gap']].head(20).to_string())

# Check whether identical component values repeat across many GEOIDs (placeholder sign)
vc = df.groupby(['transport_gap', 'building_gap', 'poi_gap']).size().sort_values(ascending=False)
print(f"\nLargest groups of identical (transport,building,poi) values:\n{vc.head(5)}")

print("\n=== SCORE BY REGION ===")
print(df.groupby('region')['coverage_gap_score'].agg(['count', 'mean', 'median', 'min', 'max']))
