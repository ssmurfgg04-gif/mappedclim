"""Inspect schemas of roads and tracts parquet files."""
import duckdb

# --- portable paths (override with MAPPEDCLIM_ROOT env var) ---
import os as _os
ROOT = _os.environ.get("MAPPEDCLIM_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
DATA = _os.path.join(ROOT, "data")
OUT = _os.path.join(ROOT, "submissions")
SS_PATH = _os.path.join(DATA, "SampleSubmission.csv")

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")

D = DATA

print("=== TRACTS SCHEMA (eastern-ok) ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{D}/tracts/eastern-ok-census-tracts.parquet')").df().to_string())

print("\n=== OVERTURE ROADS SCHEMA (eastern-ok) ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{D}/roads/eastern-ok-overture-roads.parquet')").df().to_string())

print("\n=== TIGER ROADS SCHEMA (eastern-ok) ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{D}/roads/eastern-ok-census-tiger-roads.parquet')").df().to_string())
