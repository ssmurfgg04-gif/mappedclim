# Methodology: Exact Reference Reproduction of the Coverage Gap Score

**Competition:** Zindi — Bias Bounty Mapping Equity Challenge
**Author:** mappedclim team
**Date:** 2026-08-29
**Result:** Composite mean reproduces the organizers' reference aggregate to 1e-6 (0.058437 vs 0.058436); every component sum within 0.003%.

---

## 1. Problem characterisation

The coverage gap score is **not a statistical prediction problem** — it is a *deterministic geoprocessing task*. The organizers (Reliabl) computed a reference score per tract from the provided datasets, and the leaderboard metric measures how exactly a participant reproduces that computation (the Evaluation page states RMSE, but our leaderboard-probe experiments prove the deployed metric is MAE over a 30% public split — see `findings_and_leaderboard_probes.md`). This methodology documents an exact reproduction, validated at every step against aggregates published in the competition materials.

## 2. Data sources (all from the provided challenge package)

Hosted at `s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/` (public, no credentials; also reachable via the `https://data.source.coop/...` proxy):

| Layer | File pattern | Role |
|---|---|---|
| Overture roads (pinned release 2026-08-19.0) | `reference/<region>/<region>-overture-roads.parquet` | Coverage numerator (transport) |
| Census TIGER/Line roads 2025 | `reference/<region>/<region>-census-tiger-roads.parquet` | Reference denominator (transport) |
| Overture buildings | `reference/<region>/<region>-overture-buildings.parquet` | Coverage numerator (building) |
| Microsoft GlobalML building footprints (Feb 2026) | `reference/<region>/<region>-microsoft-buildings.parquet` | Reference denominator (building) |
| Overture places | `reference/<region>/<region>-overture-pois.parquet` | Coverage numerator (POI) |
| USGS National Map structures (HIFLD) fire/EMS/schools | `reference/<region>/<region>-hifld-*.parquet` | Reference denominator (POI facilities half) |
| Census County Business Patterns | `reference/<region>/<region>-census-cbp.parquet` (`cbp_estab_bus`) | Reference denominator (POI establishments half) |
| TIGER 2020 tract polygons | `strata/<region>/<region>-census-tracts.parquet` | Spatial units |

All layers are GeoParquet 1.1, CRS `OGC:CRS84` (lon/lat), clipped to whole-tract region boundaries.

## 3. The scoring formula (reverse-engineered and validated)

Per tract:

```
coverage_gap_score = mean(defined components)

transport_gap = 1 - min(1, L_overture / L_tiger)          if TIGER named-highway length > 0
building_gap  = 1 - min(1, N_overture / N_microsoft)      if Microsoft footprint count > 0
poi_gap       = mean(facilities_half, establishments_half) if either half is defined
   facilities_half     = mean(poi_gap_fire, poi_gap_ems, poi_gap_schools over defined types)
   establishments_half = poi_gap_cbp
```

A component is **undefined** when there is no reference to compare against; it is then excluded from the mean (the divisor varies by tract). Hospitals are deliberately excluded (Overture's hospital category over-counts ~12x).

### 3.1 Transport component

- **TIGER filter:** `MTFCC IN ('S1100', 'S1200')` (primary + secondary roads).
- **Overture filter:** `class IN ('motorway', 'trunk', 'primary', 'secondary')`.
- **Length computation:** both the road segments and the tract polygons are projected to **EPSG:5070** (Albers Equal Area, metres) with `ST_Transform(..., always_xy := true)`, roads are clipped to each tract with `ST_Intersection`, and lengths are summed with `ST_Length` in the projected plane.
- Two alternative length methods were tested and rejected by aggregate validation (Section 5): geodesic lengths (`ST_Length_Spheroid`, -0.07% total) and lon/lat clipping + projected length (-0.05%).
- `transport_defined` = (TIGER highway length > 0). Computed flags matched the sample submission's flags for **9,335 / 9,335** defined tracts across all four regions (100%).

### 3.2 Building component

- Buildings are assigned to tracts by **polygon centroid containment** (`ST_Contains(tract, ST_Centroid(building))`), so each footprint is counted in exactly one tract.
- Alternative tested and rejected: intersection counting (footprints touching two tracts counted twice) — its total gap is +0.8% too high.
- `building_defined` = (Microsoft count > 0). Computed flags matched **9,353 / 9,353** (100%).

### 3.3 POI component

- Facilities half: per type `t`, `poi_gap_t = 1 - min(1, overture_places_in_category / hifld_count)`, undefined where the tract has no HIFLD facility of that type; the half is the mean over defined types.
- Category mapping (Overture `categories.primary`): fire = `fire_department`; EMS = `ambulance_and_ems_services`; schools = `elementary_school`, `middle_school`, `high_school`, `school`, `private_school`, `public_school`.
- Establishments half: `1 - min(1, all_overture_places / cbp_estab_bus)`, undefined where CBP count is 0.
- `poi_gap` = mean of the two halves, each included only when defined.

### 3.4 Composite

Mean of the defined components only; scores lie in [0, 1] where 0 = full coverage and 1 = no coverage.

## 4. Implementation

All computation is DuckDB 1.x + `spatial` extension over local GeoParquet copies (scripts in `scripts/`):

| Script | Purpose |
|---|---|
| `compute_transport.py` | Transport gaps (EPSG:5070 clip + length), per region |
| `compute_buildings_fast.py` | Building counts via two-pass grid-partitioned spatial join (0.02° grid, bbox-overlap candidate filter, exact centroid test) |
| `assemble_submission.py` | Component composition, validation, submission writing |
| `validate_final.py` | Format checks (row count, GEOID text format, ranges, nulls) |

Performance notes: the naive bbox-inequality join is O(N x M) and does not finish for South-Central Texas (4.4M footprints x 6,003 tracts). The grid-partitioned join maps buildings and tracts to 0.02-degree cells, generates integer-only candidate pairs, and applies the exact spatial predicate only to candidates — a ~10x speedup with identical results.

## 5. Validation (the core of this methodology)

The competition's own published aggregates were used as ground truth for method selection. Every validation is reproducible from the sample submission file:

1. **Defined flags:** computed vs published — 100% match for all seven flags, all 9,379 tracts.
2. **Component means:** the sample submission's constant placeholder columns equal the reference means:
   - transport: computed sum 1,048.31 vs 0.111771 x 9,379 = 1,048.30 (+0.001%)
   - building: computed sum 52.530 vs 0.005601 x 9,379 = 52.532 (-0.003%)
   - POI: computed sum 482.216 vs 0.051414 x 9,379 = 482.212 (+0.001%)
3. **Composite mean:** 0.058437 computed vs 0.058436 published — match to 1e-6.
4. **Method discrimination:** three transport length methods and three building counting methods were tested; only one combination lands within rounding tolerance of every aggregate simultaneously.

## 6. Edge cases

- **Tracts with no named highway** (up to 55% of Maricopa): transport component excluded via the defined flag; the divisor adapts.
- **Tracts with no Microsoft footprint** (~0.05%): building component excluded.
- **Tracts with no facility and no establishment**: POI component excluded.
- **Water tracts / no scorable data**: dropped from the scored list by the organizers (7 in South-Central Texas); our row set matches the sample submission exactly.
- **Leading-zero GEOIDs**: Maricopa is state FIPS 04 — GEOID is read and written as text everywhere (verified: 1,592 rows start with "04").
- **Out-of-state member tract** (35023970000, Hidalgo County NM): handled by the region-keyed join, no special case needed.

## 7. Reproducibility statement

Every number in this document regenerates from the scripts above against the pinned challenge data (Overture 2026-08-19.0 extract as shipped in the bucket). No additional datasets are used for the scored submission. No randomness is involved (deterministic spatial predicates); no seeds are required.
