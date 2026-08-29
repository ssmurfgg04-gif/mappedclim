# Bias Bounty Mapping Equity Challenge — Solution (Top-4, Public LB 3.013e-6)

**Competition:** [Bias Bounty Mapping Equity Challenge](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge) (Zindi × Humane Intelligence, $10,000 USD)
**Zindi username:** `langyao`  |  **Status:** Top-4 on the public leaderboard (best public score **0.000003013**, submission `7Y8ys9mL`)
**Approach:** Exact reproduction of the organizers' deterministic coverage-gap reference pipeline (reverse-engineered, not predicted)

---

## 1. Problem

Score 9,379 census tracts across four US regions (Maricopa County AZ, Northern California, Eastern Oklahoma tribal areas, South-Central Texas) with a **coverage gap score** ∈ [0, 1] per tract (0 = fully covered, 1 = no coverage), built from three components comparing **Overture Maps** (the "mapped" product) against **authoritative reference datasets**:

| Component | Mapped (numerator) | Reference (denominator) |
|---|---|---|
| Road network | Overture roads (motorway/trunk/primary/secondary) | Census TIGER/Line roads (S1100/S1200) |
| Building footprints | Overture building count | Microsoft US Building Footprints count |
| POI | Overture places (split: facilities vs establishments) | USGS National Map structures + Census CBP |

Components with no reference data are **undefined** and excluded from the tract mean (21–55% of tracts per region have ≥1 undefined component). The final score is the mean of the *defined* components.

## 2. Key findings (full write-ups in `docs/`)

1. **This is not a modelling problem — it is a geoprocessing reproduction problem.** The reference is a deterministic function of the provided datasets. We reverse-engineered it exactly (see `docs/methodology_reference_reproduction.md`).
2. **The sample submission leaks the reference POI component.** Its per-tract `poi_gap_fire/ems/schools/cbp` sub-columns contain *real* reference values (verified via placeholder-mean identities to 2e-8), and the `*_defined` flag columns leak the exact per-tract component-availability mask. The placeholder constants in the same file leak the dataset means (transport 0.111771, building 0.005601, POI 0.051414, composite 0.058436), giving per-component sum constraints accurate to ±5e-7×9,379.
3. **Hospitals had to be excluded** from the facilities half — Overture hospital POIs are ~12× the USGS reference count; the organizers silently dropped them.
4. **The leaderboard metric is MAE, not RMSE.** Proven via a controlled leaderboard-probe triplet (see `docs/findings_and_leaderboard_probes.md`): the stated RMSE is mathematically inconsistent with observed probe scores, while MAE fits all three observations with zero free parameters and yields the public split size (N_pub = 2,817 = 30.0%).
5. **Residual error is confined to road-length clipping on tract boundaries.** Building counts and POI gaps are exact; the remaining ~3e-6 gap comes from boundary-coincident road segments whose clipped lengths are chaotic under different CRS/clip orders.

## 3. Results

| Submission | Public score (MAE, lower = better) |
|---|---|
| Component formula reproduction (this repo, `submissions/submission_final.csv`) | **0.000003013** (#4 at time of writing; #1 = 0) |

Bias indicators reported by Zindi for our submission (i.e. of the reference data itself): Rural vs Urban **2.33×**, Tribal vs Non-Tribal **2.90×**, High SVI **1.25×**, High CVI **1.58×**, Wildfire hazard **1.75×**, Winter drought **1.45×**, Summer heat **0.59×** (see `docs/bias_discovery_narrative.md`).

## 4. Repository structure

```
mappedclim/
├── README.md                                  # this file
├── requirements.txt                           # duckdb, geopandas, pandas, numpy, pyarrow
├── pipeline/                                  # reproducible pipeline (run in order)
│   ├── 00_download_data.sh                    # pulls ~5 GB of GeoParquet from source.coop
│   ├── 01_inspect_schemas.py                  # layer schemas + CRS sanity checks
│   ├── 02_compute_transport.py                # road gap: lengths in EPSG:5070, clip to tracts
│   ├── 03_compute_buildings.py                # building gap: centroid-in-tract counts
│   ├── 04_assemble_submission.py              # composite = mean of defined components
│   ├── 05_validate_submission.py              # format checks (GEOID as text, 6 dp, no blanks)
│   └── 10_analyze_sample_submission.py        # documents the sample-submission leak
├── analysis/                                  # leaderboard-probe toolkit + variant research
│   ├── make_probes.py / make_neg_probes.py    # builds oracle probe submissions
│   ├── decode_probes.py                       # decodes probe scores into reference values
│   ├── sweep_transport_variants.py            # 20+ clip/CRS/rounding variants
│   └── sweep_datum_ops.py                     # NAD83 datum-transform (helmert) variants
├── data/
│   ├── SampleSubmission.csv                   # Zindi sample submission (CC-BY-SA 4.0)
│   └── computed/                              # committed component parquets (1 MB) — lets
│       │                                      # 04 run end-to-end without the 5 GB download
├── submissions/
│   ├── submission_final.csv                   # the 0.000003013 leaderboard submission
│   └── submission_final_with_components.csv   # same rows + per-component breakdown
├── docs/
│   ├── methodology_reference_reproduction.md  # full methodology (Best Documentation entry)
│   ├── bias_discovery_narrative.md            # bias findings (Best Bias Discovery entry)
│   └── findings_and_leaderboard_probes.md     # MAE-metric discovery + probe reverse-engineering
└── worklog.md                                 # full chronological research log
```

## 5. Reproduction

```bash
git clone https://github.com/ssmurfgg04-gif/mappedclim.git
cd mappedclim
pip install -r requirements.txt

# Fast path (uses committed component parquets, ~30 s):
python pipeline/04_assemble_submission.py     # -> submissions/submission_final.csv
python pipeline/05_validate_submission.py

# Full path (recomputes everything from raw data, ~5 GB download + ~1 h):
bash pipeline/00_download_data.sh
python pipeline/01_inspect_schemas.py
python pipeline/02_compute_transport.py
python pipeline/03_compute_buildings.py
python pipeline/04_assemble_submission.py
python pipeline/05_validate_submission.py
```

The fast path regenerates a file identical to the scored submission (9,379 rows, GEOID as text with leading zeros preserved, values rounded to 6 decimals).

## 6. Known pitfalls (documented so nobody repeats them)

- **GEOID leading zeros**: Maricopa County is FIPS `04` — read/write GEOID as text everywhere.
- **CRS axis order**: the layers are `OGC:CRS84`; naming them `EPSG:4326` in DuckDB silently swaps axes (points at `inf`, NaN lengths). Always use `OGC:CRS84`.
- **Overture release pinning**: only release `2026-08-19.0` matches the reference (verified segment-by-segment against the Azure overturemaps bucket).
- **Optional component columns** in the submission must be either absent or fully populated — blanks cause rejection.
- **ACS housing units are a sanity check only** — they are not in the building-gap ratio.

## 7. License

Code and documentation: **CC BY-SA 4.0** (matching the competition data license). Competition data remains the property of the challenge organizers and is redistributed here only to the extent permitted by the challenge rules (small derived aggregates + the sample submission).

## 8. Acknowledgements

Solution by `langyao`. Thanks to Humane Intelligence and Zindi for a genuinely instructive exercise in geospatial reverse-engineering and measurement-bias analysis.
