# Bias Bounty Mapping Equity Challenge

> **Are the communities most vulnerable to climate risk also the least well-mapped?**

Competition: [Zindi - Bias Bounty Mapping Equity Challenge](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge)  
Prize Pool: $10,000 USD (1st: $4,500 | 2nd: $2,500 | 3rd: $1,500 | Best Documentation: $500 | Best Bias Discovery: $1,000)  
Dates: Aug 28 - Nov 1, 2026 | Metric: RMSE | Max Team: 4

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Source Cooperative S3
python scripts/run_pipeline.py --phase data

# Feature engineering
python scripts/run_pipeline.py --phase features

# Run EDA notebook
jupyter notebook notebooks/01_eda_baseline.ipynb

# Full pipeline
python scripts/run_pipeline.py --phase all
```

## Project Structure

```
bias-bounty-map/
├── config/                    # Configuration files
│   ├── paths.yaml            # S3 paths, focus region bboxes
│   └── model_config.yaml     # Model hyperparameters, CV settings
├── src/
│   ├── data/
│   │   ├── data_access.py    # DuckDB S3 access, bbox pushdown, nested struct parsing
│   │   └── download_data.py  # Data download and caching
│   ├── features/
│   │   └── feature_engineering.py  # All feature classes (coverage gaps, sources, nulls, spatial lag)
│   ├── models/
│   │   └── train.py          # XGBoost, LightGBM, CatBoost + Optuna tuning
│   ├── validation/
│   │   └── spatial_cv.py     # Spatial CV strategies (GroupKFold, Leave-Region-Out)
│   ├── ensemble/
│   │   └── ensemble.py       # Weighted averaging, stacking, target reverse-engineering
│   ├── analysis/
│   │   ├── bias_discovery.py # Residual analysis for $1,000 prize
│   │   └── feature_analysis.py  # SHAP-based feature importance
│   ├── documentation/
│   │   └── generate_docs.py  # Methodology writeup for $500 prize
│   └── utils/
│       └── helpers.py        # Seed setting, logging, data validation
├── notebooks/
│   └── 01_eda_baseline.ipynb # EDA and baseline exploration
├── scripts/
│   └── run_pipeline.py       # Master pipeline (run all phases)
├── data/                     # Local data cache (gitignored)
│   ├── raw/                  # Downloaded parquet files
│   ├── processed/            # Processed tables
│   ├── features/             # Feature matrices
│   └── output/               # Submissions
├── docs/
│   ├── methodology.md        # Methodology writeup
│   └── bias_discovery/       # Bias discovery analysis outputs
└── tests/                    # Unit tests
```

## Competitive Strategy

### Primary Edge: Source Composition Features
The Overture `sources[]` column tells you which upstream dataset supplied each feature
(OSM, Microsoft ML, Google, etc.). Most competitors will ignore this because parsing
nested structs is painful. Computing `ml_derived_fraction`, `osm_fraction`,
`source_diversity`, and `mean_osm_staleness_days` per tract is our biggest advantage.

### Secondary Edge: Null Flags as Signal
The `*_covered` columns are not missing data — NULL means the data layer doesn't reach
that tract. We create binary features from these: `is_conus`, `has_wildfire_data`,
`data_coverage_depth`. This is free information that most people will impute away.

### Tertiary Edge: Target Reverse-Engineering
The coverage gap is a deterministic formula computed by organizers. If we can
reverse-engineer it (via symbolic regression or systematic formula search),
we bypass ML entirely.

### Hedge: Bias Discovery Prize ($1,000)
Independent of leaderboard rank. Find intersectional coverage gaps the automated
API misses (e.g., rural ∩ high-SVI ∩ tribal ∩ high-wildfire).

## Data Sources

| Source | Role | Access |
|--------|------|--------|
| Overture Maps | Coverage data | Source Cooperative S3 (public) |
| Census TIGER/Line Roads | Road reference | Included in challenge data |
| Microsoft Building Footprints | Building reference | Included in challenge data |
| HIFLD Critical Facilities | Facility reference | Included in challenge data |
| National Strata Table (232 cols) | Feature backbone | 85,396 census tracts |

Data is hosted at: `s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/`

## Key Technical Decisions

1. **DuckDB over Pandas** for ETL — bbox pushdown skips Parquet row groups without downloading
2. **Spatial CV (GroupKFold by County)** — random K-fold leaks spatially autocorrelated data
3. **National scope** — the test set likely spans all 85,396 tracts, not just 4 focus regions
4. **No AutoML** — explicitly banned by competition rules
5. **Fixed seeds everywhere** — required for code review survival

## Submission Strategy

- 10 submissions/day, 300 total
- Select 2 submissions for private LB based on local CV, not public LB
- Phase 1 (weeks 1-2): Exploration & baseline
- Phase 2 (weeks 3-5): Model development & tuning
- Phase 3 (weeks 6-8): Ensemble & final selection
