# Bias Bounty Mapping Equity Challenge — Methodology

## Approach Overview

Our approach combines spatial feature engineering, gradient-boosted tree ensembles, and systematic bias discovery to predict coverage gap scores for US Census tracts. The methodology is designed around three key principles:

1. **Feature richness over model complexity**: 300+ engineered features per tract capture the full information content of the Overture Maps + reference data
2. **Spatial integrity**: All evaluation uses GroupKFold by county to prevent spatial autocorrelation leakage
3. **Bias-aware optimization**: We jointly optimize prediction RMSE and equity across demographic strata

## Feature Engineering (300+ features per tract)

### Coverage Gap Features
- Building count ratio (Overture / Microsoft) and gap (1 - ratio)
- Road length ratio (Overture / TIGER) and gap
- Road segment count ratio
- POI count vs HIFLD facility count
- Building count vs ACS housing units
- Log transforms and squared terms for nonlinear relationships

### Source Composition Features (unique differentiator)
- Per-tract ML-derived fraction (% of buildings from Microsoft ML, not human-verified)
- OSM fraction, Google fraction, Esri fraction
- Source diversity (count of distinct sources)
- POI mean confidence and low-confidence fraction

### Strata Table Features (from national 85,396-tract table)
- SVI (Social Vulnerability Index) overall and sub-indices
- CVI (Climate Vulnerability Index)
- RUCA rural/urban classification and population density
- Tribal tract indicators
- Wildfire risk (USFS, NIFC, MTBS)
- Drought risk (USDM)
- All `*_covered` flags encoded as: 1=covered, 0=not covered, -1=null (data doesn't reach)
- Data coverage depth and fraction

### Spatial Lag Features
- K-nearest neighbor (k=5,10,20) mean aggregates
- Spatial difference from neighbors (local outlier detection)
- County-level mean and standard deviation
- Tract deviation from county mean

### Interaction Features
- SVI × rural/urban
- SVI × building gap, SVI × road gap
- Tribal × hazard exposure
- Tribal × building gap
- Compound risk score (building gap + road gap + low data coverage)
- Coverage gap × ML-derived fraction
- Log and squared transforms

## Model Architecture

### Base Models
1. **XGBoost**: 500 estimators, max_depth=6, learning_rate=0.05, with spatial GroupKFold
2. **LightGBM**: 500 estimators, max_depth=6, num_leaves=31, with spatial GroupKFold
3. **CatBoost**: For categorical feature handling (when target available)

### Ensemble
- Optimized weighted average (scipy SLSQP) minimizing RMSE on OOF predictions
- Stacking ensemble with Ridge meta-learner (backup)

### Spatial Cross-Validation
- GroupKFold by county FIPS (5 folds)
- Prevents spatial autocorrelation leakage
- Ensures model generalizes to unseen counties

## Self-Evolving Pipeline

The pipeline iteratively improves through:
1. Train all models with current best hyperparameters
2. Evaluate with spatial CV + bias penalty (RMSE + 0.3 × bias_score)
3. Auto-tune with random search (20 trials per iteration)
4. Analyze residuals for systematic strata patterns
5. Generate new interaction features if residuals correlate with strata
6. Repeat until convergence (RMSE improvement < 0.001)

## Current Results (Self-Supervised, Proxy Targets)

| Model | Target | CV RMSE | CV R² |
|-------|--------|---------|-------|
| XGBoost | Building Gap | 0.139 | 0.69 |
| LightGBM | Building Gap | 0.140 | 0.69 |
| Blended | Building Gap | 0.139 | 0.69 |
| XGBoost | Road Gap | 0.195 | 0.94 |

## Bias Discovery ($1,000 Prize)

### Individual Strata Disparities
| Dimension | Building Gap Disparity | Road Gap Disparity |
|-----------|----------------------|-------------------|
| Rural/Urban | 0.14 | 1.28 |
| Tribal | 0.09 | 0.77 |
| Wildfire Risk | 0.07 | 0.89 |
| Drought Risk | 0.10 | 0.37 |

### Key Intersectional Gaps
The Bias Scoring API evaluates individual strata, but misses compounding effects:
- **Rural + High-SVI**: Systematically larger road AND building gaps
- **Tribal + High Wildfire**: Double vulnerability — harder to map AND more likely to need emergency mapping
- **Border + Colonias**: Microsoft detects structures but Overture conflation excludes them

### Real-World Impact
1. **Emergency dispatch**: Missing roads in tribal wildfire corridors delay 911 response
2. **FEMA assessment**: Colonias in South-Central TX invisible to damage assessment
3. **ML verification gap**: 88% of buildings in Eastern OK are ML-derived (never human-verified)
4. **Compound deprivation**: Rural high-SVI tracts face ALL forms of mapping deprivation simultaneously

## Target Reverse-Engineering Strategy

When the competition releases the target variable (Aug 28), we will:
1. Test linear formula candidates (weighted averages of sub-metrics)
2. Test nonlinear candidates (geometric mean, harmonic mean, max gap)
3. Use symbolic regression (PySR) if simple formulas don't match
4. If exact formula found, skip ML entirely for that component

## GPU Training (Kaggle)

For Optuna tuning with 100+ trials, we use Kaggle T4x2 GPUs:
- XGBoost with `tree_method='gpu_hist'` and `device='cuda'`
- LightGBM with `device='gpu'`
- CatBoost with `task_type='GPU'`
- 50 Optuna trials per model, 3-fold spatial CV each

## Reproducibility

All code, configs, and random seeds (42) are version-controlled. The pipeline is fully resumable — state is saved after each iteration. Feature engineering is deterministic (no random sampling).
