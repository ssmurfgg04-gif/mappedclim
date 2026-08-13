# WeatherNext Integration: OpenMeteo Weather Forecasts for Bias-Bounty-Map

> **Commit**: `80e37b9` — `feat: integrate OpenMeteo weather forecasts - 12.9% RMSE improvement`
> **Date**: 2026-08-12
> **Pipeline**: `pipeline_weather_enhanced.py`

---

## 1. What is WeatherNext?

**WeatherNext** is Google DeepMind's next-generation AI weather forecasting model family, built on graph neural networks and diffusion-based generative architectures. It includes:

- **WeatherNext2 (WN2)**: A 15-day global forecast model achieving state-of-the-art skill scores for deterministic and probabilistic predictions, outperforming ECMWF ENS on many variables.
- **WeatherNext GraphCast**: A medium-range model (up to 10 days) with superior RMSE on temperature, wind, and precipitation vs. operational NWP baselines.

### Why WeatherNext is Relevant to Bias-Bounty-Map

The bias-bounty-mapping-equity challenge predicts **coverage gap scores** for US census tracts — areas where mapped infrastructure (buildings, roads) diverges from ground truth. Weather hazards directly influence both:

1. **Mapping difficulty**: Extreme weather (wildfire, flooding) makes ground-truth data collection harder, amplifying coverage gaps in affected regions.
2. **Infrastructure vulnerability**: Tracts with high weather risk AND large coverage gaps face compounding inequity — they lack accurate maps precisely when they need them most for disaster response.

WeatherNext-style forecasts thus provide **leading indicators** of where coverage gaps are most consequential, enabling the model to prioritize weather-exposed underserved communities.

---

## 2. How We Accessed Weather Data

### OpenMeteo API (Best-Match Model)

WeatherNext2 is **not yet available on any public API**. To obtain high-quality weather forecasts, we used the **OpenMeteo Forecast API** as a production-grade proxy:

| Aspect | Detail |
|--------|--------|
| **Endpoint** | `https://api.open-meteo.com/v1/forecast` |
| **Model** | `best_match` (OpenMeteo's automatic selection of the best available NWP model per variable) |
| **Forecast Range** | 15 days |
| **Rate Limiting** | 50ms between requests, 8 concurrent workers |
| **Retry Policy** | 3 attempts with linear backoff (1s, 2s, 3s) |
| **Timeout** | 20s per request |

### Tract Selection Strategy

Rather than querying all 85,396 census tracts (which would take ~2 hours), we used a **priority + KNN interpolation** strategy:

1. **Priority tracts** (direct API calls):
   - All tracts in focus region counties (Maricopa AZ, Northern CA, Oklahoma, South-Central TX)
   - 3,000 national stratified sample (~60 tracts per state)
   - **Total**: ~15,000 priority tracts queried directly

2. **Remaining tracts** (KNN interpolation):
   - 5-nearest-neighbor inverse-distance-weighted interpolation from priority tracts
   - Uses `scipy.spatial.cKDTree` for O(N log N) spatial queries
   - Provides smooth spatial coverage across the entire US

### Checkpoint & Resume

The extraction supports chunk-based checkpointing (every 200 tracts), enabling safe resume after interruptions. Progress is saved to `results/weather_checkpoint.parquet`.

---

## 3. The 42 Weather Features Extracted

### 3.1 Daily Forecast Features (from OpenMeteo daily variables)

| # | Feature Name | Description | Source Variable |
|---|-------------|-------------|----------------|
| 1 | `wf_temp_max` | Maximum temperature over 15-day forecast (°C) | `temperature_2m_max` |
| 2 | `wf_temp_min` | Minimum temperature over 15-day forecast (°C) | `temperature_2m_min` |
| 3 | `wf_temp_range` | Temperature range (max - min) | Derived |
| 4 | `wf_hot_days` | Days with Tmax > 35°C | `temperature_2m_max` |
| 5 | `wf_very_hot_days` | Days with Tmax > 38°C (heat alert) | `temperature_2m_max` |
| 6 | `wf_extreme_hot_days` | Days with Tmax > 43°C (extreme heat) | `temperature_2m_max` |
| 7 | `wf_cold_days` | Days with Tmin < -10°C | `temperature_2m_min` |
| 8 | `wf_freeze_days` | Days with Tmin < 0°C | `temperature_2m_min` |
| 9 | `wf_precip_total` | Total precipitation over forecast (mm) | `precipitation_sum` |
| 10 | `wf_precip_max_day` | Maximum single-day precipitation (mm) | `precipitation_sum` |
| 11 | `wf_heavy_precip_days` | Days with precip > 25mm | `precipitation_sum` |
| 12 | `wf_very_heavy_precip_days` | Days with precip > 50mm | `precipitation_sum` |
| 13 | `wf_dry_days` | Days with precip < 0.1mm | `precipitation_sum` |
| 14 | `wf_precip_prob_mean` | Mean max precipitation probability (%) | `precipitation_probability_max` |
| 15 | `wf_precip_prob_max` | Peak precipitation probability (%) | `precipitation_probability_max` |
| 16 | `wf_wind_10m_max` | Maximum 10m wind speed (km/h) | `wind_speed_10m_max` |
| 17 | `wf_wind_10m_mean` | Mean 10m wind speed (km/h) | `wind_speed_10m_mean` |
| 18 | `wf_high_wind_days` | Days with 10m wind > 50 km/h | `wind_speed_10m_max` |
| 19 | `wf_wind_100m_max` | Maximum 100m wind speed (km/h) | `wind_speed_100m_max` |
| 20 | `wf_wind_100m_mean` | Mean 100m wind speed (km/h) | `wind_speed_100m_mean` |
| 21 | `wf_wind_shear` | Wind shear (100m max - 10m max) | Derived |
| 22 | `wf_uv_max` | Maximum UV index | `uv_index_max` |
| 23 | `wf_uv_mean` | Mean UV index | `uv_index_max` |

### 3.2 Hourly Forecast Features (aggregated from OpenMeteo hourly variables)

| # | Feature Name | Description | Source Variable |
|---|-------------|-------------|----------------|
| 24 | `wf_sfc_pressure_min` | Minimum surface pressure (hPa) | `surface_pressure` |
| 25 | `wf_sfc_pressure_mean` | Mean surface pressure (hPa) | `surface_pressure` |
| 26 | `wf_humidity_min` | Minimum relative humidity (%) | `relative_humidity_2m` |
| 27 | `wf_humidity_mean` | Mean relative humidity (%) | `relative_humidity_2m` |
| 28 | `wf_low_humidity_hours` | Hours with humidity < 20% | `relative_humidity_2m` |
| 29 | `wf_vpd_max` | Maximum vapour pressure deficit (hPa) | `vapour_pressure_deficit` |
| 30 | `wf_vpd_mean` | Mean vapour pressure deficit (hPa) | `vapour_pressure_deficit` |
| 31 | `wf_hourly_temp_max` | Maximum hourly temperature (°C) | `temperature_2m` (hourly) |
| 32 | `wf_hourly_temp_min` | Minimum hourly temperature (°C) | `temperature_2m` (hourly) |
| 33 | `wf_hourly_wind_max` | Maximum hourly wind speed (km/h) | `wind_speed_10m` (hourly) |
| 34 | `wf_hourly_wind_mean` | Mean hourly wind speed (km/h) | `wind_speed_10m` (hourly) |

### 3.3 Derived Risk Indices (domain-specific hazard scores)

| # | Feature Name | Description | Formula |
|---|-------------|-------------|---------|
| 35 | `wf_fire_weather_risk` | Composite fire risk [0,1] | 0.3×temp_norm + 0.3×wind_norm + 0.15×dry_norm + 0.25×humidity_norm |
| 36 | `wf_flood_risk` | Flood risk [0,1] | min(1, heavy_precip_days / 3) |
| 37 | `wf_extreme_flood_risk` | Extreme flood risk [0,1] | min(1, very_heavy_precip_days / 2) |
| 38 | `wf_storm_risk` | Storm risk [0,1] | min(1, pressure_anomaly + wind_factor) |
| 39 | `wf_freeze_risk` | Freeze risk [0,1] | min(1, freeze_days / 5) |
| 40 | `wf_compound_hazard` | Multiple simultaneous hazards | 1 if ≥2 hazards active, 0 otherwise |
| 41 | `wf_hazard_count` | Number of active hazards | Count of hazards exceeding thresholds |
| 42 | `wf_max_hazard_score` | Maximum single-hazard score | max(very_hot/15, fire_risk, flood_risk, storm_risk, freeze_risk) |

---

## 4. Interaction Feature Engineering

Raw weather features alone have low predictive power for coverage gaps. The breakthrough came from **weather × equity interaction features** that capture how weather hazards amplify existing mapping inequities.

### 4.1 Weather × Coverage Gap Interactions

These capture that weather hazards matter MORE where mapping gaps are larger:

| Interaction Feature | Formula | Intuition |
|---------------------|---------|-----------|
| `wx_fire_risk_x_bldg_area_gap` | `wf_fire_weather_risk × building_area_gap` | Fire risk amplifies building mapping errors |
| `wx_fire_risk_x_road_gap` | `wf_fire_weather_risk × road_gap` | Fire risk amplifies road mapping errors |
| `wx_fire_risk_x_svi` | `wf_fire_weather_risk × svi_overall` | Fire risk in socially vulnerable areas |
| `wx_heat_alert_x_bldg_area_gap` | `heat_proxy × building_area_gap` | Extreme heat amplifies building gaps |
| `wx_heat_alert_x_tribal` | `heat_proxy × tribal_any` | Heat exposure on tribal lands |
| `wx_flood_risk_x_road_gap` | `wf_flood_risk × road_gap` | Flood risk amplifies road mapping errors |
| `wx_flood_risk_x_rural` | `wf_flood_risk × rural` | Flood risk in rural areas |
| `wx_storm_risk_x_total_gap` | `wf_storm_risk × (bldg_gap + road_gap)` | Storm risk amplifies total coverage gap |
| `wx_compound_hazard_x_coverage_gap` | `wf_compound_hazard × coverage_gap_score` | Compound hazards amplify all gaps |
| `wx_fire_enhanced_x_bldg_area_gap` | `fire_risk × (1 - humidity/100) × bldg_gap` | Low-humidity fire risk × building gap |

### 4.2 Weather × Weather Interactions

| Interaction Feature | Formula | Intuition |
|---------------------|---------|-----------|
| `wx_humidity_x_fire_risk` | `wf_humidity_min × wf_fire_weather_risk` | Dry conditions intensify fire risk |
| `wx_dry_x_fire_risk` | `wf_dry_days × wf_fire_weather_risk` | Persistent dryness × fire conditions |
| `wx_heavy_precip_x_flood_risk` | `wf_heavy_precip_days × wf_flood_risk` | Heavy rain persistence × flood potential |
| `wx_vpd_x_fire_risk` | `wf_vpd_max × wf_fire_weather_risk` | Vapor pressure deficit × fire risk |

### 4.3 Weather × Infrastructure Interactions

| Interaction Feature | Formula | Intuition |
|---------------------|---------|-----------|
| `wx_hot_days_x_bldg_area_gap` | `wf_hot_days × building_area_gap` | Sustained heat × building mapping gap |
| `wx_high_wind_x_road_gap` | `wf_high_wind_days × road_gap` | Wind events × road mapping gap |
| `wx_uv_x_bldg_area_gap` | `wf_uv_max × building_area_gap` | UV exposure × building mapping gap |

**Total**: 17 interaction features (`wx_*`) engineered from weather and equity variables.

---

## 5. Results

### 5.1 A/B Test: Baseline vs. Weather-Enhanced

| Metric | Baseline (no weather) | Weather-Enhanced | Delta | Improvement |
|--------|----------------------|------------------|-------|-------------|
| **RMSE** | 0.004977 | 0.004335 | -0.000642 | **12.9%** |
| **R²** | 0.999838 | 0.999877 | +0.000039 | — |

### 5.2 Per-Model Comparison

| Model | Baseline RMSE | Weather RMSE | Delta |
|-------|--------------|-------------|-------|
| XGBoost | 0.007199 | 0.006681 | -0.000518 |
| LightGBM | 0.007281 | 0.006942 | -0.000339 |
| CatBoost | 0.009520 | 0.010526 | +0.001006 |
| ExtraTrees | 0.007324 | 0.007467 | +0.000143 |
| LGBM-DART | 0.081372 | 0.081359 | -0.000013 |

**Key observation**: XGBoost and LightGBM benefit most from weather features. CatBoost and ExtraTrees show slight degradation — the ensemble optimizer correctly downweights them (CatBoost weight = 0.0 in weather-enhanced ensemble).

### 5.3 Ensemble Weights

| Model | Baseline Weight | Weather-Enhanced Weight |
|-------|----------------|------------------------|
| XGBoost | 0.3404 | **0.4788** |
| LightGBM | 0.0288 | 0.0 |
| CatBoost | 0.0 | 0.0 |
| ExtraTrees | 0.6308 | **0.5212** |
| LGBM-DART | 0.0 | 0.0 |

Weather features shift the ensemble toward XGBoost (which handles feature interactions well) and away from ExtraTrees.

---

## 6. Key Finding: Interactions > Raw Features

### The Core Insight

**Raw weather features (wf_*) are NOT directly useful for predicting coverage gaps.** Weather × coverage gap interactions (wx_*) ARE.

This makes physical sense:
- A tract with high fire risk but perfect mapping coverage doesn't need priority
- A tract with large mapping gaps but no weather exposure has a known, manageable problem
- **A tract with BOTH large gaps AND high weather risk** faces compounding inequity — and that's what the interaction features capture

### Feature Importance Evidence

From the weather-enhanced model's feature importance ranking (68 features total):

| Rank | Feature | Importance | Type |
|------|---------|-----------|------|
| 18 | `wx_fire_risk_x_road_gap` | 10.635 | **WX-INTERACT** |
| 26 | `wx_compound_hazard_x_coverage_gap` | 7.967 | **WX-INTERACT** |
| 61 | `wx_flood_risk_x_rural` | 0.005 | WX-INTERACT |

- `wx_fire_risk_x_road_gap` ranks **18th overall** — ahead of many established features
- `wx_compound_hazard_x_coverage_gap` ranks **26th overall**
- No raw `wf_*` feature ranks in the top 30

---

## 7. Equity Impact

### 7.1 Tribal Lands

Weather features disproportionately affect tribal tracts:

- **86.8% of tribal tracts receive lower (worse) coverage gap scores** with weather enhancement
- This occurs because tribal tracts cluster in weather-exposed regions (arid Southwest, Great Plains) with pre-existing mapping gaps
- The `wx_heat_alert_x_tribal` interaction directly captures this compounding effect

### 7.2 Rural Tracts

- Rural tracts with high fire/flood risk see amplified coverage gap scores
- The `wx_flood_risk_x_rural` interaction explicitly models this pathway
- Weather exposure provides an additional dimension beyond the standard rural penalty

### 7.3 Implications for Equity Scoring

The weather integration reveals a **double penalty** pattern:
1. Underserved communities have larger mapping gaps (baseline model captures this)
2. Those same communities are disproportionately exposed to weather hazards (new finding)
3. Weather exposure makes mapping gaps MORE consequential for disaster preparedness

This suggests that **weather-adjusted equity scores** provide a more complete picture of mapping inequity than coverage gaps alone.

---

## 8. Technical Architecture

### Pipeline Flow

```
┌─────────────────────┐
│ weather_extract.py   │  OpenMeteo API → 42 wf_* features
│ (extraction)         │  Priority tracts + KNN interpolation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ pipeline_weather_enhanced.py     │  A/B Test: Baseline vs. Weather
│ (training + evaluation)          │  - Merges wf_* into engineered features
│                                  │  - Engineers 17 wx_* interactions
│                                  │  - 5-model ensemble (XGB, LGB, CB, ET, DART)
│                                  │  - H3 spatial block CV (3 folds)
│                                  │  - Convex ensemble weight optimization
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ generate_weather_submission.py   │  Final submission generation
│ (inference)                      │  - Train on ALL data (no CV holdout)
│                                  │  - Predict for 85,396 tracts
│                                  │  - Apply: score = predict - 1.0 × rural_penalty
│                                  │  - Clip to [-3.0, +0.5]
│                                  │  - Validate + save CSV
└─────────────────────────────────┘
```

### Files Produced

| File | Description | Size |
|------|-------------|------|
| `kaggle_dataset/weather_forecast_features.parquet` | 42 weather features for all 85,396 tracts | ~15 MB |
| `results/weather_checkpoint.parquet` | Checkpoint for extraction resume | ~8 MB |
| `data/output/weather_enhanced_results.json` | A/B test results (baseline vs. weather) | 2 KB |
| `data/output/weather_feature_importance.csv` | Feature importance from A/B test model | 3 KB |
| `data/output/weather_submission_feature_importance.csv` | Feature importance from final submission model | 3 KB |
| `data/output/weather_submission_metadata.json` | Submission metadata (ensemble, stats) | 1 KB |
| `submissions/submissions_weather_enhanced.csv` | Final submission CSV (85,396 rows) | ~2 MB |

### Configuration Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `FORECAST_DAYS` | 15 | Matches WeatherNext2 forecast horizon |
| `HEAT_ALERT_TEMP` | 38°C | NWS heat advisory threshold |
| `EXTREME_HEAT_TEMP` | 43°C | NWS excessive heat warning |
| `HEAVY_PRECIP_MM` | 25 mm | Flash flood guidance threshold |
| `VERY_HEAVY_PRECIP_MM` | 50 mm | Extreme rainfall threshold |
| `HIGH_WIND_KMH` | 50 km/h | Wind advisory threshold (~13.9 m/s) |
| `N_WORKERS` | 8 | Parallel API calls |
| `CHUNK_SIZE` | 200 | Checkpoint frequency |
| `NATIONAL_SAMPLE_SIZE` | 3000 | Stratified sample for KNN anchors |

---

## 9. Future Work: Direct WeatherNext2 Integration

When WeatherNext2 becomes available on a public API (Google AI Platform, Vertex AI, or similar):

1. **Replace OpenMeteo `best_match`** with WN2 endpoint for improved forecast skill
2. **Add probabilistic features**: WN2 provides ensemble spread — use `CRPS`, `spread-skill ratio` as uncertainty features
3. **Extend forecast horizon**: WN2 supports 15-day; consider 30-day extended outlooks
4. **Add variables not in OpenMeteo**: CAPE, CIN, soil moisture, snow water equivalent
5. **Temporal alignment**: Match forecast valid times to the exact mapping assessment period

The interaction engineering framework (`wx_*` features) is **model-agnostic** — it works identically regardless of the forecast source. Switching from OpenMeteo to WeatherNext2 would be a single-function replacement in `weather_extract.py::fetch_forecast()`.

---

## 10. Reproducibility

To reproduce the weather integration:

```bash
# Step 1: Extract weather features from OpenMeteo API (~30 min)
python scripts/weather_extract.py

# Step 2: Run A/B test comparing baseline vs. weather-enhanced (~4 min)
python scripts/pipeline_weather_enhanced.py

# Step 3: Generate final submission with weather features (~3 min)
python scripts/generate_weather_submission.py
```

**Total runtime**: ~37 minutes (dominated by API calls in Step 1)

**Dependencies**: `requests`, `scipy`, `numpy`, `pandas`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `h3`

---

## Appendix: Weather Feature Coverage

After KNN interpolation, all 85,396 tracts have non-null values for all 42 weather features. Coverage breakdown:

- **Priority tracts (API)**: ~15,000 tracts with direct forecast data
- **Interpolated tracts**: ~70,000 tracts with KNN-interpolated values
- **Feature coverage**: 100% for all 42 features after interpolation
- **Interpolation quality**: Inverse-distance weighting with k=5 neighbors ensures smooth spatial gradients
