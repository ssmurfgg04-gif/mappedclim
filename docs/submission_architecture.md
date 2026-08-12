# Submission Architecture Specification

**Bias Bounty Map — Coverage Gap Prediction Competition**
**Document version:** 1.0 | **Last updated:** 2026-08-12
**Dataset drop date:** **August 28, 2026**

---

## 1. Required Columns and Types

| Column               | Type       | Constraints                                    |
|----------------------|------------|------------------------------------------------|
| `GEOID`              | `str`      | 11-digit census tract FIPS code, all numeric    |
| `coverage_gap_score` | `float64`  | Numeric, no nulls, range ∈ \[-10, +10\]        |

- **No extra columns** in the final submission file.
- **Row count:** expected ~73,000–85,000 tracts (all US census tracts with sufficient feature coverage).
- **No duplicate GEOIDs.** Each tract appears exactly once.
- **GEOID format:** Leading zeros are preserved (e.g., `01001020100`, not `1001020100`).

---

## 2. Score Range Expectations

### Empirical distribution (from proxy target, pre-competition)

| Statistic | Value        |
|-----------|-------------|
| mean      | ≈ -0.235    |
| std       | ≈ 0.391     |
| min       | ≈ -1.27     |
| median    | ≈ -0.001    |
| max       | ≈ +0.0004   |

### Expected range after real dataset drop

- **Primary range:** \[-2, +0.5\] — the vast majority of scores.
- **Hard clip:** \[-3.0, +0.5\] applied at inference time.
- **Theoretical bounds:** \[-10, +10\] — the validator's permissive envelope.
- Scores are **predominantly negative or near zero**: a score of 0 means "no coverage gap"; negative values indicate a gap (building/road/POI deficit).

### Key insight

The competition target is a *gap* score, so most tracts will have negative values (they have some gap). Only a small fraction of well-covered tracts will have scores near zero. Positive scores would indicate *over-coverage* and are rare.

---

## 3. Pipeline Version: `DETERMINISTIC_FIX`

| Property            | Value                                                |
|---------------------|------------------------------------------------------|
| Pipeline name       | `merged_proxy_honest_DETERMINISTIC_FIX`              |
| Version tag         | `DETERMINISTIC_FIX`                                  |
| Phase 1 script      | `scripts/pipeline_merged_phase1.py`                  |
| Phase 2 script      | `scripts/pipeline_merged_phase2.py`                  |
| Validator script    | `scripts/validate_submission.py`                     |
| Feature engineering | `data/output/engineered_features_merged.parquet`     |
| Pipeline state      | `data/output/pipeline_state_merged.json`             |

### What the DETERMINISTIC_FIX changed

The original pipeline trained on `proxy_merged = gap_only - 1.0 * rural_penalty`, which **baked the rural penalty into the training target**. This caused the model to double-apply the penalty (once during training, once at inference). The fix:

1. **Train on `gap_only` only** (alpha=0) — the raw coverage gap without rural penalty.
2. **Apply rural penalty at inference time only** — `final_score = model.predict(X) - 1.0 * rural_penalty`.

This ensures the rural penalty is applied exactly once, not twice.

---

## 4. Training Target

| Property         | Value                                                       |
|------------------|-------------------------------------------------------------|
| Target column    | `gap_only`                                                  |
| Alpha            | 0 (no rural penalty blended into training target)           |
| Proxy formula    | `gap_only = -mean(max(0, bg), 2·max(0, bag), max(0, rg), max(0, pfg))` |
| Where `bg`       | `building_gap` = 1 - building_ratio                         |
| Where `bag`      | `building_area_gap` (area-weighted building gap)            |
| Where `rg`       | `road_gap` = 1 - road_ratio                                 |
| Where `pfg`      | `poi_facility_gap` (POI-to-facility ratio gap)             |

The `2×` weighting on `building_area_gap` reflects that area-weighted building coverage is the most informative single gap metric.

---

## 5. Inference Formula

```
final_score = model_ensemble.predict(X) - 1.0 * rural_penalty
```

Where:
- `model_ensemble.predict(X)` = weighted average of the 3-model (or 5-model) ensemble predictions
- `rural_penalty` = computed in Phase 1 feature engineering, based on `pct_urban`, `tribal_any`, and SVI
- The `-1.0` coefficient is fixed (not learned) — it ensures rural/tribal tracts get a *larger* (more negative) gap score

After applying the formula, scores are **clipped** to \[-3.0, +0.5\].

---

## 6. Validation Checks

The validator script (`scripts/validate_submission.py`) performs the following checks:

| #  | Check                                                  | Pass condition                               |
|----|--------------------------------------------------------|----------------------------------------------|
| a  | File exists                                            | Path is readable                             |
| b  | CSV loadable                                           | `pd.read_csv` succeeds                       |
| c  | Exactly 2 columns: GEOID, coverage_gap_score           | Column list matches exactly                  |
| d  | GEOID is 11-digit FIPS                                 | All strings, length=11, all digits           |
| e  | No duplicate GEOIDs                                    | `duplicated().sum() == 0`                    |
| f  | No null values                                         | `isnull().sum() == 0` for all columns        |
| g  | Score numeric and in \[-10, +10\]                      | All values pass range check                  |
| h  | Row count reasonable (70K–120K)                        | Within expected US tract count               |
| i  | Score distribution reasonable                           | min ≥ -3.0, max ≤ +0.5, not mostly zeros    |
| j  | Tribal tracts present (>100)                           | Merge with features finds tribal_any=True    |
| k  | Rural tracts present (>100)                            | Merge with features finds pct_urban < 0.5    |
| l  | Pipeline target = gap_only                             | pipeline_state_merged.json → target          |
| m  | Pipeline includes DETERMINISTIC_FIX                    | pipeline name contains the tag               |
| n  | Inference formula includes rural_penalty               | Formula string verified                      |

**Exit code:** 0 if all checks pass, 1 otherwise.

---

## 7. How to Adapt When the Real Dataset Drops (Aug 28, 2026)

### What CHANGES

| Component              | Current (proxy)                        | Real dataset                               |
|------------------------|----------------------------------------|--------------------------------------------|
| Training target        | `gap_only` (proxy-computed)            | `coverage_gap_score` (ground truth)        |
| Target column name     | `gap_only`                             | `coverage_gap_score`                       |
| Score distribution     | Proxy-based (mean ≈ -0.235)           | Real distribution (unknown until drop)     |
| Feature engineering    | Proxy gaps (building, road, POI)       | May include new Overture/HUD features     |
| Clip range             | \[-3.0, +0.5\]                        | May need adjustment based on real range   |
| Row count              | 85,396 (all tracts with features)     | Competition test set size (TBD)            |
| Validation             | Proxy-specific checks                  | Adapt score range expectations             |

### What STAYS the Same

| Component              | Reason                                                      |
|------------------------|-------------------------------------------------------------|
| `GEOID` format (11-digit FIPS) | Census tract identifier is universal                       |
| Column schema (GEOID, coverage_gap_score) | Competition submission format is fixed                   |
| DETERMINISTIC_FIX logic | Rural penalty applied once at inference is architecturally correct |
| Ensemble framework (XGB + LGB + ET ± CatBoost ± LGB-DART) | Model diversity principle holds regardless of target      |
| H3 spatial CV          | Spatial autocorrelation exists in real data too             |
| Feature selection (top-60, corr threshold) | Adaptive to any target                                    |
| Validator script       | Same checks; update score range constants if needed         |

### Step-by-step adaptation plan

1. **Download the real dataset** on Aug 28, 2026.
2. **Inspect the target variable** — check distribution, range, null rate.
3. **Update `gap_only` column** — replace proxy target with real `coverage_gap_score`.
4. **Re-run Phase 1** — recompute features if new data columns are available.
5. **Re-run Phase 2** — retrain models on real target. The `DETERMINISTIC_FIX` still applies: train on `gap_only` (now real), apply `rural_penalty` at inference.
6. **Update validator** — adjust `SCORE_MIN`/`SCORE_MAX` if real range differs from proxy.
7. **Run validator** — confirm all checks pass before submitting.
8. **Submit** — with confidence that the architecture is sound.

---

## 8. File Naming Conventions

| File                                           | Description                                    |
|------------------------------------------------|------------------------------------------------|
| `submission_merged.csv`                        | 3-model ensemble submission (XGB+LGB+ET)       |
| `submission_merged_v2.csv`                     | 5-model ensemble submission (+CatBoost+DART)   |
| `engineered_features_merged.parquet`           | Full feature matrix with all engineered cols    |
| `pipeline_state_merged.json`                   | Pipeline metadata, weights, metrics             |
| `oof_predictions_merged.parquet`               | Out-of-fold predictions for stacking/blending   |
| `bias_findings_merged.csv`                     | Bias discovery results (SVI, tribal, rural)     |
| `model_comparison_merged.csv`                  | Per-model RMSE/R² comparison                    |

### Path structure

```
/home/z/my-project/
├── bias-bounty-map/
│   └── data/output/
│       ├── engineered_features_merged.parquet
│       ├── pipeline_state_merged.json
│       ├── oof_predictions_merged.parquet
│       ├── submission_merged.csv
│       ├── submission_merged_v2.csv
│       ├── bias_findings_merged.csv
│       └── model_comparison_merged.csv
├── download/
│   ├── submission_merged.csv          (copy for easy access)
│   ├── submission_merged_v2.csv       (copy for easy access)
│   └── submission_architecture.md     (this document)
└── scripts/
    ├── pipeline_merged_phase1.py      (feature engineering)
    ├── pipeline_merged_phase2.py      (training + submission)
    └── validate_submission.py         (this validator)
```

---

## 9. The 5-Model Ensemble

### 3-Model Ensemble (submission_merged.csv)

| Model     | Weight | CV RMSE    | CV R²    |
|-----------|--------|------------|----------|
| XGBoost   | 0.3333 | 0.004519   | 0.9603   |
| LightGBM  | 0.3333 | 0.004995   | 0.9536   |
| ExtraTrees| 0.3333 | 0.004275   | 0.9649   |

**Ensemble CV RMSE:** 0.002204 | **Ensemble CV R²:** 0.9778

### 5-Model Ensemble (submission_merged_v2.csv)

| Model      | Weight | CV RMSE    | CV R²    |
|------------|--------|------------|----------|
| XGBoost    | 0.20   | 0.004519   | 0.9603   |
| LightGBM   | 0.20   | 0.004995   | 0.9536   |
| ExtraTrees | 0.20   | 0.004275   | 0.9649   |
| CatBoost   | 0.20   | 0.005582   | 0.9439   |
| LGB-DART   | 0.20   | 0.006421   | 0.9239   |

**Ensemble CV RMSE:** 0.002383 | **Ensemble CV R²:** 0.9740

### Model role rationale

| Model      | Role                                                                    |
|------------|-------------------------------------------------------------------------|
| XGBoost    | Strong gradient booster, good at feature interactions, fast training    |
| LightGBM   | Complementary boosting (leaf-wise growth), different bias profile       |
| ExtraTrees | High-variance, low-correlation with boosters — ideal diversity member  |
| CatBoost   | Ordered boosting, robust to overfitting, handles categorical features   |
| LGB-DART   | DART boosting mode, different convergence behavior than GBDT           |

### Ensemble method

- **Convex blend** (weights sum to 1, all ≥ 0) optimized via `scipy.optimize.minimize` (SLSQP) to minimize RMSE on out-of-fold predictions.
- The 3-model ensemble currently outperforms 5-model on the proxy target (lower RMSE), but the 5-model may be more robust on the real dataset due to greater diversity.

---

## 10. Cross-Validation Strategy

| Property       | Value                    |
|----------------|--------------------------|
| Method         | H3 spatial block CV      |
| N folds        | 3                        |
| H3 resolution  | 4                        |
| N spatial blocks | 467                    |
| Seed           | 42                       |

Spatial blocks are created by assigning each tract to an H3 hexagonal cell at resolution 4 (~22 km edge length). Blocks are randomly assigned to folds, ensuring no spatial leakage between train and validation.

---

*End of specification.*
