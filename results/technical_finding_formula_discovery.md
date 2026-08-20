# Technical Finding: Deterministic Structure of the Coverage Gap Proxy

**Finding ID:** TF-001  
**Status:** VERIFIED  
**Date:** 2026-08-15  
**Significance:** Falsifies hypothesis that `proxy_merged` requires ML; redirects project toward gap prediction

---

## Abstract

We hypothesized that the coverage gap proxy (`proxy_merged`) required machine learning to predict from tract-level features. Through systematic audit—including OLS regression, formula verification, and residual analysis—we discovered that `proxy_merged` is a **deterministic linear function** of four gap features and one urbanization feature, with R² = 1.0 and zero residuals across all 85,396 census tracts. This falsifies the original hypothesis and establishes that the proxy is a constructed composite index, not an independent measurement requiring prediction. The project's ML effort is redirected toward predicting the **underlying gap features** from independent variables—a substantially different and more scientifically meaningful task.

---

## 1. The Formula

```
proxy_merged = -(bldg_gap_clip + 2·area_gap_clip + road_gap_clip + poi_gap_clip) / 4 + pct_urban - 1
```

**Verification:** OLS on 85,396 tracts yields R² = 1.0, RMSE = 0.0, max |error| = 0.0.

| Feature | OLS Coefficient | Expected from Formula | Difference |
|---------|----------------|---------------------|------------|
| bldg_gap_clip | -0.2500000000 | -0.25 | 0 |
| area_gap_clip | -0.5000000000 | -0.50 | 0 |
| road_gap_clip | -0.2500000000 | -0.25 | 0 |
| poi_gap_clip | -0.2500000000 | -0.25 | 0 |
| pct_urban | +1.0000000000 | +1.00 | 0 |
| intercept | -1.0000000000 | -1.00 | 0 |

## 2. Dominance Structure

The formula is dominated by the `pct_urban` term for the vast majority of tracts:

- **85.0% of tracts** (72,566 / 85,396) have all gap clips < 0.001
- For these tracts, `proxy_merged ≈ pct_urban - 1` (R² = 0.99999999)
- `pct_urban` alone explains **99.87%** of total variance
- The four gap features collectively add only **0.13%** over `pct_urban` alone
- Without `pct_urban`, gap features explain only **16.5%** of variance

**Interpretation:** The coverage gap proxy is essentially a measure of urbanization, with small corrections for infrastructure under-mapping. The area gap receives double weight (coefficient -0.50 vs -0.25), suggesting the proxy designers considered building area coverage more important than count, road, or POI coverage.

## 3. Sensitivity Analysis

Perturbation analysis (10,000 Monte Carlo samples, σ = 0.1) reveals:

| Feature | Sensitivity | Rank |
|---------|------------|------|
| road_gap_clip | 0.00154 | 1 (most sensitive) |
| area_gap_clip | 0.00077 | 2 |
| bldg_gap_clip | 0.00056 | 3 |
| poi_gap_clip | 0.00026 | 4 (least sensitive) |

Despite having the same OLS coefficient as `bldg_gap_clip` and `poi_gap_clip`, `road_gap_clip` is most sensitive because it has the largest variance in the data (road coverage varies more across tracts than building or POI coverage).

## 4. Implications for Prior Work

### What Was Real

| Result | Status | Why |
|--------|--------|-----|
| Formula discovery & OLS verification | ✅ Valid | Proved the target is deterministic |
| Tribal gap disparity (7.14× road_gap) | ✅ Valid | Data coverage inequity in OSM |
| pct_urban dominance (99.87% R²) | ✅ Valid | Real structural finding |
| 85% trivial-gap observation | ✅ Valid | Most tracts are well-mapped |

### What Was Circular

| Result | Status | Why |
|--------|--------|-----|
| R² = 0.9998 on proxy_merged | ❌ Circular | Model inverted a known formula |
| Two-stage classifier→regressor | ❌ Circular | Both stages learned the same linear boundary |
| SHAP importances | ❌ Misleading | Measured formula contributions, not learned patterns |
| Ensemble disagreement | ❌ Circular | Measured formula approximation error |
| Spatial CV generalization | ❌ Trivial | Formula is geography-independent |
| Prediction intervals | ❌ Misleading | Intervals around a deterministic function |

## 5. The Correct Reframing: Predict the Gaps

The formula discovery does not eliminate the ML opportunity—it **redirects** it. For tracts where gap measurements exist, the formula gives the exact answer. But for **unmapped tracts** (where we lack OSM building/road data), we must **predict the gaps** from independent features:

```
Stage 1 (ML):  Predict gaps from non-gap features
               road_gap  = f(wildfire, drought, SVI, climate, ...)
               poi_gap   = g(wildfire, drought, SVI, climate, ...)
               bldg_gap  = h(wildfire, drought, SVI, climate, ...)
               area_gap  = i(wildfire, drought, SVI, climate, ...)

Stage 2 (Algebra): Apply the deterministic formula
               proxy_merged = -(bldg_gap + 2·area_gap + road_gap + poi_gap)/4 + pct_urban - 1
```

### Baseline Predictability (OLS on top-5 non-gap features)

| Gap Target | OLS R² | Top Features | Interpretation |
|-----------|--------|-------------|----------------|
| road_gap_clip | **0.212** | GHCN stations, wildfire fires, MTBS fires | Roads are unmapped where fire risk & climate stations are sparse |
| poi_gap_clip | **0.080** | USDM drought, compound risk, SPI | POI gaps correlate with drought severity |
| bldg_gap_clip | **0.043** | CVI climate-health, SPI, PMDI | Building gaps weakly predicted by climate vulnerability |
| area_gap_clip | **0.047** | CVI climate-health, compound risk | Area gaps similarly weak |

**The road_gap prediction problem (R² = 0.21) is the most promising real ML task.** With gradient-boosted trees and proper feature engineering, R² could reach 0.3–0.4—enough to meaningfully estimate coverage gaps for unmapped tracts.

## 6. Tribal Data Equity Finding

The formula is **group-fair** (identical coefficients for all tracts), but the **data distribution** creates outcome disparities:

| Metric | Tribal Tracts | Non-Tribal Tracts | Ratio |
|--------|-------------|-------------------|-------|
| road_gap_clip mean | 0.0383 | 0.0054 | **7.14×** |
| bldg_gap_clip mean | 0.0029 | 0.0022 | 1.30× |
| area_gap_clip mean | 0.0042 | 0.0031 | 1.37× |
| pct_urban mean | 0.439 | 0.777 | 0.57× |
| proxy_merged mean | -0.573 | -0.227 | — |

**Diagnosis:** Tribal tracts are systematically under-mapped in OpenStreetMap (7× higher road gap), and are more rural (56% rural vs 22% for non-tribal). The formula correctly captures this—the proxy IS worse for tribal tracts because the underlying data IS worse. This is a **measurement coverage inequity**, not a model bias. Any model (including the formula) that faithfully reflects the data will show this disparity.

**Policy implication:** If this proxy informs resource allocation, the 7× road mapping gap in tribal areas should be documented as a **data quality limitation** that may systematically disadvantage tribal communities regardless of the modeling approach used.

## 7. Actions

1. **For known tracts:** Apply the formula directly. No ML needed.
2. **For unmapped tracts:** Predict gaps from non-gap features, then apply formula.
3. **When `coverage_gap_score` releases:** Test if the same formula applies. If yes → no ML. If no → rebuild with non-gap features only.
4. **Document tribal equity:** Include group-specific data quality metrics in any policy-facing deliverable.

---

## Appendix: Formula Derivation

The formula was originally defined in `scripts/pipeline_merged_phase1.py` as:

```python
proxy_merged = -np.mean([
    np.maximum(0, building_gap),              # bldg_gap_clip
    2.0 * np.maximum(0, building_area_gap),   # 2 * area_gap_clip (2x weight!)
    np.maximum(0, road_gap),                  # road_gap_clip
    np.maximum(0, poi_facility_gap_corrected) # poi_gap_clip
], axis=0) - 1.0 * (1 - pct_urban).clip(0, 1)
```

Expanding the mean and the rural penalty:

```
proxy_merged = -(bldg_gap_clip + 2·area_gap_clip + road_gap_clip + poi_gap_clip) / 4 - (1 - pct_urban)
            = -(bldg_gap_clip + 2·area_gap_clip + road_gap_clip + poi_gap_clip) / 4 + pct_urban - 1
```

The intermediate variables are themselves derived:
- `building_area_gap = 1.3·building_gap + 0.2·building_gap·rural` (heuristic approximation)
- `poi_facility_gap_corrected = 0.6·building_gap + 0.4·poi_signal` (confidence-weighted heuristic)

So the formula is ultimately a function of three raw measurements: `building_gap`, `road_gap`, and `pct_urban`, plus the POI signal.
