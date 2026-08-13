# Pipeline Results

## Key Files

| File | Description |
|------|-------------|
| `pipeline_lock_report.json` | Full validation + bug fix verification |
| `lean_pipeline_state.json` | Phase 2 (3-model ensemble) state |
| `expanded_pipeline_state.json` | Integrated 10x (5-model) state |
| `strategy_analysis.json` | Lean vs expanded comparison + best path |
| `diagnostic_report.json` | Full audit (zero critical bugs) |
| `strata_deep_audit.json` | 68 unused strata features analysis |
| `conformal_crossval_results.json` | Conformal prediction intervals |

## Feature Importance

| File | Description |
|------|-------------|
| `feature_importance_integrated_10x.csv` | Top features for expanded pipeline |
| `feature_importance_augmented.csv` | Top features with strata interactions |
| `feature_correlation_full.csv` | All 285 features ranked by |corr| |
| `shap_importance.csv` | SHAP-based feature importance |
| `shap_bias_decomposition.csv` | SHAP bias decomposition by stratum |

## Bias Findings

| File | Description |
|------|-------------|
| `bias_findings_merged.csv` | Disparity ratios (HighSVI, Tribal, Rural) |
| `comprehensive_bias_findings.csv` | Full bias analysis |
| `intersectional_bias_summary.csv` | Intersectional bias (tribal×rural, etc.) |

## Key Numbers

- **Tribal bias ratio**: 2.54× (expanded), 1.51× (lean)
- **Rural/Urban ratio**: 28.12× (expanded)
- **LORO R² ceiling**: +0.155 (data ceiling, 85% variance missing)
- **H3-CV R²**: 0.976 (lean), 0.967 (expanded)
- **Pearson corr between submissions**: 0.999990
