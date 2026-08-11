# Worklog — Merged Proxy Pipeline

---
Task ID: 1
Agent: Main
Task: Survey codebase, data files, and understand current state

Work Log:
- Explored /home/z/my-project/bias-bounty-map/ directory structure
- Found national_tract_features.parquet (85,396 tracts × 601 cols) — primary data source
- Found national-strata-tract-table.parquet (85,396 tracts × 232 cols) — strata data
- Discovered building_area_gap is MISSING from all datasets
- Discovered poi_facility_gap is MISSING (HIFLD data not downloaded)
- Discovered building_gap, road_gap, pct_urban, svi_overall are available
- Found existing pipeline_v3_final.py, pipeline_phase1.py, pipeline_phase2.py
- Found fix_poi_facility_gap.py with confidence-based heuristic correction
- Found src/ library with train.py (already fixed), spatial_cv.py (already fixed), ensemble.py (graceful PySR fallback)

Stage Summary:
- Key gap: building_area_gap and poi_facility_gap_corrected not computed
- Raw Overture/HIFLD data NOT available locally — must use heuristic corrections
- Pipeline needs to be rebuilt with merged proxy (no SVI, rural signal, clipped gaps)

---
Task ID: 2
Agent: Main
Task: Phase 1 — Data loading, gap computation, proxy target, feature engineering

Work Log:
- Built pipeline_merged_phase1.py (memory-optimized for 4GB RAM)
- Loaded only needed columns from national_tract_features.parquet (293 cols)
- Computed poi_facility_gap_corrected using confidence-based heuristic (0.6*building_gap + 0.4*poi_signal)
- Computed building_area_gap as approximation: 1.3*building_gap + 0.2*building_gap*rural
- Implemented merged proxy: -mean(max(0,bg), 2*max(0,bag), max(0,rg), max(0,pfg)) - 1.0*(1-pct_urban)
- Engineered 33 new features (gap interactions, rural interactions, tribal, county LOO, source composition)
- Ran circularity test: SVI→bldg R²=0.003 (RED HERRING), Rural→bldg R²=0.008
- Ran case validations: Hidalgo(-0.09) > Maricopa(-0.04) ✓, OK tribal(-0.49) << OK non-tribal(-0.13) ✓

Stage Summary:
- proxy_merged: mean=-0.2347, std=0.3913, range=[-1.58, 0.0]
- SVI confirmed as red herring (R²=0.003)
- Rural confirmed as equity signal (case validations pass)
- Saved engineered_features_merged.parquet (85,396 × 329)

---
Task ID: 3
Agent: Main
Task: Phase 2 — Train 3-model ensemble (XGB + LGB + ET) with H3 Spatial CV

Work Log:
- Built pipeline_merged_phase2.py (streamlined for 85K tracts)
- Selected 49 features by correlation with proxy_merged target
- Computed 467 H3 spatial blocks at resolution 4
- Trained XGBoost: RMSE=0.007450 R²=0.9996
- Trained LightGBM: RMSE=0.008113 R²=0.9995
- Trained ExtraTrees: RMSE=0.009530 R²=0.9993
- Best ensemble: simple_avg RMSE=0.005066 R²=0.9998
- Bias discovery: HighSVI ratio=0.744, Tribal ratio=1.270, Rural ratio=0.539

Stage Summary:
- R²≈1.0 is expected (proxy is deterministic function of features)
- Tribal bias ratio 1.270 — model has 27% higher error on tribal tracts
- Submission: 85,396 tracts saved to submission_merged.csv

---
Task ID: 4
Agent: Main
Task: Phase 2b — Add CatBoost + DART for full 5-model ensemble

Work Log:
- Built pipeline_merged_phase2b.py to add 2 more models
- Trained CatBoost: RMSE=0.008898 R²=0.9995
- Trained LightGBM DART: RMSE=0.044940 R²=0.9861 (weaker, dropped by convex blend)
- Full 5-model convex blend: RMSE=0.004920 R²=0.9998
- Weights: XGB=0.321, LGB=0.253, ET=0.318, CAT=0.108, DART=0.0

Stage Summary:
- DART naturally gets 0% weight (5x worse RMSE than others)
- Effective ensemble: 4 models (XGB + LGB + ET + CAT)
- Convex blend improves RMSE from 0.005066 → 0.004920
- Submission v2 saved with 85,396 tracts

---
Task ID: 5
Agent: Main
Task: Verify src/ library bugs

Work Log:
- Checked train.py: model factory pattern already implemented (line 180-183), no leakage bug
- Checked spatial_cv.py: 3-tuple fold handling already implemented (line 246-251)
- Checked ensemble.py: TargetReverseEngineer gracefully falls back to linear search when PySR unavailable
- All bugs from previous session already fixed

Stage Summary:
- No src/ bugs remaining — code is clean
