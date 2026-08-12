#!/usr/bin/env python3
"""
National-scale proxy_v1 pipeline: Predict on all 85,396 US census tracts.

Uses the proxy_v1 models trained on 9,496 focus-region tracts to predict
coverage gap scores for all 85,396 national tracts.

Steps:
1. Load national strata table (85,396 tracts)
2. Load national features (601 columns)
3. Compute proxy_v1 target for all tracts
4. Apply same feature selection as training pipeline
5. Predict with saved proxy_v1 models
6. Generate national submission
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import h3

SEED = 42; np.random.seed(SEED)

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
DL = Path("/home/z/my-project/download"); DL.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJ / "models"

print("=" * 72)
print("NATIONAL-SCALE PROXY_V1: Predict on all 85,396 US census tracts")
print("=" * 72)
t0 = time.time()

# ── Step 1: Load national strata ──────────────────────────────────────────────
print("\n[1] Loading national strata table...")
strata = pd.read_parquet(PROJ / "kaggle_dataset/national-strata-tract-table.parquet")
print(f"    Strata: {strata.shape[0]:,} tracts × {strata.shape[1]} cols")

# ── Step 2: Load national features ────────────────────────────────────────────
print("\n[2] Loading national features...")
national = pd.read_parquet(PROJ / "data/features/national_tract_features.parquet")
print(f"    National: {national.shape[0]:,} tracts × {national.shape[1]} cols")

# ── Step 3: Ensure proxy_v1 columns exist ─────────────────────────────────────
print("\n[3] Computing proxy_v1 for national tracts...")

# Need building_gap, road_gap, poi_facility_gap_corrected, svi_overall
# Check which gap columns exist in national features
gap_cols_available = [c for c in ['building_gap', 'road_gap', 'road_length_gap',
                                   'poi_facility_gap', 'poi_facility_gap_corrected']
                     if c in national.columns]
print(f"    Gap columns available: {gap_cols_available}")

# Merge from strata if needed
if 'svi_overall' not in national.columns or national['svi_overall'].isna().all():
    if 'svi_overall' in strata.columns:
        svi_map = strata.set_index('GEOID')['svi_overall']
        national['svi_overall'] = national['GEOID'].astype(str).map(svi_map)
        print(f"    Merged SVI from strata: non-null={national['svi_overall'].notna().sum()}")

# Compute proxy_v1
building_gap = national['building_gap'].fillna(0) if 'building_gap' in national.columns else pd.Series(0, index=national.index)
road_gap = national['road_gap'].fillna(0) if 'road_gap' in national.columns else pd.Series(0, index=national.index)

if 'poi_facility_gap_corrected' in national.columns:
    poi_gap = national['poi_facility_gap_corrected'].fillna(0)
elif 'poi_facility_gap' in national.columns:
    poi_gap = national['poi_facility_gap'].fillna(0)
else:
    poi_gap = road_gap.copy()

svi_overall = national['svi_overall'].fillna(0.5) if 'svi_overall' in national.columns else pd.Series(0.5, index=national.index)

gap_mean = np.mean([building_gap.values, road_gap.values, poi_gap.values], axis=0)
proxy_v1 = -gap_mean - 2.0 * svi_overall.values
national['proxy_v1'] = proxy_v1

print(f"    proxy_v1: mean={proxy_v1.mean():.4f}, std={proxy_v1.std():.4f}, "
      f"range=[{proxy_v1.min():.4f}, {proxy_v1.max():.4f}]")

# ── Step 4: Prepare features (same as training) ──────────────────────────────
print("\n[4] Preparing feature matrix...")

# Load training features to get the same column set
train_feat = pd.read_parquet(OUT / "engineered_features_v3.parquet")
print(f"    Training features: {train_feat.shape}")

# Define the same drop list as training
drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
        'building_gap','road_gap','building_ratio','road_ratio','building_count_ratio',
        'building_count_gap','road_count_ratio','road_count_gap','road_length_ratio',
        'road_length_gap','poi_facility_gap','poi_to_facility_ratio',
        'poi_facility_gap_corrected','poi_to_facility_ratio_corrected',
        'coverage_gap_score','coverage_gap','gap_score','coverage_score',
        'proxy_simple_avg','proxy_svi_weighted','proxy_max_gap','proxy_pop_weighted',
        'proxy_v1']

# Get the feature columns used in training
train_feat_clean = train_feat.loc[:, ~train_feat.columns.duplicated()]
train_fc = [c for c in train_feat_clean.columns if c not in drop and pd.api.types.is_numeric_dtype(train_feat_clean[c])]

# Use same feature selection as training
y_train = train_feat_clean['proxy_v1'] if 'proxy_v1' in train_feat_clean.columns else train_feat_clean['building_gap']
X_train_tmp = train_feat_clean[train_fc].copy()
v_train = y_train.notna()
X_train_tmp = X_train_tmp[v_train]
y_train = y_train[v_train]
X_train_tmp = X_train_tmp.fillna(-999).replace([np.inf, -np.inf], -999)
s_train = X_train_tmp.std()
X_train_tmp = X_train_tmp[s_train[s_train > 1e-10].index]
cs_train = X_train_tmp.corrwith(y_train).abs().fillna(0)
top_n = min(80, len(cs_train))
selected_features = cs_train.sort_values(ascending=False).head(top_n).index.tolist()

# Remove highly correlated
cm = X_train_tmp[selected_features].corr().abs()
up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
td = [c for c in up.columns if any(up[c] > 0.98)]
selected_features = [c for c in selected_features if c not in td]

print(f"    Selected {len(selected_features)} features (same as training)")

# Apply to national
national_clean = national.loc[:, ~national.columns.duplicated()]

# Rename national columns to match training feature names
# Training uses: svi_x_road_gap (national has: svi_x_road)
# Training uses: bldg_gap_sq (national has: building_gap_sq)
rename_map = {}
if 'svi_x_road' in national_clean.columns and 'svi_x_road_gap' not in national_clean.columns:
    rename_map['svi_x_road'] = 'svi_x_road_gap'
if 'building_gap_sq' in national_clean.columns and 'bldg_gap_sq' not in national_clean.columns:
    rename_map['building_gap_sq'] = 'bldg_gap_sq'
if rename_map:
    national_clean = national_clean.rename(columns=rename_map)
    print(f"    Renamed columns: {rename_map}")

# Check which features exist
available_features = [c for c in selected_features if c in national_clean.columns]
missing_features = [c for c in selected_features if c not in national_clean.columns]
print(f"    Available in national: {len(available_features)}/{len(selected_features)}")
if missing_features:
    print(f"    Missing (will fill with -999): {missing_features[:10]}...")
    for c in missing_features:
        national_clean[c] = -999

# Reorder to match training feature order exactly
X_national = national_clean[selected_features].copy()
X_national = X_national.fillna(-999).replace([np.inf, -np.inf], -999)

# Final fix: rename columns to match model's expected feature names
# The model was trained with svi_x_road_gap and bldg_gap_sq
# but the data might have svi_x_road and building_gap_sq
final_rename = {}
if 'svi_x_road' in X_national.columns and 'svi_x_road_gap' not in X_national.columns:
    final_rename['svi_x_road'] = 'svi_x_road_gap'
if 'building_gap_sq' in X_national.columns and 'bldg_gap_sq' not in X_national.columns:
    final_rename['building_gap_sq'] = 'bldg_gap_sq'
if final_rename:
    X_national = X_national.rename(columns=final_rename)
    print(f"    Final rename for model compatibility: {final_rename}")

print(f"    National feature matrix: {X_national.shape}")

# Verify feature alignment
if X_national.shape[1] != len(selected_features):
    print(f"    WARNING: Feature count mismatch! National has {X_national.shape[1]}, expected {len(selected_features)}")
else:
    print(f"    Feature alignment verified: {X_national.shape[1]} features")

# ── Step 5: Load and predict with saved models ────────────────────────────────
print("\n[5] Predicting with saved proxy_v1 models...")

model_files = {
    'xgb': MODELS_DIR / "proxy_v1_xgb.pkl",
    'lgb': MODELS_DIR / "proxy_v1_lgb.pkl",
    'cat': MODELS_DIR / "proxy_v1_cat.pkl",
    'et': MODELS_DIR / "proxy_v1_et.pkl",
    'lgb_dart': MODELS_DIR / "proxy_v1_lgb_dart.pkl",
}

preds = {}
for name, path in model_files.items():
    if path.exists():
        with open(path, 'rb') as f:
            model = pickle.load(f)
        preds[name] = model.predict(X_national)
        print(f"    {name}: mean={preds[name].mean():.4f}, std={preds[name].std():.4f}")
    else:
        print(f"    WARNING: {name} model not found at {path}")

# Load stacking meta-learner
meta_path = MODELS_DIR / "proxy_v1_stacking_meta.pkl"
if meta_path.exists():
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Stacking prediction
    ns = list(preds.keys())
    pred_mat = np.column_stack([preds[n] for n in ns])
    stack_pred = meta.predict(pred_mat)
    print(f"    Stacking: mean={stack_pred.mean():.4f}, std={stack_pred.std():.4f}")

    # Also compute convex blend
    # Load weights from pipeline state
    state_path = OUT / "pipeline_state_proxy_v1.json"
    if state_path.exists():
        state = json.load(open(state_path))
        cw = state['ensemble']['convex']['weights']
        # Reorder weights to match model order
        weights = np.array([cw.get(n, 0) for n in ns])
        weights = weights / weights.sum()  # normalize
        convex_pred = pred_mat @ weights
        print(f"    Convex: mean={convex_pred.mean():.4f}, std={convex_pred.std():.4f}")
    else:
        convex_pred = None
else:
    print("    WARNING: Stacking meta-learner not found, using simple average")
    ns = list(preds.keys())
    pred_mat = np.column_stack([preds[n] for n in ns])
    stack_pred = pred_mat.mean(axis=1)
    convex_pred = None

# Use stacking as final prediction (it was best in training)
final_pred = stack_pred

# ── Step 6: Generate national submission ───────────────────────────────────────
print("\n[6] Generating national submission...")

submission = pd.DataFrame({
    'GEOID': national['GEOID'].astype(str).values,
    'coverage_gap_score': final_pred,
})

# Save
submission.to_csv(OUT / "submission_national_proxy_v1.csv", index=False)
submission.to_csv(DL / "submission_national_proxy_v1.csv", index=False)
print(f"    National submission: {len(submission):,} tracts")
print(f"    Pred stats: mean={final_pred.mean():.4f}, std={final_pred.std():.4f}, "
      f"min={final_pred.min():.4f}, max={final_pred.max():.4f}")

# Also compute H3 spatial blocks for national tracts
print("\n[7] Computing H3 spatial blocks for national tracts...")
if 'INTPTLAT' in strata.columns and 'INTPTLON' in strata.columns:
    lats = pd.to_numeric(strata['INTPTLAT'], errors='coerce')
    lons = pd.to_numeric(strata['INTPTLON'], errors='coerce')
    h3_blocks = pd.Series(
        [h3.latlng_to_cell(float(la), float(lo), 4) if not (np.isnan(la) or np.isnan(lo)) else 'unk'
         for la, lo in zip(lats.values, lons.values)],
        index=strata.index
    )
    print(f"    H3 blocks: {h3_blocks.nunique():,} unique blocks at resolution 4")
else:
    print("    WARNING: No lat/lon in strata for H3 blocks")

# ── Step 8: Save pipeline state ───────────────────────────────────────────────
print("\n[8] Saving national pipeline state...")

state = {
    'pipeline': 'national_proxy_v1',
    'target': 'proxy_v1',
    'n_tracts_national': int(len(submission)),
    'n_features': int(X_national.shape[1]),
    'models_used': list(preds.keys()),
    'prediction_stats': {
        'mean': round(float(final_pred.mean()), 4),
        'std': round(float(final_pred.std()), 4),
        'min': round(float(final_pred.min()), 4),
        'max': round(float(final_pred.max()), 4),
    },
    'feature_selection': {
        'n_selected': len(selected_features),
        'n_available_in_national': len(available_features),
        'n_missing_filled': len(missing_features),
    },
}

with open(OUT / "pipeline_state_national.json", 'w') as f:
    json.dump(state, f, indent=2)
print(f"    Saved to {OUT / 'pipeline_state_national.json'}")

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print(f"DONE in {elapsed:.0f}s")
print(f"National submission: {len(submission):,} tracts")
print(f"{'=' * 72}")
