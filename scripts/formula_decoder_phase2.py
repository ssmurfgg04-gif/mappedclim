#!/usr/bin/env python3
"""
Formula Decoder Phase 2: 1M Stability + Analysis Experiments
=============================================================
Runs 5 experiments building on Phase 1's exact formula discovery.

Exact formula from Phase 1:
  proxy_merged = -(bldg_gap_clip + 2*area_gap_clip + road_gap_clip + poi_gap_clip)/4 - (1 - pct_urban)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data/output/engineered_features_merged.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "formula_decoder_phase2.json"


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: 1M Random Split Stability Test (Joblib parallel)
# ═══════════════════════════════════════════════════════════════════════════

def _ridge_one_split(seed, feat, tgt, n_rows, n_feat, train_frac, n_train, eye):
    """Single Ridge split — called via joblib."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=n_train, replace=False)
    mask = np.zeros(n_rows, dtype=bool)
    mask[idx] = True
    Xtr, ytr = feat[mask], tgt[mask]
    Xte, yte = feat[~mask], tgt[~mask]
    beta = np.linalg.solve(Xtr.T @ Xtr + eye, Xtr.T @ ytr)
    yp = Xte @ beta
    ss_res = np.sum((yte - yp)**2)
    ss_tot = np.sum((yte - yte.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def experiment_a_stability(df):
    """1M random 80/20 Ridge regression splits."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: 1M RANDOM SPLIT STABILITY TEST")
    print("=" * 70)

    from joblib import Parallel, delayed
    from multiprocessing import cpu_count

    core_features = [
        'building_gap', 'road_gap', 'pct_urban', 'tribal_any',
        'rural_indicator', 'svi_overall', 'cvi_overall',
        'bldg_osm_fraction', 'bldg_source_diversity'
    ]
    optional = ['building_area_gap', 'poi_facility_gap_corrected', 'pop_total', 'ALAND']
    for f in optional:
        if f in df.columns and f not in core_features:
            core_features.append(f)

    print(f"Core features ({len(core_features)}): {core_features}")

    feature_data = df[core_features].values.astype(np.float64)
    target_data = df['proxy_merged'].values.astype(np.float64)
    n = len(df)
    n_features = len(core_features)

    # Subsample for speed
    SUBSAMPLE = 10_000
    if n > SUBSAMPLE:
        print(f"Subsampling from {n:,} to {SUBSAMPLE:,} rows for speed")
        rng_sub = np.random.default_rng(123)
        sub_idx = rng_sub.choice(n, size=SUBSAMPLE, replace=False)
        feature_data = feature_data[sub_idx]
        target_data = target_data[sub_idx]
        n = SUBSAMPLE

    fmask = np.isfinite(feature_data).all(axis=1) & np.isfinite(target_data)
    if fmask.sum() < n:
        print(f"WARNING: {(~fmask).sum()} rows with NaN/Inf")
        feature_data = feature_data[fmask]
        target_data = target_data[fmask]
        n = len(feature_data)

    N_SPLITS = 1_000_000
    TRAIN_FRAC = 0.8
    N_WORKERS = min(8, cpu_count())
    n_train = int(n * TRAIN_FRAC)
    eye = np.eye(n_features)

    print(f"Running {N_SPLITS:,} random splits with {N_WORKERS} workers (subsample n={n:,})")

    rng = np.random.default_rng(42)
    all_seeds = rng.integers(0, 2**31, size=N_SPLITS)

    # Process in chunks using joblib
    CHUNK = 50_000
    all_coefs = np.zeros((N_SPLITS, n_features))
    all_r2s = np.zeros(N_SPLITS)
    t0 = time.time()

    for chunk_start in range(0, N_SPLITS, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N_SPLITS)
        chunk_seeds = all_seeds[chunk_start:chunk_end]

        results_chunk = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=0)(
            delayed(_ridge_one_split)(s, feature_data, target_data, n, n_features, TRAIN_FRAC, n_train, eye)
            for s in chunk_seeds
        )

        for i, (beta, r2) in enumerate(results_chunk):
            all_coefs[chunk_start + i] = beta
            all_r2s[chunk_start + i] = r2

        done = chunk_end
        if done % 100_000 == 0 or done == N_SPLITS:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (N_SPLITS - done) / rate
            print(f"  {done:>10,} / {N_SPLITS:,} | {rate:,.0f} splits/s | ETA: {eta/60:.1f}min | R²={all_r2s[:done].mean():.6f}")

    done = N_SPLITS

    # ═══ Compute stability statistics ═══
    print("\n--- Stability Statistics ---")
    abs_coefs = np.abs(all_coefs)

    results_a = {
        "n_splits": N_SPLITS,
        "features": core_features,
        "n_features": n_features,
        "subsample_size": n,
        "r2_stats": {
            "mean": float(all_r2s.mean()),
            "std": float(all_r2s.std()),
            "min": float(all_r2s.min()),
            "max": float(all_r2s.max()),
            "median": float(np.median(all_r2s)),
            "p5": float(np.percentile(all_r2s, 5)),
            "p95": float(np.percentile(all_r2s, 95)),
        },
        "per_feature": {}
    }

    for j, feat_name in enumerate(core_features):
        coef_j = all_coefs[:, j]
        abs_coef_j = abs_coefs[:, j]

        n_positive = int((coef_j > 0).sum())
        n_negative = int((coef_j < 0).sum())
        sign_stability = max(n_positive, n_negative) / N_SPLITS
        dominant_sign = "positive" if n_positive > n_negative else "negative"

        for_split_top1 = (abs_coefs == abs_coefs.max(axis=1, keepdims=True))
        top1_count = int(for_split_top1[:, j].sum())

        sorted_idx = abs_coefs.argsort(axis=1)[:, -3:]
        top3_count = int(np.any(sorted_idx == j, axis=1).sum())

        feat_stats = {
            "mean_abs_coef": float(abs_coef_j.mean()),
            "std_abs_coef": float(abs_coef_j.std()),
            "median_abs_coef": float(np.median(abs_coef_j)),
            "mean_coef": float(coef_j.mean()),
            "std_coef": float(coef_j.std()),
            "pct_top1": float(top1_count / N_SPLITS * 100),
            "pct_top3": float(top3_count / N_SPLITS * 100),
            "sign_stability_pct": float(sign_stability * 100),
            "dominant_sign": dominant_sign,
        }
        results_a["per_feature"][feat_name] = feat_stats

        print(f"  {feat_name:30s} | |coef|={abs_coef_j.mean():.6f}±{abs_coef_j.std():.6f} | "
              f"top1={top1_count/N_SPLITS*100:.1f}% top3={top3_count/N_SPLITS*100:.1f}% | "
              f"sign={dominant_sign} ({sign_stability*100:.1f}%)")

    print(f"\n  R²: mean={all_r2s.mean():.6f}, std={all_r2s.std():.6f}, "
          f"range=[{all_r2s.min():.6f}, {all_r2s.max():.6f}]")

    results_a["elapsed_seconds"] = float(time.time() - t0)
    return results_a


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 70)
    print("FORMULA DECODER PHASE 2: 1M STABILITY + ANALYSIS")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} cols")

    bg_clip = df['building_gap'].clip(lower=0)
    bag_clip = df['building_area_gap'].clip(lower=0)
    rg_clip = df['road_gap'].clip(lower=0)
    pg_clip = df['poi_facility_gap_corrected'].clip(lower=0)
    exact = -(bg_clip + 2*bag_clip + rg_clip + pg_clip)/4 - (1 - df['pct_urban'])
    max_err = (df['proxy_merged'] - exact).abs().max()
    print(f"Exact formula verification: max error = {max_err:.1e}")

    df['gap_only'] = -(bg_clip + 2*bag_clip + rg_clip + pg_clip)/4
    df['rural_penalty'] = -(1 - df['pct_urban'])
    df['is_well_mapped'] = (df['proxy_merged'] > -0.01).astype(int)
    df['is_gapped'] = (df['proxy_merged'] < -0.1).astype(int)

    print(f"Well-mapped (proxy > -0.01): {df['is_well_mapped'].sum():,}")
    print(f"Gapped (proxy < -0.1): {df['is_gapped'].sum():,}")
    print(f"gap_only range: [{df['gap_only'].min():.4f}, {df['gap_only'].max():.4f}], mean={df['gap_only'].mean():.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Perfect Mapping Decoder
# ═══════════════════════════════════════════════════════════════════════════

def experiment_b_perfect_mapping(df):
    print("\n" + "=" * 70)
    print("EXPERIMENT B: PERFECT MAPPING DECODER")
    print("=" * 70)

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import precision_score, recall_score, f1_score

    key_features = [
        'pct_urban', 'rural_indicator', 'building_gap', 'road_gap',
        'building_area_gap', 'poi_facility_gap_corrected',
        'svi_overall', 'cvi_overall', 'tribal_any',
        'bldg_osm_fraction', 'bldg_source_diversity'
    ]
    key_features = [f for f in key_features if f in df.columns]

    X = df[key_features].values.astype(np.float64)
    y = df['is_well_mapped'].values

    print(f"Well-mapped: {y.sum():,} / {len(y):,} ({y.mean()*100:.1f}%)")

    # 1. Decision Tree
    print("\n--- Decision Tree (max_depth=3) ---")
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=42)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Tree depth: {dt.get_depth()}, Leaves: {dt.get_n_leaves()}")

    tree_text = export_text(dt, feature_names=key_features, decimals=3)
    print(f"\nDecision Tree Rules:\n{tree_text}")

    imp_df = pd.DataFrame({
        'feature': key_features, 'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    for _, row in imp_df.head(5).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # 2. Single threshold rules
    print("\n--- Single Threshold Rules ---")
    best_rule = None
    best_f1 = 0

    for feat in key_features:
        vals = df[feat].values.astype(np.float64)
        unique_vals = np.percentile(vals, np.arange(1, 100))
        for thresh in unique_vals:
            for direction in ['gt', 'lt']:
                pred = (vals > thresh).astype(int) if direction == 'gt' else (vals < thresh).astype(int)
                p = precision_score(y, pred, zero_division=0)
                r = recall_score(y, pred, zero_division=0)
                f = f1_score(y, pred, zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_rule = {'feature': feat, 'threshold': float(thresh), 'direction': direction,
                                'precision': float(p), 'recall': float(r), 'f1': float(f)}

    d_str = ">" if best_rule['direction'] == 'gt' else "<"
    print(f"\nBest single rule: {best_rule['feature']} {d_str} {best_rule['threshold']:.4f}")
    print(f"  Precision: {best_rule['precision']:.4f}, Recall: {best_rule['recall']:.4f}, F1: {best_rule['f1']:.4f}")

    # 3. Two-threshold AND rules
    print("\n--- Two-Threshold AND Rules ---")
    best_two_rule = None
    best_two_f1 = 0
    top_feats = imp_df.head(4)['feature'].tolist()

    for i, f1_name in enumerate(top_feats):
        for f2_name in top_feats[i+1:]:
            v1 = df[f1_name].values.astype(np.float64)
            v2 = df[f2_name].values.astype(np.float64)
            t1_vals = np.percentile(v1, [10, 25, 50, 75, 90])
            t2_vals = np.percentile(v2, [10, 25, 50, 75, 90])
            for t1 in t1_vals:
                for t2 in t2_vals:
                    for d1 in ['gt', 'lt']:
                        for d2 in ['gt', 'lt']:
                            c1 = (v1 > t1) if d1 == 'gt' else (v1 < t1)
                            c2 = (v2 > t2) if d2 == 'gt' else (v2 < t2)
                            pred = (c1 & c2).astype(int)
                            p = precision_score(y, pred, zero_division=0)
                            r = recall_score(y, pred, zero_division=0)
                            f = f1_score(y, pred, zero_division=0)
                            if f > best_two_f1:
                                best_two_f1 = f
                                best_two_rule = {
                                    'feature1': f1_name, 'threshold1': float(t1), 'direction1': d1,
                                    'feature2': f2_name, 'threshold2': float(t2), 'direction2': d2,
                                    'precision': float(p), 'recall': float(r), 'f1': float(f)
                                }

    d1_str = ">" if best_two_rule['direction1'] == 'gt' else "<"
    d2_str = ">" if best_two_rule['direction2'] == 'gt' else "<"
    print(f"\nBest AND rule: {best_two_rule['feature1']} {d1_str} {best_two_rule['threshold1']:.4f} AND "
          f"{best_two_rule['feature2']} {d2_str} {best_two_rule['threshold2']:.4f}")
    print(f"  Precision: {best_two_rule['precision']:.4f}, Recall: {best_two_rule['recall']:.4f}, F1: {best_two_rule['f1']:.4f}")

    return {
        "decision_tree": {
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "depth": dt.get_depth(), "n_leaves": dt.get_n_leaves(),
            "tree_rules": tree_text,
            "feature_importances": {row['feature']: float(row['importance']) for _, row in imp_df.iterrows()}
        },
        "best_single_threshold": best_rule,
        "best_two_threshold_and": best_two_rule,
        "well_mapped_count": int(y.sum()),
        "well_mapped_pct": float(y.mean() * 100)
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Residual Structure Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_morans_i(values, coords, k=8):
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k+1)
    n = len(values)
    z = values - values.mean()
    lag = np.zeros(n)
    w_sum = 0.0
    for i in range(n):
        neighbors = indices[i, 1:]
        w = 1.0 / (distances[i, 1:] + 1e-10)
        w = w / w.sum()
        lag[i] = np.sum(w * z[neighbors])
        w_sum += w.sum()
    num = np.sum(z * lag)
    den = np.sum(z**2)
    if den == 0: return 0.0, 1.0
    I = (n / w_sum) * (num / den)
    E_I = -1.0 / (n - 1)
    z_score = (I - E_I) / max(0.01, np.sqrt(abs(1.0 / n)))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return float(I), float(p_value)


def experiment_c_residuals(df):
    print("\n" + "=" * 70)
    print("EXPERIMENT C: RESIDUAL STRUCTURE ANALYSIS")
    print("=" * 70)

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    exclude = ['proxy_merged', 'gap_only', 'rural_penalty', 'is_well_mapped', 'is_gapped',
               'GEOID', 'centroid_lat', 'centroid_lon', 'INTPTLAT', 'INTPTLON']
    feature_cols = [c for c in numeric_cols if c not in exclude and df[c].notna().sum() > 100]
    feature_cols = [c for c in feature_cols if 'proxy' not in c.lower() and 'gap_only' not in c.lower()]

    print(f"Using {len(feature_cols)} numeric features for XGBoost")

    X = df[feature_cols].values.astype(np.float32)
    y = df['proxy_merged'].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print("Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {'max_depth': 6, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'verbosity': 0}
    model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')], verbose_eval=50)

    dall = xgb.DMatrix(X)
    y_pred_all = model.predict(dall)
    residuals = y - y_pred_all

    r2_full = r2_score(y, y_pred_all)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_all))
    print(f"\nXGBoost on all data: R²={r2_full:.6f}, RMSE={rmse_full:.6f}")
    print(f"Residuals: mean={residuals.mean():.6f}, std={residuals.std():.6f}, max_abs={np.abs(residuals).max():.6f}")

    # Moran's I
    print("\n--- Spatial Autocorrelation (Moran's I) ---")
    coords = df[['INTPTLAT', 'INTPTLON']].values
    valid_coord = (coords[:, 0] != 0) & (coords[:, 1] != 0)
    morans_i, morans_p = 0.0, 1.0
    if valid_coord.sum() > 1000:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(np.where(valid_coord)[0], size=min(10000, valid_coord.sum()), replace=False)
        morans_i, morans_p = compute_morans_i(residuals[sub_idx], coords[sub_idx], k=8)
        print(f"  Moran's I = {morans_i:.4f} (p ≈ {morans_p:.4f})")
        interp = "positive_clustering" if morans_i > 0.1 else ("mild" if morans_i > 0.01 else "none")
        print(f"  → {interp}")

    # Correlation scan
    print("\n--- Correlation Scan (|r| > 0.1 with residuals) ---")
    corr_results = []
    for col in feature_cols:
        vals = df[col].values.astype(np.float64)
        valid = np.isfinite(vals) & np.isfinite(residuals)
        if valid.sum() < 100: continue
        r, p = stats.pearsonr(vals[valid], residuals[valid])
        if abs(r) > 0.1:
            corr_results.append({'feature': col, 'correlation': float(r), 'abs_corr': float(abs(r)), 'p_value': float(p)})
    corr_results.sort(key=lambda x: x['abs_corr'], reverse=True)
    print(f"  Found {len(corr_results)} features with |r| > 0.1")
    for item in corr_results[:10]:
        print(f"    {item['feature']:35s}: r={item['correlation']:+.4f}")

    # Residual predictability
    print("\n--- Residual Predictability ---")
    res_r2, res_rmse = 0.0, 0.0
    if residuals.std() > 1e-6:
        X_tr, X_te, y_tr, y_te = train_test_split(X, residuals, test_size=0.2, random_state=123)
        dr_tr = xgb.DMatrix(X_tr, label=y_tr)
        dr_te = xgb.DMatrix(X_te, label=y_te)
        model_r = xgb.train(params, dr_tr, num_boost_round=100, evals=[(dr_te, 'test')], verbose_eval=0)
        ypr = model_r.predict(dr_te)
        res_r2 = r2_score(y_te, ypr)
        res_rmse = np.sqrt(mean_squared_error(y_te, ypr))
        print(f"  XGBoost on residuals: R²={res_r2:.6f}, RMSE={res_rmse:.6f}")
        print(f"  → {'Residuals ARE predictable!' if res_r2 > 0.1 else 'Residuals largely unpredictable'}")

    return {
        "xgboost_full": {"r2": float(r2_full), "rmse": float(rmse_full), "n_features": len(feature_cols)},
        "residual_stats": {"mean": float(residuals.mean()), "std": float(residuals.std()),
                          "max_abs": float(np.abs(residuals).max()), "median_abs": float(np.median(np.abs(residuals)))},
        "morans_i": {"I": morans_i, "p_value": morans_p,
                    "interpretation": "positive_clustering" if morans_i > 0.1 else "no_significant_clustering"},
        "residual_correlations": corr_results[:20],
        "residual_predictability": {"r2": float(res_r2), "rmse": float(res_rmse)}
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT D: Hidden Non-Linear Terms Search
# ═══════════════════════════════════════════════════════════════════════════

def experiment_d_nonlinear(df):
    print("\n" + "=" * 70)
    print("EXPERIMENT D: HIDDEN NON-LINEAR TERMS SEARCH")
    print("=" * 70)

    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression

    gap_only = df['gap_only'].values
    bg, bag, rg, pg = df['building_gap'].values, df['building_area_gap'].values, df['road_gap'].values, df['poi_facility_gap_corrected'].values
    bg_clip, bag_clip, rg_clip, pg_clip = np.maximum(0, bg), np.maximum(0, bag), np.maximum(0, rg), np.maximum(0, pg)

    results_d = {}

    # Test 1: Linear (3 gaps)
    print("\n--- Test 1: Linear (3 gaps) ---")
    X1 = np.column_stack([bg, rg, pg])
    lr1 = LinearRegression().fit(X1, gap_only)
    p1 = lr1.predict(X1)
    rmse1, r2_1 = np.sqrt(mean_squared_error(gap_only, p1)), r2_score(gap_only, p1)
    print(f"  RMSE: {rmse1:.6f}, R²: {r2_1:.6f}")
    results_d["linear_3gap"] = {"rmse": float(rmse1), "r2": float(r2_1),
        "coefs": {"bg": float(lr1.coef_[0]), "rg": float(lr1.coef_[1]), "pg": float(lr1.coef_[2]), "intercept": float(lr1.intercept_)}}

    # Test 2: Linear (4 gaps)
    print("--- Test 2: Linear (4 gaps) ---")
    X2 = np.column_stack([bg, bag, rg, pg])
    lr2 = LinearRegression().fit(X2, gap_only)
    p2 = lr2.predict(X2)
    rmse2, r2_2 = np.sqrt(mean_squared_error(gap_only, p2)), r2_score(gap_only, p2)
    print(f"  RMSE: {rmse2:.6f}, R²: {r2_2:.6f}")
    results_d["linear_4gap"] = {"rmse": float(rmse2), "r2": float(r2_2),
        "coefs": {"bg": float(lr2.coef_[0]), "bag": float(lr2.coef_[1]), "rg": float(lr2.coef_[2]), "pg": float(lr2.coef_[3]), "intercept": float(lr2.intercept_)}}

    # Test 3: Exact clip formula
    print("--- Test 3: Exact clip formula ---")
    exact_gap = -(bg_clip + 2*bag_clip + rg_clip + pg_clip)/4
    rmse3, r2_3 = np.sqrt(mean_squared_error(gap_only, exact_gap)), r2_score(gap_only, exact_gap)
    print(f"  RMSE: {rmse3:.6f}, R²: {r2_3:.6f}")
    results_d["exact_clip"] = {"rmse": float(rmse3), "r2": float(r2_3),
        "formula": "-(max(0,bg) + 2*max(0,bag) + max(0,rg) + max(0,pg))/4"}

    # Test 4: Bldg + Road only (clipped)
    print("--- Test 4: Bldg+Road only (clipped) ---")
    X4 = np.column_stack([bg_clip, rg_clip])
    lr4 = LinearRegression().fit(X4, gap_only)
    p4 = lr4.predict(X4)
    rmse4, r2_4 = np.sqrt(mean_squared_error(gap_only, p4)), r2_score(gap_only, p4)
    print(f"  RMSE: {rmse4:.6f}, R²: {r2_4:.6f}")
    results_d["bldg_road_only"] = {"rmse": float(rmse4), "r2": float(r2_4)}

    # Test 5: Quadratic + interactions
    print("--- Test 5: Quadratic+interactions ---")
    X5 = np.column_stack([bg_clip, bag_clip, rg_clip, pg_clip,
        bg_clip**2, bag_clip**2, rg_clip**2, pg_clip**2,
        bg_clip*bag_clip, bg_clip*rg_clip, bg_clip*pg_clip,
        bag_clip*rg_clip, bag_clip*pg_clip, rg_clip*pg_clip])
    lr5 = LinearRegression().fit(X5, gap_only)
    p5 = lr5.predict(X5)
    rmse5, r2_5 = np.sqrt(mean_squared_error(gap_only, p5)), r2_score(gap_only, p5)
    print(f"  RMSE: {rmse5:.6f}, R²: {r2_5:.6f}")
    results_d["quadratic"] = {"rmse": float(rmse5), "r2": float(r2_5)}

    # Test 6: Power-law
    print("--- Test 6: Power-law ---")
    best_pw = None
    for pe in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        Xp = np.column_stack([np.sign(a) * np.abs(a)**pe for a in [bg_clip, bag_clip, rg_clip, pg_clip]])
        lrp = LinearRegression().fit(Xp, gap_only)
        pp = lrp.predict(Xp)
        rp = np.sqrt(mean_squared_error(gap_only, pp))
        r2p = r2_score(gap_only, pp)
        if best_pw is None or rp < best_pw[0]:
            best_pw = (rp, r2p, pe, [float(c) for c in lrp.coef_], float(lrp.intercept_))
    print(f"  Best exponent: {best_pw[2]}, RMSE: {best_pw[0]:.6f}, R²: {best_pw[1]:.6f}")
    results_d["best_power_law"] = {"exponent": best_pw[2], "rmse": float(best_pw[0]), "r2": float(best_pw[1])}

    # Summary
    print("\n--- Formula Comparison Summary ---")
    for name, rmse, r2 in sorted([
        ("Linear (3 gaps)", rmse1, r2_1), ("Linear (4 gaps)", rmse2, r2_2),
        ("Exact clip", rmse3, r2_3), ("Bldg+Road", rmse4, r2_4),
        ("Quadratic+inter", rmse5, r2_5), (f"Power (exp={best_pw[2]})", best_pw[0], best_pw[1]),
    ], key=lambda x: x[1]):
        print(f"  {name:20s}: RMSE={rmse:.6f}, R²={r2:.6f}")

    return results_d


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT E: Coefficient Perturbation
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e_perturbation(df):
    print("\n" + "=" * 70)
    print("EXPERIMENT E: COEFFICIENT PERTURBATION")
    print("=" * 70)

    from sklearn.metrics import mean_squared_error

    exact_coefs = np.array([0.25, 0.5, 0.25, 0.25, 1.0])
    coef_names = ['building_gap_clip', 'building_area_gap_clip', 'road_gap_clip', 'poi_gap_clip', 'rural_penalty']

    bg_clip = np.maximum(0, df['building_gap'].values)
    bag_clip = np.maximum(0, df['building_area_gap'].values)
    rg_clip = np.maximum(0, df['road_gap'].values)
    pg_clip = np.maximum(0, df['poi_facility_gap_corrected'].values)
    rural = (1 - df['pct_urban'].values).clip(0, 1)
    target = df['proxy_merged'].values
    components = np.column_stack([bg_clip, bag_clip, rg_clip, pg_clip, rural])

    exact_pred = -components @ exact_coefs
    exact_rmse = np.sqrt(mean_squared_error(target, exact_pred))
    print(f"Exact formula RMSE: {exact_rmse:.6f}")

    RMSE_THRESHOLD = 0.01

    # Individual sensitivity
    print("\n--- Individual Coefficient Sensitivity ---")
    sensitivity = {}
    for j, name in enumerate(coef_names):
        deltas = np.linspace(-0.5 * exact_coefs[j], 0.5 * exact_coefs[j], 200)
        rmse_profile = np.array([
            np.sqrt(mean_squared_error(target, -components @ np.array([
                exact_coefs[k] + (deltas[i] if k == j else 0) for k in range(5)
            ]))) for i in range(len(deltas))
        ])

        exceed_idx = np.where(rmse_profile > RMSE_THRESHOLD)[0]
        center = len(deltas) // 2
        if len(exceed_idx) > 0:
            left_ex = exceed_idx[exceed_idx < center]
            right_ex = exceed_idx[exceed_idx > center]
            left_max = abs(deltas[left_ex[-1]]) if len(left_ex) > 0 else abs(deltas[0])
            right_max = abs(deltas[right_ex[0]]) if len(right_ex) > 0 else abs(deltas[-1])
            max_delta = min(left_max, right_max)
        else:
            max_delta = abs(deltas[-1])
        rel_tol = max_delta / exact_coefs[j]

        sensitivity[name] = {
            "exact_coef": float(exact_coefs[j]),
            "max_abs_delta": float(max_delta),
            "relative_tolerance": float(rel_tol),
            "rmse_at_plus_50pct": float(rmse_profile[-1]),
            "rmse_at_minus_50pct": float(rmse_profile[0]),
        }
        tag = "ESSENTIAL" if rel_tol < 0.05 else ("IMPORTANT" if rel_tol < 0.2 else "DECORATIVE")
        print(f"  {name:30s}: exact={exact_coefs[j]:.3f}, ±{rel_tol*100:.1f}% tolerance, "
              f"RMSE at ±50%: {rmse_profile[-1]:.4f}/{rmse_profile[0]:.4f} → {tag}")

    # Multi-dimensional perturbation
    print("\n--- Multi-Dimensional Perturbation (10K) ---")
    N_P = 10_000
    rng = np.random.default_rng(42)
    p_rmse = np.zeros(N_P)
    p_deltas = np.zeros((N_P, 5))
    for i in range(N_P):
        delta = rng.standard_normal(5) * rng.uniform(0, 0.5) * exact_coefs
        p_deltas[i] = delta
        p_rmse[i] = np.sqrt(mean_squared_error(target, -components @ (exact_coefs + delta)))

    rmse_inc = p_rmse - exact_rmse
    dim_sens = {}
    print("\n  Correlation |δ| ↔ RMSE increase:")
    for j, name in enumerate(coef_names):
        c, _ = stats.pearsonr(np.abs(p_deltas[:, j]), rmse_inc)
        dim_sens[name] = float(c)
        print(f"    {name:30s}: {c:.4f}")

    safe = p_rmse < 0.001
    print(f"\n  Safe perturbations (RMSE < 0.001): {safe.sum():,} / {N_P:,}")
    if safe.sum() > 10:
        for j, name in enumerate(coef_names):
            print(f"    {name:30s}: mean Δ = {p_deltas[safe, j].mean():.4f}, std Δ = {p_deltas[safe, j].std():.4f}")

    # Removal test
    print("\n--- Coefficient Removal Test ---")
    removal = {}
    for j, name in enumerate(coef_names):
        c = exact_coefs.copy(); c[j] = 0.0
        pred = -components @ c
        rmse_r = np.sqrt(mean_squared_error(target, pred))
        ss_res = np.sum((target - pred)**2); ss_tot = np.sum((target - target.mean())**2)
        r2_r = 1 - ss_res / ss_tot
        removal[name] = {"rmse": float(rmse_r), "r2": float(r2_r)}
        tag = "CRITICAL" if rmse_r > 0.1 else ("SIGNIFICANT" if rmse_r > 0.01 else "MARGINAL")
        print(f"  Remove {name:30s}: RMSE={rmse_r:.4f}, R²={r2_r:.6f} → {tag}")

    return {
        "exact_coefficients": {n: float(c) for n, c in zip(coef_names, exact_coefs)},
        "exact_rmse": float(exact_rmse),
        "individual_sensitivity": sensitivity,
        "dimension_sensitivity_correlation": dim_sens,
        "removal_test": removal,
        "perturbation_stats": {
            "n_perturbations": N_P,
            "mean_rmse": float(p_rmse.mean()), "std_rmse": float(p_rmse.std()),
            "max_rmse": float(p_rmse.max()),
            "pct_below_0_001": float((p_rmse < 0.001).mean() * 100)
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    df = load_data()
    results = {"timestamp": datetime.now().isoformat(), "data_shape": list(df.shape)}
    errors = {}

    for exp_name, exp_func in [
        ("experiment_a", experiment_a_stability),
        ("experiment_b", experiment_b_perfect_mapping),
        ("experiment_c", experiment_c_residuals),
        ("experiment_d", experiment_d_nonlinear),
        ("experiment_e", experiment_e_perturbation),
    ]:
        try:
            results[exp_name] = exp_func(df)
        except Exception as e:
            print(f"\nERROR in {exp_name}: {e}")
            traceback.print_exc()
            errors[exp_name] = str(e)

    if errors:
        results["errors"] = errors

    elapsed = time.time() - t_start
    results["total_elapsed_seconds"] = elapsed

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    if "experiment_a" in results and "per_feature" in results.get("experiment_a", {}):
        print("\n[A] 1M Stability:")
        pf = results["experiment_a"]["per_feature"]
        for name, s in sorted(pf.items(), key=lambda x: x[1]['pct_top1'], reverse=True)[:5]:
            print(f"  {name:30s}: top1={s['pct_top1']:.1f}%, |coef|={s['mean_abs_coef']:.6f}±{s['std_abs_coef']:.6f}, sign={s['sign_stability_pct']:.1f}%")
        r2s = results["experiment_a"]["r2_stats"]
        print(f"  R²: {r2s['mean']:.6f} ± {r2s['std']:.6f}")

    if "experiment_b" in results:
        b = results["experiment_b"]
        r = b.get('best_single_threshold', {})
        if r:
            d = ">" if r.get('direction') == 'gt' else "<"
            print(f"\n[B] Best rule: {r.get('feature','?')} {d} {r.get('threshold',0):.4f} (F1={r.get('f1',0):.4f})")

    if "experiment_c" in results:
        c = results["experiment_c"]
        print(f"\n[C] XGBoost R²={c['xgboost_full']['r2']:.6f}, Moran's I={c['morans_i']['I']:.4f}, Residual R²={c['residual_predictability']['r2']:.6f}")

    if "experiment_d" in results:
        d = results["experiment_d"]
        print(f"\n[D] Gap: linear R²={d.get('linear_3gap',{}).get('r2',0):.4f}, exact R²={d.get('exact_clip',{}).get('r2',0):.4f}")

    if "experiment_e" in results:
        e = results["experiment_e"]
        print(f"\n[E] Removal:")
        for name, s in e.get("removal_test", {}).items():
            tag = "CRITICAL" if s['rmse'] > 0.1 else ("SIGNIFICANT" if s['rmse'] > 0.01 else "MARGINAL")
            print(f"  {name:30s}: RMSE={s['rmse']:.4f}, R²={s['r2']:.6f} → {tag}")

    # Save
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(RESULTS_FILE, 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()
