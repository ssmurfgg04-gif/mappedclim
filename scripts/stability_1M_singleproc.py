#!/usr/bin/env python3
"""
1M-Iteration Stability Test — SINGLE-PROCESS (NO multiprocessing)
=================================================================
Vectorized numpy throughout. No Pool, no threading, no subprocess.

KEY OPTIMIZATION: Precompute x_i ⊗ x_i and y_i · x_i for all rows.
Then each subsampled Ridge regression is just a SUM + SOLVE, 
eliminating the O(np²) matrix multiply from the inner loop.
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/z/my-project/bias-bounty-map")
DATA_FILE    = PROJECT_ROOT / "data/output/engineered_features_merged.parquet"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "formula_decoder_1M.json"


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(n_sub=10000, seed=42):
    print(f"[LOAD] Reading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    print(f"[LOAD] Full shape: {df.shape}")

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n_sub, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    print(f"[LOAD] Subsampled to {len(df)} rows (seed={seed})")

    core_features = [
        'building_gap', 'road_gap', 'pct_urban', 'tribal_any',
        'svi_overall', 'cvi_overall', 'bldg_osm_fraction',
        'bldg_source_diversity', 'pop_total', 'ALAND',
    ]
    extended_features = [
        'building_area_gap', 'poi_facility_gap_corrected',
        'bldg_gap_clip', 'area_gap_clip', 'road_gap_clip', 'poi_gap_clip',
    ]
    available_extended = [f for f in extended_features if f in df.columns]
    all_features = core_features + available_extended

    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[LOAD] WARNING: missing features dropped: {missing}")
        all_features = [f for f in all_features if f in df.columns]

    X = df[all_features].values.astype(np.float64)
    y = df['proxy_merged'].values.astype(np.float64)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    df_clean = df.loc[mask].copy()
    print(f"[LOAD] After cleaning: {len(X)} rows, {X.shape[1]} features")
    print(f"[LOAD] Features: {all_features}")

    # Standardize
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std  = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    return X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std, all_features, df_clean


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: 1M STABILITY TEST — Precomputed Outer Products
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1M_stability(X_norm, y_norm, feature_names, n_iter=1_000_000, alpha=1.0):
    """
    1M subsampled Ridge regressions using precomputed outer products.
    
    For each row i, precompute:
      xxT[i] = x_i @ x_i^T   (p × p matrix)
      yx[i]  = y_i * x_i      (p vector)
    
    Then for a random subset S of rows:
      X_S^T @ X_S = sum_{i in S} xxT[i]
      X_S^T @ y_S = sum_{i in S} yx[i]
      coefs = (X_S^T X_S + αI)^{-1} X_S^T y_S
    
    This turns each iteration from O(n·p²) matmul → O(|S|·p²) sum,
    which is ~2x faster and avoids allocating X_sub each time.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: 1M-Iteration Stability Test (Precomputed Outer Products)")
    print("="*70)

    n, p = X_norm.shape
    train_size = int(0.8 * n)
    print(f"  n={n}, p={p}, train_size={train_size}, alpha={alpha}")

    # ── Precompute outer products ──
    t0_precomp = time.time()
    # xxT: shape (n, p, p) — each xxT[i] = x_i ⊗ x_i
    xxT = np.einsum('ni,nj->nij', X_norm, X_norm)  # (n, p, p)
    # yx: shape (n, p) — each yx[i] = y_i * x_i
    yx = y_norm[:, np.newaxis] * X_norm  # (n, p)
    print(f"  Precomputed outer products in {time.time()-t0_precomp:.2f}s")
    print(f"  xxT shape: {xxT.shape}, yx shape: {yx.shape}")
    print(f"  Memory: xxT={xxT.nbytes/1e6:.1f}MB, yx={yx.nbytes/1e6:.1f}MB")

    # ── Accumulators ──
    coef_sum   = np.zeros(p)
    coef_sqsum = np.zeros(p)
    abs_coef_sum   = np.zeros(p)
    abs_coef_sqsum = np.zeros(p)
    sign_pos_count = np.zeros(p)
    top1_count = np.zeros(p, dtype=np.int64)
    top3_count = np.zeros(p, dtype=np.int64)

    total_iter = 0
    t0 = time.time()
    rng = np.random.RandomState(12345)
    alpha_I = alpha * np.eye(p)

    progress_interval = 100_000
    batch_idx = 0

    while total_iter < n_iter:
        # Generate train indices for this iteration
        train_idx = rng.choice(n, size=train_size, replace=False)

        # Compute XtX and Xty via precomputed sums
        XtX = xxT[train_idx].sum(axis=0) + alpha_I  # (p, p)
        Xty = yx[train_idx].sum(axis=0)              # (p,)

        # Solve
        try:
            coefs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coefs = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        abs_coefs = np.abs(coefs)

        # Accumulate
        coef_sum   += coefs
        coef_sqsum += coefs ** 2
        abs_coef_sum   += abs_coefs
        abs_coef_sqsum += abs_coefs ** 2
        sign_pos_count += (coefs > 0).astype(np.float64)

        sorted_idx = np.argsort(abs_coefs)[::-1]
        top1_count[sorted_idx[0]] += 1
        top3_count[sorted_idx[:3]] += 1

        total_iter += 1

        # Progress
        if total_iter % progress_interval == 0:
            elapsed = time.time() - t0
            rate = total_iter / elapsed
            eta = (n_iter - total_iter) / rate
            print(f"  [{total_iter:>10,} / {n_iter:,}]  "
                  f"{elapsed:.1f}s  {rate:.0f} iter/s  ETA {eta:.0f}s")

    # Final statistics
    mean_coef   = coef_sum / total_iter
    std_coef    = np.sqrt(np.maximum(coef_sqsum / total_iter - mean_coef**2, 0))
    mean_abs    = abs_coef_sum / total_iter
    std_abs     = np.sqrt(np.maximum(abs_coef_sqsum / total_iter - mean_abs**2, 0))
    sign_stab   = np.maximum(sign_pos_count / total_iter, 1.0 - sign_pos_count / total_iter)
    top1_pct    = top1_count / total_iter
    top3_pct    = top3_count / total_iter

    print("\n  ─── Coefficient Stability Summary ───")
    print(f"  {'Feature':<30s} {'MeanCoef':>9s} {'StdCoef':>9s} {'Mean|c|':>9s} {'Std|c|':>9s} {'Sign%':>7s} {'Top1%':>7s} {'Top3%':>7s}")
    print("  " + "-"*88)
    results_list = []
    for i, fname in enumerate(feature_names):
        row = {
            'feature': fname,
            'mean_coef': float(mean_coef[i]),
            'std_coef':  float(std_coef[i]),
            'mean_abs':  float(mean_abs[i]),
            'std_abs':   float(std_abs[i]),
            'sign_stability': float(sign_stab[i]),
            'top1_pct':  float(top1_pct[i]),
            'top3_pct':  float(top3_pct[i]),
        }
        results_list.append(row)
        print(f"  {fname:<30s} {mean_coef[i]:>9.4f} {std_coef[i]:>9.4f} "
              f"{mean_abs[i]:>9.4f} {std_abs[i]:>9.4f} {sign_stab[i]:>6.3f} "
              f"{top1_pct[i]:>6.3f} {top3_pct[i]:>6.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s  ({total_iter/elapsed:.0f} iter/s)")

    return {
        'n_iterations': int(total_iter),
        'elapsed_seconds': round(elapsed, 2),
        'feature_results': results_list,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: PERFECT MAPPING TREE
# ═══════════════════════════════════════════════════════════════════════════

def experiment_mapping_tree(df, y_raw):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Perfect Mapping Decision Tree")
    print("="*70)

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    is_well_mapped = (y_raw > -0.01).astype(int)
    n_pos = int(is_well_mapped.sum())
    n_neg = len(is_well_mapped) - n_pos
    print(f"  is_well_mapped: {n_pos} well-mapped ({100*n_pos/len(is_well_mapped):.1f}%), "
          f"{n_neg} not ({100*n_neg/len(is_well_mapped):.1f}%)")

    tree_features = ['building_gap', 'road_gap', 'pct_urban', 'tribal_any', 'svi_overall', 'cvi_overall']
    available_tree = [f for f in tree_features if f in df.columns]
    X_tree = df[available_tree].values.astype(np.float64)
    mask = np.isfinite(X_tree).all(axis=1) & np.isfinite(is_well_mapped)
    X_tree = X_tree[mask]
    y_tree = is_well_mapped[mask]

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_tree, y_tree)

    tree_text = export_text(clf, feature_names=available_tree, decimals=3)
    print("\n  ─── Decision Tree Rules ───")
    for line in tree_text.split('\n'):
        print(f"  {line}")

    y_pred = clf.predict(X_tree)
    acc  = accuracy_score(y_tree, y_pred)
    prec = precision_score(y_tree, y_pred, zero_division=0)
    rec  = recall_score(y_tree, y_pred, zero_division=0)
    f1   = f1_score(y_tree, y_pred, zero_division=0)
    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Tree depth: {clf.get_depth()}, leaves: {clf.get_n_leaves()}")

    return {
        'features': available_tree,
        'depth': clf.get_depth(),
        'n_leaves': clf.get_n_leaves(),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'tree_text': tree_text,
        'n_well_mapped': n_pos,
        'n_not_mapped': n_neg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: COEFFICIENT PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════

def experiment_perturbation(df, y_raw, n_perturb=10_000):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Coefficient Perturbation Sensitivity")
    print("="*70)

    def get_col(name, default=0.0):
        if name in df.columns:
            return df[name].values.astype(np.float64)
        return np.full(len(df), default)

    bldg_clip = get_col('bldg_gap_clip')
    area_clip = get_col('area_gap_clip')
    road_clip = get_col('road_gap_clip')
    poi_clip  = get_col('poi_gap_clip')
    pct_urban = get_col('pct_urban')

    base_coefs = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    coef_names = ['c1_bldg', 'c2_area*2', 'c3_road', 'c4_poi', 'c5_rural']

    def predict_proxy(coefs):
        c1, c2, c3, c4, c5 = coefs
        gap_term = -(c1 * bldg_clip + c2 * 2 * area_clip + c3 * road_clip + c4 * poi_clip) / 4
        rural_term = -c5 * (1 - pct_urban)
        return gap_term + rural_term

    base_pred = predict_proxy(base_coefs)
    base_rmse = np.sqrt(np.mean((base_pred - y_raw)**2))
    print(f"  Base formula RMSE: {base_rmse:.6f}")

    rng = np.random.RandomState(42)
    perturb_factors = 1.0 + rng.uniform(-0.5, 0.5, size=(n_perturb, 5))
    perturbed_coefs = base_coefs[np.newaxis, :] * perturb_factors

    # Vectorized: compute all predictions at once
    # gap_term_i = -(c1_i * bldg + c2_i * 2 * area + c3_i * road + c4_i * poi) / 4
    # Shape tricks: coefs (N,5), features (M,)
    bldg_b = bldg_clip[np.newaxis, :]   # (1, M)
    area_b = area_clip[np.newaxis, :]
    road_b = road_clip[np.newaxis, :]
    poi_b  = poi_clip[np.newaxis, :]
    purb_b = pct_urban[np.newaxis, :]
    y_b    = y_raw[np.newaxis, :]

    gap_terms = -(perturbed_coefs[:, 0:1] * bldg_b +
                  perturbed_coefs[:, 1:2] * 2 * area_b +
                  perturbed_coefs[:, 2:3] * road_b +
                  perturbed_coefs[:, 3:4] * poi_b) / 4
    rural_terms = -perturbed_coefs[:, 4:5] * (1 - purb_b)
    preds = gap_terms + rural_terms  # (N, M)
    sq_err = (preds - y_b) ** 2
    rmses = np.sqrt(sq_err.mean(axis=1))  # (N,)

    rmse_mean = float(rmses.mean())
    rmse_std  = float(rmses.std())
    rmse_min  = float(rmses.min())
    rmse_max  = float(rmses.max())
    print(f"  Perturbed RMSE: mean={rmse_mean:.6f}, std={rmse_std:.6f}, "
          f"min={rmse_min:.6f}, max={rmse_max:.6f}")

    # Per-coefficient sensitivity (perturb one at a time)
    sensitivity = {}
    for j, cname in enumerate(coef_names):
        single_coefs = np.tile(base_coefs, (n_perturb, 1))
        single_coefs[:, j] = base_coefs[j] * (1.0 + rng.uniform(-0.5, 0.5, size=n_perturb))

        gap_s = -(single_coefs[:, 0:1] * bldg_b +
                  single_coefs[:, 1:2] * 2 * area_b +
                  single_coefs[:, 2:3] * road_b +
                  single_coefs[:, 3:4] * poi_b) / 4
        rural_s = -single_coefs[:, 4:5] * (1 - purb_b)
        preds_s = gap_s + rural_s
        rmses_s = np.sqrt(((preds_s - y_b) ** 2).mean(axis=1))

        sensitivity[cname] = {
            'rmse_mean': float(rmses_s.mean()),
            'rmse_std':  float(rmses_s.std()),
            'delta_rmse': float(rmses_s.mean() - base_rmse),
        }
        print(f"  {cname}: ΔRMSE={sensitivity[cname]['delta_rmse']:+.6f}, "
              f"std={sensitivity[cname]['rmse_std']:.6f}")

    sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]['delta_rmse']), reverse=True)
    print(f"\n  MOST  sensitive: {sorted_sens[0][0]} (|ΔRMSE|={abs(sorted_sens[0][1]['delta_rmse']):.6f})")
    print(f"  LEAST sensitive: {sorted_sens[-1][0]} (|ΔRMSE|={abs(sorted_sens[-1][1]['delta_rmse']):.6f})")

    return {
        'base_rmse': float(base_rmse),
        'perturb_rmse_mean': rmse_mean,
        'perturb_rmse_std': rmse_std,
        'perturb_rmse_min': rmse_min,
        'perturb_rmse_max': rmse_max,
        'sensitivity': sensitivity,
        'most_sensitive': sorted_sens[0][0],
        'least_sensitive': sorted_sens[-1][0],
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: HIDDEN FORMULA DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def experiment_formula_discovery(df, y_raw):
    print("\n" + "="*70)
    print("EXPERIMENT 4: Hidden Formula Discovery")
    print("="*70)

    from scipy.optimize import minimize

    def get_col(name, default=0.0):
        if name in df.columns:
            return df[name].values.astype(np.float64)
        return np.full(len(df), default)

    pct_urban = get_col('pct_urban')
    bg = get_col('building_gap')
    rg = get_col('road_gap')
    bag = get_col('building_area_gap')
    pg = get_col('poi_facility_gap_corrected')

    gap_only = y_raw + (1 - pct_urban)
    print(f"  gap_only stats: mean={gap_only.mean():.4f}, std={gap_only.std():.4f}")

    # --- Test 1: Linear ---
    print("\n  --- Test 1: Linear: gap_only = a*bg + b*rg + c*pg + d ---")
    X_lin = np.column_stack([bg, rg, pg, np.ones(len(df))])
    coefs_lin, _, _, _ = np.linalg.lstsq(X_lin, gap_only, rcond=None)
    pred_lin = X_lin @ coefs_lin
    rmse_lin = np.sqrt(np.mean((pred_lin - gap_only)**2))
    r2_lin = 1 - np.var(gap_only - pred_lin) / np.var(gap_only)
    print(f"    a={coefs_lin[0]:.6f}, b={coefs_lin[1]:.6f}, c={coefs_lin[2]:.6f}, d={coefs_lin[3]:.6f}")
    print(f"    RMSE={rmse_lin:.6f}, R²={r2_lin:.6f}")

    # --- Test 2: Weighted gap ---
    print("\n  --- Test 2: Weighted: gap_only = -(a*max(0,bg) + b*max(0,bag) + c*max(0,rg) + d*max(0,pg))/(a+b+c+d) ---")
    bg_pos  = np.maximum(0, bg)
    bag_pos = np.maximum(0, bag)
    rg_pos  = np.maximum(0, rg)
    pg_pos  = np.maximum(0, pg)

    def weighted_gap_loss(params):
        a, b, c, d = params
        denom = a + b + c + d
        if abs(denom) < 1e-10:
            return 1e10
        pred = -(a * bg_pos + b * bag_pos + c * rg_pos + d * pg_pos) / denom
        return np.mean((pred - gap_only)**2)

    best_loss = float('inf')
    best_params = None
    for x0 in [[1,1,1,1], [1,2,1,1], [2,1,1,1], [1,1,2,1], [0.5,1,0.5,0.5]]:
        res = minimize(weighted_gap_loss, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    pred_wt = -(best_params[0] * bg_pos + best_params[1] * bag_pos +
                best_params[2] * rg_pos + best_params[3] * pg_pos) / (best_params.sum())
    rmse_wt = np.sqrt(best_loss)
    r2_wt = 1 - np.var(gap_only - pred_wt) / np.var(gap_only)
    a, b, c, d = best_params
    s = a + b + c + d
    print(f"    a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")
    print(f"    Normalized: a={a/s:.4f}, b={b/s:.4f}, c={c/s:.4f}, d={d/s:.4f}")
    print(f"    RMSE={rmse_wt:.6f}, R²={r2_wt:.6f}")

    # --- Test 3: Extended linear with area ---
    print("\n  --- Test 3: Linear with area: gap_only = a*bg + b*bag + c*rg + d*pg + e ---")
    X_ext = np.column_stack([bg, bag, rg, pg, np.ones(len(df))])
    coefs_ext, _, _, _ = np.linalg.lstsq(X_ext, gap_only, rcond=None)
    pred_ext = X_ext @ coefs_ext
    rmse_ext = np.sqrt(np.mean((pred_ext - gap_only)**2))
    r2_ext = 1 - np.var(gap_only - pred_ext) / np.var(gap_only)
    print(f"    a(bg)={coefs_ext[0]:.6f}, b(bag)={coefs_ext[1]:.6f}, "
          f"c(rg)={coefs_ext[2]:.6f}, d(pg)={coefs_ext[3]:.6f}, e={coefs_ext[4]:.6f}")
    print(f"    RMSE={rmse_ext:.6f}, R²={r2_ext:.6f}")

    return {
        'linear': {
            'coefs': [float(x) for x in coefs_lin],
            'rmse': float(rmse_lin),
            'r2': float(r2_lin),
        },
        'weighted': {
            'params': [float(x) for x in best_params],
            'rmse': float(rmse_wt),
            'r2': float(r2_wt),
        },
        'extended_linear': {
            'coefs': [float(x) for x in coefs_ext],
            'rmse': float(rmse_ext),
            'r2': float(r2_ext),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: SURPRISE HUNT
# ═══════════════════════════════════════════════════════════════════════════

def experiment_surprise_hunt(df, y_raw):
    print("\n" + "="*70)
    print("EXPERIMENT 5: Surprise Hunt — Unexpected Correlations & Effects")
    print("="*70)

    # --- Part A: High correlations ---
    print("\n  --- Part A: Features with |r| > 0.5 ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obvious_cols = {
        'proxy_merged', 'bldg_gap_sq', 'bldg_gap_abs', 'bldg_gap_clip',
        'area_gap_abs', 'area_gap_clip', 'road_gap_abs', 'road_gap_clip',
        'poi_gap_clip', 'bldg_x_area_gap', 'bldg_minus_area_gap',
        'bldg_road_diff', 'bldg_road_product', 'rural_x_bldg',
        'rural_x_bldg_clip', 'pct_urban_x_bldg', 'rural_x_road',
        'rural_x_area_gap', 'rural_indicator', 'rural_continuous',
        'tribal_x_bldg', 'tribal_x_rural', 'svi_x_bldg',
        'rural_x_svi_x_bldg', 'cvi_x_bldg', 'wf_x_bldg',
        'compound_risk', 'rural_x_risk',
        'bldg_total_sources_x_bldg', 'bldg_source_diversity_x_bldg',
        'source_coverage_fraction_x_bldg', 'source_diversity_entropy_x_bldg',
        'bldg_county_loo_smooth',
    }

    correlations = []
    for col in numeric_cols:
        if col == 'proxy_merged':
            continue
        vals = df[col].values.astype(np.float64)
        mask = np.isfinite(vals) & np.isfinite(y_raw)
        if mask.sum() < 100:
            continue
        r = np.corrcoef(vals[mask], y_raw[mask])[0, 1]
        if np.isfinite(r) and abs(r) > 0.5:
            is_obvious = col in obvious_cols
            correlations.append((col, r, is_obvious))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    surprising = [(c, r) for c, r, o in correlations if not o]
    obvious_found = [(c, r) for c, r, o in correlations if o]

    print(f"  Found {len(correlations)} features with |r| > 0.5")
    print(f"  Of which {len(surprising)} are SURPRISING (not obviously derived):")
    for col, r in surprising[:20]:
        print(f"    {col:<45s}  r={r:+.4f}")
    if obvious_found:
        print(f"  Obviously derived (for reference):")
        for col, r in obvious_found[:10]:
            print(f"    {col:<45s}  r={r:+.4f}")

    # --- Part B: State-level effect ---
    print("\n  --- Part B: State-Level Effect ---")
    state_results = {}
    if 'GEOID' in df.columns:
        from sklearn.linear_model import Ridge

        state_fps = df['GEOID'].astype(str).str[:2]
        unique_states = state_fps.unique()
        print(f"  Found {len(unique_states)} states in data")

        feature_cols = ['building_gap', 'road_gap', 'pct_urban', 'tribal_any',
                        'svi_overall', 'cvi_overall']
        feature_cols = [f for f in feature_cols if f in df.columns]
        X_feat = df[feature_cols].values.astype(np.float64)
        mask_feat = np.isfinite(X_feat).all(axis=1) & np.isfinite(y_raw)
        X_clean = X_feat[mask_feat]
        y_clean = y_raw[mask_feat]
        state_clean = state_fps.values[mask_feat]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_clean, y_clean)
        residuals = y_clean - ridge.predict(X_clean)

        state_resid_stats = {}
        for st in unique_states:
            st_mask = state_clean == st
            if st_mask.sum() < 10:
                continue
            st_resid = residuals[st_mask]
            state_resid_stats[st] = {
                'n': int(st_mask.sum()),
                'mean': float(st_resid.mean()),
                'std':  float(st_resid.std()),
            }

        sorted_states = sorted(state_resid_stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)
        print(f"  Top 10 states by |mean residual|:")
        for st, stats_d in sorted_states[:10]:
            print(f"    State {st}: n={stats_d['n']}, mean_resid={stats_d['mean']:+.6f}, std={stats_d['std']:.6f}")

        state_means = np.array([s['mean'] for s in state_resid_stats.values()])
        state_ns    = np.array([s['n'] for s in state_resid_stats.values()])
        between_var = np.average(state_means**2, weights=state_ns)
        within_var  = np.average(np.array([s['std']**2 for s in state_resid_stats.values()]), weights=state_ns)
        f_ratio = between_var / within_var if within_var > 0 else float('inf')
        print(f"\n  Between-state variance: {between_var:.6f}")
        print(f"  Within-state variance:  {within_var:.6f}")
        print(f"  F-ratio (between/within): {f_ratio:.4f}")
        print(f"  → {'SIGNIFICANT' if f_ratio > 0.05 else 'Not significant'} state-level effect")

        state_results = {
            'n_states': len(unique_states),
            'between_var': float(between_var),
            'within_var': float(within_var),
            'f_ratio': float(f_ratio),
            'significant': bool(f_ratio > 0.05),
            'top_biased_states': [
                {'state': st, 'n': stats_d['n'], 'mean_resid': stats_d['mean']}
                for st, stats_d in sorted_states[:10]
            ],
        }

    # --- Part C: County-level effect ---
    print("\n  --- Part C: County-Level Effect ---")
    county_results = {}
    if 'GEOID' in df.columns:
        county_fps = df['GEOID'].astype(str).str[:5]
        unique_counties = county_fps.unique()
        print(f"  Found {len(unique_counties)} counties in data")

        county_clean = county_fps.values[mask_feat]
        county_resid_stats = {}
        for co in unique_counties:
            co_mask = county_clean == co
            if co_mask.sum() < 5:
                continue
            co_resid = residuals[co_mask]
            county_resid_stats[co] = {
                'n': int(co_mask.sum()),
                'mean': float(co_resid.mean()),
                'std':  float(co_resid.std()),
            }

        sorted_counties = sorted(county_resid_stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)
        print(f"  Top 10 counties by |mean residual|:")
        for co, stats_d in sorted_counties[:10]:
            print(f"    County {co}: n={stats_d['n']}, mean_resid={stats_d['mean']:+.6f}")

        county_means = np.array([s['mean'] for s in county_resid_stats.values()])
        county_ns    = np.array([s['n'] for s in county_resid_stats.values()])
        between_var_c = np.average(county_means**2, weights=county_ns)
        within_var_c  = np.average(np.array([s['std']**2 for s in county_resid_stats.values()]), weights=county_ns)
        f_ratio_c = between_var_c / within_var_c if within_var_c > 0 else float('inf')
        print(f"\n  Between-county variance: {between_var_c:.6f}")
        print(f"  Within-county variance:  {within_var_c:.6f}")
        print(f"  F-ratio: {f_ratio_c:.4f}")
        print(f"  → {'SIGNIFICANT' if f_ratio_c > 0.05 else 'Not significant'} county-level effect")

        county_results = {
            'n_counties': len(unique_counties),
            'between_var': float(between_var_c),
            'within_var': float(within_var_c),
            'f_ratio': float(f_ratio_c),
            'significant': bool(f_ratio_c > 0.05),
            'top_biased_counties': [
                {'county': co, 'n': stats_d['n'], 'mean_resid': stats_d['mean']}
                for co, stats_d in sorted_counties[:10]
            ],
        }

    return {
        'high_correlations': {
            'surprising': [{'feature': c, 'r': round(r, 4)} for c, r in surprising[:20]],
            'obvious_derived': [{'feature': c, 'r': round(r, 4)} for c, r in obvious_found[:10]],
        },
        'state_effect': state_results,
        'county_effect': county_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("1M-ITERATION STABILITY TEST — SINGLE-PROCESS (NO multiprocessing)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {}

    # Load data
    X, y, X_norm, y_norm, X_mean, X_std, y_mean, y_std, feature_names, df_clean = load_data(n_sub=10000, seed=42)
    all_results['data_info'] = {
        'n_rows': int(len(X)),
        'n_features': int(X.shape[1]),
        'features': feature_names,
        'y_mean': float(y.mean()),
        'y_std': float(y.std()),
    }

    # --- Experiment 1: 1M Stability ---
    try:
        all_results['experiment_1M_stability'] = experiment_1M_stability(
            X_norm, y_norm, feature_names, n_iter=1_000_000, alpha=1.0
        )
    except Exception as e:
        print(f"\n  EXPERIMENT 1 FAILED: {e}")
        traceback.print_exc()
        all_results['experiment_1M_stability'] = {'error': str(e)}

    # --- Experiment 2: Mapping Tree ---
    try:
        all_results['experiment_mapping_tree'] = experiment_mapping_tree(df_clean, y)
    except Exception as e:
        print(f"\n  EXPERIMENT 2 FAILED: {e}")
        traceback.print_exc()
        all_results['experiment_mapping_tree'] = {'error': str(e)}

    # --- Experiment 3: Perturbation ---
    try:
        all_results['experiment_perturbation'] = experiment_perturbation(df_clean, y, n_perturb=10_000)
    except Exception as e:
        print(f"\n  EXPERIMENT 3 FAILED: {e}")
        traceback.print_exc()
        all_results['experiment_perturbation'] = {'error': str(e)}

    # --- Experiment 4: Formula Discovery ---
    try:
        all_results['experiment_formula_discovery'] = experiment_formula_discovery(df_clean, y)
    except Exception as e:
        print(f"\n  EXPERIMENT 4 FAILED: {e}")
        traceback.print_exc()
        all_results['experiment_formula_discovery'] = {'error': str(e)}

    # --- Experiment 5: Surprise Hunt ---
    try:
        all_results['experiment_surprise_hunt'] = experiment_surprise_hunt(df_clean, y)
    except Exception as e:
        print(f"\n  EXPERIMENT 5 FAILED: {e}")
        traceback.print_exc()
        all_results['experiment_surprise_hunt'] = {'error': str(e)}

    # Save
    all_results['meta'] = {
        'total_seconds': round(time.time() - t_start, 2),
        'completed_at': datetime.now().isoformat(),
        'process_type': 'single-process (no multiprocessing)',
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] Results → {RESULTS_FILE}")

    # ═══════════════════════════════════════════════════════════════════
    # COMPREHENSIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)

    print(f"\nTotal runtime: {all_results['meta']['total_seconds']:.1f}s")
    print(f"Process type: {all_results['meta']['process_type']}")

    if 'error' not in all_results.get('experiment_1M_stability', {}):
        e1 = all_results['experiment_1M_stability']
        print(f"\n[1] 1M STABILITY: {e1['n_iterations']:,} iterations in {e1['elapsed_seconds']:.1f}s")
        fr = e1['feature_results']
        most_stable = max(fr, key=lambda x: x['sign_stability'])
        most_unstable = min(fr, key=lambda x: x['sign_stability'])
        print(f"    Most stable:  {most_stable['feature']} (sign={most_stable['sign_stability']:.3f})")
        print(f"    Least stable: {most_unstable['feature']} (sign={most_unstable['sign_stability']:.3f})")
        top1_feat = max(fr, key=lambda x: x['top1_pct'])
        print(f"    Top-1 dominant: {top1_feat['feature']} ({top1_feat['top1_pct']:.1%} of iterations)")

    if 'error' not in all_results.get('experiment_mapping_tree', {}):
        e2 = all_results['experiment_mapping_tree']
        print(f"\n[2] MAPPING TREE: depth={e2['depth']}, leaves={e2['n_leaves']}")
        print(f"    Accuracy={e2['accuracy']:.4f}, Precision={e2['precision']:.4f}, Recall={e2['recall']:.4f}")

    if 'error' not in all_results.get('experiment_perturbation', {}):
        e3 = all_results['experiment_perturbation']
        print(f"\n[3] PERTURBATION: Base RMSE={e3['base_rmse']:.6f}")
        print(f"    Perturbed RMSE: {e3['perturb_rmse_mean']:.6f} ± {e3['perturb_rmse_std']:.6f}")
        print(f"    Most sensitive:  {e3['most_sensitive']}")
        print(f"    Least sensitive: {e3['least_sensitive']}")

    if 'error' not in all_results.get('experiment_formula_discovery', {}):
        e4 = all_results['experiment_formula_discovery']
        print(f"\n[4] FORMULA DISCOVERY:")
        print(f"    Linear:  RMSE={e4['linear']['rmse']:.6f}, R²={e4['linear']['r2']:.6f}")
        print(f"    Weighted: RMSE={e4['weighted']['rmse']:.6f}, R²={e4['weighted']['r2']:.6f}")
        print(f"    Extended: RMSE={e4['extended_linear']['rmse']:.6f}, R²={e4['extended_linear']['r2']:.6f}")

    if 'error' not in all_results.get('experiment_surprise_hunt', {}):
        e5 = all_results['experiment_surprise_hunt']
        n_surprising = len(e5['high_correlations']['surprising'])
        print(f"\n[5] SURPRISE HUNT:")
        print(f"    {n_surprising} surprising high-correlation features (|r|>0.5, not obviously derived)")
        if e5.get('state_effect'):
            se = e5['state_effect']
            print(f"    State effect: F-ratio={se['f_ratio']:.4f} → {'SIGNIFICANT' if se['significant'] else 'not significant'}")
        if e5.get('county_effect'):
            ce = e5['county_effect']
            print(f"    County effect: F-ratio={ce['f_ratio']:.4f} → {'SIGNIFICANT' if ce['significant'] else 'not significant'}")

    print("\n" + "=" * 70)
    print(f"DONE. Results saved to {RESULTS_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
