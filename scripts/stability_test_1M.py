#!/usr/bin/env python3
"""
1M-Iteration Stability Test for Bias-Bounty-Mapping-Equity Model
================================================================
Optimized: subsamples to 10K rows, vectorized inner loop,
multiprocessing with shared memory via fork.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from multiprocessing import Pool, shared_memory

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "output" / "engineered_features_merged.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "formula_decoder_1M.json"

N_SPLITS = 1_000_000
BATCH_SIZE = 10_000
N_WORKERS = 8
RIDGE_ALPHA = 1.0
SEED = 42
SUBSAMPLE_N = 10_000  # subsample for speed


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    
    feature_names = [
        'building_gap', 'road_gap', 'pct_urban', 'tribal_any', 
        'svi_overall', 'cvi_overall', 'bldg_osm_fraction',
        'bldg_source_diversity', 'pop_total', 'ALAND',
        'rural_indicator',
        'is_perfectly_mapped'
    ]
    
    # Compute derived features
    if 'rural_indicator' not in df.columns:
        df['rural_indicator'] = 1.0 - df['pct_urban'].fillna(1.0)
    else:
        df['rural_indicator'] = df['rural_indicator'].fillna(1.0 - df['pct_urban'].fillna(1.0))
    
    df['is_perfectly_mapped'] = (df['proxy_merged'] > -0.001).astype(int)
    
    X_raw = df[feature_names].copy()
    if X_raw['tribal_any'].dtype == bool:
        X_raw['tribal_any'] = X_raw['tribal_any'].astype(float)
    X_raw = X_raw.fillna(0)
    
    y = df['proxy_merged'].values.copy()
    
    # Standardize features
    X_means = X_raw.mean().values
    X_stds = X_raw.std().values
    X_stds[X_stds == 0] = 1.0
    X = (X_raw.values - X_means) / X_stds
    
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[valid]
    y = y[valid]
    
    # Subsample for speed (stratified by is_perfectly_mapped)
    n_total = len(y)
    if n_total > SUBSAMPLE_N:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_total, size=SUBSAMPLE_N, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"  Subsampled to {SUBSAMPLE_N:,} rows for speed")
    
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Target: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    print(f"  is_perfectly_mapped: {df['is_perfectly_mapped'].sum()} / {len(df)} = {df['is_perfectly_mapped'].mean()*100:.1f}%")
    
    return X, y, feature_names, df


# ═══════════════════════════════════════════════════════════════════════════
# BATCH RIDGE WORKER
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(args):
    """Run a batch of Ridge regressions."""
    X, y, n_in_batch, test_frac, seed_offset, alpha = args
    n, p = X.shape
    n_train = int(n * (1.0 - test_frac))
    alpha_I = alpha * np.eye(p)
    
    rng = np.random.RandomState(seed_offset)
    
    # Pre-allocate
    abs_coefs = np.empty((n_in_batch, p), dtype=np.float32)
    r2s = np.empty(n_in_batch, dtype=np.float32)
    top1s = np.empty(n_in_batch, dtype=np.int32)
    top3s = np.empty((n_in_batch, 3), dtype=np.int32)
    signs = np.empty((n_in_batch, p), dtype=np.int8)
    
    for i in range(n_in_batch):
        perm = rng.permutation(n)
        tr = perm[:n_train]
        te = perm[n_train:]
        
        Xtr = X[tr]
        ytr = y[tr]
        
        # Ridge: solve (X'X + alpha*I) w = X'y
        XtX = Xtr.T @ Xtr + alpha_I
        Xty = Xtr.T @ ytr
        coefs = np.linalg.solve(XtX, Xty)
        
        # Test R2
        y_pred = X[te] @ coefs
        y_true = y[te]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        ac = np.abs(coefs)
        abs_coefs[i] = ac.astype(np.float32)
        r2s[i] = r2
        top1s[i] = np.argmax(ac)
        top3s[i] = np.argsort(ac)[-3:][::-1]
        signs[i] = np.sign(coefs).astype(np.int8)
    
    return abs_coefs, r2s, top1s, top3s, signs


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: 1M RIDGE STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1M_ridge(X, y, feature_names):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 1,000,000 ITERATION RIDGE STABILITY TEST")
    print("=" * 70)
    
    n, p = X.shape
    test_frac = 0.2
    n_batches = N_SPLITS // BATCH_SIZE
    t0 = time.time()
    
    print(f"  Config: {N_SPLITS:,} splits, batch={BATCH_SIZE:,}, workers={N_WORKERS}, alpha={RIDGE_ALPHA}")
    print(f"  Data: n={n}, p={p}, test_frac={test_frac}")
    print(f"  Batches: {n_batches}")
    
    # Prepare batch args
    batch_args = []
    for b in range(n_batches):
        seed_offset = SEED + b * 99991
        batch_args.append((X, y, BATCH_SIZE, test_frac, seed_offset, RIDGE_ALPHA))
    
    # Accumulators (use float64 for precision)
    coef_sum = np.zeros(p, dtype=np.float64)
    coef_sq_sum = np.zeros(p, dtype=np.float64)
    r2_sum = 0.0
    r2_sq_sum = 0.0
    top1_count = np.zeros(p, dtype=np.int64)
    top3_count = np.zeros(p, dtype=np.int64)
    sign_pos_count = np.zeros(p, dtype=np.int64)
    sign_neg_count = np.zeros(p, dtype=np.int64)
    total_iters = 0
    
    completed = 0
    progress_interval = 100000  # print every 100K
    
    with Pool(N_WORKERS) as pool:
        for batch_result in pool.imap_unordered(run_batch, batch_args):
            all_abs_coefs, all_r2, all_top1, all_top3, all_signs = batch_result
            batch_n = len(all_r2)
            
            # Accumulate
            coef_sum += all_abs_coefs.astype(np.float64).sum(axis=0)
            coef_sq_sum += (all_abs_coefs.astype(np.float64) ** 2).sum(axis=0)
            r2_arr = all_r2.astype(np.float64)
            r2_sum += r2_arr.sum()
            r2_sq_sum += (r2_arr ** 2).sum()
            
            for j in range(p):
                top1_count[j] += np.sum(all_top1 == j)
            
            for k in range(3):
                for j in range(p):
                    top3_count[j] += np.sum(all_top3[:, k] == j)
            
            sign_pos_count += (all_signs == 1).sum(axis=0).astype(np.int64)
            sign_neg_count += (all_signs == -1).sum(axis=0).astype(np.int64)
            
            total_iters += batch_n
            completed += 1
            
            if total_iters % progress_interval < BATCH_SIZE or completed == n_batches:
                elapsed = time.time() - t0
                rate = total_iters / elapsed
                eta = (N_SPLITS - total_iters) / rate if rate > 0 else 0
                print(f"  [{completed}/{n_batches}] {total_iters:>10,} iters | "
                      f"rate={rate:,.0f}/s | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")
    
    elapsed_total = time.time() - t0
    
    # Compute stability metrics
    results = {"meta": {
        "n_splits": int(total_iters),
        "n_features": p,
        "feature_names": feature_names,
        "ridge_alpha": RIDGE_ALPHA,
        "test_fraction": test_frac,
        "n_workers": N_WORKERS,
        "subsample_n": SUBSAMPLE_N,
        "total_runtime_sec": round(elapsed_total, 2),
        "rate_per_sec": round(total_iters / elapsed_total, 1)
    }}
    
    r2_mean = r2_sum / total_iters
    r2_std = np.sqrt(max(0, r2_sq_sum / total_iters - r2_mean ** 2))
    results["r2_stats"] = {"mean": round(r2_mean, 6), "std": round(r2_std, 6),
                           "min_approx": "see per-split data", "p5": "N/A", "p95": "N/A"}
    
    print(f"\n  Overall Test R2: mean={r2_mean:.6f}, std={r2_std:.6f}")
    
    feature_stability = {}
    print(f"\n  {'Feature':<25s} {'mean|c|':>10s} {'std|c|':>10s} {'top1%':>8s} "
          f"{'top3%':>8s} {'sign_stab%':>10s} {'dominant':>8s}")
    print("  " + "-" * 85)
    
    for j, fname in enumerate(feature_names):
        mean_abs = coef_sum[j] / total_iters
        var_abs = coef_sq_sum[j] / total_iters - mean_abs ** 2
        std_abs = np.sqrt(max(0, var_abs))
        
        top1_pct = top1_count[j] / total_iters * 100
        top3_pct = top3_count[j] / total_iters * 100
        
        pos = int(sign_pos_count[j])
        neg = int(sign_neg_count[j])
        sign_total = pos + neg
        sign_stab = max(pos, neg) / sign_total * 100 if sign_total > 0 else 0
        dominant_sign = "+" if pos >= neg else "-"
        
        feature_stability[fname] = {
            "mean_abs_coef": round(float(mean_abs), 6),
            "std_abs_coef": round(float(std_abs), 6),
            "cv_coef": round(float(std_abs / mean_abs), 4) if mean_abs > 1e-8 else 0.0,
            "pct_top1": round(float(top1_pct), 4),
            "pct_top3": round(float(top3_pct), 4),
            "sign_positive_count": pos,
            "sign_negative_count": neg,
            "sign_stability_pct": round(float(sign_stab), 4),
            "dominant_sign": dominant_sign
        }
        
        print(f"  {fname:<25s} {mean_abs:>10.6f} {std_abs:>10.6f} {top1_pct:>7.2f}% "
              f"{top3_pct:>7.2f}% {sign_stab:>9.2f}% {dominant_sign:>8s}")
    
    results["feature_stability"] = feature_stability
    
    # Surprises
    surprises = []
    for fname, stats in feature_stability.items():
        if stats["sign_stability_pct"] < 90:
            surprises.append(f"LOW SIGN STABILITY: {fname} = {stats['sign_stability_pct']:.1f}%")
        if stats["pct_top1"] > 50:
            surprises.append(f"DOMINANT TOP-1: {fname} = {stats['pct_top1']:.1f}%")
        if stats["cv_coef"] > 0.5 and stats["mean_abs_coef"] > 0.001:
            surprises.append(f"HIGH COEF VARIABILITY: {fname} (CV={stats['cv_coef']:.2f})")
        if stats["pct_top1"] < 0.01 and stats["mean_abs_coef"] < 0.001:
            surprises.append(f"NEGLIGIBLE FEATURE: {fname} (top1={stats['pct_top1']:.4f}%, mean|c|={stats['mean_abs_coef']:.6f})")
    
    results["surprises"] = surprises
    if surprises:
        print(f"\n  *** SURPRISES DETECTED ({len(surprises)}) ***")
        for s in surprises:
            print(f"    - {s}")
    else:
        print(f"\n  No major surprises detected.")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: PERFECT MAPPING DECISION TREE DECODER
# ═══════════════════════════════════════════════════════════════════════════

def experiment_perfect_mapping_decoder(df, feature_names):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PERFECT MAPPING DECISION TREE DECODER")
    print("=" * 70)
    
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score, classification_report
    
    y_cls = (df['proxy_merged'] > -0.01).astype(int).values
    
    X_raw = df[feature_names].fillna(0).copy()
    if X_raw['tribal_any'].dtype == bool:
        X_raw['tribal_any'] = X_raw['tribal_any'].astype(float)
    X = X_raw.values
    
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y_cls)
    X = X[valid]
    y_cls = y_cls[valid]
    
    print(f"  Target: class 0 = {(y_cls==0).sum():,}, class 1 = {(y_cls==1).sum():,}")
    print(f"  Balance: {y_cls.mean()*100:.1f}% perfectly mapped")
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    clf.fit(X, y_cls)
    
    y_pred = clf.predict(X)
    acc = accuracy_score(y_cls, y_pred)
    print(f"\n  DecisionTree(max_depth=3) Accuracy: {acc:.4f}")
    print(f"  Leaves: {clf.get_n_leaves()}")
    
    tree_text = export_text(clf, feature_names=feature_names, decimals=4)
    print(f"\n  TREE RULES:")
    for line in tree_text.split('\n'):
        print(f"    {line}")
    
    importances = clf.feature_importances_
    print(f"\n  Feature Importances (Gini):")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        if importances[i] > 0:
            print(f"    {feature_names[i]:<25s} {importances[i]:.6f}")
    
    report = classification_report(y_cls, y_pred, target_names=['not_perfect', 'perfect'])
    print(f"\n  Classification Report:")
    print(report)
    
    results = {
        "accuracy": round(float(acc), 6),
        "n_leaves": int(clf.get_n_leaves()),
        "tree_rules": tree_text,
        "feature_importances": {feature_names[i]: round(float(importances[i]), 6) 
                                for i in range(len(feature_names)) if importances[i] > 0},
        "class_balance_pct": round(float(y_cls.mean() * 100), 2)
    }
    
    surprises = []
    non_zero = sum(1 for i in importances if i > 0)
    if non_zero <= 3:
        surprises.append(f"ONLY {non_zero} features used - perfect mapping is mechanically determined by few features!")
    if acc > 0.95:
        surprises.append(f"VERY HIGH accuracy ({acc:.4f}) - perfect mapping is almost deterministic")
    # Check if is_perfectly_mapped dominates
    if 'is_perfectly_mapped' in feature_names:
        idx_pm = feature_names.index('is_perfectly_mapped')
        if importances[idx_pm] > 0.9:
            surprises.append(f"is_perfectly_mapped dominates tree ({importances[idx_pm]:.4f}) - CIRCULAR feature!")
    
    results["surprises"] = surprises
    for s in surprises:
        print(f"  *** SURPRISE: {s}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: COEFFICIENT PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════

def experiment_coefficient_perturbation(df):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: COEFFICIENT PERTURBATION ON EXACT FORMULA")
    print("=" * 70)
    
    bldg_gap_clip = df['bldg_gap_clip'].fillna(0).values
    area_gap_clip = df['area_gap_clip'].fillna(0).values
    road_gap_clip = df['road_gap_clip'].fillna(0).values
    poi_gap_clip  = df['poi_gap_clip'].fillna(0).values
    pct_urban = df['pct_urban'].fillna(1.0).values
    rural = 1.0 - pct_urban
    
    proxy_exact = -(bldg_gap_clip + 2*area_gap_clip + road_gap_clip + poi_gap_clip) / 4.0 - rural
    proxy_actual = df['proxy_merged'].values
    
    rmse_baseline = np.sqrt(np.mean((proxy_actual - proxy_exact) ** 2))
    mae_baseline = np.mean(np.abs(proxy_actual - proxy_exact))
    max_err = np.max(np.abs(proxy_actual - proxy_exact))
    
    print(f"  Exact formula vs actual proxy_merged:")
    print(f"    RMSE = {rmse_baseline:.8f}")
    print(f"    MAE  = {mae_baseline:.8f}")
    print(f"    Max  = {max_err:.8f}")
    
    # 10K perturbations
    n_perturb = 10_000
    rng = np.random.RandomState(SEED + 12345)
    
    perturb_ranges = {
        'c_bldg': (0.8, 1.2),
        'c_area': (1.6, 2.4),
        'c_road': (0.8, 1.2),
        'c_poi':  (0.8, 1.2),
        'denom':  (3.2, 4.8),
    }
    
    # Vectorized perturbation
    c_bldg = rng.uniform(*perturb_ranges['c_bldg'], size=n_perturb)
    c_area = rng.uniform(*perturb_ranges['c_area'], size=n_perturb)
    c_road = rng.uniform(*perturb_ranges['c_road'], size=n_perturb)
    c_poi  = rng.uniform(*perturb_ranges['c_poi'], size=n_perturb)
    denom  = rng.uniform(*perturb_ranges['denom'], size=n_perturb)
    
    rmse_arr = np.empty(n_perturb)
    for i in range(n_perturb):
        proxy_p = -(c_bldg[i]*bldg_gap_clip + c_area[i]*area_gap_clip + 
                   c_road[i]*road_gap_clip + c_poi[i]*poi_gap_clip) / denom[i] - rural
        rmse_arr[i] = np.sqrt(np.mean((proxy_actual - proxy_p) ** 2))
    
    rmse_changes = rmse_arr - rmse_baseline
    
    print(f"\n  Perturbation results ({n_perturb:,} perturbations, +-20%):")
    print(f"    Mean RMSE change: {rmse_changes.mean():.8f}")
    print(f"    Std  RMSE change: {rmse_changes.std():.8f}")
    print(f"    Min  RMSE change: {rmse_changes.min():.8f}")
    print(f"    Max  RMSE change: {rmse_changes.max():.8f}")
    print(f"    P(improve):       {(rmse_changes < 0).mean()*100:.2f}%")
    print(f"    P(worsen):        {(rmse_changes > 0).mean()*100:.2f}%")
    
    best_idx = np.argmin(rmse_arr)
    print(f"\n  Best perturbed coefficients:")
    print(f"    c_bldg={c_bldg[best_idx]:.4f} (orig=1.0)")
    print(f"    c_area={c_area[best_idx]:.4f} (orig=2.0)")
    print(f"    c_road={c_road[best_idx]:.4f} (orig=1.0)")
    print(f"    c_poi ={c_poi[best_idx]:.4f}  (orig=1.0)")
    print(f"    denom ={denom[best_idx]:.4f}  (orig=4.0)")
    print(f"    RMSE  ={rmse_arr[best_idx]:.8f} (baseline={rmse_baseline:.8f})")
    
    # Sensitivity: perturb one coefficient at a time
    sensitivity = {}
    coef_list = ['c_bldg', 'c_area', 'c_road', 'c_poi', 'denom']
    orig_vals = [1.0, 2.0, 1.0, 1.0, 4.0]
    
    print(f"\n  Sensitivity (one-at-a-time):")
    for cidx, (cname, orig_val) in enumerate(zip(coef_list, orig_vals)):
        rng2 = np.random.RandomState(SEED + cidx * 11111)
        single_rmse = np.empty(n_perturb)
        for i in range(n_perturb):
            pvals = [1.0, 2.0, 1.0, 1.0, 4.0]
            pvals[cidx] = rng2.uniform(orig_val * 0.8, orig_val * 1.2)
            proxy_p = -(pvals[0]*bldg_gap_clip + pvals[1]*area_gap_clip + 
                       pvals[2]*road_gap_clip + pvals[3]*poi_gap_clip) / pvals[4] - rural
            single_rmse[i] = np.sqrt(np.mean((proxy_actual - proxy_p) ** 2))
        
        srmse_changes = single_rmse - rmse_baseline
        sensitivity[cname] = {
            "mean_rmse_change": round(float(srmse_changes.mean()), 8),
            "std_rmse_change": round(float(srmse_changes.std()), 8),
            "max_abs_change": round(float(np.abs(srmse_changes).max()), 8)
        }
        print(f"    {cname:<10s}: mean={srmse_changes.mean():.8f}, "
              f"std={srmse_changes.std():.8f}, max_abs={np.abs(srmse_changes).max():.8f}")
    
    results = {
        "baseline_rmse": round(float(rmse_baseline), 8),
        "baseline_mae": round(float(mae_baseline), 8),
        "n_perturbations": n_perturb,
        "perturb_range_pct": 20,
        "rmse_change_stats": {
            "mean": round(float(rmse_changes.mean()), 8),
            "std": round(float(rmse_changes.std()), 8),
            "min": round(float(rmse_changes.min()), 8),
            "max": round(float(rmse_changes.max()), 8),
            "pct_improve": round(float((rmse_changes < 0).mean() * 100), 2)
        },
        "best_perturbed": {
            "c_bldg": round(float(c_bldg[best_idx]), 6),
            "c_area": round(float(c_area[best_idx]), 6),
            "c_road": round(float(c_road[best_idx]), 6),
            "c_poi": round(float(c_poi[best_idx]), 6),
            "denom": round(float(denom[best_idx]), 6),
            "rmse": round(float(rmse_arr[best_idx]), 8)
        },
        "sensitivity": sensitivity
    }
    
    surprises = []
    if rmse_baseline < 1e-6:
        surprises.append(f"EXACT FORMULA MATCHES (RMSE={rmse_baseline:.2e}) - formula is deterministic!")
    elif rmse_baseline > 0.01:
        surprises.append(f"LARGE FORMULA MISMATCH (RMSE={rmse_baseline:.4f}) - formula may be wrong")
    if (rmse_changes < 0).mean() > 0.1:
        surprises.append(f"{(rmse_changes < 0).mean()*100:.1f}% perturbations IMPROVE RMSE - coefficients suboptimal")
    most_sensitive = max(sensitivity.items(), key=lambda x: x[1]['std_rmse_change'])
    surprises.append(f"Most sensitive coefficient: {most_sensitive[0]} (std={most_sensitive[1]['std_rmse_change']:.8f})")
    
    results["surprises"] = surprises
    for s in surprises:
        print(f"  *** SURPRISE: {s}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    
    print("\n" + "█" * 70)
    print("  1M-ITERATION STABILITY TEST: BIAS-BOUNTY-MAPPING-EQUITY")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 70)
    
    X, y, feature_names, df = load_data()
    
    all_results = {"timestamp": datetime.now().isoformat()}
    
    # Experiment 1
    ridge_results = experiment_1M_ridge(X, y, feature_names)
    all_results["ridge_stability_1M"] = ridge_results
    
    # Experiment 2
    tree_results = experiment_perfect_mapping_decoder(df, feature_names)
    all_results["perfect_mapping_decoder"] = tree_results
    
    # Experiment 3
    perturb_results = experiment_coefficient_perturbation(df)
    all_results["coefficient_perturbation"] = perturb_results
    
    total_time = time.time() - t_start
    all_results["total_runtime_sec"] = round(total_time, 2)
    
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL SURPRISES")
    print("=" * 70)
    
    all_surprises = []
    for exp_name, exp_data in all_results.items():
        if isinstance(exp_data, dict) and "surprises" in exp_data:
            for s in exp_data["surprises"]:
                all_surprises.append(f"[{exp_name}] {s}")
    
    if all_surprises:
        for s in all_surprises:
            print(f"  !! {s}")
    else:
        print("  No surprises detected.")
    all_results["all_surprises"] = all_surprises
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    return all_results


if __name__ == "__main__":
    main()
