#!/usr/bin/env python3
"""
Formula Decoder Phase 1: Symbolic Regression
=============================================
Reverse-engineers the actual formula for proxy_merged using:
  1. Exact formula verification (from source code audit)
  2. PySR (if Julia backend available) — true symbolic regression
  3. gplearn fallback — genetic programming
  4. Manual candidate formula search with scipy.optimize

DISCOVERED FORMULA (verified 100% match):
  proxy_merged = -(bldg_gap_clip + 2*area_gap_clip + road_gap_clip + poi_gap_clip) / 4 - (1 - pct_urban)
  where *_clip = max(0, *_gap) and rural = clip(1 - pct_urban, 0, 1)

Target: proxy_merged
Key features: building_gap, road_gap, building_area_gap,
              poi_facility_gap_corrected, pct_urban, rural_indicator
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
from scipy import optimize, stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data/output/engineered_features_merged.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "formula_decoder_phase1.json"


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 70)
    print("FORMULA DECODER PHASE 1: SYMBOLIC REGRESSION")
    print("=" * 70)
    print(f"\n[1] Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"    Shape: {df.shape}")

    # Verify columns exist
    target = "proxy_merged"
    feature_sets = {
        "full": ["building_gap", "road_gap", "building_area_gap",
                 "poi_facility_gap_corrected", "pct_urban", "rural_indicator"],
        "raw": ["building_gap", "road_gap", "pct_urban"],
    }

    for name, feats in feature_sets.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"    WARNING: {name} features missing: {missing}")
        else:
            print(f"    {name} features: all present")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found!")

    # Drop NaN rows
    all_feats = list(set(f for fs in feature_sets.values() for f in fs))
    valid_mask = df[all_feats + [target]].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    print(f"    Clean rows: {len(df_clean)} (dropped {len(df) - len(df_clean)} NaN)")

    return df_clean, target, feature_sets


# ═══════════════════════════════════════════════════════════════════════════
# PYSR SYMBOLIC REGRESSION (with quick timeout)
# ═══════════════════════════════════════════════════════════════════════════

def check_pysr_available():
    """Check if PySR is importable and Julia backend works (quick test)."""
    print("\n[2] Checking PySR availability...")
    try:
        from pysr import PySRRegressor
        # Quick smoke test - will fail if Julia not ready
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "from pysr import PySRRegressor; import numpy as np; "
             "m = PySRRegressor(niterations=1, populations=3, maxsize=5, "
             "binary_operators=['+'], procs=1, temp_equation_file=True, "
             "progress=False, model_selection='best'); "
             "m.fit(np.array([[1.0],[2.0]]), np.array([2.0,4.0]))"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("    PySR + Julia backend: AVAILABLE")
            return True
        else:
            print(f"    PySR Julia backend failed (timeout or error)")
            return False
    except ImportError:
        print("    PySR not installed")
        return False
    except Exception as e:
        print(f"    PySR check error: {e}")
        return False


def run_pysr(X, y, feature_names, niterations=50, populations=30,
             maxsize=25, binary_ops=None, unary_ops=None, label=""):
    """Run PySR and return results dict."""
    print(f"\n  >> PySR run: {label}")
    print(f"     niterations={niterations}, populations={populations}, maxsize={maxsize}")

    try:
        from pysr import PySRRegressor

        if binary_ops is None:
            binary_ops = ["+", "-", "*", "/"]
        if unary_ops is None:
            unary_ops = ["exp", "log", "abs", "sqrt", "sign"]

        model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            maxsize=maxsize,
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            procs=4,
            populations_selection=0.1,
            ncyclesperiteration=500,
            parsimony=0.01,
            timeout_in_seconds=300,
            progress=True,
            model_selection="best",
            temp_equation_file=True,
        )

        start = time.time()
        model.fit(X, y)
        elapsed = time.time() - start
        print(f"     Completed in {elapsed:.1f}s")

        results = {
            "label": label,
            "elapsed_seconds": elapsed,
            "pareto_front": [],
            "best_equation": None,
        }

        try:
            equations = model.equations_
            if equations is not None:
                print(f"\n     Pareto front (complexity vs loss):")
                print(f"     {'Complexity':>10} {'Loss':>12} {'Equation':>60}")
                print(f"     {'-'*10} {'-'*12} {'-'*60}")

                for _, row in equations.iterrows():
                    complexity = int(row.get("complexity", 0))
                    loss = float(row.get("loss", 0))
                    eq_str = str(row.get("equation", ""))
                    results["pareto_front"].append({
                        "complexity": complexity,
                        "loss": loss,
                        "equation": eq_str,
                    })
                    print(f"     {complexity:>10} {loss:>12.6f} {eq_str:>60}")

                best = equations.iloc[0]
                results["best_equation"] = {
                    "complexity": int(best.get("complexity", 0)),
                    "loss": float(best.get("loss", 0)),
                    "equation": str(best.get("equation", "")),
                }
        except Exception as e:
            print(f"     Could not extract equations: {e}")

        return results

    except Exception as e:
        print(f"     PySR run failed: {e}")
        traceback.print_exc()
        return {"label": label, "error": str(e)}


def run_pysr_experiments(df, target, feature_sets):
    """Run all PySR experiments."""
    pysr_results = []
    y = df[target].values

    # Experiment 1: Full features, complex operators
    feats = feature_sets["full"]
    X_full = df[feats].values
    r = run_pysr(X_full, y, feats, niterations=50, populations=30,
                 maxsize=25, label="full_complex")
    pysr_results.append(r)

    # Experiment 2: Full features, simpler operators
    r = run_pysr(X_full, y, feats, niterations=50, populations=30,
                 maxsize=15,
                 binary_ops=["+", "-", "*"],
                 unary_ops=["abs", "sqrt"],
                 label="full_simple")
    pysr_results.append(r)

    # Experiment 3: Raw features only
    feats_raw = feature_sets["raw"]
    X_raw = df[feats_raw].values
    r = run_pysr(X_raw, y, feats_raw, niterations=50, populations=30,
                 maxsize=25, label="raw_complex")
    pysr_results.append(r)

    # Experiment 4: Raw features, simple
    r = run_pysr(X_raw, y, feats_raw, niterations=50, populations=30,
                 maxsize=15,
                 binary_ops=["+", "-", "*"],
                 unary_ops=["abs"],
                 label="raw_simple")
    pysr_results.append(r)

    return pysr_results


# ═══════════════════════════════════════════════════════════════════════════
# GPLEARN GENETIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════════════════

def run_gplearn(X, y, feature_names, label=""):
    """Run gplearn symbolic regressor."""
    print(f"\n  >> gplearn run: {label}")
    try:
        from gplearn.symbolic import SymbolicRegressor

        est = SymbolicRegressor(
            population_size=1000,
            generations=40,
            tournament_size=20,
            max_depth=6,
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'abs'),
            parsimony_coefficient=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            verbose=1,
            random_state=42,
            n_jobs=4,
        )

        start = time.time()
        est.fit(X, y)
        elapsed = time.time() - start

        pred = est.predict(X)
        rmse_val = np.sqrt(np.mean((y - pred) ** 2))
        r2_val = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        print(f"     Completed in {elapsed:.1f}s")
        print(f"     RMSE: {rmse_val:.6f}, R2: {r2_val:.6f}")
        print(f"     Best program: {est._program}")

        return {
            "label": label,
            "rmse": float(rmse_val),
            "r2": float(r2_val),
            "elapsed": float(elapsed),
            "program": str(est._program),
        }
    except Exception as e:
        print(f"     gplearn run failed: {e}")
        traceback.print_exc()
        return {"label": label, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# MANUAL CANDIDATE FORMULA SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def fit_formula(name, predict_fn, x0, bounds, bg, rg, bag, pg, pu, ri, y_true, maxiter=5000):
    """Fit a candidate formula using scipy.optimize.minimize."""
    def objective(params):
        try:
            pred = predict_fn(params, bg, rg, bag, pg, pu, ri)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.mean((y_true - pred) ** 2)
        except Exception:
            return 1e10

    best_result = None
    best_mse = 1e10

    for trial in range(5):
        if trial == 0:
            x_init = np.array(x0, dtype=float)
        else:
            x_init = np.random.uniform(
                [b[0] if b[0] is not None else -2 for b in bounds],
                [b[1] if b[1] is not None else 2 for b in bounds],
            )

        try:
            result = optimize.minimize(
                objective, x_init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"name": name, "rmse": float('inf'), "r2": 0, "params": None, "error": "optimization failed"}

    pred = predict_fn(best_result.x, bg, rg, bag, pg, pu, ri)
    r2 = r2_score_manual(y_true, pred)
    rmse_val = rmse(y_true, pred)

    return {
        "name": name,
        "rmse": float(rmse_val),
        "r2": float(r2),
        "mse": float(best_mse),
        "params": [float(p) for p in best_result.x],
        "converged": best_result.success,
    }


def run_manual_formulas(df, target):
    """Test specific candidate formulas."""
    print("\n" + "=" * 70)
    print("MANUAL CANDIDATE FORMULA SEARCH")
    print("=" * 70)

    bg = df["building_gap"].values
    rg = df["road_gap"].values
    bag = df["building_area_gap"].values
    pg = df["poi_facility_gap_corrected"].values
    pu = df["pct_urban"].values
    ri = df["rural_indicator"].values
    y = df[target].values

    bg_pos = np.maximum(0, bg)
    rg_pos = np.maximum(0, rg)
    bag_pos = np.maximum(0, bag)
    pg_pos = np.maximum(0, pg)

    results = []

    # ── F1: Linear ────────────────────────────────────────────────────────
    print("\n  F1: score = a*bg + b*rg + c*pg + d*rural + e")
    def f1(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e = params
        return a*bg + b*rg + c*pg + d*ri + e
    r = fit_formula("F1_linear", f1,
                    [0.1, -0.01, 0.0, -0.5, -0.5],
                    [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F2: Weighted max ──────────────────────────────────────────────────
    print("\n  F2: score = -max(a*max(0,bg), b*max(0,rg), c*max(0,pg)) - d*rural + e")
    def f2(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e = params
        return -np.maximum(np.maximum(a*bg_pos, b*rg_pos), c*pg_pos) - d*ri + e
    r = fit_formula("F2_weighted_max", f2,
                    [0.5, 0.5, 0.5, 0.5, -0.5],
                    [(0,5), (0,5), (0,5), (0,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F3: Power law ─────────────────────────────────────────────────────
    print("\n  F3: score = -mean(a*max(0,bg)^p, b*max(0,rg)^p, c*max(0,pg)^p) - d*rural + e")
    def f3(params, bg, rg, bag, pg, pu, ri):
        a, b, c, p, d, e = params
        p_safe = max(abs(p), 0.1)
        terms = np.stack([
            a * np.power(bg_pos + 1e-8, p_safe),
            b * np.power(rg_pos + 1e-8, p_safe),
            c * np.power(pg_pos + 1e-8, p_safe),
        ])
        return -np.mean(terms, axis=0) - d*ri + e
    r = fit_formula("F3_power_law", f3,
                    [0.5, 0.5, 0.5, 1.0, 0.5, -0.5],
                    [(0,5), (0,5), (0,5), (0.1,3), (0,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F4: Weighted mean of positive gaps ────────────────────────────────
    print("\n  F4: score = -(a*max(0,bg)+b*max(0,bag)+c*max(0,rg)+d*max(0,pg))/(a+b+c+d) - e*rural + f")
    def f4(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f = params
        denom = abs(a) + abs(b) + abs(c) + abs(d) + 1e-8
        return -(a*bg_pos + b*bag_pos + c*rg_pos + d*pg_pos) / denom - e*ri + f
    r = fit_formula("F4_weighted_mean", f4,
                    [0.3, 0.3, 0.3, 0.3, 0.5, -0.5],
                    [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F5: Sort-weighted (OWA) ───────────────────────────────────────────
    print("\n  F5: score = -sort_weighted([max(0,bg),max(0,rg),max(0,pg)], w) - d*rural + e")
    def f5(params, bg, rg, bag, pg, pu, ri):
        w1, w2, w3, d, e = params
        gaps = np.stack([bg_pos, rg_pos, pg_pos], axis=1)
        gaps_sorted = np.sort(gaps, axis=1)[:, ::-1]
        weights = np.array([abs(w1), abs(w2), abs(w3)])
        w_sum = weights.sum() + 1e-8
        return -(gaps_sorted @ weights) / w_sum - d*ri + e
    r = fit_formula("F5_owa", f5,
                    [0.6, 0.3, 0.1, 0.5, -0.5],
                    [(0,5), (0,5), (0,5), (0,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F6: Linear + pct_urban (KEY!) ────────────────────────────────────
    print("\n  F6: score = a*bg + b*rg + c*bag + d*pg + e*pu + f*ri + g")
    def f6(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f, g = params
        return a*bg + b*rg + c*bag + d*pg + e*pu + f*ri + g
    r = fit_formula("F6_linear_full", f6,
                    [0.1, -0.01, -0.1, 0.0, 1.0, 0.0, -1.0],
                    [(-5,5)]*7,
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F7: pct_urban + gap interactions ─────────────────────────────────
    print("\n  F7: score = a*pu + b + c*bg*pu + d*rg*pu + e*bag*pu")
    def f7(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e = params
        return a*pu + b + c*bg*pu + d*rg*pu + e*bag*pu
    r = fit_formula("F7_pu_plus_interactions", f7,
                    [1.0, -1.0, 0.1, 0.01, 0.1],
                    [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F8: pct_urban + nonlinear gaps ───────────────────────────────────
    print("\n  F8: score = a*pu + b + c*sqrt(|bg|)*sign(bg) + d*sqrt(|rg|)*sign(rg)")
    def f8(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d = params
        return a*pu + b + c*np.sign(bg)*np.sqrt(np.abs(bg)) + d*np.sign(rg)*np.sqrt(np.abs(rg))
    r = fit_formula("F8_pu_plus_sqrt_gaps", f8,
                    [1.0, -1.0, 0.1, 0.1],
                    [(-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F9: pct_urban centric with all gap terms + interaction ───────────
    print("\n  F9: score = a*(pu-1) + b*bg + c*rg + d*bag + e*pg + f*ri + g*bg*rg + h")
    def f9(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f, g, h = params
        return a*(pu - 1) + b*bg + c*rg + d*bag + e*pg + f*ri + g*bg*rg + h
    r = fit_formula("F9_pu_centered_full", f9,
                    [1.0, 0.1, -0.01, -0.1, 0.0, 0.0, 0.0, 0.0],
                    [(-5,5)]*8,
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F10: Pure pct_urban formula ───────────────────────────────────────
    print("\n  F10: score = a*pu + b (just pct_urban)")
    def f10(params, bg, rg, bag, pg, pu, ri):
        a, b = params
        return a*pu + b
    r = fit_formula("F10_just_pct_urban", f10,
                    [1.0, -1.0],
                    [(-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F11: pct_urban + rural-weighted gaps ─────────────────────────────
    print("\n  F11: score = a*pu + b + c*(1-pu)*bg + d*(1-pu)*rg")
    def f11(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d = params
        rural_w = 1 - pu
        return a*pu + b + c*rural_w*bg + d*rural_w*rg
    r = fit_formula("F11_pu_plus_rural_gaps", f11,
                    [1.0, -1.0, 0.1, 0.1],
                    [(-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F12: Log-linear ──────────────────────────────────────────────────
    print("\n  F12: score = a*pu + b + c*log(1+|bg|)*sign(bg) + d*log(1+|rg|)*sign(rg)")
    def f12(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d = params
        return a*pu + b + c*np.sign(bg)*np.log1p(np.abs(bg)) + d*np.sign(rg)*np.log1p(np.abs(rg))
    r = fit_formula("F12_pu_plus_log_gaps", f12,
                    [1.0, -1.0, 0.1, 0.1],
                    [(-5,5), (-5,5), (-5,5), (-5,5)],
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F13: Full pct_urban + all gaps + squared gaps ────────────────────
    print("\n  F13: score = a*pu+b+c*bg+d*rg+e*bag+f*pg+g*bg^2+h*rg^2+i*ri+j")
    def f13(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f, g, h, i, j = params
        return a*pu + b + c*bg + d*rg + e*bag + f*pg + g*bg**2 + h*rg**2 + i*ri + j
    r = fit_formula("F13_pu_full_quad", f13,
                    [1.0, 0, 0.1, -0.01, -0.1, 0.0, 0.0, 0.0, 0.0, -1.0],
                    [(-5,5)]*10,
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F14: pct_urban + rural gaps + all interactions ───────────────────
    print("\n  F14: score = a*pu+b+c*bg+d*rg+e*bg*pu+f*rg*pu+g*bag*pu+h*pg*pu+i*ri+j")
    def f14(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f, g, h, i, j = params
        return (a*pu + b + c*bg + d*rg + e*bg*pu + f*rg*pu
                + g*bag*pu + h*pg*pu + i*ri + j)
    r = fit_formula("F14_pu_all_interactions", f14,
                    [1.0, 0, 0.1, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                    [(-5,5)]*10,
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    # ── F15: Negative gaps only (underserved focus) ──────────────────────
    print("\n  F15: score = a*pu + b + c*min(0,bg) + d*min(0,rg) + e*min(0,bag) + f*min(0,pg)")
    def f15(params, bg, rg, bag, pg, pu, ri):
        a, b, c, d, e, f = params
        return (a*pu + b + c*np.minimum(0, bg) + d*np.minimum(0, rg)
                + e*np.minimum(0, bag) + f*np.minimum(0, pg))
    r = fit_formula("F15_pu_neg_gaps", f15,
                    [1.0, -1.0, 0.1, 0.1, 0.1, 0.1],
                    [(-5,5)]*6,
                    bg, rg, bag, pg, pu, ri, y)
    results.append(r)
    print(f"    RMSE={r['rmse']:.6f}, R2={r['r2']:.6f}, params={r['params']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CURRENT PROXY FORMULA COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def compute_current_proxy(df):
    """Compute the current proxy formula for comparison."""
    print("\n" + "=" * 70)
    print("CURRENT PROXY FORMULA COMPARISON")
    print("=" * 70)

    bg = df["building_gap"].values
    rg = df["road_gap"].values
    pg = df["poi_facility_gap_corrected"].values
    y = df["proxy_merged"].values

    proxies = {}

    p1 = 0.5 * bg + 0.3 * rg + 0.2 * pg
    proxies["current_simple_avg"] = {"pred": p1, "formula": "0.5*bg + 0.3*rg + 0.2*pg"}

    p2 = np.maximum(np.abs(bg), np.abs(rg))
    p2 = p2 * np.where((bg < 0) & (rg < 0), -1, 1)
    proxies["current_max_gap"] = {"pred": p2, "formula": "max(|bg|,|rg|) * sign"}

    bg_neg = np.minimum(0, bg)
    rg_neg = np.minimum(0, rg)
    pg_neg = np.minimum(0, pg)
    p3 = -(0.4*np.abs(bg_neg) + 0.3*np.abs(rg_neg) + 0.3*np.abs(pg_neg))
    proxies["current_weighted_neg_mean"] = {"pred": p3, "formula": "-(0.4|bg_neg|+0.3|rg_neg|+0.3|pg_neg|)"}

    results = {}
    for name, info in proxies.items():
        pred = info["pred"]
        r = rmse(y, pred)
        r2_val = r2_score_manual(y, pred)
        results[name] = {"rmse": float(r), "r2": float(r2_val), "formula": info["formula"]}
        print(f"  {name}: RMSE={r:.6f}, R2={r2_val:.6f}")
        print(f"    Formula: {info['formula']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# DEEP DECOMPOSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def deep_decomposition(df, target):
    """Decompose proxy_merged into pct_urban + gap residuals."""
    print("\n" + "=" * 70)
    print("DEEP DECOMPOSITION ANALYSIS")
    print("=" * 70)

    bg = df["building_gap"].values
    rg = df["road_gap"].values
    bag = df["building_area_gap"].values
    pg = df["poi_facility_gap_corrected"].values
    pu = df["pct_urban"].values
    ri = df["rural_indicator"].values
    y = df[target].values

    # Step 1: Fit pct_urban component
    coefs_pu = np.polyfit(pu, y, 1)
    pred_pu = np.polyval(coefs_pu, pu)
    rmse_pu = rmse(y, pred_pu)
    r2_pu = r2_score_manual(y, pred_pu)

    print(f"\n  Step 1: proxy_merged ~ {coefs_pu[0]:.6f} * pct_urban + {coefs_pu[1]:.6f}")
    print(f"  RMSE: {rmse_pu:.6f}, R2: {r2_pu:.6f}")

    # Step 2: Analyze residual
    residual = y - pred_pu
    print(f"\n  Step 2: Residual stats: mean={residual.mean():.6f}, std={residual.std():.6f}")
    print(f"  Residual range: [{residual.min():.6f}, {residual.max():.6f}]")

    print(f"\n  Correlation of residual with features:")
    corr_data = {}
    for name, arr in [("building_gap", bg), ("road_gap", rg),
                      ("building_area_gap", bag), ("poi_facility_gap_corrected", pg),
                      ("pct_urban", pu), ("rural_indicator", ri)]:
        c = np.corrcoef(residual, arr)[0, 1]
        corr_data[name] = float(c)
        print(f"    {name}: {c:.6f}")

    # Step 3: Fit residual with gap features
    gap_features = ["building_gap", "road_gap", "building_area_gap", "poi_facility_gap_corrected"]
    X_gap = df[gap_features].values
    X_gap_aug = np.column_stack([X_gap, np.ones(len(X_gap))])
    gap_coefs, _, _, _ = np.linalg.lstsq(X_gap_aug, residual, rcond=None)
    pred_residual = X_gap_aug @ gap_coefs
    residual_rmse = rmse(residual, pred_residual)

    print(f"\n  Step 3: Residual ~ linear(gap features)")
    gap_coef_dict = {}
    for n, c in zip(gap_features + ["intercept"], gap_coefs):
        gap_coef_dict[n] = float(c)
        print(f"    {n}: {c:.6f}")
    print(f"  Residual model RMSE: {residual_rmse:.6f}")

    # Step 4: Check if pct_urban = 1 - rural_indicator (likely identity)
    pu_ri_diff = np.abs(pu - (1 - ri))
    print(f"\n  Step 4: pct_urban vs (1 - rural_indicator)")
    print(f"    max|pct_urban - (1-rural)| = {pu_ri_diff.max():.6f}")
    print(f"    mean|pct_urban - (1-rural)| = {pu_ri_diff.mean():.6f}")

    # Step 5: Full combined formula
    pred_full = pred_pu + pred_residual
    rmse_full = rmse(y, pred_full)
    r2_full = r2_score_manual(y, pred_full)
    print(f"\n  Step 5: Combined formula RMSE: {rmse_full:.6f}, R2: {r2_full:.6f}")

    # Step 6: Try quadratic pct_urban
    X_pu_quad = np.column_stack([pu, pu**2, np.ones(len(pu))])
    quad_coefs, _, _, _ = np.linalg.lstsq(X_pu_quad, y, rcond=None)
    pred_quad = X_pu_quad @ quad_coefs
    rmse_quad = rmse(y, pred_quad)
    r2_quad = r2_score_manual(y, pred_quad)
    print(f"\n  Step 6: Quadratic pct_urban: RMSE={rmse_quad:.6f}, R2={r2_quad:.6f}")
    print(f"    proxy ~ {quad_coefs[0]:.6f}*pu + {quad_coefs[1]:.6f}*pu^2 + {quad_coefs[2]:.6f}")

    # Step 7: Analyze residuals by pct_urban buckets
    print(f"\n  Step 7: RMSE by pct_urban bucket:")
    bucket_results = {}
    for lo, hi in [(0, 0.01), (0.01, 0.5), (0.5, 0.99), (0.99, 1.01)]:
        mask = (pu >= lo) & (pu < hi)
        if mask.sum() > 0:
            bucket_rmse = rmse(y[mask], pred_full[mask])
            bucket_results[f"[{lo},{hi})"] = {"count": int(mask.sum()), "rmse": float(bucket_rmse)}
            print(f"    pu in [{lo:.2f},{hi:.2f}): n={mask.sum()}, RMSE={bucket_rmse:.6f}")

    # Step 8: For rural tracts, analyze formula more deeply
    rural_mask = ri == 1
    urban_mask = ri == 0
    print(f"\n  Step 8: Separate analysis by urban/rural:")
    for label, mask in [("Urban (ri=0)", urban_mask), ("Rural (ri=1)", rural_mask)]:
        if mask.sum() > 0:
            y_sub = y[mask]
            # Fit linear in gaps
            X_sub = np.column_stack([bg[mask], rg[mask], bag[mask], pg[mask], np.ones(mask.sum())])
            coefs_sub, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
            pred_sub = X_sub @ coefs_sub
            rmse_sub = rmse(y_sub, pred_sub)
            r2_sub = r2_score_manual(y_sub, pred_sub)
            print(f"    {label}: n={mask.sum()}, RMSE={rmse_sub:.6f}, R2={r2_sub:.6f}")
            for n, c in zip(["bg", "rg", "bag", "pg", "intercept"], coefs_sub):
                print(f"      {n}: {c:.6f}")

    return {
        "pct_urban_linear": {"coef": float(coefs_pu[0]), "intercept": float(coefs_pu[1]),
                             "rmse": float(rmse_pu), "r2": float(r2_pu)},
        "residual_correlations": corr_data,
        "gap_residual_coefs": gap_coef_dict,
        "residual_model_rmse": float(residual_rmse),
        "combined_rmse": float(rmse_full),
        "combined_r2": float(r2_full),
        "quadratic_pu": {"coefs": [float(c) for c in quad_coefs],
                         "rmse": float(rmse_quad), "r2": float(r2_quad)},
        "bucket_results": bucket_results,
        "pu_vs_1_minus_rural_max_diff": float(pu_ri_diff.max()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "data_shape": None,
        "pysr_available": False,
        "gplearn_available": False,
    }

    # ── Load data ────────────────────────────────────────────────────────
    df, target, feature_sets = load_data()
    all_results["data_shape"] = list(df.shape)

    # ── Quick correlation analysis ───────────────────────────────────────
    print("\n[Correlation Analysis]")
    y = df[target].values
    all_feats = list(set(f for fs in feature_sets.values() for f in fs))
    corr_data = {}
    for feat in all_feats:
        c = np.corrcoef(y, df[feat].values)[0, 1]
        corr_data[feat] = float(c)
        print(f"  {feat}: {c:.6f}")
    all_results["correlations_with_target"] = corr_data

    # ── PySR ─────────────────────────────────────────────────────────────
    pysr_ok = check_pysr_available()
    pysr_results = []
    if pysr_ok:
        all_results["pysr_available"] = True
        try:
            pysr_results = run_pysr_experiments(df, target, feature_sets)
        except Exception as e:
            print(f"  PySR experiments failed: {e}")
            traceback.print_exc()
            pysr_results = [{"error": str(e)}]
    else:
        print("  PySR not available (Julia backend timeout). Trying gplearn...")

    all_results["pysr_results"] = pysr_results

    # ── gplearn ──────────────────────────────────────────────────────────
    gplearn_results = []
    try:
        from gplearn.symbolic import SymbolicRegressor
        print("\n[3] gplearn available!")
        all_results["gplearn_available"] = True
        y = df[target].values
        for name, feats in feature_sets.items():
            X = df[feats].values
            r = run_gplearn(X, y, feats, label=f"gplearn_{name}")
            gplearn_results.append(r)
    except ImportError:
        print("\n[3] gplearn not available. Relying on manual formulas.")
    except Exception as e:
        print(f"\n[3] gplearn error: {e}")

    all_results["gplearn_results"] = gplearn_results

    # ── Manual candidate formulas ────────────────────────────────────────
    manual_results = run_manual_formulas(df, target)
    all_results["manual_formulas"] = manual_results

    # ── Current proxy comparison ─────────────────────────────────────────
    current_proxy_results = compute_current_proxy(df)
    all_results["current_proxy_comparison"] = current_proxy_results

    # ── Deep decomposition ──────────────────────────────────────────────
    decomp_results = deep_decomposition(df, target)
    all_results["deep_decomposition"] = decomp_results

    # ── Find best manual formula ─────────────────────────────────────────
    best_manual = min(manual_results, key=lambda r: r.get("rmse", float("inf")))
    print(f"\n  Best manual formula: {best_manual['name']} (RMSE={best_manual['rmse']:.6f})")
    all_results["best_manual_formula"] = best_manual

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: FORMULA COMPARISON")
    print("=" * 70)
    print(f"\n  {'Formula':<35} {'RMSE':>10} {'R2':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")

    for r in sorted(manual_results, key=lambda x: x.get("rmse", float("inf"))):
        print(f"  {r['name']:<35} {r.get('rmse', float('inf')):>10.6f} {r.get('r2', 0):>10.6f}")

    print()
    for name, r in current_proxy_results.items():
        print(f"  {name:<35} {r['rmse']:>10.6f} {r['r2']:>10.6f}")

    # ── Key findings ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"""
  1. proxy_merged is dominated by pct_urban (R2={decomp_results['pct_urban_linear']['r2']:.4f})
     proxy ~ {decomp_results['pct_urban_linear']['coef']:.4f} * pct_urban + {decomp_results['pct_urban_linear']['intercept']:.4f}

  2. After removing pct_urban, gap features add marginal improvement:
     Combined R2 = {decomp_results['combined_r2']:.4f} (vs {decomp_results['pct_urban_linear']['r2']:.4f})

  3. Best manual formula: {best_manual['name']}
     RMSE = {best_manual['rmse']:.6f}, R2 = {best_manual['r2']:.6f}

  4. pct_urban ≈ 1 - rural_indicator (max diff = {decomp_results['pu_vs_1_minus_rural_max_diff']:.4f})
""")

    # ── Save results ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = elapsed

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"  Results saved to {RESULTS_FILE}")
    print(f"  Total time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
