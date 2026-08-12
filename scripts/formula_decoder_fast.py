#!/usr/bin/env python3
"""FAST formula decoder - all 7 experiments under 5 minutes."""
import time, json, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/z/my-project/bias-bounty-map")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

t0 = time.time()

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
df = pd.read_parquet(ROOT / "data/output/engineered_features_merged.parquet")
print(f"  Shape: {df.shape}  ({time.time()-t0:.1f}s)", flush=True)

# Key columns
bg  = df['bldg_gap_clip'].values
bag = df['area_gap_clip'].values
rg  = df['road_gap_clip'].values
pg  = df['poi_gap_clip'].values
pu  = df['pct_urban'].values
pm  = df['proxy_merged'].values

results = {}

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Verify the exact formula
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 1: Verify exact formula ===", flush=True)
t1 = time.time()
reconstructed = -(bg + 2*bag + rg + pg) / 4 - (1 - pu)
residuals = pm - reconstructed
rmse = np.sqrt(np.mean(residuals**2))
max_err = np.max(np.abs(residuals))
mean_err = np.mean(np.abs(residuals))
zero_gap = np.sum((bg == 0) & (bag == 0) & (rg == 0) & (pg == 0))
pct_zero = zero_gap / len(pm) * 100

exp1 = {
    "rmse": float(rmse),
    "max_error": float(max_err),
    "mean_abs_error": float(mean_err),
    "n_zero_gap": int(zero_gap),
    "pct_zero_gap": float(pct_zero),
    "formula": "proxy_merged = -(bldg_gap_clip + 2*area_gap_clip + road_gap_clip + poi_gap_clip)/4 - (1-pct_urban)",
    "verified": bool(rmse < 1e-6)
}
results["exp1_formula_verify"] = exp1
print(f"  RMSE={rmse:.2e}, MaxErr={max_err:.2e}, MeanAE={mean_err:.2e}")
print(f"  Zero-gap tracts: {zero_gap} ({pct_zero:.1f}%)")
print(f"  Formula VERIFIED: {exp1['verified']}  ({time.time()-t1:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: FAST stability test (100K iterations)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 2: Stability test (100K iters, 2K subsample) ===", flush=True)
t2 = time.time()

# Subsample and features
np.random.seed(42)
sub_idx = np.random.choice(len(pm), size=2000, replace=False)

feat_names = ['bldg_gap_clip', 'area_gap_clip', 'road_gap_clip', 'poi_gap_clip', 'pct_urban',
              'building_gap', 'building_area_gap', 'road_gap', 'poi_facility_gap_corrected',
              'bldg_x_area_gap']
X_sub = df[feat_names].values[sub_idx].astype(np.float64)
y_sub = pm[sub_idx]

# Standardize
X_mean = X_sub.mean(axis=0)
X_std = X_sub.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X_sub - X_mean) / X_std

# Precompute X'X per row (outer products)
# For Ridge: coef = (X'X + λI)^{-1} X'y
# We'll accumulate X'X and X'y for train subset
N_ITER = 100_000
ALPHA = 1.0  # Ridge penalty

coef_accum = np.zeros((N_ITER, len(feat_names)))
n = X_norm.shape[0]
train_size = int(0.8 * n)

for i in range(N_ITER):
    perm = np.random.permutation(n)
    train_idx = perm[:train_size]
    Xt = X_norm[train_idx]
    yt = y_sub[train_idx]
    
    XtX = Xt.T @ Xt + ALPHA * np.eye(len(feat_names))
    Xty = Xt.T @ yt
    coef_accum[i] = np.linalg.solve(XtX, Xty)

# Analyze stability
mean_coef = np.abs(coef_accum).mean(axis=0)
std_coef = coef_accum.std(axis=0)
sign_stable = (np.mean(coef_accum > 0, axis=0).clip(0.001, 0.999))
sign_stability = 2 * np.abs(sign_stable - 0.5) * 100  # % stable

# Top-1 frequency
abs_coefs = np.abs(coef_accum)
top1_idx = np.argmax(abs_coefs, axis=1)
top1_counts = np.bincount(top1_idx, minlength=len(feat_names))
top1_pct = top1_counts / N_ITER * 100

exp2 = {
    "n_iterations": N_ITER,
    "subsample_size": 2000,
    "features": {},
}
for j, fn in enumerate(feat_names):
    exp2["features"][fn] = {
        "mean_abs_coef": float(mean_coef[j]),
        "std_coef": float(std_coef[j]),
        "sign_stability_pct": float(sign_stability[j]),
        "top1_pct": float(top1_pct[j]),
        "mean_coef_signed": float(coef_accum[:, j].mean()),
    }
results["exp2_stability"] = exp2

for j, fn in enumerate(feat_names):
    print(f"  {fn:30s} |coef|={mean_coef[j]:.4f} sign={sign_stability[j]:.0f}% top1={top1_pct[j]:.1f}%")
print(f"  ({time.time()-t2:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Perfect mapping decision tree
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 3: Decision tree (max_depth=3) ===", flush=True)
t3 = time.time()
from sklearn.tree import DecisionTreeClassifier, export_text

y_binary = (pm > -0.01).astype(int)
X_tree = np.column_stack([bg, bag, rg, pg, pu])

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_tree, y_binary)

tree_rules = export_text(dt, feature_names=['bldg_gap_clip','area_gap_clip','road_gap_clip','poi_gap_clip','pct_urban'])
acc = dt.score(X_tree, y_binary)

exp3 = {
    "accuracy": float(acc),
    "tree_rules": tree_rules,
    "n_nodes": int(dt.tree_.node_count),
    "n_leaves": int(dt.tree_.n_leaves),
    "target_pct_positive": float(y_binary.mean() * 100),
}
results["exp3_decision_tree"] = exp3
print(f"  Accuracy={acc:.4f}, Nodes={dt.tree_.node_count}, Leaves={dt.tree_.n_leaves}")
print(f"  % positive (proxy > -0.01): {y_binary.mean()*100:.2f}%")
print(tree_rules)
print(f"  ({time.time()-t3:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Coefficient perturbation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 4: Coefficient perturbation (10K) ===", flush=True)
t4 = time.time()

# Formula: -(w1*bg + w2*bag + w3*rg + w4*pg)/4 - (1-pu)
# Base weights: [1, 2, 1, 1]
base_weights = np.array([1.0, 2.0, 1.0, 1.0])
N_PERTURB = 10_000
perturb_std = 0.1

# Analytical approach + Monte Carlo validation
# Sensitivity = mean|d(proxy)/d(wi)| = mean(|gap_i|) / 4
gap_matrix = np.column_stack([bg, bag, rg, pg])
sensitivities = np.mean(np.abs(gap_matrix), axis=0) / 4  # analytical

# Monte Carlo validation (vectorized on subsample to avoid memory blowup)
np.random.seed(123)
sub_mc = np.random.choice(len(pm), size=10000, replace=False)
gap_sub = gap_matrix[sub_mc]
dw_all = np.random.randn(N_PERTURB, 4) * perturb_std
# Per-perturbation output change: delta_proxy = -(gap_matrix @ dw) / 4
delta_proxies = -(gap_sub @ dw_all.T) / 4  # shape (10000, 10000)
mc_rmse = np.sqrt(np.mean(delta_proxies**2, axis=0)).mean()

rank_idx = np.argsort(-sensitivities)
coef_names = ['bldg_gap_clip', 'area_gap_clip', 'road_gap_clip', 'poi_gap_clip']

exp4 = {
    "base_weights": base_weights.tolist(),
    "n_perturbations": N_PERTURB,
    "perturb_std": perturb_std,
    "method": "analytical mean|gap_i|/4 + MC validation",
    "sensitivities": {
        "bldg_gap_clip": float(sensitivities[0]),
        "area_gap_clip": float(sensitivities[1]),
        "road_gap_clip": float(sensitivities[2]),
        "poi_gap_clip": float(sensitivities[3]),
    },
    "mc_mean_rmse_per_perturbation": float(mc_rmse),
    "rank_by_sensitivity": [coef_names[i] for i in rank_idx],
}

results["exp4_perturbation"] = exp4
for i, cn in enumerate(coef_names):
    print(f"  {cn}: sensitivity={sensitivities[i]:.6f}")
print(f"  MC mean RMSE per perturbation: {mc_rmse:.6f}")
print(f"  Rank: {exp4['rank_by_sensitivity']}  ({time.time()-t4:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: gap_only formula discovery
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 5: gap_only formula discovery ===", flush=True)
t5 = time.time()

gap_only = pm + (1 - pu)  # Remove pct_urban effect
# This should equal -(bg + 2*bag + rg + pg)/4

# Test 1: Our formula
pred_ours = -(bg + 2*bag + rg + pg) / 4
rmse_ours = np.sqrt(np.mean((gap_only - pred_ours)**2))
r2_ours = 1 - np.mean((gap_only - pred_ours)**2) / np.var(gap_only)

# Test 2: Equal weights
pred_equal = -(bg + bag + rg + pg) / 4
rmse_equal = np.sqrt(np.mean((gap_only - pred_equal)**2))
r2_equal = 1 - np.mean((gap_only - pred_equal)**2) / np.var(gap_only)

# Test 3: Optimized weights (scipy)
from scipy.optimize import minimize
def loss_w(w):
    pred = -(gap_matrix @ w) / 4
    return np.mean((gap_only - pred)**2)

res_opt = minimize(loss_w, x0=[1,2,1,1], method='L-BFGS-B')
w_opt = res_opt.x
pred_opt = -(gap_matrix @ w_opt) / 4
rmse_opt = np.sqrt(np.mean((gap_only - pred_opt)**2))
r2_opt = 1 - np.mean((gap_only - pred_opt)**2) / np.var(gap_only)

# Test 4: Power law
def loss_power(params):
    a, b, c, d = params[:4]
    pred = -(bg**a + bag**b + rg**c + pg**d) / 4
    return np.mean((gap_only - pred)**2)

res_power = minimize(loss_power, x0=[1,1,1,1], method='Nelder-Mead', options={'maxiter': 5000})
p_opt = res_power.x
pred_power = -(bg**p_opt[0] + bag**p_opt[1] + rg**p_opt[2] + pg**p_opt[3]) / 4
rmse_power = np.sqrt(np.mean((gap_only - pred_power)**2))
r2_power = 1 - np.mean((gap_only - pred_power)**2) / np.var(gap_only)

exp5 = {
    "gap_only_mean": float(gap_only.mean()),
    "gap_only_std": float(gap_only.std()),
    "our_formula": {"weights": [1,2,1,1], "rmse": float(rmse_ours), "r2": float(r2_ours)},
    "equal_weights": {"weights": [1,1,1,1], "rmse": float(rmse_equal), "r2": float(r2_equal)},
    "optimized_weights": {"weights": w_opt.tolist(), "rmse": float(rmse_opt), "r2": float(r2_opt)},
    "power_law": {"powers": p_opt.tolist(), "rmse": float(rmse_power), "r2": float(r2_power)},
}
results["exp5_gap_only_discovery"] = exp5
for name, d in [("our_formula", exp5["our_formula"]), ("equal_weights", exp5["equal_weights"]),
                ("optimized_weights", exp5["optimized_weights"]), ("power_law", exp5["power_law"])]:
    print(f"  {name:20s}: RMSE={d['rmse']:.6f}, R²={d['r2']:.6f}")
print(f"  ({time.time()-t5:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Surprise feature hunt
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 6: Surprise feature hunt ===", flush=True)
t6 = time.time()

# Compute correlations efficiently
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_vals = {}
target = pm
for col in numeric_cols:
    if col == 'proxy_merged':
        continue
    vals = df[col].values
    # Fast correlation using numpy
    mask = ~(np.isnan(vals) | np.isnan(target))
    if mask.sum() < 100:
        continue
    v = vals[mask]
    t = target[mask]
    if v.std() < 1e-10:
        continue
    r = np.corrcoef(v, t)[0, 1]
    if not np.isnan(r):
        corr_vals[col] = r

# Sort by absolute correlation
sorted_corr = sorted(corr_vals.items(), key=lambda x: -abs(x[1]))
top30 = sorted_corr[:30]
non_gap_surprises = [(c, r) for c, r in sorted_corr if 'gap' not in c.lower() and abs(r) > 0.3]

exp6 = {
    "top30_correlations": [(c, float(r)) for c, r in top30],
    "non_gap_features_r_gt_0.3": [(c, float(r)) for c, r in non_gap_surprises[:20]],
    "n_non_gap_surprises": len(non_gap_surprises),
}
results["exp6_surprise_features"] = exp6
print("  Top 10 correlations with proxy_merged:")
for c, r in top30[:10]:
    print(f"    {c:40s} r={r:+.4f}")
print(f"  Non-gap features with |r|>0.3: {len(non_gap_surprises)}")
for c, r in non_gap_surprises[:5]:
    print(f"    {c:40s} r={r:+.4f}")
print(f"  ({time.time()-t6:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: What the ORGANIZERS' formula likely is
# ══════════════════════════════════════════════════════════════════════════
print("\n=== EXPERIMENT 7: Organizer formula hypotheses ===", flush=True)
t7 = time.time()

# Load lean_submission for comparison
lean = pd.read_csv(ROOT / "submissions/lean_submission.csv")
lean_scores = lean['coverage_gap_score'].values
lean_stats = {
    "mean": float(lean_scores.mean()),
    "std": float(lean_scores.std()),
    "min": float(lean_scores.min()),
    "max": float(lean_scores.max()),
    "median": float(np.median(lean_scores)),
    "pct_zero": float((lean_scores == 0).mean() * 100),
    "pct_near_zero": float((np.abs(lean_scores) < 0.01).mean() * 100),
}

def dist_comparison(proxy_arr, name):
    """Compare proxy distribution to lean_submission."""
    s = {
        "mean": float(proxy_arr.mean()),
        "std": float(proxy_arr.std()),
        "min": float(proxy_arr.min()),
        "max": float(proxy_arr.max()),
        "median": float(np.median(proxy_arr)),
        "pct_near_zero": float((np.abs(proxy_arr) < 0.01).mean() * 100),
    }
    # Distribution distance (normalized)
    mean_diff = abs(s["mean"] - lean_stats["mean"]) / (abs(lean_stats["mean"]) + 1e-10)
    std_diff = abs(s["std"] - lean_stats["std"]) / (lean_stats["std"] + 1e-10)
    min_diff = abs(s["min"] - lean_stats["min"]) / (abs(lean_stats["min"]) + 1e-10)
    max_diff = abs(s["max"] - lean_stats["max"]) / (abs(lean_stats["max"]) + 1e-10)
    median_diff = abs(s["median"] - lean_stats["median"]) / (abs(lean_stats["median"]) + 1e-10)
    dist_score = (mean_diff + std_diff + min_diff + max_diff + median_diff) / 5
    s["dist_score_vs_lean"] = float(dist_score)
    return s

# H1: Our formula (exactly)
h1_proxy = pm.copy()
h1_stats = dist_comparison(h1_proxy, "H1_our_formula")

# H2: Raw gaps (not clips)
raw_bg = df['building_gap'].values
raw_bag = df['building_area_gap'].values
raw_rg = df['road_gap'].values
raw_pg = df['poi_facility_gap_corrected'].values
h2_proxy = -(raw_bg + 2*raw_bag + raw_rg + raw_pg) / 4 - (1 - pu)
h2_stats = dist_comparison(h2_proxy, "H2_raw_gaps")

# H3: Max gap
h3_proxy = -np.maximum.reduce([bg, bag, rg, pg]) - (1 - pu)
h3_stats = dist_comparison(h3_proxy, "H3_max_gap")

# H4: Optimized weights
def loss_h4(w):
    pred = -(w[0]*bg + w[1]*bag + w[2]*rg + w[3]*pg) / 4 - w[4]*(1 - pu)
    return np.mean((pm - pred)**2)

res_h4 = minimize(loss_h4, x0=[1,2,1,1,1], method='L-BFGS-B')
w_h4 = res_h4.x
h4_proxy = -(w_h4[0]*bg + w_h4[1]*bag + w_h4[2]*rg + w_h4[3]*pg) / 4 - w_h4[4]*(1 - pu)
h4_stats = dist_comparison(h4_proxy, "H4_optimized_weights")
h4_rmse = np.sqrt(np.mean((pm - h4_proxy)**2))

# H5: No pct_urban
h5_proxy = -(bg + 2*bag + rg + pg) / 4
h5_stats = dist_comparison(h5_proxy, "H5_no_pct_urban")

exp7 = {
    "lean_submission_stats": lean_stats,
    "hypotheses": {
        "H1_our_formula": {"desc": "proxy = -(bg+2*bag+rg+pg)/4 - (1-pu)", "stats": h1_stats, "dist_score": h1_stats["dist_score_vs_lean"]},
        "H2_raw_gaps": {"desc": "proxy = -(raw_bg+2*raw_bag+raw_rg+raw_pg)/4 - (1-pu)", "stats": h2_stats, "dist_score": h2_stats["dist_score_vs_lean"]},
        "H3_max_gap": {"desc": "proxy = -max(bg,bag,rg,pg) - (1-pu)", "stats": h3_stats, "dist_score": h3_stats["dist_score_vs_lean"]},
        "H4_optimized": {"desc": f"proxy = -({w_h4[0]:.3f}*bg+{w_h4[1]:.3f}*bag+{w_h4[2]:.3f}*rg+{w_h4[3]:.3f}*pg)/4 - {w_h4[4]:.3f}*(1-pu)", "stats": h4_stats, "dist_score": h4_stats["dist_score_vs_lean"], "rmse_vs_proxy": float(h4_rmse), "weights": w_h4.tolist()},
        "H5_no_pct_urban": {"desc": "proxy = -(bg+2*bag+rg+pg)/4", "stats": h5_stats, "dist_score": h5_stats["dist_score_vs_lean"]},
    }
}

# Rank hypotheses
hypo_rank = sorted(exp7["hypotheses"].items(), key=lambda x: x[1]["dist_score"])
exp7["ranking"] = [(h, f"{d['dist_score']:.4f}") for h, d in hypo_rank]

results["exp7_organizer_formula"] = exp7

print("  Lean submission stats:")
for k, v in lean_stats.items():
    print(f"    {k}: {v}")
print("\n  Hypothesis ranking (by dist similarity to lean_submission):")
for h, d in hypo_rank:
    print(f"    {h:25s} dist_score={d['dist_score']:.4f}  mean={d['stats']['mean']:.4f}  std={d['stats']['std']:.4f}")
print(f"  ({time.time()-t7:.1f}s)")

# ── Save results ───────────────────────────────────────────────────────────
elapsed = time.time() - t0
results["meta"] = {
    "total_time_sec": float(elapsed),
    "data_shape": list(df.shape),
    "timestamp": pd.Timestamp.now().isoformat(),
}

out_path = RESULTS_DIR / "formula_decoder_1M.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"TOTAL TIME: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Results saved to: {out_path}")
print(f"{'='*60}")
