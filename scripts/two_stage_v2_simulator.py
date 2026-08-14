#!/usr/bin/env python3
"""
TWO-STAGE V2 SIMULATOR: 200K Batched Perturbation Analysis
===========================================================
Loads v2 results and submission, performs batched perturbation analysis
on sampled tracts to assess prediction stability.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time, gc, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"; OUT.mkdir(parents=True, exist_ok=True)
RESULTS = PROJ / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("TWO-STAGE V2 SIMULATOR: 200K Batched Perturbation Analysis")
print("=" * 78)
t0 = time.time()

# ========================================================================
# 1. LOAD V2 RESULTS + SUBMISSION
# ========================================================================
print("\n[1] Loading v2 results and submission...")

with open(RESULTS / 'two_stage_v2_results.json') as f:
    v2_results = json.load(f)
print(f"  Pipeline: {v2_results['pipeline']}")
print(f"  Tracts: {v2_results['data']['n_total']}")
print(f"  Non-zero gap: {v2_results['data']['n_nonzero_gap']} ({100 - v2_results['data']['pct_zero_gap']:.1f}%)")

sub = pd.read_csv(PROJ / "submissions" / "submission_two_stage_v2.csv")
sub['GEOID'] = sub['GEOID'].astype(str)
print(f"  Submission: {len(sub)} tracts")

# ========================================================================
# 2. LOAD BASE DATA FOR TARGETS
# ========================================================================
print("\n[2] Loading base data for rural_penalty and gap_only...")

feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")
feat['GEOID'] = feat['GEOID'].astype(str)

y_proxy = feat['proxy_merged'].copy()
rural_penalty = (1 - feat['pct_urban'].fillna(0.5)).clip(0, 1)
gap_only = y_proxy + rural_penalty
has_gap = (gap_only.abs() > 1e-10).astype(int)
pct_urban = feat['pct_urban'].fillna(0.5)

# Merge submission scores
data = pd.DataFrame({
    'GEOID': feat['GEOID'].astype(str),
    'proxy_merged': y_proxy.values,
    'rural_penalty': rural_penalty.values,
    'gap_only': gap_only.values,
    'has_gap': has_gap.values,
    'pct_urban': pct_urban.values,
})
data = data.merge(sub[['GEOID', 'coverage_gap_score']], on='GEOID', how='left')
data = data.dropna(subset=['coverage_gap_score'])

print(f"  Merged data: {len(data)} tracts")
print(f"  Has gap: {data['has_gap'].sum()} ({data['has_gap'].mean()*100:.1f}%)")

del feat; gc.collect()

# ========================================================================
# 3. SAMPLE TRACTS
# ========================================================================
N_SAMPLE = 5000
N_ITER = 200000
BATCH = 5000

print(f"\n[3] Sampling {N_SAMPLE} tracts...")

# Stratified sampling: proportional from has_gap and no_gap
rng = np.random.RandomState(SEED)

gap_tracts = data[data['has_gap'] == 1]
no_gap_tracts = data[data['has_gap'] == 0]

# Sample proportionally
n_gap_sample = min(int(N_SAMPLE * data['has_gap'].mean()), len(gap_tracts))
n_nogap_sample = N_SAMPLE - n_gap_sample

gap_sampled = gap_tracts.sample(n=n_gap_sample, random_state=rng)
nogap_sampled = no_gap_tracts.sample(n=n_nogap_sample, random_state=rng)

sampled = pd.concat([gap_sampled, nogap_sampled]).reset_index(drop=True)
print(f"  Sampled: {len(sampled)} tracts ({sampled['has_gap'].sum()} has_gap, {(~sampled['has_gap'].astype(bool)).sum()} no_gap)")

# ========================================================================
# 4. PREPARE ARRAYS
# ========================================================================
print("\n[4] Preparing arrays...")

scores = sampled['coverage_gap_score'].values.astype(np.float64)
gap_only_vals = sampled['gap_only'].values.astype(np.float64)
rural_penalty_vals = sampled['rural_penalty'].values.astype(np.float64)
has_gap_vals = sampled['has_gap'].values.astype(np.float64)

n_tracts = len(sampled)
print(f"  Tracts: {n_tracts}")
print(f"  Scores: mean={scores.mean():.4f} std={scores.std():.4f}")
print(f"  Gap_only: mean={gap_only_vals.mean():.4f} std={gap_only_vals.std():.4f}")
print(f"  Rural_penalty: mean={rural_penalty_vals.mean():.4f}")

# ========================================================================
# 5. BATCHED PERTURBATION ANALYSIS
# ========================================================================
print(f"\n[5] Running {N_ITER} perturbation iterations (batch={BATCH})...")

# Epsilon range
eps_values = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
n_eps = len(eps_values)

# Results accumulators
perturb_score_deltas = {f"eps_{e:.2f}": [] for e in eps_values}
perturb_rural_deltas = {f"eps_{e:.2f}": [] for e in eps_values}
perturb_cls_deltas = {f"eps_{e:.2f}": [] for e in eps_values}

# Track sign flips and magnitude changes
sign_flips_score = {f"eps_{e:.2f}": 0 for e in eps_values}
sign_flips_cls = {f"eps_{e:.2f}": 0 for e in eps_values}
large_deltas_score = {f"eps_{e:.2f}": 0 for e in eps_values}  # |delta| > 0.01

# Per-tract stability scores
tract_stability = np.zeros(n_tracts)

# Classifier probability proxy: use abs(score) as proxy for P(has_gap)
# In the two-stage model, score = P(has_gap) * E[gap|has_gap] - rural_penalty
# We approximate P(has_gap) from the score
cls_proba_proxy = np.clip(1.0 - np.abs(scores) / (np.abs(scores).max() + 1e-10), 0, 1)

n_batches = N_ITER // BATCH
remaining = N_ITER % BATCH
if remaining > 0:
    n_batches += 1

print(f"  Batches: {n_batches} ({BATCH} each)")

for batch_idx in range(n_batches):
    if batch_idx == n_batches - 1 and remaining > 0:
        batch_size = remaining
    else:
        batch_size = BATCH

    # Generate random perturbation indices and signs
    tract_indices = rng.randint(0, n_tracts, size=batch_size)
    eps_indices = rng.randint(0, n_eps, size=batch_size)
    signs = rng.choice([-1.0, 1.0], size=batch_size)

    # Random perturbation type: 0=score, 1=rural_penalty, 2=classifier
    perturb_types = rng.randint(0, 3, size=batch_size)

    for i in range(batch_size):
        t_idx = tract_indices[i]
        e_idx = eps_indices[i]
        eps = eps_values[e_idx]
        sign = signs[i]
        ptype = perturb_types[i]
        ekey = f"eps_{eps:.2f}"

        orig_score = scores[t_idx]
        orig_rural = rural_penalty_vals[t_idx]
        orig_cls = cls_proba_proxy[t_idx]

        if ptype == 0:
            # Perturb coverage_gap_score
            perturbed = orig_score + sign * eps
            delta = perturbed - orig_score
            perturb_score_deltas[ekey].append(delta)
            if orig_score * perturbed < 0:
                sign_flips_score[ekey] += 1
            if abs(delta) > 0.01:
                large_deltas_score[ekey] += 1
            tract_stability[t_idx] += abs(delta)

        elif ptype == 1:
            # Perturb rural_penalty
            perturbed_rural = np.clip(orig_rural + sign * eps, 0, 1)
            delta = perturbed_rural - orig_rural
            # Score changes: coverage_gap_score = gap_only_pred - rural_penalty
            # So delta_score = -delta_rural
            perturb_rural_deltas[ekey].append(-delta)
            tract_stability[t_idx] += abs(delta)

        else:
            # Perturb classifier probability
            perturbed_cls = np.clip(orig_cls + sign * eps, 0, 1)
            delta_cls = perturbed_cls - orig_cls
            perturb_cls_deltas[ekey].append(delta_cls)
            if orig_cls * perturbed_cls < 0:
                sign_flips_cls[ekey] += 1
            tract_stability[t_idx] += abs(delta_cls)

    if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
        pct = (batch_idx + 1) / n_batches * 100
        print(f"    Batch {batch_idx+1}/{n_batches} ({pct:.0f}%)")

gc.collect()

# ========================================================================
# 6. COMPUTE STABILITY METRICS
# ========================================================================
print("\n[6] Computing stability metrics...")

stability_results = {
    'config': {
        'n_sample': N_SAMPLE,
        'n_iter': N_ITER,
        'batch_size': BATCH,
        'eps_values': [float(e) for e in eps_values],
        'seed': SEED,
    },
    'score_perturbation': {},
    'rural_penalty_perturbation': {},
    'classifier_perturbation': {},
}

# Score perturbation stats
print("\n  Score Perturbation:")
for e in eps_values:
    ekey = f"eps_{e:.2f}"
    deltas = np.array(perturb_score_deltas[ekey])
    if len(deltas) > 0:
        stats = {
            'n': len(deltas),
            'mean': float(np.mean(deltas)),
            'std': float(np.std(deltas)),
            'abs_mean': float(np.mean(np.abs(deltas))),
            'max': float(np.max(np.abs(deltas))),
            'sign_flips': sign_flips_score[ekey],
            'large_deltas': large_deltas_score[ekey],
            'pct_sign_flips': float(sign_flips_score[ekey] / len(deltas) * 100),
            'pct_large_deltas': float(large_deltas_score[ekey] / len(deltas) * 100),
        }
        stability_results['score_perturbation'][ekey] = stats
        print(f"    {ekey}: abs_mean={stats['abs_mean']:.6f}, sign_flips={stats['sign_flips']} ({stats['pct_sign_flips']:.2f}%)")

# Rural penalty perturbation stats
print("\n  Rural Penalty Perturbation:")
for e in eps_values:
    ekey = f"eps_{e:.2f}"
    deltas = np.array(perturb_rural_deltas[ekey])
    if len(deltas) > 0:
        stats = {
            'n': len(deltas),
            'mean': float(np.mean(deltas)),
            'std': float(np.std(deltas)),
            'abs_mean': float(np.mean(np.abs(deltas))),
            'max': float(np.max(np.abs(deltas))),
        }
        stability_results['rural_penalty_perturbation'][ekey] = stats
        print(f"    {ekey}: abs_mean={stats['abs_mean']:.6f}, std={stats['std']:.6f}")

# Classifier perturbation stats
print("\n  Classifier Probability Perturbation:")
for e in eps_values:
    ekey = f"eps_{e:.2f}"
    deltas = np.array(perturb_cls_deltas[ekey])
    if len(deltas) > 0:
        stats = {
            'n': len(deltas),
            'mean': float(np.mean(deltas)),
            'std': float(np.std(deltas)),
            'abs_mean': float(np.mean(np.abs(deltas))),
            'max': float(np.max(np.abs(deltas))),
            'sign_flips': sign_flips_cls[ekey],
            'pct_sign_flips': float(sign_flips_cls[ekey] / len(deltas) * 100),
        }
        stability_results['classifier_perturbation'][ekey] = stats
        print(f"    {ekey}: abs_mean={stats['abs_mean']:.6f}, sign_flips={stats['sign_flips']} ({stats['pct_sign_flips']:.2f}%)")

# ========================================================================
# 7. TRACT-LEVEL STABILITY RANKING
# ========================================================================
print("\n[7] Tract-level stability ranking...")

# Normalize stability scores (average per perturbation)
avg_stability = tract_stability / (N_ITER / n_tracts)

# Categorize tracts
stability_quartiles = np.percentile(avg_stability, [25, 50, 75])
n_fragile = (avg_stability > stability_quartiles[2]).sum()
n_stable = (avg_stability <= stability_quartiles[0]).sum()

print(f"  Stability quartiles: Q25={stability_quartiles[0]:.6f}, Q50={stability_quartiles[1]:.6f}, Q75={stability_quartiles[2]:.6f}")
print(f"  Stable tracts (Q1): {n_stable} ({n_stable/n_tracts*100:.1f}%)")
print(f"  Fragile tracts (Q4): {n_fragile} ({n_fragile/n_tracts*100:.1f}%)")

# Top 10 most fragile tracts
fragile_idx = np.argsort(avg_stability)[-10:][::-1]
print(f"\n  Top 10 most fragile tracts:")
for idx in fragile_idx:
    row = sampled.iloc[idx]
    print(f"    GEOID={row['GEOID']}, score={row['coverage_gap_score']:.4f}, stability={avg_stability[idx]:.6f}, has_gap={int(row['has_gap'])}")

# Top 10 most stable tracts
stable_idx = np.argsort(avg_stability)[:10]
print(f"\n  Top 10 most stable tracts:")
for idx in stable_idx:
    row = sampled.iloc[idx]
    print(f"    GEOID={row['GEOID']}, score={row['coverage_gap_score']:.4f}, stability={avg_stability[idx]:.6f}, has_gap={int(row['has_gap'])}")

stability_results['tract_analysis'] = {
    'n_tracts': n_tracts,
    'stability_quartiles': {
        'q25': float(stability_quartiles[0]),
        'q50': float(stability_quartiles[1]),
        'q75': float(stability_quartiles[2]),
    },
    'n_stable': int(n_stable),
    'n_fragile': int(n_fragile),
    'overall_mean_stability': float(np.mean(avg_stability)),
    'overall_std_stability': float(np.std(avg_stability)),
}

# ========================================================================
# 8. AGGREGATE STABILITY SCORE
# ========================================================================
print("\n[8] Aggregate stability score...")

# Compute robustness: fraction of perturbations that don't cause sign flip or large delta
# Higher = more robust
score_robust = []
for e in eps_values:
    ekey = f"eps_{e:.2f}"
    sp = stability_results['score_perturbation'].get(ekey, {})
    total = sp.get('n', 0)
    if total > 0:
        robust = 1.0 - sp.get('pct_sign_flips', 0) / 100.0
        score_robust.append(robust)

cls_robust = []
for e in eps_values:
    ekey = f"eps_{e:.2f}"
    cp = stability_results['classifier_perturbation'].get(ekey, {})
    total = cp.get('n', 0)
    if total > 0:
        robust = 1.0 - cp.get('pct_sign_flips', 0) / 100.0
        cls_robust.append(robust)

overall_score_robust = np.mean(score_robust) if score_robust else 0.0
overall_cls_robust = np.mean(cls_robust) if cls_robust else 0.0
overall_robust = 0.5 * overall_score_robust + 0.5 * overall_cls_robust

print(f"  Score robustness: {overall_score_robust:.4f}")
print(f"  Classifier robustness: {overall_cls_robust:.4f}")
print(f"  Overall robustness: {overall_robust:.4f}")

stability_results['aggregate'] = {
    'score_robustness': float(overall_score_robust),
    'classifier_robustness': float(overall_cls_robust),
    'overall_robustness': float(overall_robust),
}

# ========================================================================
# 9. SAVE RESULTS
# ========================================================================
print("\n[9] Saving simulator results...")

stability_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
stability_results['elapsed_sec'] = round(time.time() - t0, 1)

with open(RESULTS / 'two_stage_v2_simulator.json', 'w') as f:
    json.dump(stability_results, f, indent=2)
print(f"  Saved: two_stage_v2_simulator.json")

el = time.time() - t0
print(f"\n{'='*78}")
print(f"DONE in {el:.0f}s")
print(f"Perturbations: {N_ITER} iterations on {N_SAMPLE} sampled tracts")
print(f"Epsilon range: [{eps_values[0]:.2f}, {eps_values[-1]:.2f}]")
print(f"Overall robustness: {overall_robust:.4f}")
print(f"Stable tracts: {n_stable} ({n_stable/n_tracts*100:.1f}%), Fragile: {n_fragile} ({n_fragile/n_tracts*100:.1f}%)")
print(f"{'='*78}")
