#!/usr/bin/env python3
"""
PIPELINE VALIDATION + LOCK REPORT
Validates both submissions, produces final comparison, and locks the pipeline.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np, pandas as pd, json, time
from pathlib import Path

PROJ = Path("/home/z/my-project/bias-bounty-map")
OUT = PROJ / "data/output"
DL = Path("/home/z/my-project/download")

print("=" * 72)
print("PIPELINE VALIDATION + LOCK REPORT")
print("=" * 72)

# ── 1. Load pipeline states ──
print("\n[1] Loading pipeline states...")

states = {}
for name, path in [
    ('Lean (Phase 2)', OUT / 'pipeline_state_merged.json'),
    ('Expanded (Integrated 10x)', OUT / 'pipeline_state_integrated_10x.json'),
]:
    try:
        with open(path) as f:
            states[name] = json.load(f)
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ {name} — not found")

# ── 2. Validate submissions ──
print("\n[2] Validating submissions...")

validation_results = {}
for name, path in [
    ('Lean', DL / 'submission_merged.csv'),
    ('Expanded', DL / 'submission_integrated_10x.csv'),
]:
    try:
        sub = pd.read_csv(path)
        checks = {}
        
        # Basic checks
        checks['n_tracts'] = len(sub)
        checks['has_geoid'] = 'GEOID' in sub.columns
        checks['has_score'] = 'coverage_gap_score' in sub.columns
        
        # Score range
        scores = sub['coverage_gap_score']
        checks['score_min'] = float(scores.min())
        checks['score_max'] = float(scores.max())
        checks['score_mean'] = float(scores.mean())
        checks['score_std'] = float(scores.std())
        checks['score_nan'] = int(scores.isna().sum())
        checks['score_inf'] = int(np.isinf(scores).sum())
        
        # Distribution checks
        checks['pct_negative'] = float((scores < 0).mean())
        checks['pct_zero'] = float((scores == 0).mean())
        checks['pct_positive'] = float((scores > 0).mean())
        
        # GEOID checks
        checks['geoid_unique'] = int(sub['GEOID'].nunique())
        checks['geoid_len_ok'] = int(sub['GEOID'].astype(str).str.len().isin([11, 12]).sum())
        
        validation_results[name] = checks
        
        # Print summary
        print(f"\n  {name} submission:")
        print(f"    Tracts:     {checks['n_tracts']:,}")
        print(f"    Score range: [{checks['score_min']:.4f}, {checks['score_max']:.4f}]")
        print(f"    Score mean:  {checks['score_mean']:.4f} ± {checks['score_std']:.4f}")
        print(f"    NaN/Inf:     {checks['score_nan']}/{checks['score_inf']}")
        print(f"    <0 / =0 / >0: {checks['pct_negative']:.1%} / {checks['pct_zero']:.1%} / {checks['pct_positive']:.1%}")
        print(f"    GEOID unique: {checks['geoid_unique']:,}")
        
    except Exception as e:
        print(f"  ✗ {name} — {e}")
        validation_results[name] = {'error': str(e)}

# ── 3. Compare predictions between pipelines ──
print("\n[3] Comparing predictions between pipelines...")

try:
    lean = pd.read_csv(DL / 'submission_merged.csv')
    expanded = pd.read_csv(DL / 'submission_integrated_10x.csv')
    
    merged = lean.merge(expanded, on='GEOID', suffixes=('_lean', '_exp'))
    
    score_lean = merged['coverage_gap_score_lean']
    score_exp = merged['coverage_gap_score_exp']
    
    corr = np.corrcoef(score_lean, score_exp)[0, 1]
    rmse_diff = np.sqrt(np.mean((score_lean - score_exp) ** 2))
    mean_diff = (score_exp - score_lean).mean()
    
    print(f"  Pearson correlation: {corr:.6f}")
    print(f"  RMSE between predictions: {rmse_diff:.6f}")
    print(f"  Mean difference (expanded - lean): {mean_diff:.6f}")
    
    # Where do they differ most?
    diff = np.abs(score_exp - score_lean)
    top_diff_idx = diff.nlargest(5).index
    print(f"\n  Top 5 largest disagreements:")
    for idx in top_diff_idx:
        geoid = merged.loc[idx, 'GEOID']
        print(f"    GEOID {geoid}: lean={score_lean[idx]:.4f}, exp={score_exp[idx]:.4f}, diff={diff[idx]:.4f}")
    
    comparison = {
        'pearson_corr': float(corr),
        'rmse_between': float(rmse_diff),
        'mean_diff': float(mean_diff),
        'n_agreement_99pct': int((diff < 0.01).sum()),
        'n_agreement_95pct': int((diff < 0.05).sum()),
    }
except Exception as e:
    print(f"  Comparison failed: {e}")
    comparison = {'error': str(e)}

# ── 4. Bug fix verification ──
print("\n[4] Bug fix verification...")

feat = pd.read_parquet(OUT / "engineered_features_merged.parquet")

# Bug #1: is_perfectly_mapped exists
has_perfect = 'is_perfectly_mapped' in feat.columns
n_perfect = int(feat['is_perfectly_mapped'].sum()) if has_perfect else 0
print(f"  Bug Fix #1 (is_perfectly_mapped): {'✓ PRESENT' if has_perfect else '✗ MISSING'}")
if has_perfect:
    print(f"    {n_perfect:,} tracts flagged as perfectly mapped ({n_perfect/len(feat)*100:.1f}%)")

# Bug #2: building_gap clipped
bg = feat['building_gap'].fillna(0)
bg_min, bg_max = float(bg.min()), float(bg.max())
clipped_ok = bg_min >= -4.0 and bg_max <= 1.0
print(f"  Bug Fix #2 (building_gap clipped): {'✓ VERIFIED' if clipped_ok else '✗ RANGE VIOLATION'}")
print(f"    Range: [{bg_min:.4f}, {bg_max:.4f}]")

# Deterministic fix: gap_only and rural_penalty exist
has_gap_only = 'gap_only' in feat.columns
has_rural = 'rural_penalty' in feat.columns
print(f"  Deterministic fix (gap_only): {'✓ PRESENT' if has_gap_only else '✗ MISSING'}")
print(f"  Deterministic fix (rural_penalty): {'✓ PRESENT' if has_rural else '✗ MISSING'}")

# Interaction features
interaction_cols = [c for c in feat.columns if any(x in c for x in 
    ['ws_dist_x_', 'cvi_ext_x_', 'carbon_x_', 'svi_min_x_', 'burn_pct_x_',
     'fire_acres_x_', 'tribal_legal_x_', 'tribal_x_fire', 'tribal_x_burn',
     'cvi_health_x_', 'carb_bldg_x_'])]
print(f"  Interaction features: {'✓ PRESENT' if len(interaction_cols) > 0 else '✗ MISSING'}")
print(f"    {len(interaction_cols)} interaction features found: {interaction_cols[:5]}...")

del feat

# ── 5. Pipeline comparison table ──
print("\n[5] Final pipeline comparison:")

print(f"\n  {'Metric':<30} {'Lean (Phase 2)':>18} {'Expanded (10x)':>18}")
print(f"  {'─'*30} {'─'*18} {'─'*18}")

for metric, lean_key, exp_key in [
    ('N features', 'n_features', 'n_features'),
    ('Best R²', 'best_r2', 'best_r2'),
    ('Best RMSE', 'best_rmse', 'best_rmse'),
    ('N models', None, None),
    ('Ensemble type', 'best_ensemble', 'best_ensemble'),
]:
    lean_val = states.get('Lean (Phase 2)', {}).get(lean_key, '?') if lean_key else '?'
    exp_val = states.get('Expanded (Integrated 10x)', {}).get(exp_key, '?') if exp_key else '?'
    
    if metric == 'N models':
        lean_val = len(states.get('Lean (Phase 2)', {}).get('models', {}))
        exp_val = len(states.get('Expanded (Integrated 10x)', {}).get('models', {}))
    
    if isinstance(lean_val, float):
        lean_s = f"{lean_val:.4f}" if lean_val < 1 else f"{lean_val:.6f}"
    else:
        lean_s = str(lean_val)
    if isinstance(exp_val, float):
        exp_s = f"{exp_val:.4f}" if exp_val < 1 else f"{exp_val:.6f}"
    else:
        exp_s = str(exp_val)
    
    print(f"  {metric:<30} {lean_s:>18} {exp_s:>18}")

# ── 6. Bias comparison ──
print(f"\n  {'Bias dimension':<30} {'Lean':>18} {'Expanded':>18}")
print(f"  {'─'*30} {'─'*18} {'─'*18}")

for dim in ['HighSVI vs LowSVI', 'Tribal vs Non', 'Rural vs Urban']:
    lean_biases = states.get('Lean (Phase 2)', {}).get('bias_findings', [])
    exp_biases = states.get('Expanded (Integrated 10x)', {}).get('bias_findings', [])
    lean_ratio = next((b['ratio'] for b in lean_biases if isinstance(b, dict) and b.get('stratum') == dim), '?')
    exp_ratio = next((b['ratio'] for b in exp_biases if isinstance(b, dict) and b.get('stratum') == dim), '?')
    lean_s = f"{lean_ratio:.3f}" if isinstance(lean_ratio, (int, float)) else str(lean_ratio)
    exp_s = f"{exp_ratio:.3f}" if isinstance(exp_ratio, (int, float)) else str(exp_ratio)
    print(f"  {dim:<30} {lean_s:>18} {exp_s:>18}")

# ── 7. Build lock report ──
print("\n[6] Building lock report...")

lock_report = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'status': 'LOCKED',
    'bug_fixes': {
        'fix_1_is_perfectly_mapped': {'applied': has_perfect, 'n_tracts': n_perfect},
        'fix_2_building_gap_clipped': {'applied': clipped_ok, 'range': [bg_min, bg_max]},
    },
    'deterministic_fix': {
        'train_on_gap_only': has_gap_only,
        'rural_penalty_at_inference': has_rural,
    },
    'interaction_features': {
        'n_features': len(interaction_cols),
        'features': interaction_cols,
    },
    'pipelines': {
        'lean': {
            'features': states.get('Lean (Phase 2)', {}).get('n_features'),
            'r2': states.get('Lean (Phase 2)', {}).get('best_r2'),
            'rmse': states.get('Lean (Phase 2)', {}).get('best_rmse'),
            'risk': 'LOW',
            'use': 'Primary submission',
        },
        'expanded': {
            'features': states.get('Expanded (Integrated 10x)', {}).get('n_features'),
            'r2': states.get('Expanded (Integrated 10x)', {}).get('best_r2'),
            'rmse': states.get('Expanded (Integrated 10x)', {}).get('best_rmse'),
            'risk': 'MEDIUM',
            'use': 'Secondary submission (hedge)',
        },
    },
    'validation': validation_results,
    'comparison': comparison,
    'next_steps': [
        'Wait for Aug 28 real labels',
        'Submit both lean and expanded to public LB',
        'If expanded beats lean → interactions are real signal',
        'If lean wins → interactions are proxy-specific, stay lean',
        'Retrain winner on full data for final submission',
    ],
}

with open(OUT / 'pipeline_lock_report.json', 'w') as f:
    json.dump(lock_report, f, indent=2, default=str)
with open(DL / 'pipeline_lock_report.json', 'w') as f:
    json.dump(lock_report, f, indent=2, default=str)

print(f"  Lock report saved to {DL / 'pipeline_lock_report.json'}")

print(f"\n{'=' * 72}")
print(f"PIPELINE LOCKED")
print(f"{'=' * 72}")
print(f"\n  Two submissions ready:")
print(f"    Lean:      {DL / 'submission_merged.csv'}")
print(f"    Expanded:  {DL / 'submission_integrated_10x.csv'}")
print(f"\n  Next: Wait for Aug 28, submit both, let public LB decide.")
print(f"{'=' * 72}")
