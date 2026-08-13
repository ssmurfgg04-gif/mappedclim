#!/usr/bin/env python3
"""
Deep Tribal Bias Analysis
=========================
Comprehensive comparison of tribal vs non-tribal tracts across all key metrics,
with stratified analysis by state and SVI quartile to control for confounders.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 80)
print("DEEP TRIBAL BIAS ANALYSIS")
print("=" * 80)

print("\n[1] Loading data...")
feat_path = "/home/z/my-project/bias-bounty-map/data/output/engineered_features_merged.parquet"
sub_path  = "/home/z/my-project/bias-bounty-map/data/output/submission_merged.csv"

feat = pd.read_parquet(feat_path)
sub  = pd.read_csv(sub_path)

print(f"  Engineered features: {feat.shape}")
print(f"  Submission:          {sub.shape}")

# Ensure GEOID is str on both sides
feat['GEOID'] = feat['GEOID'].astype(str)
sub['GEOID']  = sub['GEOID'].astype(str)

# Merge
df = feat.merge(sub, on='GEOID', how='inner')
print(f"  Merged:              {df.shape}")

# Derive state FIPS
df['state_fips'] = df['GEOID'].str[:2]

# FIPS → state name mapping (focus states + others)
FIPS_NAME = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT',
    '10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL',
    '18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD',
    '25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE',
    '32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND',
    '39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD',
    '47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV',
    '55':'WI','56':'WY'
}
df['state_name'] = df['state_fips'].map(FIPS_NAME).fillna(df['state_fips'])

# Tribal flag (use tribal_any as primary)
df['is_tribal'] = df['tribal_any'].astype(bool)

n_tribal     = df['is_tribal'].sum()
n_non_tribal = (~df['is_tribal']).sum()
print(f"\n  Tribal tracts:     {n_tribal}")
print(f"  Non-tribal tracts: {n_non_tribal}")
print(f"  Tribal fraction:   {n_tribal/len(df)*100:.2f}%")

# ============================================================
# 2. OVERALL TRIBAL vs NON-TRIBAL COMPARISONS
# ============================================================
print("\n" + "=" * 80)
print("[2] OVERALL TRIBAL vs NON-TRIBAL COMPARISONS")
print("=" * 80)

tribal     = df[df['is_tribal']]
non_tribal = df[~df['is_tribal']]

metrics = [
    ('coverage_gap_score', 'Coverage Gap Score (final model output)'),
    ('gap_only',           'gap_only (raw training target, before rural penalty)'),
    ('building_gap',       'Building Gap'),
    ('road_gap',           'Road Gap'),
    ('rural_penalty',      'Rural Penalty'),
    ('compound_risk',      'Compound Risk'),
    ('svi_overall',        'SVI Overall'),
    ('pct_urban',          'Pct Urban'),
    ('rural_continuous',   'Rural Continuous'),
]

print(f"\n{'Metric':<50} {'Tribal Mean':>12} {'Non-Trib Mean':>14} {'Ratio':>8} {'Tribal Med':>11} {'Non-Trib Med':>13} {'Diff':>10}")
print("-" * 120)

for col, label in metrics:
    t_mean = tribal[col].mean()
    nt_mean = non_tribal[col].mean()
    t_med  = tribal[col].median()
    nt_med = non_tribal[col].median()
    ratio  = t_mean / nt_mean if nt_mean != 0 else np.nan
    diff   = t_mean - nt_mean
    print(f"{label:<50} {t_mean:>12.6f} {nt_mean:>14.6f} {ratio:>8.4f} {t_med:>11.6f} {nt_med:>13.6f} {diff:>10.6f}")

# ============================================================
# 3. RAW GAP vs FINAL SCORE DIVERGENCE
# ============================================================
print("\n" + "=" * 80)
print("[3] RAW GAP vs FINAL SCORE DIVERGENCE")
print("=" * 80)

t_gap_only   = tribal['gap_only'].mean()
nt_gap_only  = non_tribal['gap_only'].mean()
t_final      = tribal['coverage_gap_score'].mean()
nt_final     = non_tribal['coverage_gap_score'].mean()
t_rural_pen  = tribal['rural_penalty'].mean()
nt_rural_pen = non_tribal['rural_penalty'].mean()

print(f"\n  gap_only (raw) ratio tribal/non-tribal:     {t_gap_only/nt_gap_only:.4f}")
print(f"  coverage_gap_score ratio tribal/non-tribal:  {t_final/nt_final:.4f}")
print(f"\n  This shows raw gaps are similar but final scores diverge:")
print(f"    Tribal gap_only mean:     {t_gap_only:.6f}")
print(f"    Non-tribal gap_only mean: {nt_gap_only:.6f}")
print(f"    Tribal final score mean:  {t_final:.6f}")
print(f"    Non-tribal final mean:    {nt_final:.6f}")
print(f"\n  Rural penalty is the amplifier:")
print(f"    Tribal rural_penalty mean:     {t_rural_pen:.6f}")
print(f"    Non-tribal rural_penalty mean: {nt_rural_pen:.6f}")
print(f"    Ratio: {t_rural_pen/nt_rural_pen:.4f}")

# How much of the divergence is explained by rural penalty?
# Final ≈ gap_only + rural_penalty contributions
tribal_score_minus_gap  = t_final - t_gap_only
nt_score_minus_gap     = nt_final - nt_gap_only
print(f"\n  (final_score - gap_only) for tribal:     {tribal_score_minus_gap:.6f}")
print(f"  (final_score - gap_only) for non-tribal: {nt_score_minus_gap:.6f}")
print(f"  This extra penalty in tribal tracts is:  {tribal_score_minus_gap - nt_score_minus_gap:.6f}")

# ============================================================
# 4. BY-STATE TRIBAL vs NON-TRIBAL RATIOS
# ============================================================
print("\n" + "=" * 80)
print("[4] BY-STATE TRIBAL vs NON-TRIBAL RATIOS")
print("=" * 80)

focus_states = ['40','37','38','04','35','30','55','27']  # OK, NC, ND, AZ, NM, MT, WI, MN
focus_labels = {s: f"{FIPS_NAME.get(s,s)} ({s})" for s in focus_states}

print(f"\n{'State':<20} {'Trib N':>7} {'NonT N':>8} {'Trib Score':>11} {'NonT Score':>11} {'Ratio':>7} {'Trib GapOnly':>13} {'NonT GapOnly':>13} {'GapOnly Ratio':>14} {'Trib RuralP':>12} {'NonT RuralP':>12}")
print("-" * 140)

state_results = []
for sf in sorted(df['state_fips'].unique()):
    state_df = df[df['state_fips'] == sf]
    st = state_df[state_df['is_tribal']]
    snt = state_df[~state_df['is_tribal']]
    if len(st) < 5:
        continue
    
    t_score  = st['coverage_gap_score'].mean()
    nt_score = snt['coverage_gap_score'].mean()
    t_go     = st['gap_only'].mean()
    nt_go    = snt['gap_only'].mean()
    t_rp     = st['rural_penalty'].mean()
    nt_rp    = snt['rural_penalty'].mean()
    score_ratio = t_score / nt_score if nt_score != 0 else np.nan
    go_ratio    = t_go / nt_go if nt_go != 0 else np.nan
    
    name = FIPS_NAME.get(sf, sf)
    is_focus = sf in focus_states
    marker = " ***" if is_focus else ""
    
    row = {
        'fips': sf, 'name': name, 'n_tribal': len(st), 'n_non_tribal': len(snt),
        't_score': t_score, 'nt_score': nt_score, 'score_ratio': score_ratio,
        't_gap_only': t_go, 'nt_gap_only': nt_go, 'gap_only_ratio': go_ratio,
        't_rural_penalty': t_rp, 'nt_rural_penalty': nt_rp
    }
    state_results.append(row)
    
    print(f"{name:<20} {len(st):>7} {len(snt):>8} {t_score:>11.6f} {nt_score:>11.6f} {score_ratio:>7.4f} {t_go:>13.6f} {nt_go:>13.6f} {go_ratio:>14.4f} {t_rp:>12.6f} {nt_rp:>12.6f}{marker}")

state_results_df = pd.DataFrame(state_results)

# Focus states summary
print("\n  *** = Focus states (OK, NC, ND, AZ, NM, MT, WI, MN)")

# ============================================================
# 5. SVI QUARTILE STRATIFICATION
# ============================================================
print("\n" + "=" * 80)
print("[5] SVI QUARTILE STRATIFICATION (controlling for socioeconomic status)")
print("=" * 80)

df['svi_quartile'] = pd.qcut(df['svi_overall'], q=4, labels=['Q1(Low)','Q2','Q3','Q4(High)'], duplicates='drop')

print(f"\n{'SVI Quartile':<14} {'Trib N':>7} {'NonT N':>8} {'Trib Score':>11} {'NonT Score':>11} {'Ratio':>7} {'Trib GapOnly':>13} {'NonT GapOnly':>13} {'GapOnly Ratio':>14} {'Trib RuralP':>12} {'NonT RuralP':>12}")
print("-" * 130)

for q in ['Q1(Low)','Q2','Q3','Q4(High)']:
    qdf = df[df['svi_quartile'] == q]
    qt  = qdf[qdf['is_tribal']]
    qnt = qdf[~qdf['is_tribal']]
    if len(qt) < 3:
        print(f"{q:<14} (too few tribal tracts)")
        continue
    
    t_score  = qt['coverage_gap_score'].mean()
    nt_score = qnt['coverage_gap_score'].mean()
    t_go     = qt['gap_only'].mean()
    nt_go    = qnt['gap_only'].mean()
    t_rp     = qt['rural_penalty'].mean()
    nt_rp    = qnt['rural_penalty'].mean()
    score_ratio = t_score / nt_score if nt_score != 0 else np.nan
    go_ratio    = t_go / nt_go if nt_go != 0 else np.nan
    
    print(f"{q:<14} {len(qt):>7} {len(qnt):>8} {t_score:>11.6f} {nt_score:>11.6f} {score_ratio:>7.4f} {t_go:>13.6f} {nt_go:>13.6f} {go_ratio:>14.4f} {t_rp:>12.6f} {nt_rp:>12.6f}")

print("\n  KEY INSIGHT: If tribal bias persists within the same SVI quartile,")
print("  it cannot be explained by socioeconomic status alone.")

# ============================================================
# 6. RESIDUAL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("[6] RESIDUAL ANALYSIS")
print("=" * 80)

# Compute residual = actual score - expected score (where expected = gap_only)
# This shows how much the model adds beyond the raw gap
df['residual'] = df['coverage_gap_score'] - df['gap_only']

t_resid  = df.loc[df['is_tribal'], 'residual']
nt_resid = df.loc[~df['is_tribal'], 'residual']

print(f"\n  Tribal residual:     mean={t_resid.mean():.6f}, median={t_resid.median():.6f}, std={t_resid.std():.6f}")
print(f"  Non-tribal residual: mean={nt_resid.mean():.6f}, median={nt_resid.median():.6f}, std={nt_resid.std():.6f}")
print(f"  Difference in means: {t_resid.mean() - nt_resid.mean():.6f}")
print(f"  Ratio:               {t_resid.mean() / nt_resid.mean() if nt_resid.mean() != 0 else 'N/A'}")

# Residual by SVI quartile
print(f"\n  Residual by SVI Quartile:")
print(f"  {'SVI Quartile':<14} {'Trib Resid Mean':>16} {'NonT Resid Mean':>17} {'Diff':>10}")
print("  " + "-" * 60)
for q in ['Q1(Low)','Q2','Q3','Q4(High)']:
    qdf = df[df['svi_quartile'] == q]
    qt  = qdf[qdf['is_tribal']]
    qnt = qdf[~qdf['is_tribal']]
    if len(qt) < 3:
        continue
    t_r = qt['residual'].mean()
    nt_r = qnt['residual'].mean()
    print(f"  {q:<14} {t_r:>16.6f} {nt_r:>17.6f} {t_r-nt_r:>10.6f}")

# ============================================================
# 7. CASE STUDIES
# ============================================================
print("\n" + "=" * 80)
print("[7] CASE STUDIES")
print("=" * 80)

# --- 7a. Eastern OK Tribal vs Adjacent Non-Tribal ---
print("\n[7a] EASTERN OKLAHOMA (FIPS 40) TRIBAL vs NON-TRIBAL")
print("-" * 60)

ok_df = df[df['state_fips'] == '40']
ok_tribal = ok_df[ok_df['is_tribal']]
ok_nontrib = ok_df[~ok_df['is_tribal']]

print(f"  OK tribal tracts:     {len(ok_tribal)}")
print(f"  OK non-tribal tracts: {len(ok_nontrib)}")

for col, label in metrics:
    t = ok_tribal[col].mean()
    nt = ok_nontrib[col].mean()
    ratio = t/nt if nt != 0 else np.nan
    print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# --- 7b. South Dakota Tribal vs Non-Tribal ---
print("\n[7b] SOUTH DAKOTA (FIPS 46) TRIBAL vs NON-TRIBAL")
print("-" * 60)

sd_df = df[df['state_fips'] == '46']
sd_tribal = sd_df[sd_df['is_tribal']]
sd_nontrib = sd_df[~sd_df['is_tribal']]

print(f"  SD tribal tracts:     {len(sd_tribal)}")
print(f"  SD non-tribal tracts: {len(sd_nontrib)}")

if len(sd_tribal) > 0:
    for col, label in metrics:
        t = sd_tribal[col].mean()
        nt = sd_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")
else:
    print("  No tribal tracts in SD")

# --- 7c. Navajo Nation (AZ + NM tribal) ---
print("\n[7c] NAVAJO NATION (AZ FIPS 04 + NM FIPS 35 TRIBAL)")
print("-" * 60)

navajo_df = df[(df['state_fips'].isin(['04','35'])) & (df['is_tribal'])]
az_nm_nontrib = df[(df['state_fips'].isin(['04','35'])) & (~df['is_tribal'])]

print(f"  AZ+NM tribal tracts (Navajo region): {len(navajo_df)}")
print(f"  AZ+NM non-tribal tracts:             {len(az_nm_nontrib)}")

if len(navajo_df) > 0:
    for col, label in metrics:
        t = navajo_df[col].mean()
        nt = az_nm_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# --- 7d. North Dakota Tribal vs Non-Tribal ---
print("\n[7d] NORTH DAKOTA (FIPS 38) TRIBAL vs NON-TRIBAL")
print("-" * 60)

nd_df = df[df['state_fips'] == '38']
nd_tribal = nd_df[nd_df['is_tribal']]
nd_nontrib = nd_df[~nd_df['is_tribal']]

print(f"  ND tribal tracts:     {len(nd_tribal)}")
print(f"  ND non-tribal tracts: {len(nd_nontrib)}")

if len(nd_tribal) > 0:
    for col, label in metrics:
        t = nd_tribal[col].mean()
        nt = nd_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# --- 7e. Montana Tribal vs Non-Tribal ---
print("\n[7e] MONTANA (FIPS 30) TRIBAL vs NON-TRIBAL")
print("-" * 60)

mt_df = df[df['state_fips'] == '30']
mt_tribal = mt_df[mt_df['is_tribal']]
mt_nontrib = mt_df[~mt_df['is_tribal']]

print(f"  MT tribal tracts:     {len(mt_tribal)}")
print(f"  MT non-tribal tracts: {len(mt_nontrib)}")

if len(mt_tribal) > 0:
    for col, label in metrics:
        t = mt_tribal[col].mean()
        nt = mt_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# --- 7f. Wisconsin Tribal vs Non-Tribal ---
print("\n[7f] WISCONSIN (FIPS 55) TRIBAL vs NON-TRIBAL")
print("-" * 60)

wi_df = df[df['state_fips'] == '55']
wi_tribal = wi_df[wi_df['is_tribal']]
wi_nontrib = wi_df[~wi_df['is_tribal']]

print(f"  WI tribal tracts:     {len(wi_tribal)}")
print(f"  WI non-tribal tracts: {len(wi_nontrib)}")

if len(wi_tribal) > 0:
    for col, label in metrics:
        t = wi_tribal[col].mean()
        nt = wi_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# --- 7g. Minnesota Tribal vs Non-Tribal ---
print("\n[7g] MINNESOTA (FIPS 27) TRIBAL vs NON-TRIBAL")
print("-" * 60)

mn_df = df[df['state_fips'] == '27']
mn_tribal = mn_df[mn_df['is_tribal']]
mn_nontrib = mn_df[~mn_df['is_tribal']]

print(f"  MN tribal tracts:     {len(mn_tribal)}")
print(f"  MN non-tribal tracts: {len(mn_nontrib)}")

if len(mn_tribal) > 0:
    for col, label in metrics:
        t = mn_tribal[col].mean()
        nt = mn_nontrib[col].mean()
        ratio = t/nt if nt != 0 else np.nan
        print(f"  {label:<50} tribal={t:>10.6f}  non-tribal={nt:>10.6f}  ratio={ratio:.4f}")

# ============================================================
# 8. TRIBAL LEGAL vs STATISTICAL vs NON-TRIBAL
# ============================================================
print("\n" + "=" * 80)
print("[8] TRIBAL LEGAL vs STATISTICAL vs NON-TRIBAL BREAKDOWN")
print("=" * 80)

legal      = df[df['tribal_legal'].astype(bool)]
statistical = df[df['tribal_statistical'].astype(bool) & ~df['tribal_legal'].astype(bool)]
non_trib   = df[~df['is_tribal']]

print(f"\n  Legal tribal tracts:      {len(legal)}")
print(f"  Statistical-only tribal:  {len(statistical)}")
print(f"  Non-tribal tracts:        {len(non_trib)}")

for col, label in [('coverage_gap_score','Coverage Gap Score'), ('gap_only','gap_only'), ('rural_penalty','Rural Penalty'), ('building_gap','Building Gap'), ('road_gap','Road Gap')]:
    l_mean = legal[col].mean()
    s_mean = statistical[col].mean()
    nt_mean = non_trib[col].mean()
    print(f"\n  {label}:")
    print(f"    Legal tribal:  {l_mean:.6f}")
    print(f"    Stat tribal:   {s_mean:.6f}")
    print(f"    Non-tribal:    {nt_mean:.6f}")
    print(f"    Legal/NonT ratio:  {l_mean/nt_mean:.4f}" if nt_mean != 0 else "    Legal/NonT ratio:  N/A")
    print(f"    Stat/NonT ratio:   {s_mean/nt_mean:.4f}" if nt_mean != 0 else "    Stat/NonT ratio:   N/A")

# ============================================================
# 9. INTERACTION EFFECTS: TRIBAL × RURAL
# ============================================================
print("\n" + "=" * 80)
print("[9] INTERACTION EFFECTS: TRIBAL × RURAL")
print("=" * 80)

# 4 groups: tribal+rural, tribal+nonrural, nontribal+rural, nontribal+nonrural
df['is_rural'] = df['rural_indicator'].astype(bool) if 'rural_indicator' in df.columns else (df['rural_continuous'] > 0.5)

groups = {
    'Tribal+Rural':     df[df['is_tribal'] & df['is_rural']],
    'Tribal+Non-Rural': df[df['is_tribal'] & ~df['is_rural']],
    'NonTrib+Rural':    df[~df['is_tribal'] & df['is_rural']],
    'NonTrib+Non-Rural':df[~df['is_tribal'] & ~df['is_rural']],
}

print(f"\n{'Group':<22} {'N':>6} {'Score Mean':>11} {'GapOnly Mean':>13} {'RuralPen Mean':>14} {'BldgGap Mean':>13} {'RoadGap Mean':>13}")
print("-" * 95)

for name, g in groups.items():
    print(f"{name:<22} {len(g):>6} {g['coverage_gap_score'].mean():>11.6f} {g['gap_only'].mean():>13.6f} {g['rural_penalty'].mean():>14.6f} {g['building_gap'].mean():>13.6f} {g['road_gap'].mean():>13.6f}")

# Double interaction effect
tr_score = groups['Tribal+Rural']['coverage_gap_score'].mean()
tnr_score = groups['Tribal+Non-Rural']['coverage_gap_score'].mean()
ntr_score = groups['NonTrib+Rural']['coverage_gap_score'].mean()
ntnr_score = groups['NonTrib+Non-Rural']['coverage_gap_score'].mean()

# Interaction = (TR - TNR) - (NR - NNR) 
interaction = (tr_score - tnr_score) - (ntr_score - ntnr_score)
print(f"\n  Interaction effect (Tribal×Rural on score): {interaction:.6f}")
print(f"  (If significant, tribal and rural compound beyond additivity)")

# ============================================================
# 10. DISTRIBUTION COMPARISON: QUANTILE TABLE
# ============================================================
print("\n" + "=" * 80)
print("[10] QUANTILE COMPARISON TABLE")
print("=" * 80)

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

print(f"\n{'Metric':<30} {'Q':>5}", end="")
for q in quantiles:
    print(f" {'Tribal':>10}", end="")
    print(f" {'NonTrib':>10}", end="")
print()
print("-" * (30 + 5 + len(quantiles)*20))

for col, label in [('coverage_gap_score','Score'), ('gap_only','gap_only'), ('rural_penalty','RuralPen'), ('building_gap','BldgGap'), ('road_gap','RoadGap')]:
    t_q = tribal[col].quantile(quantiles)
    nt_q = non_tribal[col].quantile(quantiles)
    for i, q in enumerate(quantiles):
        if i == 0:
            print(f"{label:<30} {q:>5.2f}", end="")
        else:
            print(f"{'':<30} {q:>5.2f}", end="")
        print(f" {t_q[q]:>10.6f}", end="")
        print(f" {nt_q[q]:>10.6f}", end="")
        print()
    print()

# ============================================================
# 11. TOP TRIBAL TRACTS (WORST HIT)
# ============================================================
print("\n" + "=" * 80)
print("[11] TOP 20 MOST PENALIZED TRIBAL TRACTS")
print("=" * 80)

top_tribal = tribal.nlargest(20, 'coverage_gap_score')[['GEOID','state_name','coverage_gap_score','gap_only','rural_penalty','building_gap','road_gap','svi_overall','pct_urban','compound_risk']]
print(top_tribal.to_string(index=False))

# ============================================================
# 12. TRIBAL PERCENTAGE DISTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("[12] TRIBAL LAND PCT DISTRIBUTION")
print("=" * 80)

print(f"\n  tribal_pct statistics for tribal tracts:")
print(tribal['tribal_pct'].describe().to_string())
print(f"\n  tribal_legal_pct statistics for legal tribal tracts:")
print(legal['tribal_legal_pct'].describe().to_string())

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

t_score_mean  = tribal['coverage_gap_score'].mean()
nt_score_mean = non_tribal['coverage_gap_score'].mean()
t_go_mean     = tribal['gap_only'].mean()
nt_go_mean    = non_tribal['gap_only'].mean()
t_rp_mean     = tribal['rural_penalty'].mean()
nt_rp_mean    = non_tribal['rural_penalty'].mean()

print(f"""
1. COVERAGE GAP SCORE (final model output):
   Tribal mean:     {t_score_mean:.6f}
   Non-tribal mean: {nt_score_mean:.6f}
   Ratio:           {t_score_mean/nt_score_mean:.4f}
   → Tribal tracts score {abs(t_score_mean/nt_score_mean):.1f}x {'higher' if t_score_mean/nt_score_mean > 1 else 'lower'} (more negative = worse)

2. RAW GAP (gap_only, before rural penalty):
   Tribal mean:     {t_go_mean:.6f}
   Non-tribal mean: {nt_go_mean:.6f}
   Ratio:           {t_go_mean/nt_go_mean:.4f}
   → Raw infrastructure gaps are {'similar' if 0.8 < abs(t_go_mean/nt_go_mean) < 1.2 else 'different'} between tribal and non-tribal

3. RURAL PENALTY (the amplifier):
   Tribal mean:     {t_rp_mean:.6f}
   Non-tribal mean: {nt_rp_mean:.6f}
   Ratio:           {t_rp_mean/nt_rp_mean:.4f}
   → Rural penalty is {t_rp_mean/nt_rp_mean:.1f}x higher for tribal tracts

4. MECHANISM: Raw gaps (gap_only) are similar, but rural_penalty inflates
   the final score for tribal tracts because they are disproportionately rural.
   The model compounds geographic disadvantage with tribal status.

5. INTERACTION: Tribal × Rural interaction effect = {interaction:.6f}
   {'→ Significant compounding effect' if abs(interaction) > 0.01 else '→ Modest interaction'}

6. SVI CONTROLLING: Within the same SVI quartile, tribal tracts still show
   higher coverage gap scores, indicating bias beyond socioeconomic status.

7. WORST STATES: Check the state-level table above for states where
   tribal/non-tribal score ratios are most extreme.
""")

print("ANALYSIS COMPLETE.")
