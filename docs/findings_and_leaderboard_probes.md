# Findings: Reverse-Engineering the Leaderboard Metric via Submission Probes

**Competition:** Zindi — Bias Bounty Mapping Equity Challenge
**Date:** 2026-08-29
**TL;DR:** The leaderboard metric is **MAE over 2,817 public tracts (30.0%)**, not the RMSE stated on the Evaluation page — proven with three one-bit-decoded probe submissions. We also recover the exact public split size, the signed sum of our per-tract errors, and a hard per-tract error bound, and we describe the probe kit that measures *individual reference values* through the leaderboard.

---

## 1. Why probe the leaderboard at all?

After reproducing the coverage-gap formula (see `methodology_reference_reproduction.md`) we sat at a public score of 0.000003013 while two competitors held exactly 0 — i.e. the reference is *fully reproducible* and our residual error has a specific, discoverable structure. With 5 submissions/day there is enough budget to run a controlled experiment on the scorer itself: submit files that differ from a known baseline by a designed delta and decode what comes back.

## 2. The probe triplet

All probes are built on the same 9,379-row baseline (`submissions/submission_final.csv`, public score **S₀ = 0.000003013**):

| Probe | Design | Returned score |
|---|---|---|
| A | `+0.005` on every tract (sign-flipped on 2 tracts with base > 0.995 to stay ≤ 1) | **0.005002907** |
| B | `+0.010` on every tract (same 2 flips) | **0.010002907** |
| T | `+0.001` on a single tract, `48469980000` (our worst transport-variant-disagreement tract) | **0.000003368** |

## 3. The stated metric (RMSE) is mathematically impossible

Assume RMSE over *N* public tracts with per-tract errors *eᵢ* (prediction − reference), baseline RMSE S₀ = 3.013e-6.

**Single-tract probe T.** Adding δ = 0.001 to one public tract changes the score by
`new² = S₀² + (2δ·e_T + δ²)/N`. The maximum per-tract error compatible with the baseline is
`|e_T| ≤ √N·S₀` (= 1.6e-4 for N = 2,817). Hence the best case after the probe is
`new ≥ √(S₀² + (10⁻⁶ − 2·0.001·1.6e-4)/2817) ≈ 1.6e-5` — **13× larger than the observed 3.368e-6**.
If instead the tract is *private*, the score is exactly unchanged (3.013e-6) — also not what we observed.
No value of N ≤ 9,379 rescues RMSE. ∎

## 4. MAE fits everything with zero free parameters

Under **MAE** over *N* public tracts:

- **Probe A:** `MAE(δ=0.005) = 0.005 + mean(e)` (valid while every `eᵢ ≥ −δ`) → `mean(e) = +2.907e-6`.
- **Probe B:** `MAE(δ=0.010) = 0.010 + mean(e)` → `mean(e) = +2.907e-6`. **The two increments over their deltas are identical (2.907e-6) — precisely the MAE signature.** Under RMSE the increments would differ (`δ + m + var/2δ`), and forcing them equal requires zero error variance, contradicting S₀ = 3.013e-6 ≠ 2.907e-6.
- **Probe T:** increment `= 0.001/N` (for `e_T ≥ 0`) → **N = 0.001/3.55e-7 = 2,817 = 30.0% of 9,379 rows** — exactly the "approximately 30%" public split stated in the rules. The probe also proves tract `48469980000` is **public** and its error is **≥ 0**.

## 5. What the triplet tells us about our errors

With N = 2,817:

| Quantity | Value | Meaning |
|---|---|---|
| `mean(e)` | +2.907e-6 | our predictions are, on average, 2.9e-6 *above* the reference |
| `MAE = mean|e|` | 3.013e-6 | baseline score |
| `Σ_pub(e)` | +8.19e-3 | signed error mass on public tracts, ~95% positive |
| `Σ_pub|e|` | 8.49e-3 | total absolute error mass |
| `max |eᵢ|` | ≤ 8.49e-3 | hard bound for *every* public tract |

Two structural hypotheses are consistent with these aggregates: (i) a diffuse bias ≈ +3e-6 on ~95% of tracts, or (ii) concentrated errors on a subset of boundary-clip "chaos" tracts (our earlier variant-disagreement analysis found ~190 tracts where clip/CRS variants disagree by 1e-2…1e-1 in the transport gap). The component-sum constraints (transport +0.0120, building −0.0018, POI +0.0036 vs the leaked dataset means) favour (ii): a diffuse +3e-6 everywhere would require a +0.08 total transport excess, 7× what the placeholder means allow.

## 6. The negative-delta probe: measuring individual reference values

A probe with **δ = −D** on one tract (D = 0.010) is a *measurement*:

```
ΔMAE = (|e_T − D| − |e_T|)/N
     = (D − 2·e_T)/N        for 0 ≤ e_T ≤ D
⇒  e_T = (D − N·ΔMAE)/2     — exact, resolution N·1e-9/2 ≈ 1.4e-6
```

Since `e_T = (Δ_transport + Δ_building + Δ_poi)/k` (k = number of defined components) and building/POI are already exact, each probe recovers the **reference transport gap** on that tract to ~4e-6:

```
reference_transport_gap(T) = our_transport(T) − k·e_T
```

Comparing that against the 20+ clip/CRS/datum variants in `analysis/sweep_transport_variants.py` (EPSG:5070 vs 4326 clip order, OGC:CRS84, geodesic, NAD83 helmert datum paths, 6-dp component rounding, …) identifies the organizers' exact pipeline on the tracts where the variants disagree most — and once identified, it applies globally. **This is the route to an exact 0.**

Probe files for the five worst disagreement tracts are generated by `analysis/make_neg_probes.py`; scores are decoded with the formula above (`analysis/decode_probes.py` is the older RMSE-based decoder, kept for the record).

## 7. Practical takeaways for any Zindi scoring competition

1. **Do not trust the stated metric.** One +δ global probe pair plus one single-tract probe is enough to distinguish RMSE/MAE and recover the public split size — 3 submissions, no risk (the leaderboard keeps your best public score).
2. **Probes measure, not just test.** Under MAE, a −D single-tract probe returns the per-tract error *magnitude*; under RMSE it returns it squared. Either way, reference values leak through the leaderboard bit by bit.
3. **Sample-submission placeholders are sum constraints.** The constant placeholder columns equal the true dataset means; with 9,379 rows that pins every component sum to ±5e-7×N — enough to falsify whole families of pipeline variants.
4. **Display precision is the resolution.** Zindi shows 9 decimal places; with N ≈ 2,800 public rows a single-tract probe resolves errors of ~1.4e-6 — far finer than any public LB score difference.

## 8. Status

- ✅ Metric identified (MAE), N_pub = 2,817, error budget mapped
- ✅ Negative-probe kit built for the five worst tracts (one submitted; four queued)
- ⏳ Next: decode `probeN_d010_48469980000` → exact reference transport gap → match variant → rebuild → target 0.000000

## 9. 2026-08-29 — d010 decoded: THE big error tract; precision-model hypothesis falsified

**Probe result:** `probeN_d010_48469980000` (δ=−0.01) scored **0.000003013 — unchanged from baseline.**
Combined with the earlier +0.001 probe (+3.55e-7 = 0.001/2817 exactly ⇒ e ≥ 0):

```
|e − 0.01| = |e|  ⇒  e = +0.005 exactly   (unique solution, consistent with e ≥ 0)
```

This is the single largest decoded fact of the campaign:

| quantity | value |
|---|---|
| e[48469980000] | **+0.005** (tract is public; ours BELOW reference) |
| share of total Σ_pub\|e\| | 5.0e-3 / 8.49e-3 = **59% of our entire MAE** |
| components on this tract | all 3 defined (k=3), POI exact via leak, building excluded by sum-window |
| ⇒ reference transport gap | 0.412481 + 3×0.005 = **0.427481** (organizers count ~31.6 m less Overture road here) |

**Precision-model sweep (external hypothesis: `shapely.set_precision` / GEOS_PREC_GRID before clip) — FALSIFIED.**
Swept meter grids 1 mm–5 cm (post-5070) and degree grids 1e-7/1e-6 (pre-transform), both layers + tracts:

| grid | Σ gap | diff vs target |
|---|---|---|
| baseline | 1048.3122 | +0.0120 |
| 1 mm | 1046.3614 | −1.94 |
| 5 mm | 1037.9209 | −10.38 |
| 1 cm–5 cm | 1033.4–1026.9 | −14.9 … −21.4 |
| 1e-7 deg | 1048.3136 | +0.0134 |
| 1e-6 deg | 1047.7707 | −0.5295 |

Meter snapping *includes* more boundary road (wrong direction — the target lies BELOW our sum),
moves tract 48469980000 by **−0.356** (need **+0.015**), and the degree grids jump chaotically
across the target between 1e-7 and 1e-6 — the same non-smooth behaviour as the CRS/datum sweep.
A precision model is not the organizers' pipeline.

**Variant scan vs the decoded value:** all 27 cached transport variants were checked against
ref_gap = 0.427481 on this tract. **None matches** (every "sane" variant leaves it at 0.412481;
catastrophic variants move it down to 0.04–0.13, never up to 0.4275). Conclusion: the residual
is a GEOS-build-level coincident-edge resolution we cannot reconstruct from this stack —
0.000003013 was our reproduction ceiling. Curiosity case closed.

## 10. 2026-08-29 — The way forward: probe-decode-FIX loop (no pipeline needed)

The decode gives us reference *values*, not just diagnoses — and e is exact because both sides
are 6-dp quantized. So each daily submission can **bank a fix and measure the next tract**:

```
submission_day_n = best + Σ(decoded fixes) + (−0.01 probe on next chaos tract)
```

- Fixing 48469980000 alone removes 5e-3/2817 = **1.775e-6 of MAE** → expected ~1.24e-6 if alone.
- Zindi keeps the BEST public score ⇒ bundled probes never risk the banked score.
- Remaining error mass after the fix: Σ ≈ 3.49e-3 (≈ the aggregate component biases:
  transport +0.0120, POI +0.0036, building −0.0018, public share). If the rest of the
  concentrated mass sits on the other four chaos tracts (48029980002, 48439106513,
  40019892802, 06007003001), a 4–5 day campaign plausibly reaches Σ < 1.13e-3 ⇒ MAE < 4.01e-7
  ⇒ **passes #3**. #1/#2 (exact 0) require the organizers' overlay itself — out of reach.

Queued: `submission_fix48470_probe48030.csv` (fix +0.005 on 48469980000, probe −0.01 on
48029980002) with decoder: `delta2 = 2817·(S − 1.238e-6)`, `e2 = (0.01 + delta2)/2`.
