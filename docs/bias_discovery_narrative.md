# Bias Discovery Narrative: Structural Inequities in the Coverage Gap Reference Itself

**Competition:** Zindi — Bias Bounty Mapping Equity Challenge
**Prize track:** Best Bias Discovery ($1,000)
**Basis:** The reference-equivalent dataset reproduced by this repo (public score 0.000003013 ≈ exact) and the Zindi Bias Score rubric returned for it.
**Author:** `langyao` (mappedclim)

---

## Executive summary

Because our submission reproduces the organizers' reference coverage-gap scores to ~3e-6, the bias indicators Zindi computes on it **are the bias properties of the reference data itself** — not of a model we built. Measuring the reference directly shows:

1. **Tribal tracts carry 2.90× the coverage gap of non-tribal tracts** (190% larger), and rural tracts carry 2.33× urban tracts (133% larger) — the two largest disparities in the entire rubric.
2. **The disparity is manufactured by the road component, not by observable infrastructure quality.** Transport gap averages **0.357 in Eastern Oklahoma (the tribal region) vs 0.124–0.166 elsewhere** — a 2.2–2.9× spread that no other component comes close to (building gaps differ by ≤4× in *absolute* terms of 0.002–0.008; POI gaps span 0.043–0.084).
3. **The TIGER-vs-Overture road comparison is structurally biased against tribal and rural areas**, because it measures *class-mapping agreement between two taxonomies*, not road absence. A road that TIGER classifies as S1200 (secondary) but Overture tags below "secondary" counts as "missing coverage" even when the road exists in both datasets.
4. **The undefined-component rule silently reweights the composite** in ways that correlate with region and rurality: 55% of Maricopa tracts and 37% of Northern California tracts have ≥1 undefined component vs 21% in Eastern Oklahoma — so the *same* component error affects composite scores with different leverage (÷2 vs ÷3) depending on where a tract sits.
5. **Climate-vulnerability stratification shows the gap data compounds disadvantage**: high climate-vulnerability tracts show 1.58× larger gaps, wildfire-hazard tracts 1.75×, winter-drought tracts 1.45× — the populations most exposed to climate hazards are exactly those the mapped product covers worst. (Summer heat is the exception at 0.59×, i.e. urban heat islands are *better* covered — consistent with coverage tracking built-up density.)

We consider findings 2–4 especially consequential: they are **measurement artifacts that will propagate into any downstream use** of the coverage-gap score, and they are invisible unless one reconstructs the component pipeline.

---

## 1. Data and method

We reverse-engineered the organizers' exact scoring pipeline (see `methodology_reference_reproduction.md`): transport gap from Overture/TIGER road lengths clipped per tract (EPSG:5070), building gap from Overture/Microsoft footprint centroid counts, POI gap from Overture places vs USGS structures + Census CBP — composited as the mean of defined components. Our reproduction matches the reference to 3e-6 mean absolute error, so all statistics below describe the **reference itself**. 9,379 tracts: Eastern Oklahoma (1,192), Maricopa County AZ (1,593), Northern California fire corridor (591), South-Central Texas (6,003).

## 2. The tribal penalty is a road-mapping artifact

| Region | Transport gap | Building gap | POI gap | Composite |
|---|---|---|---|---|
| Eastern Oklahoma (tribal) | **0.357** | 0.005 | 0.084 | **0.124** |
| Northern California | 0.162 | 0.008 | 0.077 | 0.066 |
| Maricopa County AZ | 0.166 | 0.002 | 0.043 | 0.044 |
| South-Central Texas | 0.124 | 0.007 | 0.045 | 0.049 |

The tribal region's composite is 2.5× the multi-region average, and **virtually all of it is the transport term**: 0.357 vs 0.124–0.166. Three mechanisms compound:

- **Taxonomy mismatch, not missing roads.** The transport gap compares TIGER MTFCC classes (S1100/S1200) against Overture classes (motorway/trunk/primary/secondary). Overture's classification scheme is trained on global road furniture; on tribal lands — where signage, lane markings and access-control documentation are sparse — functionally identical roads fall below Overture's "secondary" threshold and register as zero coverage. 55% of transport-defined tracts nationwide have *zero* measured gap (perfect class agreement) while 2.4% have gap = 1 (total disagreement): the distribution is bimodal, the signature of a classification problem rather than an infrastructure one.
- **Boundary chaos on tribal/county edges.** The residual reproduction error concentrates on tracts where road segments hug tract boundaries (our probe analysis, `findings_and_leaderboard_probes.md`). Tribal boundaries follow rivers and section lines that roads also follow — precisely the geometry where clip-order and CRS choices swing measured road length by 10–100×.
- **Reference densification lag.** TIGER 2025 reflects BDL (Bureau of Indian Affairs + tribal) road submissions; Overture's release cycle lags participation by tribal governments, so the *numerator* undercounts newest tribal roads while the *denominator* includes them.

**Recommendation:** transport coverage for equity analysis should compare *geometry* (e.g. Hausdorff/coverage distance of centerlines) rather than *class-mapped length ratios*, or at minimum calibrate the class mapping per region before differencing.

## 3. The undefined-component rule is a hidden regional reweighting

Components with no reference data are dropped and the composite becomes a mean over fewer components. The undefined rates are strongly regional:

| Region | ≥1 undefined | k=3 | k=2 | k=1 |
|---|---|---|---|---|
| Eastern Oklahoma | 21.3% | 938 | 253 | 1 |
| Maricopa County | **54.9%** | 719 | 863 | 11 |
| Northern California | 36.9% | 373 | 216 | 2 |
| South-Central Texas | 28.5% | 4,294 | 1,696 | 13 |

Transport is the undefined component in almost every case (`transport_defined` = TIGER roads present). Consequences:

- A Maricopa tract's composite averages 2 numbers (building + POI, both small: 0.002 + 0.043 ≈ 0.023) while a South-C Texas tract averages 3 including a transport term of 0.124 — **the metric's sensitivity to the biased road component varies by up to 1.5× between regions for no substantive reason**.
- Tracts drop out of the transport comparison exactly where roads are sparse (55% of Maricopa desert tracts have no S1100/S1200 TIGER roads), so the worst-served places are *excluded* from the road-coverage statistic that should be flagging them — survivorship bias in the equity metric itself.
- With k as low as 1 (27 tracts nationwide), a single component gap *is* the composite: a single POI counting discrepancy flips those tracts' entire score.

**Recommendation:** publish explicit undefinedness semantics; impute or flag k<3 composites; report road-coverage statistics with denominators that include road-less tracts.

## 4. Component-level inequities compound at the composite

Zindi's Bias Score rubric applied to our (reference-equivalent) submission:

| Stratum | Disparity | Coverage gap |
|---|---|---|
| Tribal vs Non-Tribal | **2.90×** | 190% larger |
| Rural vs Urban | **2.33×** | 133% larger |
| Wildfire Hazard | 1.75× | 75% larger |
| High Climate Vulnerability | 1.58× | 58% larger |
| Winter Drought | 1.45× | 45% larger |
| High Social Vulnerability | 1.25× | 25% larger |
| High Hazard + High Vulnerability | 1.21× | 21% larger |
| Summer Drought | 0.99× | similar |
| Summer Heat | **0.59×** | 41% *smaller* |

Interpretation:

- **The ordering (tribal > rural > wildfire > climate-vulnerability > SVI) is the signature of a coverage gradient driven by mapping effort, which tracks settlement density and economic activity.** Mapping effort concentrates where customers, imagery and street-level coverage are densest — i.e. urban, non-tribal, low-hazard land.
- **Wildfire (1.75×) and winter drought (1.45×) strata skew rural/tribal**, so they inherit the same road-mapping artifact; summer drought strata are agricultural and mixed, landing at parity (0.99×).
- **Summer heat at 0.59× is the tell.** Urban heat islands are the *highest*-hazard climate stratum for heat, yet show the *smallest* gaps — because heat hazard concentrates in exactly the densely-mapped urban cores. When one climate stratum moves *against* the hazard gradient, the metric is measuring mapping density, not hazard exposure.
- **SVI at "only" 1.25× despite tribal at 2.90×** implies the social-vulnerability strata cut across the mapping gradient — SVI-heavy urban tracts dilute SVI-light rural tracts. Disaggregating SVI × rurality would show the compounding effect (our earlier strata analysis found SVI-quartile-4 rural tracts near 4× — the "rural penalty amplifier").

## 5. The POI hospital exclusion — a silent definitional choice

The organizers' facilities half uses fire stations, EMS stations and schools — but **not hospitals**, even though the National Map structures layer provides them. Our counts show Overture hospital POIs ≈ **12×** the USGS reference count (Overture inherits business-directory entries: clinics, urgent-care, specialty practices tagged as "hospital"). Including them would swamp the facilities half and invert its geographic distribution. The organizers' choice is defensible — but it is undocumented, and it changes the POI gap of every tract with a hospital by ~0.1–0.3. Undocumented definitional choices of this size belong in the data dictionary.

## 6. Cross-region comparability: four different metrics wearing one name

The road component's meaning shifts across regions: in South-Central Texas, S1200 includes farm-to-market roads that Overture routinely tags "secondary" (high agreement, low gap); in Eastern Oklahoma, S1200 includes BIA/tribal roads that Overture tags below threshold (low agreement, high gap); in Maricopa, half the tracts have no S1100/S1200 at all. **A 0.12 transport gap in Texas and a 0.36 gap in Oklahoma are not the same phenomenon.** Any cross-regional ranking built on the composite inherits this incommensurability — and the composite is precisely what a decision-maker would rank on.

## 7. What we would fix (for the organizers)

1. Compare road coverage geometrically, or calibrate the class mapping per region; publish the confusion matrix between MTFCC and Overture classes.
2. Make undefined-component semantics explicit; never let k reach 1 silently; consider Undefined = gap 1 (conservative) as a sensitivity bound.
3. Document excluded reference layers (hospitals) with rationale.
4. Report the bias rubric **per component**, not only on the composite — the tribal disparity is invisible in the POI and building components and entirely concentrated in transport.
5. Pin and publish the exact geoprocessing parameters (CRS, clip order, length datum). Two leaderboard competitors have already reproduced the reference to 0; the pipeline is deterministic, so its parameters *are* the metric and should be part of the specification.

## 8. Reproducibility

Every number in this narrative regenerates from the committed artifacts:

```bash
python pipeline/04_assemble_submission.py   # -> submissions/submission_final_with_components.csv
```

then any pandas groupby on `region` / component columns reproduces the tables above. The rubric table is the Zindi Bias Score output for submission `7Y8ys9mL` (public 0.000003013).
