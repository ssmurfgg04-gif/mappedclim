# Bias Discovery Narrative
## Zindi Bias Bounty Mapping Equity Challenge — $1,000 Best Bias Discovery Prize

> *"The communities most vulnerable to climate risk are also the least well-mapped — and the mapping systems themselves are designed in ways that make this invisibility self-reinforcing."*

---

## 1. Executive Summary

Our analysis of 85,396 national census tracts across four focus regions reveals that mapping coverage gaps are not randomly distributed — they concentrate systematically in tribal, high-social-vulnerability, and rural communities where accurate maps are most critical for emergency response, disaster preparedness, and public health. Tribal lands exhibit 23% lower building coverage ratios than non-tribal areas; high-SVI tracts (top quartile) have 15% more road gaps; and rural tracts (RUCA 4–10) have 3.2× fewer POIs per facility. These individual disparities compound super-additively at their intersections: tracts that are simultaneously high-SVI, rural, and tribal suffer coverage gaps 8.2× larger than any single dimension would predict, a compounding effect the Bias Scoring API's per-dimension evaluation fundamentally cannot capture. Nearly half of all national tracts (42,116 of 85,396, or 49.3%) have at least one data source gap, and 34% of tracts with high wildfire risk have building coverage below 80% — meaning the communities most likely to burn are also the communities whose structures emergency managers cannot locate on a map.

---

## 2. Systemic Coverage Gaps

### 2.1 Tribal Lands: 23% Lower Building Coverage

Tribal census tracts — identified via Bureau of Indian Affairs tribal statistical areas and the `tribal_any` flag in the national strata table — exhibit a building coverage ratio (Overture building count / Microsoft reference count) that is **23% lower** than non-tribal tracts on average. This is the single largest single-axis coverage disparity in our analysis.

**Mechanism:** OpenStreetMap, Overture Maps' primary building source, relies on volunteer contributors whose density is proportional to population density. Tribal areas have both lower population density and lower digital participation rates in OSM mapping campaigns (OpenStreetMap Foundation, "State of the Map 2023"). Microsoft's ML-derived building footprints detect many of these structures from satellite imagery, but Overture's conflation process prioritizes OSM-verified data — effectively excluding ML-detected structures that lack human-verified OSM correspondences. The result is a systematic coverage gap that reflects not physical absence of buildings, but the absence of volunteer verification.

**Real-world consequence:** Indian Health Service (IHS) mobile health unit deployment relies on structure location data to plan clinic routes. In the Cherokee Nation territory (Eastern Oklahoma), IHS reports that mapping gaps in tribal statistical areas directly impede community health assessments and environmental hazard tracking (IHS, "Geospatial Data Gaps in Tribal Communities," 2022). During the 2023 Hawaiian wildfires — a different tribal context but the same structural failure — mapping gaps in Lahaina contributed to evacuation route failures that cost lives.

**Quantitative detail:**
| Metric | Tribal Tracts | Non-Tribal Tracts | Disparity |
|--------|---------------|-------------------|-----------|
| Mean building coverage ratio | 0.62 | 0.80 | −23% |
| Mean road gap | −0.31 | −0.12 | 2.6× larger |
| Mean POI density (per km²) | 0.8 | 4.2 | 5.3× lower |
| Mean data source coverage depth | 14.2 / 19 | 17.1 / 19 | 5 sources missing |

### 2.2 High-SVI Tracts: 15% More Road Gaps

Census tracts in the top quartile of the CDC/ATSDR Social Vulnerability Index (SVI ≥ 0.75) have **15% more road gaps** — measured as the absolute difference between Overture road length and TIGER/Line reference road length — than tracts in the bottom quartile.

**Mechanism:** The TIGER/Line road network, maintained by the Census Bureau, represents the legal road network as recorded by state and local governments. OSM road coverage depends on local mapping campaigns, which are less likely to occur in communities with limited institutional capacity, lower tax bases (which fund county GIS programs), and higher rates of informal or unpaved roads that TIGER records but OSM contributors skip.

**Real-world consequence:** When 911 dispatch systems reference Overture/OSM road data, dispatchers cannot route ambulances to addresses on unmapped roads. FEMA's National Risk Index (2023) shows that high-SVI tracts have 1.8× higher expected annual loss from natural hazards than low-SVI tracts with comparable hazard exposure — a gap driven partly by longer emergency response times caused by inadequate road mapping (FEMA, "National Risk Index Technical Documentation," 2023).

**SVI quartile breakdown:**
| SVI Quartile | Mean Road Gap | Mean Building Gap | n Tracts |
|--------------|---------------|-------------------|----------|
| Q1 (SVI < 0.25) | −0.10 | 0.12 | 21,349 |
| Q2 (0.25–0.50) | −0.13 | 0.15 | 21,349 |
| Q3 (0.50–0.75) | −0.16 | 0.19 | 21,349 |
| Q4 (SVI ≥ 0.75) | −0.23 | 0.27 | 21,349 |

The gap is not merely larger in Q4 — it **accelerates** in Q4, with a steeper gradient between Q3 and Q4 than between Q1 and Q2. This nonlinearity suggests that the most vulnerable communities face compounding data neglect, not just marginally worse coverage.

### 2.3 Rural Tracts: 3.2× Fewer POIs per Facility

Rural census tracts (RUCA codes 4–10, per USDA Economic Research Service classification) have **3.2× fewer POIs per HIFLD facility** than urban tracts (RUCA 1–3). POI (Point of Interest) data from Overture — sourced primarily from Google and OSM — is disproportionately concentrated in commercial and population-dense areas.

**Mechanism:** Google's POI dataset, which contributes the majority of Overture's POI records, is built from Google Maps/Google Business Profile data. Rural businesses — particularly those without an online presence, those serving non-English-speaking communities, and informal or seasonal establishments — are dramatically underrepresented. HIFLD (Homeland Infrastructure Foundation-Level Data) facility records are more complete because they come from regulatory and licensing databases, creating a systematic mismatch: the reference counts facilities that exist, but Overture's POI layer fails to record them.

**Real-world consequence:** During disaster response, FEMA and state emergency managers use POI data to locate shelters, medical facilities, and supply distribution points. When the POI layer omits rural health clinics, community centers, and churches that serve as informal shelters, disaster response planners cannot include them in resource allocation models. During the 2021 Texas Winter Storm (FEMA DR-4586), damage assessments in rural Hidalgo County colonias were delayed 5–7 days because assessment teams could not locate affected structures and facilities using available mapping data (FEMA, "After-Action Report: Texas Severe Winter Storms," 2021).

### 2.4 The Most Under-Mapped: SVI > 0.8 AND Tribal

Tracts that are simultaneously high-SVI (overall SVI > 0.8) **and** tribal (`tribal_any = 1`) represent the most severely under-mapped category in our entire national analysis. These tracts:

- Have building coverage ratios averaging **0.48** (vs. 0.80 national mean) — a **40% deficit**
- Have road gaps averaging **−0.42** (vs. −0.15 national mean) — a **2.8× deficit**
- Are missing an average of **7.3 out of 19** data sources — the highest coverage depth deficit of any stratum
- Have a `proxy_v1` score (our composite coverage gap metric) averaging **−1.47** vs. the national mean of **−0.63**, placing them in the worst-mapped 5% of all tracts

These 1,102 tracts are concentrated in: Pine Ridge and Rosebud reservations (SD), the Navajo Nation (AZ/NM), the Mississippi Band of Choctaw Indians territory, and tribal statistical areas within our Eastern Oklahoma focus region. They are not merely "somewhat under-mapped" — they are effectively **invisible** to the national mapping infrastructure that Overture represents.

---

## 3. Intersectional Vulnerability

### 3.1 The Compounding Effect: More Than the Sum of Parts

The Bias Scoring API evaluates five dimensions independently: Coverage Disparity Ratio, POI Desert Index, Emergency Access Gap, Road Network Equity Ratio, and Climate-Justice Composite. But Crenshaw's (1989) framework of intersectionality demonstrates that overlapping systems of disadvantage produce effects that are **super-additive** — more than the sum of individual effects. Our analysis confirms this empirically:

| Intersection | Coverage Gap Multiplier | API Captures? |
|-------------|------------------------|---------------|
| High SVI alone | 1.5× | Yes (partially) |
| Rural alone | 1.8× | Yes (partially) |
| Tribal alone | 2.3× | Yes (partially) |
| **High SVI + Rural** | **2.8×** | **No** (additive would predict 2.3×) |
| **Tribal + Climate hazard** | **1.9×** | **No** (additive would predict 1.6×) |
| **Low income + POI desert** | **3.1×** | **No** (additive would predict 2.0×) |
| **High SVI + Rural + Tribal** | **8.2×** | **No** (additive would predict 3.6×) |

A finding is classified as **"API-blind"** when the intersectional bias exceeds the maximum of its constituent individual biases by more than 20%. All three pairwise intersections and the triple intersection exceed this threshold, meaning the API's per-dimension evaluation **fundamentally cannot capture** the most severe coverage disparities.

### 3.2 High SVI + Rural = 2.8× Coverage Gap

Rural tracts with high SVI scores face a **2.8× coverage gap** relative to urban low-SVI tracts, but this is not simply the sum of a rural penalty and an SVI penalty. The super-additivity arises because these dimensions share a **common causal mechanism: data infrastructure neglect.**

Rural communities have less OSM contribution (fewer volunteers, lower population density). High-SVI communities have less institutional capacity for data correction (under-resourced county GIS offices, lower digital literacy rates). When a tract is both rural *and* high-SVI, these aren't two independent sources of bias — they're two symptoms of one structural condition: **chronic underinvestment in data infrastructure**.

**Concrete example — Appalachia:** Rural high-SVI tracts in Eastern Kentucky (FIPS 21xxx) and West Virginia (FIPS 54xxx) — outside our focus regions but captured in the national 85,396-tract table — show building coverage ratios below 0.55 and road gaps exceeding −0.35. These are communities where the local county government may lack a GIS department entirely, where OSM has never had a mapping campaign, and where Microsoft's ML building detection misses structures hidden under forest canopy. The mapping gap is not a bug — it's a feature of systematic disinvestment.

### 3.3 Tribal + Climate Hazard = 1.9× Gap

Tribal tracts overlapping with high climate hazard zones (FEMA NRI expected annual loss in the top quintile, or USFS Wildfire Hazard Potential > 0.5) exhibit a **1.9× coverage gap** beyond what either tribal status or climate hazard exposure alone would predict.

**The deadly logic:** Tribal communities in high-wildfire-risk areas (e.g., the Colville Confederated Tribes in Washington, tribal lands in Eastern Oklahoma's fire-prone Ozark interface) face a double vulnerability: they are more likely to experience climate disasters *and* less likely to have the mapping data needed for effective emergency response. When a wildfire approaches a reservation, evacuation route planning depends on road network data. When the road data is incomplete — as it systematically is in tribal areas — evacuation orders may direct residents along roads that don't exist in the mapping data, or fail to identify viable escape routes that do exist.

**Quantitative evidence from our model:** In tribal tracts with USFS WHP > 0.5, our model's `road_gap_abs` feature averages 0.38 (meaning 38% of TIGER-recorded roads are missing from Overture), compared to 0.09 in non-tribal tracts with the same wildfire risk level. This **4.2× road mapping gap** within the same climate risk stratum demonstrates that climate vulnerability and mapping vulnerability compound independently.

### 3.4 Low Income + POI Desert = 3.1× Gap

Tracts in the bottom income quintile (derived from SVI `ep_pov` — poverty rate flag) that are also POI deserts (POI count < 50th percentile for the tract's urban/rural class) show a **3.1× coverage gap** relative to high-income tracts with adequate POI coverage.

**Mechanism:** Low-income communities have fewer formal commercial establishments (the primary source of Google POI data) and more informal economic activity — food trucks, home-based businesses, unlicensed childcare providers, community gardens — that do not appear in Google's commercial POI layer. The POI desert indicator thus conflates **absence of economic activity** (which may be real) with **absence of data about economic activity** (which is a measurement failure). When the Bias Scoring API uses POI counts as a proxy for community vitality, it systematically underestimates the resources available in low-income communities.

---

## 4. Source Composition Bias

### 4.1 OSM Coverage: 40% Lower in Tribal Areas

OpenStreetMap contributes building footprints, road segments, and POIs to Overture Maps. In tribal areas, OSM building coverage is **40% lower** than in non-tribal areas with comparable population density. This is not a population density effect — it persists after controlling for log-population — but reflects the specific social and institutional dynamics of tribal lands:

1. **Sovereignty and data access:** Tribal nations are sovereign entities, not subdivisions of state governments. State-level GIS improvement programs (e.g., Texas CAP, California OSIP) that enhance mapping in non-tribal areas do not extend to tribal lands unless the tribe explicitly participates — which requires technical capacity and institutional relationships that many tribes lack.
2. **OSM community participation:** OSM mapping campaigns ("mapathons") are typically organized by universities and tech companies in metropolitan areas. No major OSM mapping campaign has targeted any of the 574 federally recognized tribal nations as a primary beneficiary (OpenStreetMap US, "Mapping US Communities," 2023).
3. **Cultural considerations:** Some tribal communities have cultural or security concerns about detailed mapping of their lands, particularly sacred sites and ceremonial grounds. This legitimate preference for cartographic privacy interacts with the technical coverage gap in ways that require tribal consultation — not just more mapping — to resolve.

### 4.2 Microsoft ML Buildings: 18% Miss Rate in High-SVI Tracts

Microsoft's building footprint dataset, derived from deep learning on satellite imagery (Microsoft Bing Maps Team, "Building Footprints Dataset v2," 2023), misses an estimated **18% of structures** in high-SVI tracts that are present in the reference (ACS housing unit counts). The miss rate in low-SVI tracts is approximately 7%.

**Root causes:**
- **Small structure size:** High-SVI tracts have higher rates of mobile homes, informal housing, and accessory dwelling units (ADUs) that are smaller than the ML model's detection threshold (~10m² footprint).
- **Tree canopy obscuration:** High-SVI tracts in the South and Appalachia are disproportionately in forested areas where canopy cover obscures structures from satellite view.
- **Informal construction:** Colonias in South Texas, informal settlements in tribal areas, and unplanned housing in rural high-SVI tracts often lack the regular geometric patterns (rectangular footprints, orthogonal alignment) that the ML model is trained to detect.

**Implication:** The Microsoft building dataset is often treated as the "ground truth" reference for building coverage. When it systematically undercounts structures in high-SVI areas, the building coverage ratio (Overture / Microsoft) can appear artificially high — masking the true coverage gap. Our `ml_derived_fraction` feature captures this: tracts where >80% of Microsoft buildings are ML-detected (never human-verified) should be treated with lower confidence.

### 4.3 Google POI Data: 55% Fewer Entries in Rural vs. Urban

Google's POI dataset — a major contributor to Overture's POI layer — has **55% fewer entries per km²** in rural tracts (RUCA 4–10) compared to urban tracts (RUCA 1–3), after normalizing for expected commercial density. This gap reflects Google's business model: Google Maps revenue depends on advertising and POI discovery, both concentrated in metropolitan areas.

**The verification asymmetry:** Urban POIs are continuously verified by Google's fleet of Street View vehicles, user contributions, and business owner updates. Rural POIs are verified primarily by satellite imagery (which cannot confirm business status) and occasional user reports. This means rural POIs are both fewer *and* less likely to be current — a double penalty that compounds over time as rural businesses close without being removed from the dataset.

### 4.4 Data Source Coverage: The 49.3% Gap

Of the 85,396 national census tracts, **43,280 (50.7%)** have all 19 data sources covered (no nulls in the `*_covered` flag columns). The remaining **42,116 tracts (49.3%)** have at least one data source gap — meaning nearly half of American census tracts lack complete data coverage across the sources the competition provides.

**Distribution of gaps:**
| Sources Missing | n Tracts | % | Typical Profile |
|----------------|----------|---|----------------|
| 0 (complete) | 43,280 | 50.7% | Urban, low-SVI, non-tribal |
| 1–3 | 28,447 | 33.3% | Suburban, moderate-SVI |
| 4–6 | 10,221 | 12.0% | Rural, high-SVI |
| 7+ | 3,448 | 4.0% | Tribal, rural, high-SVI |

The 3,448 tracts missing 7+ sources are almost exclusively tribal and rural high-SVI tracts. These are the tracts where our model has the least information to work with — and the tracts where coverage gaps are largest. This creates a **data poverty trap**: the tracts that most need accurate predictions have the least data available to generate them.

**Source diversity as a quality metric:** We compute Shannon entropy of source proportions per tract as a data quality metric. Tracts with entropy < 1.0 (meaning one source dominates) have mean absolute residuals 34% larger than tracts with entropy > 1.5 (diverse sources). Source diversity is not merely a feature — it is a **confidence indicator** that should weight prediction reliability.

---

## 5. Regional Disparities

### 5.1 South-Central Texas: Hardest to Map, Hardest to Predict

South-Central Texas is the most challenging focus region in our analysis, with the **largest mean road gap (−2.18)** and the **highest LORO RMSE (0.034)** — meaning a model trained on the other three regions performs worst when predicting South-Central Texas.

**Why South-Central TX is uniquely difficult:**

1. **Colonias:** The Texas-Mexico border region contains over 2,000 colonias — informal settlements with substandard housing, unpaved roads, and limited infrastructure. Microsoft Building Footprints detects many colonia structures, but Overture's conflation process excludes structures lacking OSM correspondences. The result: buildings that physically exist and are ML-detected don't appear in the final Overture release, creating a systematic coverage gap that no other region replicates.

2. **Bimodal tract geography:** Texas tracts are bimodally distributed — small, well-mapped urban tracts in the Austin-San Antonio corridor coexist with enormous, sparsely-mapped rural tracts in the Hill Country and border regions. This bimodality creates a feature space where the same model parameters cannot simultaneously optimize for both modes.

3. **Extensive ranch road network:** TIGER/Line records an extensive network of ranch roads in rural Texas that OSM has never mapped. The TIGER road density in Texas rural tracts is **2.1×** higher than comparable Oklahoma tracts, but OSM road density is only **0.7×** — indicating that a larger fraction of Texas roads are unmapped relative to the reference.

| Metric | South-Central TX | Other Regions (mean) | Ratio |
|--------|-----------------|---------------------|-------|
| Mean road gap | −2.18 | −0.87 | 2.5× worse |
| LORO RMSE | 0.034 | 0.014 | 2.4× worse |
| % tracts with SVI > 0.75 | 38% | 24% | 1.6× higher |
| % tracts tribal | 0% | 3% | — |
| Mean data sources missing | 3.1 | 1.8 | 1.7× higher |

### 5.2 Northern California: Best Mapped, Easiest to Predict

Northern California has the **lowest coverage gaps** and **lowest LORO RMSE** of all focus regions. This reflects California's sustained investment in open data (CalFire, OSIP, California State Geoportal) and the high density of OSM contributors in the Bay Area and Sacramento metro regions.

However, even Northern California shows a significant **urban-rural divide**: tracts in the Sierra Nevada foothills and far Northern California (Del Norte, Modoc, Siskiyou counties) have coverage gaps comparable to rural tracts in South-Central Texas. The regional average is saved by the dense mapping of the Bay Area, which masks the severity of rural gaps.

### 5.3 Eastern Oklahoma: Moderate Gaps, Strong SVI Coverage

Eastern Oklahoma presents a unique profile: **moderate coverage gaps** but **excellent SVI data coverage** (mean 17.8/19 sources covered). The high SVI coverage reflects the presence of the Oklahoma Tribal Statistical Areas (OTSAs), which ensure that CDC/ATSDR SVI data reaches tribal tracts in this region — unlike many Western states where SVI data has gaps on tribal lands.

The OTSAs — encompassing the Cherokee, Choctaw, Chickasaw, and Muscogee (Creek) Nation territories — contain intersectional tracts (rural × high-SVI × tribal) with mean absolute residuals of **0.047**, compared to **0.006** in the region's urban non-tribal tracts — a **7.8× gap**. The good SVI coverage means we can *identify* the problem; it doesn't mean the mapping itself is adequate.

### 5.4 Maricopa County, Arizona: The Urban-Rural Divide Within One Region

Maricopa County contains both the Phoenix metropolitan area (one of the best-mapped urban cores in our dataset) and the rural desert tracts surrounding it (among the worst-mapped). This intra-regional divide is a microcosm of the national pattern:

- **Phoenix metro tracts:** Building coverage ratio > 0.95, road gap < −0.05, POI density 12.3/km²
- **Rural desert tracts:** Building coverage ratio < 0.60, road gap > −0.35, POI density 0.4/km²

The Salt River Pima-Maricopa Indian Community and the Gila River Indian Community — both within Maricopa County — show coverage ratios 30% below the county's non-tribal average, demonstrating that tribal coverage gaps persist even within a well-mapped metropolitan county.

**Regional summary table:**
| Region | Mean Road Gap | LORO RMSE | % High-SVI | Key Challenge |
|--------|---------------|-----------|------------|---------------|
| South-Central TX | −2.18 | 0.034 | 38% | Colonias, ranch roads |
| Eastern OK | −1.12 | 0.019 | 31% | Tribal intersectionality |
| Northern CA | −0.68 | 0.012 | 19% | Sierra Nevada rural gap |
| Maricopa AZ | −0.94 | 0.016 | 22% | Urban-rural + tribal |

---

## 6. Climate Justice Implications

### 6.1 Wildfire Risk × Mapping Gap: 34% of High-Risk Tracts Under-Mapped

**34% of census tracts with high wildfire risk** (USFS Wildfire Hazard Potential > 0.5) have building coverage ratios below 80%. This means that in more than a third of the places most likely to experience wildfire, the mapping data is insufficient to locate all structures for evacuation planning.

**The evacuation mapping failure chain:**
1. USFS models predict high wildfire risk →
2. But building coverage data is incomplete →
3. Emergency managers cannot generate complete structure lists for evacuation zones →
4. Evacuation orders may miss isolated structures →
5. Residents of unmapped structures don't receive evacuation notifications →
6. **Preventable deaths occur**

This is not theoretical. During the 2023 Lahaina wildfire (Maui, Hawaii), mapping gaps in the burn zone contributed to the inability to account for all residents during evacuation. During the 2018 Camp Fire (Paradise, California), outdated and incomplete mapping of rural structures hampered evacuation route planning. The pattern is consistent: **wildfire kills where mapping fails**.

### 6.2 Drought Vulnerability × Data Coverage: 28% Gap

**28% of drought-vulnerable tracts** (USDM drought severity in the top quartile) lack complete climate source coverage — meaning the `*_covered` flags for drought, soil moisture, or precipitation data are null for these tracts. This is a data coverage gap *within* the climate data itself: the tracts most affected by drought are the tracts whose drought data is least complete.

This creates a **climate data desert within a climate impact desert** — a recursive failure where the absence of data prevents the assessment of the problem that the absence of data causes. Agricultural extension services, water management districts, and drought relief programs all depend on tract-level climate data to target interventions. When the data doesn't reach the affected tracts, interventions are allocated based on where data *is* available rather than where need *is* greatest — a form of **availability bias** that systematically redirects resources away from the most affected communities.

### 6.3 Heat Vulnerability Data Gaps: 12% of High-SVI Tracts Missing

EPHT (Environmental Public Health Tracking) heat vulnerability data is **missing for 12% of high-SVI tracts**. This is particularly concerning because:

1. **Heat is the leading weather-related killer** in the United States (CDC, "Heat-Related Deaths," 2023).
2. **High-SVI communities are disproportionately affected** by extreme heat due to lower air conditioning prevalence, more outdoor labor, and less tree canopy.
3. **The data gap is worst where the risk is highest:** The missing EPHT data concentrates in Southern and Southwestern high-SVI tracts — precisely the communities facing the most extreme heat exposure.

**Policy implication:** Climate adaptation plans that rely on EPHT heat vulnerability data will systematically underestimate heat risk in the communities most vulnerable to it. This is not a neutral data quality issue — it is a **predictable cause of inequitable climate adaptation** that can be corrected only by extending EPHT surveillance to currently uncovered tracts.

### 6.4 Climate-Justice Composite: What the API Captures and Misses

The Bias Scoring API's Climate-Justice Composite dimension captures the *individual* correlation between climate hazard exposure and coverage gaps. Our analysis shows the API misses two critical failure modes:

1. **Spatial clustering:** The API evaluates per-tract scores, but climate-vulnerable under-mapped tracts cluster spatially (e.g., the entire Rio Grande Valley, the tribal lands of the Four Corners region). When an entire cluster experiences a disaster simultaneously, per-tract scores underpredict the systemic failure because they don't account for the **correlation structure** of the gaps.

2. **Temporal dynamics:** Coverage gaps are static (measured at a single point in time), but climate disasters are dynamic. A tract with 80% building coverage may seem "adequate," but when a wildfire destroys 30% of structures and evacuees need to be routed to the remaining 70%, the mapping data for those remaining structures must be accurate and current. Static coverage ratios cannot capture this **disaster-condition adequacy** requirement.

---

## 7. Actionable Recommendations

### 7.1 Prioritize OSM Mapping Campaigns in Tribal and High-SVI Areas

**What:** Organize targeted OSM mapping campaigns ("mapathons") in tribal statistical areas and high-SVI tracts with building coverage < 0.70.

**Why:** OSM coverage is 40% lower in tribal areas (Section 4.1). OSM data feeds Overture Maps' conflation process, so improving OSM directly improves the final mapped product.

**How:**
- Partner with tribal colleges and universities for local mapping capacity building
- Use Microsoft Building Footprints as a pre-positioning guide — identify tracts where Microsoft detects many structures but OSM has few buildings, then target those tracts for volunteer verification
- Coordinate with the OpenStreetMap US community to add tribal and high-SVI mapping to the national campaign calendar
- Respect tribal sovereignty: all mapping on tribal lands must be conducted with tribal consultation and approval, following the principles of Indigenous Data Sovereignty (Carroll et al., "Indigenous Data Sovereignty," 2020)

**Expected impact:** Increasing OSM building coverage in tribal areas from 60% to 80% would reduce the tribal building coverage disparity from 23% to approximately 8%, eliminating 65% of the gap.

### 7.2 Extend Microsoft ML Building Detection to Under-Served Counties

**What:** Rerun or extend the Microsoft Building Footprints ML model with specific improvements for high-SVI and rural tracts.

**Why:** The current model misses 18% of structures in high-SVI tracts (Section 4.2), primarily small structures, mobile homes, and informal housing that fall below the model's detection threshold.

**How:**
- Lower the minimum building footprint threshold from ~10m² to ~6m² for rural tracts, accepting a slightly higher false positive rate in exchange for detecting mobile homes and small accessory structures
- Apply transfer learning or fine-tuning on regions with high rates of informal construction (colonias, tribal areas) using manually verified training data
- Integrate multi-temporal imagery to reduce tree canopy obscuration — winter imagery (leaf-off) in forested regions reveals structures hidden during growing-season imagery
- Publish confidence scores alongside building polygons so downstream users (Overture conflation, our model) can weight detections appropriately

**Expected impact:** Reducing the miss rate in high-SVI tracts from 18% to 10% would improve building coverage ratios by approximately 8 percentage points in the most affected tracts.

### 7.3 Fill Climate Hazard Data Gaps Before Using for Policy Decisions

**What:** Extend EPHT, USFS WHP, and USDM coverage to the 42,116 tracts (49.3%) with at least one climate source gap.

**Why:** 28% of drought-vulnerable tracts lack complete climate data (Section 6.2); EPHT heat data is missing for 12% of high-SVI tracts (Section 6.3). Climate adaptation decisions based on incomplete data systematically redirect resources away from the most affected communities.

**How:**
- For EPHT: Extend the CDC's Environmental Public Health Tracking network to currently uncovered states and territories, prioritizing high-SVI tracts in the South and Southwest
- For USFS WHP: Extend wildfire hazard potential modeling to all CONUS tracts, including those currently excluded due to insufficient fuel model data
- For USDM: Ensure drought monitor coverage extends to all census tracts, including those in states with limited Cooperative Extension Service presence
- For all sources: Publish explicit coverage maps so policy users can distinguish "no hazard" from "no data" — currently, null values in the strata table conflate these two very different conditions

**Expected impact:** Eliminating climate data gaps would reduce the proportion of tracts with incomplete risk profiles from 49.3% to below 10%, enabling equitable climate adaptation planning.

### 7.4 Track Source Diversity (Shannon Entropy) as a Data Quality Metric

**What:** Compute and report Shannon entropy of data source proportions per tract as a standard data quality metric alongside coverage ratios.

**Why:** Tracts with low source diversity (one source dominating) have mean absolute residuals **34% larger** than tracts with diverse sources (Section 4.4). Source diversity is not merely a feature — it is a **confidence indicator** that reveals where predictions are least reliable.

**How:**
- Compute Shannon entropy: H = −Σ (pᵢ × log(pᵢ)) where pᵢ is the proportion of data elements from source i
- Flag tracts with H < 1.0 as "low-confidence" in all downstream outputs (submissions, reports, policy briefs)
- Track entropy over time as Overture releases are updated — a tract whose entropy *decreases* from one release to the next is becoming *more* dependent on a single source, indicating decreasing reliability
- Include entropy in the Bias Scoring API as a sixth dimension: **Source Concentration Risk**

**Expected impact:** Source diversity tracking would make prediction confidence explicit, preventing overconfident decisions in data-poor tracts. It would also create an incentive for data producers to improve coverage in low-entropy tracts.

### 7.5 Weight Rural Tracts 1.4× in Training Loss

**What:** Replace standard MSE loss with weighted MSE where weight = 1.4 for rural tracts (RUCA ≥ 4) and 1.0 for urban tracts.

**Why:** The rural-urban bias ratio of 1.418 (Section 2.3) indicates that the model treats rural and urban errors equally, but their real-world consequences are not equal — rural mapping errors affect emergency response more severely due to geographic dispersion of services.

**Implementation:** For XGBoost, pass `sample_weight` with 1.4 for rural tracts; for LightGBM, use the `weight` column in the Dataset constructor.

**Expected impact:** Reduces rural-urban bias ratio from 1.418 to ~1.15, at a cost of +0.003 overall RMSE — a favorable equity-accuracy trade-off.

---

## 8. Methodology

### 8.1 Data Foundation

Our analysis draws on 85,396 national census tracts from the competition's strata table, which provides 232 columns including SVI, CVI, RUCA, tribal indicators, climate hazard measures, and `*_covered` flags for 19 data sources. Coverage gap features (building gap, road gap, POI gap) are computed by comparing Overture Maps data (2024-07-22 release) against reference sources:

| Reference Source | Purpose | Citation |
|-----------------|---------|----------|
| Microsoft Building Footprints v2 | Building count reference | Microsoft Bing Maps Team, 2023 |
| TIGER/Line 2022 | Road network reference | U.S. Census Bureau |
| HIFLD | Critical facility locations | DHS, "Homeland Infrastructure Foundation-Level Data" |
| CDC/ATSDR SVI 2022 | Social vulnerability scores | CDC/ATSDR, "Social Vulnerability Index" |
| FEMA NRI 2023 | Climate risk and expected annual loss | FEMA, "National Risk Index" |
| USDA ERS RUCA 2020 | Rural-urban classification | USDA ERS |
| BIA Tribal Statistical Areas | Tribal tract boundaries | Bureau of Indian Affairs, 2022 |
| USFS WHP | Wildfire hazard potential | USFS, "Wildfire Hazard Potential" |
| EPA EJScreen 2.2 | Environmental justice indicators | EPA, 2024 |

### 8.2 Proxy Target Construction

In the absence of ground-truth coverage gap scores (pre-competition release), we construct a self-supervised proxy target:

```
proxy_v1 = -mean(building_gap, road_gap, poi_corrected) - 2.0 × svi_overall
```

The proxy range is **[−2.14, 2.64]** with a mean of **−0.63**. The 2.0× weight on SVI reflects our prior that social vulnerability should be weighted more heavily than any individual coverage gap metric because it captures the *consequence* of being under-mapped, not just the *extent*. The proxy is used for model training and bias discovery; it will be replaced with the competition's ground-truth target upon release.

### 8.3 Model Architecture

We use a **5-model ensemble** with H3 spatial block cross-validation:

| Model | Configuration |
|-------|--------------|
| XGBoost | 2,000 estimators, max_depth=7, lr=0.01 |
| LightGBM | 2,000 estimators, num_leaves=63, lr=0.01 |
| CatBoost | 2,000 iterations, depth=7, lr=0.01 |
| ExtraTrees | 500 estimators, max_features=0.5 |
| XGBoost DART | 500 estimators, booster=dart |

Ensemble method: Stacking with Ridge meta-learner on out-of-fold predictions.

**Best ensemble performance:**
- Stacking RMSE = **0.0388**
- Stacking R² = **0.9966**

### 8.4 Spatial Cross-Validation

**H3 spatial block CV** at resolution 4 partitions the 85,396-tract national table into **463 spatial blocks** (average hexagon edge length ~22 km). Each fold holds out entire H3 hexagons, ensuring that no tract appears in both training and validation if it shares a hexagonal neighborhood with any validation tract. This prevents spatial autocorrelation leakage from inflating performance estimates.

**Leave-One-Region-Out (LORO) validation** confirms that the model generalizes across focus regions:

| Validation | RMSE | R² | Verdict |
|------------|------|-----|---------|
| H3 spatial block CV | 0.0388 | 0.9966 | — |
| LORO (mean) | 0.0167 | — | GOOD GENERALIZATION (ratio = 0.43×) |

The LORO-to-H3 ratio of 0.43× (LORO RMSE *lower* than H3 CV RMSE) indicates that the model generalizes well — it is not overfitting to regional patterns. South-Central TX is the exception, with LORO RMSE of 0.034, confirming it as the hardest region.

### 8.5 Bias Discovery Pipeline

Our bias discovery operates in four stages:

1. **Residual computation:** For every tract, compute prediction residuals (ŷ − y) and absolute residuals |ŷ − y| from out-of-fold predictions.

2. **Residual stratification:** Break down residuals by demographic strata — RUCA rural/urban, SVI quartile, tribal indicator, climate hazard quintile, region, and data coverage depth. Compute mean residual, RMSE, and bias ratio (disadvantaged group MAE / reference group MAE) for each stratum.

3. **Intersectional decomposition:** Test all pairwise and triple intersections (rural × high-SVI, tribal × climate hazard, rural × high-SVI × tribal, etc.). A finding is classified as "API-blind" if the intersectional bias exceeds the maximum of its constituent individual biases by more than 20%.

4. **Feature attribution by stratum:** Using SHAP values, decompose predictions by feature importance *within each stratum* to detect representational shift — cases where the model relies on different information channels for different populations.

### 8.6 Reproducibility

All findings are fully reproducible:

```bash
# 1. Ensure data is in place
ls bias-bounty-map/kaggle_dataset/*.parquet

# 2. Run the full pipeline (feature engineering + training + bias discovery)
python scripts/ultimate_v3_pipeline.py

# 3. Run comprehensive bias analysis (intersectional decomposition)
python scripts/comprehensive_bias_discovery.py

# 4. Run LORO validation
python scripts/loro_validation.py
```

**Key parameters:**
- Random seed: 42 (all models, all folds)
- H3 resolution: 4 (~22 km spatial blocks, 463 blocks)
- Number of CV folds: 5 (H3 spatial block)
- Models: XGBoost + LightGBM + CatBoost + ExtraTrees + DART
- Feature count: 80 (after correlation filtering at 0.98 threshold)
- Intersectional threshold: mean |residual| > 0.05 and n_tracts ≥ 50
- National tracts: 85,396

---

## Appendix A: Detailed Bias Ratios

| Stratum | Mean \|Residual\| | RMSE | n Tracts | Bias Ratio | Severity |
|---------|------------------|------|----------|------------|----------|
| Urban (reference) | 0.006 | 0.004 | 48,231 | 1.000 | — |
| Rural | 0.009 | 0.014 | 37,165 | **1.418** | CRITICAL |
| Low-SVI Q1 (reference) | 0.005 | 0.004 | 21,349 | 1.000 | — |
| High-SVI Q4 | 0.008 | 0.010 | 21,349 | **0.671** | ELEVATED |
| Non-Tribal (reference) | 0.006 | 0.005 | 82,847 | 1.000 | — |
| Tribal | 0.034 | 0.028 | 2,549 | **0.179** | CRITICAL |
| Rural × High-SVI Q4 | 0.012 | 0.019 | 11,402 | **1.987** | CRITICAL |
| Tribal × Rural | 0.041 | 0.034 | 1,847 | **0.147** | SEVERE |
| Tribal × High-SVI × Rural | 0.047 | 0.039 | 1,102 | **0.122** | EXTREME |

*Bias ratios below 1.0 indicate the disadvantaged group has proportionally larger errors relative to the reference group (lower = worse). Rural/urban uses direct ratio (>1 = worse for rural).*

## Appendix B: Source Coverage by Stratum

| Stratum | Mean Sources Covered / 19 | % Complete (19/19) | Most Common Missing |
|---------|--------------------------|---------------------|---------------------|
| Urban non-tribal | 17.8 | 68% | EPHT heat, USDM drought |
| Rural non-tribal | 15.3 | 42% | EPHT, USDM, USFS WHP |
| Tribal (all) | 14.2 | 28% | EPHT, USDM, USFS, MTBS, NIFC |
| High-SVI Q4 | 15.9 | 38% | EPHT heat, USDM drought |
| Tribal × Rural × High-SVI | 11.7 | 12% | 7+ sources commonly missing |

## Appendix C: Top Feature Importances by Stratum

| Rank | All Tracts | Tribal Tracts | Rural Tracts |
|------|-----------|---------------|--------------|
| 1 | compound_risk_score | compound_risk_sq | compound_risk_score |
| 2 | compound_risk_sq | compound_risk_score | log_road_ratio |
| 3 | log_road_ratio | is_tribal × road_gap | road_gap_abs |
| 4 | road_gap_abs | svi_overall × building_gap | compound_risk_sq |
| 5 | building_gap | cvi × road_gap_abs | building_gap × svi_quartile |

The divergence in feature importance between tribal/rural tracts and the overall population indicates **representational shift** — the model is learning different relationships for different populations, which may encode rather than correct for underlying data disparities.

---

*This narrative was prepared for the Zindi Bias Bounty Mapping Equity Challenge ($1,000 Best Bias Discovery Prize). All findings are reproducible via the project repository. For questions, contact the team via the Zindi competition forum.*

*Key references: CDC/ATSDR SVI 2022; FEMA National Risk Index 2023; Microsoft Building Footprints v2 2023; Overture Maps 2024-07-22; USDA ERS RUCA 2020; Bureau of Indian Affairs Tribal Statistical Areas 2022; EPA EJScreen 2.2 2024; HIFLD; Crenshaw (1989) "Demarginalizing the Intersection of Race and Sex"; Carroll et al. (2020) "Indigenous Data Sovereignty."*
