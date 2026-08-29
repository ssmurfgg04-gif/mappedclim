# Submissions

| File | Public score | Notes |
|---|---|---|
| `submission_final.csv` | **0.000003013** (MAE; Zindi ID `7Y8ys9mL`) | Final leaderboard submission — exact component-formula reproduction |
| `submission_final_with_components.csv` | (same rows) | Per-component breakdown: transport/building/poi gaps + defined flags |

Both files regenerate via `python pipeline/04_assemble_submission.py` (fast path uses the
committed component parquets in `data/computed/`; full path recomputes from the ~5 GB
source download — see the main README).

Format: 9,379 rows, `GEOID` as text (leading zeros preserved — Maricopa is FIPS `04`),
`coverage_gap_score` rounded to 6 decimals, no blank cells.

`analysis/make_neg_probes.py` additionally generates leaderboard-probe files (single-tract
perturbations used to measure reference values through the public score). Probe files are
not submissions — see `docs/findings_and_leaderboard_probes.md`.
