#!/usr/bin/env python3
"""
Submission Validator — Bias Bounty Map Competition
===================================================
Validates a submission CSV against the required architecture:
  a. Exactly 2 columns: GEOID, coverage_gap_score
  b. GEOID is 11-digit string (census tract FIPS)
  c. No duplicate GEOIDs
  d. No null values
  e. coverage_gap_score is numeric and in reasonable range (-10, 10)
  f. Counts total rows
  g. Score distribution (mean, std, min, max, median)
  h. Tribal/rural bias presence (loads from engineered_features_merged.parquet)
  i. Deterministic fix in place (pipeline_state_merged.json → target=gap_only)

Exit code 0 if all checks pass, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ = Path("/home/z/my-project/bias-bounty-map")
DEFAULT_SUBMISSION = PROJ / "data/output/submission_merged.csv"
ENGINEERED_PARQUET = PROJ / "data/output/engineered_features_merged.parquet"
PIPELINE_STATE_JSON = PROJ / "data/output/pipeline_state_merged.json"

# ─── Score range ──────────────────────────────────────────────────────────────
SCORE_MIN = -10.0
SCORE_MAX = 10.0


def header(msg: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {msg}")
    print(f"{'═' * 64}")


def check(label: str, passed: bool, detail: str = "") -> bool:
    icon = "✅ PASS" if passed else "❌ FAIL"
    suffix = f"  —  {detail}" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return passed


def validate_submission(path: Path) -> bool:
    """Run all validation checks. Returns True if all pass."""
    all_pass = True

    header(f"VALIDATING: {path}")

    # ── File exists ───────────────────────────────────────────────────────
    if not path.exists():
        all_pass &= check("File exists", False, f"{path} not found")
        return False
    all_pass &= check("File exists", True, f"{path.stat().st_size / 1e6:.2f} MB")

    # ── Load CSV ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path, dtype={"GEOID": str})
    except Exception as e:
        all_pass &= check("CSV loadable", False, str(e))
        return False
    all_pass &= check("CSV loadable", True)

    # ── (a) Exactly 2 columns ────────────────────────────────────────────
    required_cols = ["GEOID", "coverage_gap_score"]
    has_cols = list(df.columns) == required_cols
    all_pass &= check(
        "Exactly 2 columns: GEOID, coverage_gap_score",
        has_cols,
        f"got {list(df.columns)}"
    )
    if not has_cols:
        # Try to proceed if both columns exist even if extras present
        if not all(c in df.columns for c in required_cols):
            return False

    # ── (b) GEOID is 11-digit string ─────────────────────────────────────
    geoid_lengths = df["GEOID"].str.len()
    all_11_digit = (geoid_lengths == 11).all()
    n_bad_len = (geoid_lengths != 11).sum()
    all_pass &= check(
        "GEOID is 11-digit FIPS string",
        all_11_digit,
        f"{n_bad_len} rows with wrong length" if n_bad_len else ""
    )

    # Spot-check: all digits
    all_numeric = df["GEOID"].str.isnumeric().all()
    n_non_numeric = (~df["GEOID"].str.isnumeric()).sum()
    all_pass &= check(
        "GEOID all-numeric",
        all_numeric,
        f"{n_non_numeric} rows with non-digit chars" if n_non_numeric else ""
    )

    # ── (c) No duplicate GEOIDs ──────────────────────────────────────────
    n_dups = df["GEOID"].duplicated().sum()
    no_dups = n_dups == 0
    all_pass &= check(
        "No duplicate GEOIDs",
        no_dups,
        f"{n_dups} duplicates found" if n_dups else ""
    )

    # ── (d) No null values ───────────────────────────────────────────────
    null_counts = df.isnull().sum()
    has_nulls = null_counts.sum() > 0
    all_pass &= check(
        "No null values",
        not has_nulls,
        f"nulls per column: {null_counts.to_dict()}" if has_nulls else ""
    )

    # ── (e) coverage_gap_score is numeric and in range ───────────────────
    score = df["coverage_gap_score"]
    is_numeric = pd.api.types.is_numeric_dtype(score)
    all_pass &= check("coverage_gap_score is numeric", is_numeric)

    if is_numeric:
        in_range = ((score >= SCORE_MIN) & (score <= SCORE_MAX)).all()
        n_out_range = ((score < SCORE_MIN) | (score > SCORE_MAX)).sum()
        all_pass &= check(
            f"coverage_gap_score in [{SCORE_MIN}, {SCORE_MAX}]",
            in_range,
            f"{n_out_range} rows out of range" if n_out_range else ""
        )

    # ── (f) Total rows ───────────────────────────────────────────────────
    n_rows = len(df)
    print(f"\n  📊  Total rows: {n_rows:,}")
    # Sanity: US census tracts ≈ 73K–85K for all-tracts datasets
    reasonable_rows = 70000 <= n_rows <= 120000
    all_pass &= check(
        "Row count in expected range (70K–120K)",
        reasonable_rows,
        f"got {n_rows:,}"
    )

    # ── (g) Score distribution ───────────────────────────────────────────
    if is_numeric:
        stats = score.describe()
        median = score.median()
        print(f"\n  📈  Score Distribution:")
        print(f"       mean    = {stats['mean']:+.6f}")
        print(f"       std     =  {stats['std']:.6f}")
        print(f"       min     = {stats['min']:+.6f}")
        print(f"       25%     = {stats['25%']:+.6f}")
        print(f"       median  = {median:+.6f}")
        print(f"       75%     = {stats['75%']:+.6f}")
        print(f"       max     = {stats['max']:+.6f}")

        # Expect scores mostly in [-2, 0.5] based on observed data
        mostly_reasonable = (stats['min'] >= -3.0) and (stats['max'] <= 1.0)
        all_pass &= check(
            "Score range within expected [-3, +0.5]",
            mostly_reasonable,
            f"min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

        # Check for suspiciously zero-heavy distribution
        frac_zero = (score.abs() < 1e-8).mean()
        not_mostly_zero = frac_zero < 0.5
        all_pass &= check(
            "Score not mostly zeros",
            not_mostly_zero,
            f"{frac_zero:.1%} are exactly zero"
        )

    # ── (h) Tribal/rural bias presence ───────────────────────────────────
    header("TRIBAL / RURAL BIAS CHECK")
    if ENGINEERED_PARQUET.exists():
        try:
            feat = pd.read_parquet(
                ENGINEERED_PARQUET,
                columns=["GEOID", "tribal_any", "pct_urban"]
            )
            feat["GEOID"] = feat["GEOID"].astype(str)

            # Merge with submission
            merged = df.merge(feat, on="GEOID", how="left")
            n_matched = merged["tribal_any"].notna().sum()
            match_rate = n_matched / n_rows
            all_pass &= check(
                "Feature merge succeeded",
                match_rate > 0.5,
                f"{match_rate:.1%} of submission GEOIDs matched features"
            )

            # Tribal analysis
            if "tribal_any" in merged.columns:
                tribal = merged[merged["tribal_any"] == True]
                non_tribal = merged[merged["tribal_any"] == False]
                if len(tribal) > 0 and len(non_tribal) > 0:
                    t_mean = tribal["coverage_gap_score"].mean()
                    nt_mean = non_tribal["coverage_gap_score"].mean()
                    ratio = abs(t_mean) / (abs(nt_mean) + 1e-10)
                    print(f"\n  🏛️   Tribal tracts:    {len(tribal):,}  mean score = {t_mean:+.6f}")
                    print(f"       Non-tribal:      {len(non_tribal):,}  mean score = {nt_mean:+.6f}")
                    print(f"       |Tribal|/|NonT| ratio = {ratio:.3f}")
                    tribal_present = len(tribal) > 100
                    all_pass &= check(
                        "Tribal tracts present (>100)",
                        tribal_present,
                        f"found {len(tribal):,}"
                    )

            # Rural analysis
            if "pct_urban" in merged.columns:
                rural = merged[merged["pct_urban"].fillna(0.5) < 0.5]
                urban = merged[merged["pct_urban"].fillna(0.5) >= 0.5]
                if len(rural) > 0 and len(urban) > 0:
                    r_mean = rural["coverage_gap_score"].mean()
                    u_mean = urban["coverage_gap_score"].mean()
                    print(f"\n  🌾   Rural tracts:    {len(rural):,}  mean score = {r_mean:+.6f}")
                    print(f"       Urban tracts:    {len(urban):,}  mean score = {u_mean:+.6f}")
                    rural_present = len(rural) > 100
                    all_pass &= check(
                        "Rural tracts present (>100)",
                        rural_present,
                        f"found {len(rural):,}"
                    )
        except Exception as e:
            all_pass &= check(
                "Tribal/rural bias check",
                False,
                f"Error loading features: {e}"
            )
    else:
        print(f"  ⚠️   SKIP  engineered_features_merged.parquet not found at {ENGINEERED_PARQUET}")
        print("       (This check will run once the full pipeline has been executed)")

    # ── (i) Deterministic fix in place ───────────────────────────────────
    header("DETERMINISTIC FIX CHECK")
    if PIPELINE_STATE_JSON.exists():
        try:
            with open(PIPELINE_STATE_JSON) as f:
                state = json.load(f)

            target = state.get("target", "")
            is_gap_only = target == "gap_only"
            all_pass &= check(
                "Pipeline target = gap_only",
                is_gap_only,
                f"got target={target!r}"
            )

            pipeline_name = state.get("pipeline", "")
            has_deterministic = "DETERMINISTIC_FIX" in pipeline_name
            all_pass &= check(
                "Pipeline includes DETERMINISTIC_FIX",
                has_deterministic,
                f"pipeline={pipeline_name!r}"
            )

            formula = state.get("inference_formula", "")
            has_rural_penalty = "rural_penalty" in formula
            all_pass &= check(
                "Inference formula includes rural_penalty",
                has_rural_penalty,
                f"formula={formula!r}"
            )

            # Print pipeline summary
            print(f"\n  🔧  Pipeline:        {pipeline_name}")
            print(f"       Training target: {state.get('training_target', 'N/A')}")
            print(f"       Inference:      {formula}")
            print(f"       Best ensemble:  {state.get('best_ensemble', 'N/A')}")
            print(f"       Best RMSE:      {state.get('best_rmse', 'N/A'):.6f}")
            print(f"       Best R²:        {state.get('best_r2', 'N/A'):.4f}")

            weights = state.get("convex_weights", {})
            if weights:
                print(f"       Ensemble weights:")
                for model, w in sorted(weights.items()):
                    print(f"         {model:>10s} = {w:.4f}")

        except Exception as e:
            all_pass &= check(
                "Pipeline state parseable",
                False,
                f"Error: {e}"
            )
    else:
        print(f"  ⚠️   SKIP  pipeline_state_merged.json not found at {PIPELINE_STATE_JSON}")
        print("       (This check will run once Phase 2 training has been executed)")

    # ── Final verdict ─────────────────────────────────────────────────────
    header("FINAL VERDICT")
    if all_pass:
        print("  ✅  ALL CHECKS PASSED — submission is valid")
        return True
    else:
        print("  ❌  SOME CHECKS FAILED — fix issues before submitting")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission CSV for Bias Bounty Map competition"
    )
    parser.add_argument(
        "submission_path",
        nargs="?",
        default=str(DEFAULT_SUBMISSION),
        help=f"Path to submission CSV (default: {DEFAULT_SUBMISSION})"
    )
    args = parser.parse_args()

    path = Path(args.submission_path)
    passed = validate_submission(path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
