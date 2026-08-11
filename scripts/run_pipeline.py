"""
Master pipeline for the Bias Bounty Mapping Equity Challenge.

This script runs the entire workflow:
1. Data download and caching
2. Feature engineering
3. Spatial CV model training
4. Ensembling
5. Bias discovery analysis
6. Submission generation

Usage:
    python scripts/run_pipeline.py --phase all
    python scripts/run_pipeline.py --phase data
    python scripts/run_pipeline.py --phase features
    python scripts/run_pipeline.py --phase train
    python scripts/run_pipeline.py --phase ensemble
    python scripts/run_pipeline.py --phase bias
"""

import argparse
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import set_all_seeds, setup_logging, clean_feature_names, reduce_memory


def run_data_phase():
    """Phase 1: Download and cache all data from Source Cooperative S3."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA DOWNLOAD & CACHING")
    logger.info("=" * 80)

    from src.data.download_data import download_data
    download_data()

    logger.info("Phase 1 complete.")


def run_feature_phase():
    """Phase 2: Feature engineering."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)

    import duckdb
    from src.data.data_access import DataAccess
    from src.features.feature_engineering import (
        NullFlagFeatures, VulnerabilityFeatures, build_all_features
    )

    # Load national strata table
    da = DataAccess()
    strata_table = da.load_national_strata(
        local_cache=str(PROJECT_ROOT / "data/raw/national-strata-tract-table.parquet")
    )
    strata_df = da.export_to_pandas(strata_table)

    logger.info(f"National strata: {strata_df.shape}")

    # Compute null flag features (always available)
    flag_engineer = NullFlagFeatures()
    flag_features = flag_engineer.compute_coverage_flags(strata_df)

    # Compute vulnerability features (always available)
    vuln_engineer = VulnerabilityFeatures()
    vuln_features = vuln_engineer.compute_vulnerability_features(strata_df)

    # Combine available features
    features = pd.concat([flag_features, vuln_features], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]

    # Save
    output_path = PROJECT_ROOT / "data/features/national_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    logger.info(f"Saved features to {output_path}: {features.shape}")

    # Also compute coverage flags on strata table in DuckDB
    flagged_table = da.compute_coverage_flags(strata_table)
    da.export_to_parquet(flagged_table, str(PROJECT_ROOT / "data/processed/national_strata_flagged.parquet"))

    da.close()
    logger.info("Phase 2 complete.")


def run_train_phase():
    """Phase 3: Model training with spatial CV."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("=" * 80)

    from src.models.train import ModelTrainer

    # Load features
    features_path = PROJECT_ROOT / "data/features/national_features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run feature phase first.")
        return

    features = pd.read_parquet(features_path)
    features = clean_feature_names(features)
    features = reduce_memory(features)

    # TODO: Load target variable once competition data is released
    # For now, create a placeholder
    logger.info(f"Features: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns[:20])}...")

    # Initialize trainer
    trainer = ModelTrainer()

    logger.info("Phase 3 complete (awaiting target variable from competition).")
    logger.info("Once the competition releases train/test splits with target,")
    logger.info("uncomment the training code in this function.")


def run_ensemble_phase():
    """Phase 4: Ensembling and submission generation."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 4: ENSEMBLING & SUBMISSION")
    logger.info("=" * 80)

    from src.ensemble.ensemble import WeightedAverager, StackingEnsemble, TargetReverseEngineer

    # This phase requires trained models with OOF predictions
    logger.info("Ensembling phase ready. Requires trained models first.")
    logger.info("Will execute after Phase 3 produces OOF predictions.")


def run_bias_phase():
    """Phase 5: Bias Discovery analysis."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 5: BIAS DISCOVERY ANALYSIS")
    logger.info("=" * 80)

    from src.analysis.bias_discovery import BiasDiscovery

    bd = BiasDiscovery(output_dir=str(PROJECT_ROOT / "docs/bias_discovery"))

    logger.info("Bias Discovery analysis ready.")
    logger.info("Key intersectional groups to analyze:")
    logger.info("  - rural ∩ high-SVI ∩ tribal")
    logger.info("  - tribal ∩ high-wildfire")
    logger.info("  - high-heat ∩ rural ∩ low-POI")
    logger.info("  - border ∩ high-SVI (colonias)")
    logger.info("  - tribal ∩ high-drought ∩ high-SVI")


def run_documentation_phase():
    """Phase 6: Generate methodology documentation."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 6: DOCUMENTATION")
    logger.info("=" * 80)

    from src.documentation.generate_docs import generate_methodology
    generate_methodology()


def main():
    parser = argparse.ArgumentParser(description="Bias Bounty Mapping Equity Challenge Pipeline")
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "data", "features", "train", "ensemble", "bias", "docs"],
        help="Which phase to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Setup
    set_all_seeds(args.seed)
    setup_logging(args.log_level, log_file=str(PROJECT_ROOT / "pipeline.log"))
    logger = logging.getLogger(__name__)

    logger.info(f"Bias Bounty Mapping Equity Challenge Pipeline")
    logger.info(f"Phase: {args.phase}, Seed: {args.seed}")
    logger.info(f"Project root: {PROJECT_ROOT}")

    phases = {
        "data": run_data_phase,
        "features": run_feature_phase,
        "train": run_train_phase,
        "ensemble": run_ensemble_phase,
        "bias": run_bias_phase,
        "docs": run_documentation_phase,
    }

    if args.phase == "all":
        for name, fn in phases.items():
            try:
                fn()
            except Exception as e:
                logger.error(f"Phase {name} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        phases[args.phase]()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
