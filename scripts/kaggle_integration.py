"""
Kaggle integration utilities.

Handles:
1. Pushing datasets to Kaggle for GPU notebook access
2. Pulling training results back
3. Creating/updating Kaggle notebooks
4. Managing competition submissions
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def run_cmd(cmd: str) -> str:
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def push_dataset_to_kaggle(
    dataset_name: str = "bias-bounty-features",
    data_dir: Optional[str] = None,
):
    """
    Push feature files to Kaggle as a dataset.
    This makes them available to GPU notebooks without re-downloading.
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data/features")
    
    # Create Kaggle dataset metadata
    meta_dir = PROJECT_ROOT / "kaggle_dataset"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "title": "Bias Bounty Mapping Equity - Features",
        "id": f"jackblessed/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Pre-computed enhanced features for Bias Bounty Mapping Equity Challenge",
    }
    
    with open(meta_dir / "dataset-metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Copy feature files
    import shutil
    features_src = Path(data_dir)
    
    for f in features_src.glob("*_enhanced_features.parquet"):
        shutil.copy2(f, meta_dir / f.name)
        logger.info(f"Copied: {f.name}")
    
    # Copy national strata
    national_src = PROJECT_ROOT / "data/raw/strata/national/national-strata-tract-table.parquet"
    if national_src.exists():
        shutil.copy2(national_src, meta_dir / "national-strata-tract-table.parquet")
    
    # Push to Kaggle
    cmd = f"cd {meta_dir} && kaggle datasets create -p . --dir-mode zip"
    output = run_cmd(cmd)
    logger.info(f"Kaggle dataset push: {output}")
    
    return output


def push_notebook_to_kaggle(
    notebook_path: Optional[str] = None,
    notebook_name: str = "bias-bounty-gpu-training",
):
    """
    Push a notebook to Kaggle for GPU execution.
    """
    if notebook_path is None:
        notebook_path = str(PROJECT_ROOT / "notebooks/kaggle_gpu_training.ipynb")
    
    # Create Kaggle kernel metadata
    meta = {
        "id": f"ssmurfgg04-gif/{notebook_name}",
        "title": notebook_name,
        "code_file": Path(notebook_path).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["jackblessed/bias-bounty-features"],
        "competition_sources": [],
        "keywords": ["bias", "mapping", "equity", "coverage-gap"],
    }
    
    meta_dir = PROJECT_ROOT / "kaggle_kernel"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with open(meta_dir / "kernel-metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Copy notebook
    import shutil
    shutil.copy2(notebook_path, meta_dir / Path(notebook_path).name)
    
    # Push to Kaggle
    cmd = f"cd {meta_dir} && kaggle kernels push -p ."
    output = run_cmd(cmd)
    logger.info(f"Kaggle kernel push: {output}")
    
    return output


def pull_kaggle_results(
    notebook_name: str = "bias-bounty-gpu-training",
    output_dir: Optional[str] = None,
):
    """
    Pull training results from a completed Kaggle notebook.
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data/output/kaggle_results")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pull output
    cmd = f"kaggle kernels output ssmurfgg04-gif/{notebook_name} -p {output_dir}"
    output = run_cmd(cmd)
    logger.info(f"Kaggle kernel output: {output}")
    
    # Check for results file
    results_path = Path(output_dir) / "training_results.csv"
    if results_path.exists():
        import pandas as pd
        results = pd.read_csv(results_path)
        logger.info(f"Training results:\n{results.to_string()}")
        return results
    
    return None


def create_zindi_submission(
    predictions_path: str,
    output_path: Optional[str] = None,
):
    """
    Create a submission file in Zindi format.
    
    Zindi expects: GEOID, coverage_gap_score
    """
    import pandas as pd
    
    if output_path is None:
        output_path = str(PROJECT_ROOT / "data/output/zindi_submission.csv")
    
    preds = pd.read_parquet(predictions_path)
    
    # Zindi format
    submission = preds[['GEOID']].copy()
    if 'coverage_gap' in preds.columns:
        submission['coverage_gap_score'] = preds['coverage_gap']
    elif 'building_gap_pred' in preds.columns:
        submission['coverage_gap_score'] = preds['building_gap_pred']
    else:
        # Use blended prediction
        pred_cols = [c for c in preds.columns if 'pred' in c.lower()]
        if pred_cols:
            submission['coverage_gap_score'] = preds[pred_cols].mean(axis=1)
        else:
            raise ValueError("No prediction columns found")
    
    # Clip to reasonable range
    submission['coverage_gap_score'] = submission['coverage_gap_score'].clip(-1, 1)
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Zindi submission saved: {output_path} ({len(submission)} rows)")
    logger.info(f"Stats: mean={submission['coverage_gap_score'].mean():.4f}, "
               f"std={submission['coverage_gap_score'].std():.4f}")
    
    return submission


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "push-dataset":
            push_dataset_to_kaggle()
        elif action == "push-notebook":
            push_notebook_to_kaggle()
        elif action == "pull-results":
            pull_kaggle_results()
        elif action == "submission":
            create_zindi_submission(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("Usage: python kaggle_integration.py [push-dataset|push-notebook|pull-results|submission]")
