"""
Utility functions: seed setting, logging, data validation.
"""

import numpy as np
import pandas as pd
import random
import os
import logging
from typing import Optional


def set_all_seeds(seed: int = 42):
    """Set random seeds for reproducibility. CRITICAL for Zindi code review."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging with file and console output."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def validate_geoid(geoid: str) -> bool:
    """Validate census tract GEOID format (11 digits)."""
    return isinstance(geoid, str) and len(geoid) == 11 and geoid.isdigit()


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature names for XGBoost/LightGBM compatibility."""
    df.columns = [
        col.replace("[", "_").replace("]", "_").replace("<", "_lt_")
           .replace(">", "_gt_").replace(" ", "_").replace(",", "_")
        for col in df.columns
    ]
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory usage by downcasting numeric types."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != "object":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df


def check_data_leakage(train_idx: np.ndarray, test_idx: np.ndarray,
                        geoids: np.ndarray, county_fips: np.ndarray):
    """Check for spatial data leakage between train and test sets."""
    train_counties = set(county_fips[train_idx])
    test_counties = set(county_fips[test_idx])
    overlap = train_counties & test_counties
    
    if overlap:
        logging.warning(f"DATA LEAKAGE: {len(overlap)} counties appear in both train and test!")
        return False
    return True
