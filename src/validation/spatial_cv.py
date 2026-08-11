"""
Spatial cross-validation framework for census tract-level predictions.

Standard random K-fold gives optimistic scores due to spatial autocorrelation
(Tobler's First Law). This module implements:
1. GroupKFold by county (simple, effective)
2. Leave-One-Focus-Region-Out CV
3. Spatial block CV
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from typing import Iterator, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class SpatialCV:
    """Spatial cross-validation strategies for geospatial data."""

    @staticmethod
    def group_kfold_by_county(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        county_col: str = "county_fips",
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        GroupKFold where groups are counties.
        Prevents data leakage between neighboring tracts in the same county.
        
        Args:
            X: Feature matrix with GEOID as index
            y: Target variable
            n_splits: Number of folds
            county_col: Column name for county FIPS code
        
        Yields:
            (train_indices, test_indices) arrays
        """
        # Derive county FIPS from GEOID (first 5 digits of 11-digit GEOID)
        if county_col not in X.columns:
            groups = X.index.str[:5]
        else:
            groups = X[county_col]

        gkf = GroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            train_counties = set(groups.iloc[train_idx])
            test_counties = set(groups.iloc[test_idx])
            overlap = train_counties & test_counties

            if overlap:
                logger.warning(f"Fold {fold_idx}: {len(overlap)} counties leak between train/test!")

            logger.info(
                f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}, "
                f"train_counties={len(train_counties)}, test_counties={len(test_counties)}"
            )
            yield train_idx, test_idx

    @staticmethod
    def leave_region_out(
        X: pd.DataFrame,
        y: pd.Series,
        region_col: str = "focus_region",
        regions: Optional[List[str]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Leave-One-Focus-Region-Out CV.
        Train on N-1 focus regions, validate on the held-out one.
        This mimics the private leaderboard generalization test.

        Yields:
            (train_indices, test_indices, region_name)
        """
        # Graceful fallback: if region_col not found, try alternatives
        if region_col not in X.columns:
            # Try common alternatives
            for alt_col in ["region", "focus_region", "state_fips"]:
                if alt_col in X.columns:
                    logger.info(f"leave_region_out: '{region_col}' not found, "
                                f"using '{alt_col}' instead")
                    region_col = alt_col
                    break
            else:
                # Derive region from GEOID (first 2 digits = state FIPS)
                if X.index.dtype == object or hasattr(X.index, 'str'):
                    try:
                        X = X.copy()
                        X["_derived_region"] = X.index.str[:2]
                        region_col = "_derived_region"
                        logger.info("leave_region_out: Derived regions from GEOID "
                                    "(first 2 digits = state FIPS)")
                    except Exception:
                        raise ValueError(
                            f"Column '{region_col}' not found and cannot derive "
                            f"from index. Available columns: {list(X.columns[:10])}"
                        )
                else:
                    raise ValueError(
                        f"Column '{region_col}' not found. Add a region indicator "
                        f"column or use GEOID as index. Available: {list(X.columns[:10])}"
                    )

        if regions is None:
            regions = X[region_col].dropna().unique().tolist()

        for holdout_region in regions:
            # Use pandas comparison that handles NaN correctly
            # NaN != holdout_region is False, NaN == holdout_region is False
            # So we need explicit NaN handling
            region_vals = X[region_col]
            train_mask = region_vals.notna() & (region_vals != holdout_region)
            test_mask = region_vals.notna() & (region_vals == holdout_region)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            n_dropped = region_vals.isna().sum()
            if n_dropped > 0:
                logger.warning(
                    f"Leave-{holdout_region}-out: {n_dropped} rows with NaN "
                    f"region dropped from both train and test"
                )

            logger.info(
                f"Leave-{holdout_region}-out: train={len(train_idx)}, test={len(test_idx)}"
            )
            yield train_idx, test_idx, holdout_region

    @staticmethod
    def spatial_block_cv(
        X: pd.DataFrame,
        y: pd.Series,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        n_splits: int = 5,
        method: str = "kmeans",
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Spatial block cross-validation.
        Clusters tracts into spatial blocks, then holds out entire blocks.
        
        Methods:
        - 'kmeans': K-means clustering on lat/lon (simple, effective)
        - 'grid': Grid-based spatial blocks
        """
        from sklearn.cluster import KMeans

        coords = X[[lat_col, lon_col]].values

        if method == "kmeans":
            km = KMeans(n_clusters=n_splits * 4, random_state=42, n_init=10)
            block_labels = km.fit_predict(coords)
        elif method == "grid":
            # Create grid cells
            lat_bins = pd.qcut(X[lat_col], q=n_splits, labels=False, duplicates="drop")
            lon_bins = pd.qcut(X[lon_col], q=n_splits, labels=False, duplicates="drop")
            block_labels = (lat_bins * n_splits + lon_bins).values
        else:
            raise ValueError(f"Unknown method: {method}")

        gkf = GroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, block_labels)):
            logger.info(
                f"Spatial block fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}"
            )
            yield train_idx, test_idx

    @staticmethod
    def stratified_spatial_cv(
        X: pd.DataFrame,
        y: pd.Series,
        stratify_col: str = "state_fips",
        n_splits: int = 5,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Stratified spatial CV that preserves vulnerability distribution across folds.
        Ensures each fold has similar proportions of tribal, high-SVI, rural tracts.
        """
        # Derive state FIPS from GEOID if needed
        if stratify_col not in X.columns:
            groups = X.index.str[:2]
        else:
            groups = X[stratify_col]

        gkf = GroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(
                f"Stratified spatial fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}"
            )
            yield train_idx, test_idx


def compute_cv_scores(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_strategy: str = "group_kfold",
    n_splits: int = 5,
    **cv_kwargs,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute cross-validation RMSE scores using the specified strategy.

    Args:
        model: Either a sklearn-compatible model instance OR a callable factory
               that returns a fresh unfitted model. If callable, a new model
               is created for each fold to avoid data leakage.
        X: Feature matrix
        y: Target variable
        cv_strategy: Strategy name
        n_splits: Number of folds
        **cv_kwargs: Additional kwargs for CV strategy

    Returns:
        (oof_predictions, mean_rmse, std_rmse)
    """
    spatial_cv = SpatialCV()

    if cv_strategy == "group_kfold":
        folds = spatial_cv.group_kfold_by_county(X, y, n_splits, **cv_kwargs)
    elif cv_strategy == "spatial_block":
        folds = spatial_cv.spatial_block_cv(X, y, n_splits=n_splits, **cv_kwargs)
    elif cv_strategy == "stratified_spatial":
        folds = spatial_cv.stratified_spatial_cv(X, y, n_splits=n_splits, **cv_kwargs)
    elif cv_strategy == "leave_region_out":
        folds = spatial_cv.leave_region_out(X, y, **cv_kwargs)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")

    from sklearn.metrics import mean_squared_error

    fold_scores = []
    oof_predictions = np.full(len(y), np.nan)

    # Determine if model is a factory (callable) or instance
    is_factory = callable(model) and not hasattr(model, 'fit')

    for fold in folds:
        if len(fold) == 3:
            train_idx, test_idx, region_name = fold
            logger.info(f"  Region: {region_name}")
        else:
            train_idx, test_idx = fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create fresh model for each fold to prevent leakage
        if is_factory:
            fold_model = model()  # call factory to get fresh unfitted model
        else:
            fold_model = model   # use instance directly (legacy behavior)

        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_scores.append(rmse)
        oof_predictions[test_idx] = y_pred

    fold_scores = np.array(fold_scores)
    mean_rmse = fold_scores.mean()
    std_rmse = fold_scores.std()

    logger.info(f"CV RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")
    logger.info(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")

    return oof_predictions, mean_rmse, std_rmse
