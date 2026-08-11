"""
Self-supervised baseline model for Bias Bounty Mapping Equity Challenge.

Since the competition target (coverage_gap_score) isn't released yet,
we use a self-supervised approach:
1. Use building_gap and road_gap as proxy targets
2. Train models to predict these from strata features
3. Use residual analysis to discover bias patterns
4. Prepare the full modeling pipeline for when targets are released

This approach is strategic: even without the target, we can:
- Identify which strata features are most predictive
- Discover systematic bias patterns
- Build the CV pipeline and verify it works
- Be ready to train instantly when targets drop
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class SelfSupervisedBaseline:
    """
    Train baseline models using coverage gap ratios as proxy targets.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models = {}
        self.results = {}
    
    def prepare_features(
        self, 
        features_df: pd.DataFrame,
        target_col: str = 'building_gap',
        drop_cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and proxy target."""
        # Default columns to exclude
        default_drop = ['GEOID', 'region', 'county_fips', 'state_fips',
                        'centroid_lat', 'centroid_lon',
                        'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
                        'building_count_ratio', 'building_count_gap',
                        'road_count_ratio', 'road_count_gap',
                        'road_length_ratio', 'road_length_gap',
                        'poi_facility_gap', 'poi_to_facility_ratio']
        
        if drop_cols:
            default_drop.extend(drop_cols)
        
        # Remove target leakage columns
        feature_cols = [c for c in features_df.columns 
                       if c not in default_drop and features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()
        
        # Drop rows with NaN target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        features_df_clean = features_df[valid_mask]
        logger.info(f"  Dropped {(~valid_mask).sum()} tracts with NaN target ({valid_mask.mean()*100:.1f}% kept)")
        
        # Remove columns with >80% null
        null_frac = X.isnull().mean()
        good_cols = null_frac[null_frac < 0.8].index.tolist()
        X = X[good_cols]
        
        # Fill remaining NaN (tree models can handle NaN but sklearn can't)
        X = X.fillna(-999)
        
        # Remove constant columns
        std = X.std()
        varying_cols = std[std > 0].index.tolist()
        X = X[varying_cols]
        
        # Store valid mask for later use
        self._valid_mask = valid_mask
        
        logger.info(f"  Features: {X.shape[1]} columns, {X.shape[0]} tracts")
        logger.info(f"  Target ({target_col}): mean={y.mean():.4f}, std={y.std():.4f}")
        
        return X, y
    
    def train_with_spatial_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        geoids: pd.Series,
        model_name: str = 'xgboost',
        n_splits: int = 5,
    ) -> Dict:
        """Train a model with spatial GroupKFold by county."""
        # Derive county groups
        groups = geoids.str[:5]
        
        # Initialize model
        model = self._get_model(model_name)
        
        # Spatial CV
        gkf = GroupKFold(n_splits=n_splits)
        fold_scores = {'rmse': [], 'r2': [], 'mae': []}
        oof_predictions = np.full(len(y), np.nan)
        feature_importances = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model_fold = self._get_model(model_name)
            model_fold.fit(X_train, y_train)
            
            y_pred = model_fold.predict(X_test)
            oof_predictions[test_idx] = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            fold_scores['rmse'].append(rmse)
            fold_scores['r2'].append(r2)
            fold_scores['mae'].append(mae)
            
            # Feature importance
            if hasattr(model_fold, 'feature_importances_'):
                feature_importances.append(model_fold.feature_importances_)
            
            logger.info(f"  Fold {fold_idx}: RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")
        
        # Aggregate results
        results = {
            'model_name': model_name,
            'cv_rmse_mean': np.mean(fold_scores['rmse']),
            'cv_rmse_std': np.std(fold_scores['rmse']),
            'cv_r2_mean': np.mean(fold_scores['r2']),
            'cv_mae_mean': np.mean(fold_scores['mae']),
            'oof_predictions': oof_predictions,
            'fold_scores': fold_scores,
        }
        
        if feature_importances:
            mean_importance = np.mean(feature_importances, axis=0)
            fi_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_importance,
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = fi_df
        
        logger.info(f"\n  {model_name} CV Results:")
        logger.info(f"    RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}")
        logger.info(f"    R2:   {results['cv_r2_mean']:.4f}")
        logger.info(f"    MAE:  {results['cv_mae_mean']:.4f}")
        
        return results
    
    def _get_model(self, model_name: str):
        """Get a model instance by name."""
        if model_name == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                tree_method='hist', random_state=self.seed,
            )
        elif model_name == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, num_leaves=31,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                verbose=-1, random_state=self.seed,
            )
        elif model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=300, max_depth=12,
                min_samples_leaf=5, random_state=self.seed,
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=self.seed,
            )
        elif model_name == 'ridge':
            return Ridge(alpha=1.0)
        elif model_name == 'elasticnet':
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def run_all_models(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'building_gap',
    ) -> Dict:
        """Run all baseline models and compare results."""
        X, y = self.prepare_features(features_df, target_col)
        geoids = features_df.loc[self._valid_mask, 'GEOID']
        
        model_names = ['ridge', 'elasticnet', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
        
        all_results = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*40}")
            
            try:
                result = self.train_with_spatial_cv(X, y, geoids, model_name)
                all_results[model_name] = result
                self.models[model_name] = result
            except Exception as e:
                logger.error(f"  {model_name} failed: {e}")
        
        # Comparison table
        if all_results:
            comparison = pd.DataFrame({
                name: {
                    'CV RMSE': res['cv_rmse_mean'],
                    'CV R2': res['cv_r2_mean'],
                    'CV MAE': res['cv_mae_mean'],
                }
                for name, res in all_results.items()
            }).T.sort_values('CV RMSE')
        else:
            comparison = pd.DataFrame(columns=['CV RMSE', 'CV R2', 'CV MAE'])
        
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(comparison.to_string())
        
        return all_results, comparison
    
    def compute_blended_predictions(
        self,
        all_results: Dict,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal blended OOF predictions."""
        from scipy.optimize import minimize
        
        model_names = list(all_results.keys())
        n_models = len(model_names)
        
        # Stack OOF predictions
        oof_matrix = np.column_stack([
            all_results[name]['oof_predictions'] for name in model_names
        ])
        
        # Optimize weights
        def objective(weights):
            pred = oof_matrix @ weights
            return np.sqrt(mean_squared_error(y_true, pred))
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        blended_rmse = result.fun
        
        logger.info(f"\nOptimal blend weights:")
        for name, weight in zip(model_names, optimal_weights):
            logger.info(f"  {name}: {weight:.4f}")
        logger.info(f"Blended CV RMSE: {blended_rmse:.6f}")
        
        blended_oof = oof_matrix @ optimal_weights
        
        return blended_oof, optimal_weights, blended_rmse


def run_baseline_analysis():
    """Run the full baseline analysis pipeline."""
    # Load enhanced features
    features_path = PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet"
    
    if not features_path.exists():
        # Fall back to combining per-region enhanced features
        regions = ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']
        dfs = []
        for region in regions:
            path = PROJECT_ROOT / f"data/features/{region}_enhanced_features.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
        
        if not dfs:
            # Fall back to base features
            for region in regions:
                path = PROJECT_ROOT / f"data/features/{region}_tract_features.parquet"
                if path.exists():
                    dfs.append(pd.read_parquet(path))
        
        if not dfs:
            logger.error("No feature files found. Run enhanced_feature_pipeline.py first.")
            return
        
        features = pd.concat(dfs, ignore_index=True)
    else:
        features = pd.read_parquet(features_path)
    
    logger.info(f"Loaded features: {features.shape[0]} tracts, {features.shape[1]} columns")
    
    # Run baseline models
    baseline = SelfSupervisedBaseline()
    bldg_results = {}
    road_results = {}
    
    # Use building_gap as proxy target
    if 'building_gap' in features.columns:
        logger.info("\n" + "="*60)
        logger.info("TARGET: Building Coverage Gap (proxy)")
        logger.info("="*60)
        bldg_results, bldg_comparison = baseline.run_all_models(features, 'building_gap')
        
        # Save results
        (PROJECT_ROOT / "data/output").mkdir(parents=True, exist_ok=True)
        bldg_comparison.to_csv(PROJECT_ROOT / "data/output/building_gap_model_comparison.csv")
    
    # Use road_gap as proxy target
    if 'road_gap' in features.columns:
        logger.info("\n" + "="*60)
        logger.info("TARGET: Road Coverage Gap (proxy)")
        logger.info("="*60)
        road_results, road_comparison = baseline.run_all_models(features, 'road_gap')
        
        road_comparison.to_csv(PROJECT_ROOT / "data/output/road_gap_model_comparison.csv")
    
    # Compute blended predictions for building_gap
    if bldg_results and len(bldg_results) > 1:
        try:
            X, y = baseline.prepare_features(features, 'building_gap')
            blended, weights, rmse = baseline.compute_blended_predictions(
                bldg_results, y.values
            )
            
            # Save OOF predictions
            valid_mask = baseline._valid_mask
            oof_df = pd.DataFrame({
                'GEOID': features.loc[valid_mask, 'GEOID'].values,
                'building_gap_true': features.loc[valid_mask, 'building_gap'].values,
                'building_gap_pred': blended,
                'residual': features.loc[valid_mask, 'building_gap'].values - blended,
            })
            oof_df.to_parquet(PROJECT_ROOT / "data/output/oof_predictions.parquet")
        except Exception as e:
            logger.error(f"Blended predictions failed: {e}")
    
    # Save feature importance from best model
    for model_name, result in bldg_results.items():
        if 'feature_importance' in result:
            fi = result['feature_importance'].head(30)
            logger.info(f"\nTop 30 features ({model_name}):")
            for _, row in fi.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\nBaseline analysis complete!")
    return baseline


if __name__ == "__main__":
    run_baseline_analysis()
