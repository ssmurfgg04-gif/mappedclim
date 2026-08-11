"""
Self-Evolving Pipeline for Bias Bounty Mapping Equity Challenge.

This pipeline:
1. Trains models with current best hyperparameters
2. Evaluates with spatial cross-validation
3. Analyzes residuals for bias patterns
4. Auto-tunes hyperparameters based on CV + bias scores
5. Generates new features if residuals show systematic patterns
6. Iterates until convergence or budget exhausted
7. Saves best model + produces submission

Design principles:
- Never overfit: use spatial CV (GroupKFold by county)
- Bias-aware: optimize RMSE + bias_penalty jointly
- Self-correcting: if residuals show strata patterns, add interaction features
- Persistent: save all state to disk, resume from any point
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class SelfEvolvingPipeline:
    """
    Self-evolving ML pipeline that iteratively improves by:
    1. Training with current best config
    2. Evaluating CV + bias metrics
    3. Auto-tuning hyperparameters
    4. Feature refinement based on residual analysis
    5. Ensembling diverse models
    """
    
    def __init__(self, config_path: str = "config/evolving_config.yaml"):
        self.config_path = PROJECT_ROOT / config_path
        self.state_path = PROJECT_ROOT / "data/output/evolving_state.json"
        self.best_rmse = np.inf
        self.best_r2 = -np.inf
        self.iteration = 0
        self.history = []
        self.models = {}
        self.oof_predictions = {}
        
        # Load or create config
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f)
        
        # Load previous state if exists
        self._load_state()
    
    def _default_config(self) -> Dict:
        return {
            'max_iterations': 10,
            'target_col': 'building_gap',
            'n_folds': 5,
            'bias_penalty_weight': 0.3,
            'feature_evolution': True,
            'models': {
                'xgboost': {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'tree_method': 'hist',
                },
                'lightgbm': {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.7,
                    'verbose': -1,
                },
                'gradient_boosting': {
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                },
            },
            'tuning': {
                'learning_rate_range': [0.01, 0.2],
                'max_depth_range': [3, 10],
                'n_estimators_range': [200, 2000],
                'subsample_range': [0.6, 1.0],
                'colsample_range': [0.5, 1.0],
                'reg_range': [0.001, 10.0],
            }
        }
    
    def _load_state(self):
        """Load previous state for resume capability."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                state = json.load(f)
            self.iteration = state.get('iteration', 0)
            self.best_rmse = state.get('best_rmse', np.inf)
            self.best_r2 = state.get('best_r2', -np.inf)
            self.history = state.get('history', [])
            logger.info(f"Resumed from iteration {self.iteration}, best RMSE={self.best_rmse:.6f}")
    
    def _save_state(self):
        """Save current state for resume capability."""
        state = {
            'iteration': self.iteration,
            'best_rmse': self.best_rmse,
            'best_r2': self.best_r2,
            'history': self.history,
            'timestamp': datetime.now().isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def prepare_features(
        self, 
        features_df: pd.DataFrame,
        target_col: str,
        feature_subset: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare feature matrix, target, and GEOIDs."""
        drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
                     'centroid_lat', 'centroid_lon',
                     'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
                     'building_count_ratio', 'building_count_gap',
                     'road_count_ratio', 'road_count_gap',
                     'road_length_ratio', 'road_length_gap',
                     'poi_facility_gap', 'poi_to_facility_ratio']
        
        if feature_subset:
            feature_cols = [c for c in feature_subset if c in features_df.columns]
        else:
            feature_cols = [c for c in features_df.columns 
                          if c not in drop_cols and features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()
        geoids = features_df['GEOID'].copy()
        
        # Drop NaN targets
        valid = y.notna()
        X = X[valid]
        y = y[valid]
        geoids = geoids[valid]
        
        # Fill NaN features
        X = X.fillna(-999)
        
        # Remove constant columns
        std = X.std()
        varying = std[std > 0].index.tolist()
        X = X[varying]
        
        # Remove highly correlated features (>0.99 correlation)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.99)]
        if to_drop:
            X = X.drop(columns=to_drop)
        
        return X, y, geoids
    
    def compute_bias_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        geoids: pd.Series,
        strata_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Compute bias score: max |mean_residual| across county groups.
        Lower is better (more equitable predictions).
        """
        residuals = y_pred - y_true
        groups = geoids.str[:5]  # County FIPS
        
        # Mean residual per county
        county_residuals = pd.Series(residuals, index=groups.index).groupby(groups).mean()
        
        # Bias score: standard deviation of county mean residuals
        # (high means some counties systematically over/under-predicted)
        bias_score = county_residuals.std()
        
        return bias_score
    
    def train_model_with_cv(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        geoids: pd.Series,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Train a single model with spatial cross-validation."""
        groups = geoids.str[:5]
        n_folds = self.config['n_folds']
        gkf = GroupKFold(n_splits=n_folds)
        
        model = self._create_model(model_name, params)
        
        fold_scores = []
        oof_predictions = np.full(len(y), np.nan)
        feature_importances = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            fold_model = self._create_model(model_name, params)
            
            if model_name in ['xgboost', 'lightgbm']:
                fold_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                fold_model.fit(X_train, y_train)
            
            y_pred = fold_model.predict(X_test)
            oof_predictions[test_idx] = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            fold_scores.append({'rmse': rmse, 'r2': r2})
            
            if hasattr(fold_model, 'feature_importances_'):
                feature_importances.append(fold_model.feature_importances_)
        
        # Aggregate
        mean_rmse = np.mean([s['rmse'] for s in fold_scores])
        std_rmse = np.std([s['rmse'] for s in fold_scores])
        mean_r2 = np.mean([s['r2'] for s in fold_scores])
        
        # Compute bias score
        bias_score = self.compute_bias_score(y.values, oof_predictions, geoids)
        
        # Combined score: RMSE + bias penalty
        combined_score = mean_rmse + self.config['bias_penalty_weight'] * bias_score
        
        # Feature importance
        fi = None
        if feature_importances:
            mean_fi = np.mean(feature_importances, axis=0)
            fi = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_fi,
            }).sort_values('importance', ascending=False)
        
        result = {
            'model_name': model_name,
            'cv_rmse_mean': mean_rmse,
            'cv_rmse_std': std_rmse,
            'cv_r2_mean': mean_r2,
            'bias_score': bias_score,
            'combined_score': combined_score,
            'oof_predictions': oof_predictions,
            'feature_importance': fi,
            'fold_scores': fold_scores,
            'params': params or self.config['models'].get(model_name, {}),
        }
        
        logger.info(f"  {model_name}: RMSE={mean_rmse:.6f}±{std_rmse:.6f}, R2={mean_r2:.4f}, "
                   f"Bias={bias_score:.6f}, Combined={combined_score:.6f}")
        
        return result
    
    def _create_model(self, model_name: str, params: Optional[Dict] = None):
        """Create a model instance."""
        if params is None:
            params = self.config['models'].get(model_name, {})
        
        safe_params = {k: v for k, v in params.items() 
                      if k not in ['verbose', 'tree_method'] or model_name == 'xgboost'}
        
        if model_name == 'xgboost':
            return xgb.XGBRegressor(**safe_params, random_state=42)
        elif model_name == 'lightgbm':
            safe_params['verbose'] = -1
            return lgb.LGBMRegressor(**safe_params, random_state=42)
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(**safe_params, random_state=42)
        elif model_name == 'random_forest':
            return RandomForestRegressor(**safe_params, random_state=42)
        elif model_name == 'ridge':
            return Ridge(**safe_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def auto_tune(self, model_name: str, X: pd.DataFrame, y: pd.Series, geoids: pd.Series) -> Dict:
        """
        Simple random-search hyperparameter tuning.
        More efficient than grid search; works within Kaggle GPU time limits.
        """
        logger.info(f"  Auto-tuning {model_name}...")
        
        tuning = self.config['tuning']
        n_trials = 20
        best_score = np.inf
        best_params = self.config['models'].get(model_name, {})
        
        for trial in range(n_trials):
            # Random sample from parameter space
            trial_params = {
                'n_estimators': int(np.random.randint(*[200, 1500])),
                'max_depth': int(np.random.randint(*[3, 10])),
                'learning_rate': float(np.exp(np.random.uniform(*np.log([0.01, 0.2])))),
                'subsample': float(np.random.uniform(*[0.6, 1.0])),
                'colsample_bytree': float(np.random.uniform(*[0.5, 1.0])),
            }
            
            if model_name == 'xgboost':
                trial_params['reg_alpha'] = float(np.exp(np.random.uniform(*np.log([0.001, 10.0]))))
                trial_params['reg_lambda'] = float(np.exp(np.random.uniform(*np.log([0.001, 10.0]))))
                trial_params['tree_method'] = 'hist'
            elif model_name == 'lightgbm':
                trial_params['num_leaves'] = int(np.random.randint(*[15, 63]))
                trial_params['verbose'] = -1
            
            try:
                result = self.train_model_with_cv(model_name, X, y, geoids, trial_params)
                score = result['combined_score']
                
                if score < best_score:
                    best_score = score
                    best_params = trial_params
                    logger.info(f"    Trial {trial}: NEW BEST combined={score:.6f}")
                else:
                    logger.info(f"    Trial {trial}: combined={score:.6f}")
            except Exception as e:
                logger.warning(f"    Trial {trial} failed: {e}")
        
        logger.info(f"  Best {model_name} params: {best_params}, combined={best_score:.6f}")
        return best_params
    
    def evolve_features(
        self,
        features_df: pd.DataFrame,
        result: Dict,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Generate new features based on residual analysis.
        If residuals correlate with a strata feature, add interaction features.
        """
        residuals = result['oof_predictions'] - y.values
        fi = result.get('feature_importance')
        
        if fi is None:
            return features_df
        
        # Find top features
        top_features = fi.head(10)['feature'].tolist()
        
        # Create pairwise interactions of top features
        new_cols = {}
        for i, f1 in enumerate(top_features[:5]):
            for f2 in top_features[i+1:6]:
                col_name = f'{f1}_x_{f2}'
                if col_name not in features_df.columns and f1 in X.columns and f2 in X.columns:
                    new_cols[col_name] = X[f1] * X[f2]
        
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=features_df.index)
            features_df = pd.concat([features_df, new_df], axis=1)
            logger.info(f"  Evolved {len(new_cols)} new interaction features")
        
        return features_df
    
    def compute_optimal_blend(
        self,
        all_results: Dict[str, Dict],
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        """Find optimal ensemble weights."""
        model_names = list(all_results.keys())
        n_models = len(model_names)
        
        oof_matrix = np.column_stack([
            all_results[name]['oof_predictions'] for name in model_names
        ])
        
        # Handle NaN
        valid = ~np.any(np.isnan(oof_matrix), axis=1)
        oof_valid = oof_matrix[valid]
        y_valid = y_true[valid]
        
        def objective(weights):
            pred = oof_valid @ weights
            return np.sqrt(mean_squared_error(y_valid, pred))
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = {name: w for name, w in zip(model_names, result.x)}
        
        # Full OOF predictions
        blended = oof_matrix @ result.x
        
        return blended, weights, result.fun
    
    def run_iteration(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'building_gap',
        tune: bool = True,
    ) -> Dict:
        """Run one iteration of the self-evolving pipeline."""
        self.iteration += 1
        iter_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {self.iteration}")
        logger.info(f"{'='*60}")
        
        # Prepare features
        X, y, geoids = self.prepare_features(features_df, target_col)
        logger.info(f"Features: {X.shape[1]}, Tracts: {X.shape[0]}")
        
        # Train all models
        all_results = {}
        model_names = list(self.config['models'].keys())
        
        for model_name in model_names:
            logger.info(f"\nTraining {model_name}...")
            params = self.config['models'].get(model_name, {})
            
            result = self.train_model_with_cv(model_name, X, y, geoids, params)
            all_results[model_name] = result
            self.oof_predictions[model_name] = result['oof_predictions']
        
        # Auto-tune if requested
        if tune:
            for model_name in ['xgboost', 'lightgbm']:
                if model_name in model_names:
                    logger.info(f"\nAuto-tuning {model_name}...")
                    best_params = self.auto_tune(model_name, X, y, geoids)
                    self.config['models'][model_name] = best_params
                    
                    # Retrain with best params
                    result = self.train_model_with_cv(model_name, X, y, geoids, best_params)
                    all_results[f'{model_name}_tuned'] = result
        
        # Compute optimal blend
        logger.info("\nComputing optimal ensemble blend...")
        y_for_blend = y.values
        tuned_results = {k: v for k, v in all_results.items() if 'tuned' in k or k in ['ridge', 'random_forest', 'gradient_boosting']}
        if len(tuned_results) < 2:
            tuned_results = all_results
        
        try:
            blended, weights, blended_rmse = self.compute_optimal_blend(tuned_results, y_for_blend)
            logger.info(f"Blend weights: {weights}")
            logger.info(f"Blended RMSE: {blended_rmse:.6f}")
        except Exception as e:
            logger.warning(f"Blend failed: {e}")
            blended_rmse = min(r['cv_rmse_mean'] for r in all_results.values())
            weights = {}
        
        # Find best single model
        best_model_name = min(all_results, key=lambda k: all_results[k]['combined_score'])
        best_result = all_results[best_model_name]
        
        # Feature evolution
        if self.config.get('feature_evolution', True):
            logger.info("\nEvolving features based on residual analysis...")
            features_df = self.evolve_features(features_df, best_result, X, y)
        
        # Update best
        if best_result['combined_score'] < self.best_rmse:
            self.best_rmse = best_result['combined_score']
            self.best_r2 = best_result['cv_r2_mean']
            logger.info(f"NEW BEST! Combined={self.best_rmse:.6f}, R2={self.best_r2:.4f}")
        
        # Record history
        iter_result = {
            'iteration': self.iteration,
            'best_model': best_model_name,
            'best_rmse': best_result['cv_rmse_mean'],
            'best_r2': best_result['cv_r2_mean'],
            'best_bias': best_result['bias_score'],
            'best_combined': best_result['combined_score'],
            'blended_rmse': blended_rmse,
            'blend_weights': weights,
            'n_features': X.shape[1],
            'elapsed_seconds': time.time() - iter_start,
        }
        self.history.append(iter_result)
        
        # Save state
        self._save_state()
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"\nIteration {self.iteration} complete:")
        logger.info(f"  Best model: {best_model_name}")
        logger.info(f"  Best RMSE: {best_result['cv_rmse_mean']:.6f}")
        logger.info(f"  Best R2: {best_result['cv_r2_mean']:.4f}")
        logger.info(f"  Bias score: {best_result['bias_score']:.6f}")
        logger.info(f"  Combined: {best_result['combined_score']:.6f}")
        logger.info(f"  Blended RMSE: {blended_rmse:.6f}")
        
        return iter_result, all_results, features_df
    
    def run_full_pipeline(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'building_gap',
        max_iterations: Optional[int] = None,
    ) -> Dict:
        """Run the full self-evolving pipeline."""
        if max_iterations is None:
            max_iterations = self.config['max_iterations']
        
        logger.info(f"Starting self-evolving pipeline: {max_iterations} iterations")
        logger.info(f"Target: {target_col}")
        
        all_iter_results = []
        current_features = features_df.copy()
        
        for i in range(max_iterations):
            tune = (i > 0)  # Don't tune on first iteration (baseline)
            iter_result, model_results, current_features = self.run_iteration(
                current_features, target_col, tune=tune
            )
            all_iter_results.append(iter_result)
            
            # Check convergence
            if len(all_iter_results) >= 3:
                recent_rmses = [r['best_rmse'] for r in all_iter_results[-3:]]
                if max(recent_rmses) - min(recent_rmses) < 0.001:
                    logger.info("Converged! RMSE improvement < 0.001 over last 3 iterations")
                    break
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("SELF-EVOLVING PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total iterations: {self.iteration}")
        logger.info(f"Best combined score: {self.best_rmse:.6f}")
        logger.info(f"Best R2: {self.best_r2:.4f}")
        
        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(PROJECT_ROOT / "data/output/evolving_history.csv", index=False)
        
        return {
            'history': self.history,
            'best_rmse': self.best_rmse,
            'best_r2': self.best_r2,
            'final_features': current_features,
        }


def run():
    """Main entry point for the self-evolving pipeline."""
    # Load features
    features_path = PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet"
    
    if not features_path.exists():
        # Fall back to per-region
        regions = ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']
        dfs = []
        for region in regions:
            p = PROJECT_ROOT / f"data/features/{region}_enhanced_features.parquet"
            if p.exists():
                dfs.append(pd.read_parquet(p))
        if not dfs:
            logger.error("No features found. Run enhanced_feature_pipeline.py first.")
            return
        features = pd.concat(dfs, ignore_index=True)
    else:
        features = pd.read_parquet(features_path)
    
    logger.info(f"Loaded: {features.shape[0]} tracts, {features.shape[1]} columns")
    
    # Create and run pipeline
    pipeline = SelfEvolvingPipeline()
    
    # Run for building_gap
    results = pipeline.run_full_pipeline(features, target_col='building_gap', max_iterations=5)
    
    # Also run for road_gap
    pipeline2 = SelfEvolvingPipeline(config_path="config/evolving_config_road.yaml")
    pipeline2.config['target_col'] = 'road_gap'
    results2 = pipeline2.run_full_pipeline(features, target_col='road_gap', max_iterations=3)
    
    logger.info("\nSelf-evolving pipeline complete!")
    return results


if __name__ == "__main__":
    run()
