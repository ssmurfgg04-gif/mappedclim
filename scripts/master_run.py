"""
Master run script: Self-evolving pipeline → Bias discovery → Submission CSV.

This script:
1. Loads enhanced features from all regions
2. Runs the self-evolving pipeline (train, tune, iterate, improve)
3. Runs comprehensive bias discovery for the $1,000 prize
4. Generates the Zindi submission CSV
5. Saves all outputs for documentation ($500 prize)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    """Load enhanced features, trying multiple paths."""
    # Try combined file first
    paths_to_try = [
        PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet",
        PROJECT_ROOT / "kaggle_dataset/all_regions_enhanced_features.parquet",
    ]
    for p in paths_to_try:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info(f"Loaded {p.name}: {df.shape[0]} tracts × {df.shape[1]} cols")
            return df
    
    # Fall back to per-region
    regions = ['maricopa-az', 'northern-ca', 'eastern-ok', 'south-central-tx']
    dfs = []
    for region in regions:
        for base in [PROJECT_ROOT / "data/features", PROJECT_ROOT / "kaggle_dataset"]:
            p = base / f"{region}_enhanced_features.parquet"
            if p.exists():
                dfs.append(pd.read_parquet(p))
                break
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined per-region: {df.shape[0]} tracts × {df.shape[1]} cols")
        return df
    
    raise FileNotFoundError("No feature files found!")


def prepare_features(
    features_df: pd.DataFrame,
    target_col: str = 'building_gap',
) -> tuple:
    """Prepare feature matrix and target."""
    drop_cols = ['GEOID', 'region', 'county_fips', 'state_fips',
                'centroid_lat', 'centroid_lon',
                'building_gap', 'road_gap', 'building_ratio', 'road_ratio',
                'building_count_ratio', 'building_count_gap',
                'road_count_ratio', 'road_count_gap',
                'road_length_ratio', 'road_length_gap',
                'poi_facility_gap', 'poi_to_facility_ratio']
    
    feature_cols = [c for c in features_df.columns 
                   if c not in drop_cols 
                   and features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]]
    
    X = features_df[feature_cols].copy()
    y = features_df[target_col].copy()
    geoids = features_df['GEOID'].copy()
    
    # Drop NaN targets
    valid = y.notna()
    X, y, geoids = X[valid], y[valid], geoids[valid]
    
    # Fill NaN features
    X = X.fillna(-999)
    
    # Remove constant columns
    std = X.std()
    X = X[std[std > 0].index]
    
    # Remove highly correlated (>0.98)
    if X.shape[1] > 50:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
        if to_drop:
            X = X.drop(columns=to_drop)
            logger.info(f"  Dropped {len(to_drop)} highly correlated features")
    
    logger.info(f"  Prepared: {X.shape[1]} features, {X.shape[0]} tracts")
    logger.info(f"  Target ({target_col}): mean={y.mean():.4f}, std={y.std():.4f}")
    
    return X, y, geoids, valid


def train_with_spatial_cv(model, X, y, geoids, n_folds=5):
    """Train model with spatial GroupKFold."""
    groups = geoids.str[:5]
    gkf = GroupKFold(n_splits=n_folds)
    
    oof_preds = np.full(len(y), np.nan)
    fold_scores = []
    importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_model = type(model)(**model.get_params())
        
        if isinstance(fold_model, xgb.XGBRegressor):
            fold_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        elif isinstance(fold_model, lgb.LGBMRegressor):
            fold_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            fold_model.fit(X_train, y_train)
        
        y_pred = fold_model.predict(X_test)
        oof_preds[test_idx] = y_pred
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        fold_scores.append({'rmse': rmse, 'r2': r2})
        
        if hasattr(fold_model, 'feature_importances_'):
            importances.append(fold_model.feature_importances_)
        
        logger.info(f"    Fold {fold_idx}: RMSE={rmse:.6f}, R²={r2:.4f}")
    
    mean_rmse = np.mean([s['rmse'] for s in fold_scores])
    mean_r2 = np.mean([s['r2'] for s in fold_scores])
    
    fi_df = None
    if importances:
        fi_df = pd.DataFrame({
            'feature': X.columns,
            'importance': np.mean(importances, axis=0),
        }).sort_values('importance', ascending=False)
    
    return {
        'cv_rmse': mean_rmse,
        'cv_r2': mean_r2,
        'oof_predictions': oof_preds,
        'feature_importance': fi_df,
        'fold_scores': fold_scores,
    }


def compute_bias_score(y_true, y_pred, geoids):
    """Compute bias: std of county mean residuals."""
    residuals = y_pred - y_true
    groups = geoids.str[:5]
    county_residuals = pd.Series(residuals, index=groups.index).groupby(groups).mean()
    return county_residuals.std()


def optimal_blend(all_oof, y_true):
    """Find optimal blend weights."""
    model_names = list(all_oof.keys())
    n = len(model_names)
    oof_matrix = np.column_stack([all_oof[m] for m in model_names])
    
    valid = ~np.any(np.isnan(oof_matrix), axis=1)
    oof_v, y_v = oof_matrix[valid], y_true[valid]
    
    def objective(w):
        return np.sqrt(mean_squared_error(y_v, oof_v @ w))
    
    res = minimize(objective, np.ones(n)/n, method='SLSQP',
                  bounds=[(0,1)]*n, constraints={'type':'eq','fun':lambda w:sum(w)-1})
    
    weights = {m: w for m, w in zip(model_names, res.x)}
    blended = oof_matrix @ res.x
    return blended, weights, res.fun


def run_self_evoving_iteration(X, y, geoids, iteration, prev_best=np.inf):
    """One iteration: train models, tune, evaluate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SELF-EVOLVING ITERATION {iteration}")
    logger.info(f"{'='*60}")
    
    models_to_train = {
        'xgboost': xgb.XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', random_state=42,
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=800, max_depth=6, num_leaves=31,
            learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, random_state=42,
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        'ridge': Ridge(alpha=1.0),
    }
    
    all_results = {}
    all_oof = {}
    
    for name, model in models_to_train.items():
        logger.info(f"\n  Training {name}...")
        result = train_with_spatial_cv(model, X, y, geoids, n_folds=5)
        all_results[name] = result
        all_oof[name] = result['oof_predictions']
        bias = compute_bias_score(y.values, result['oof_predictions'], geoids)
        all_results[name]['bias_score'] = bias
        logger.info(f"  {name}: RMSE={result['cv_rmse']:.6f}, R²={result['cv_r2']:.4f}, Bias={bias:.6f}")
    
    # Optimal blend
    blended, weights, blended_rmse = optimal_blend(all_oof, y.values)
    blended_r2 = r2_score(y.values[~np.isnan(blended)], blended[~np.isnan(blended)])
    blended_bias = compute_bias_score(y.values, blended, geoids)
    
    logger.info(f"\n  ENSEMBLE: RMSE={blended_rmse:.6f}, R²={blended_r2:.4f}, Bias={blended_bias:.6f}")
    logger.info(f"  Weights: {weights}")
    
    improved = blended_rmse < prev_best
    
    return {
        'all_results': all_results,
        'blended_oof': blended,
        'blend_weights': weights,
        'blended_rmse': blended_rmse,
        'blended_r2': blended_r2,
        'blended_bias': blended_bias,
        'improved': improved,
    }


def auto_tune_xgboost(X, y, geoids, n_trials=15):
    """Random search hyperparameter tuning for XGBoost."""
    logger.info("\nAuto-tuning XGBoost...")
    best_score = np.inf
    best_params = None
    
    for trial in range(n_trials):
        params = {
            'n_estimators': int(np.random.randint(500, 1500)),
            'max_depth': int(np.random.randint(4, 9)),
            'learning_rate': float(np.exp(np.random.uniform(np.log(0.01), np.log(0.1)))),
            'subsample': float(np.random.uniform(0.6, 1.0)),
            'colsample_bytree': float(np.random.uniform(0.5, 0.9)),
            'reg_alpha': float(np.exp(np.random.uniform(np.log(0.01), np.log(5.0)))),
            'reg_lambda': float(np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))),
            'tree_method': 'hist',
            'random_state': 42,
        }
        
        try:
            model = xgb.XGBRegressor(**params)
            result = train_with_spatial_cv(model, X, y, geoids, n_folds=3)
            score = result['cv_rmse']
            
            if score < best_score:
                best_score = score
                best_params = params
                logger.info(f"    Trial {trial}: NEW BEST RMSE={score:.6f}")
            else:
                logger.info(f"    Trial {trial}: RMSE={score:.6f}")
        except Exception as e:
            logger.warning(f"    Trial {trial} failed: {e}")
    
    logger.info(f"  Best XGBoost RMSE: {best_score:.6f}")
    logger.info(f"  Best params: {best_params}")
    return best_params, best_score


def auto_tune_lightgbm(X, y, geoids, n_trials=15):
    """Random search hyperparameter tuning for LightGBM."""
    logger.info("\nAuto-tuning LightGBM...")
    best_score = np.inf
    best_params = None
    
    for trial in range(n_trials):
        params = {
            'n_estimators': int(np.random.randint(500, 1500)),
            'max_depth': int(np.random.randint(4, 9)),
            'num_leaves': int(np.random.randint(15, 63)),
            'learning_rate': float(np.exp(np.random.uniform(np.log(0.01), np.log(0.1)))),
            'subsample': float(np.random.uniform(0.6, 1.0)),
            'colsample_bytree': float(np.random.uniform(0.5, 0.9)),
            'reg_alpha': float(np.exp(np.random.uniform(np.log(0.01), np.log(5.0)))),
            'reg_lambda': float(np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))),
            'verbose': -1,
            'random_state': 42,
        }
        
        try:
            model = lgb.LGBMRegressor(**params)
            result = train_with_spatial_cv(model, X, y, geoids, n_folds=3)
            score = result['cv_rmse']
            
            if score < best_score:
                best_score = score
                best_params = params
                logger.info(f"    Trial {trial}: NEW BEST RMSE={score:.6f}")
            else:
                logger.info(f"    Trial {trial}: RMSE={score:.6f}")
        except Exception as e:
            logger.warning(f"    Trial {trial} failed: {e}")
    
    logger.info(f"  Best LightGBM RMSE: {best_score:.6f}")
    return best_params, best_score


def run_bias_discovery(features_df, predictions, geoids_col='GEOID'):
    """Run bias discovery analysis across strata."""
    logger.info("\n" + "="*60)
    logger.info("BIAS DISCOVERY ANALYSIS ($1,000 Prize)")
    logger.info("="*60)
    
    findings = []
    
    # Load national strata for rich demographic data
    strata_path = None
    for base in [PROJECT_ROOT / "data/features", PROJECT_ROOT / "kaggle_dataset"]:
        p = base / "national-strata-tract-table.parquet"
        if p.exists():
            strata_path = p
            break
    
    if strata_path:
        strata = pd.read_parquet(strata_path)
        logger.info(f"Loaded strata table: {strata.shape}")
        
        # Merge predictions with strata
        if 'GEOID' in strata.columns:
            strata['GEOID'] = strata['GEOID'].astype(str)
            merged = features_df[[geoids_col]].copy()
            merged['prediction'] = predictions
            merged[geoids_col] = merged[geoids_col].astype(str)
            
            # Ensure GEOID formats match
            strata_geoid_col = 'GEOID'
            
            # Analyze by rural/urban
            rural_cols = [c for c in strata.columns if 'rural' in c.lower() or 'urban' in c.lower()]
            if rural_cols:
                logger.info(f"\n  Rural/Urban columns found: {rural_cols[:5]}")
            
            # Analyze by SVI
            svi_cols = [c for c in strata.columns if 'svi' in c.lower() and 'flag' not in c.lower()]
            if svi_cols:
                logger.info(f"  SVI columns found: {svi_cols[:5]}")
                
                # Find tracts with high SVI (social vulnerability)
                for svi_col in svi_cols[:3]:
                    try:
                        strata[svi_col] = pd.to_numeric(strata[svi_col], errors='coerce')
                        high_svi = strata[strata[svi_col] > strata[svi_col].quantile(0.75)]
                        low_svi = strata[strata[svi_col] < strata[svi_col].quantile(0.25)]
                        findings.append({
                            'dimension': svi_col,
                            'high_group_mean': high_svi[svi_col].mean(),
                            'low_group_mean': low_svi[svi_col].mean(),
                            'gap': high_svi[svi_col].mean() - low_svi[svi_col].mean(),
                            'high_count': len(high_svi),
                            'low_count': len(low_svi),
                        })
                    except Exception as e:
                        logger.warning(f"  SVI analysis failed for {svi_col}: {e}")
            
            # Analyze by tribal status
            tribal_cols = [c for c in strata.columns if 'tribal' in c.lower() or 'aiannh' in c.lower()]
            if tribal_cols:
                logger.info(f"  Tribal columns found: {tribal_cols[:5]}")
                for tc in tribal_cols[:2]:
                    try:
                        tribal_tracts = strata[strata[tc] == 1]
                        non_tribal = strata[strata[tc] == 0]
                        findings.append({
                            'dimension': tc,
                            'tribal_count': len(tribal_tracts),
                            'non_tribal_count': len(non_tribal),
                            'tribal_pct': len(tribal_tracts) / len(strata) * 100,
                        })
                    except Exception as e:
                        logger.warning(f"  Tribal analysis failed for {tc}: {e}")
            
            # Analyze by hazard exposure
            hazard_cols = [c for c in strata.columns if 'hazard' in c.lower() or 'wildfire' in c.lower() or 'flood' in c.lower()]
            if hazard_cols:
                logger.info(f"  Hazard columns found: {hazard_cols[:5]}")
            
            # Analyze covered flags
            covered_cols = [c for c in strata.columns if '_covered' in c.lower()]
            if covered_cols:
                logger.info(f"  Covered flag columns: {len(covered_cols)}")
                # NULL means data doesn't reach tract = coverage gap signal
                for cc in covered_cols[:5]:
                    null_count = strata[cc].isna().sum()
                    findings.append({
                        'dimension': cc,
                        'null_count': null_count,
                        'null_pct': null_count / len(strata) * 100,
                        'interpretation': 'NULL = data layer does not reach tract (coverage gap signal)',
                    })
    
    # Regional bias analysis
    if 'region' in features_df.columns:
        for region in features_df['region'].unique():
            mask = features_df['region'] == region
            region_preds = predictions[mask.values[:len(predictions)]] if len(predictions) >= mask.sum() else predictions[:mask.sum()]
            findings.append({
                'dimension': f'region_{region}',
                'mean_prediction': np.mean(region_preds),
                'std_prediction': np.std(region_preds),
                'count': mask.sum(),
            })
    
    # County-level bias analysis
    if 'GEOID' in features_df.columns:
        counties = features_df['GEOID'].str[:5]
        county_means = pd.Series(predictions, index=features_df.index[:len(predictions)]).groupby(counties).mean()
        findings.append({
            'dimension': 'county_level_bias',
            'max_county_mean': county_means.max(),
            'min_county_mean': county_means.min(),
            'county_spread': county_means.max() - county_means.min(),
            'n_counties': len(county_means),
        })
    
    # Save findings
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(OUTPUT_DIR / "bias_discovery_findings.csv", index=False)
    logger.info(f"\n  Found {len(findings)} bias dimensions")
    logger.info(f"  Saved to bias_discovery_findings.csv")
    
    return findings


def generate_submission_csv(features_df, predictions, valid_mask, target_col='building_gap'):
    """
    Generate Zindi submission CSV.
    
    Since the actual test set isn't released yet, we produce:
    1. A sample submission for all focus region tracts
    2. OOF predictions for validation
    
    Format: GEOID,coverage_gap_score (the competition target)
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING ZINDI SUBMISSION CSV")
    logger.info("="*60)
    
    # Get GEOIDs for all tracts (focus regions)
    all_geoids = features_df['GEOID'].values
    
    # Create submission DataFrame
    # For now, use our self-supervised predictions as the coverage_gap_score
    # When the actual target is released, we'll retrain
    
    # Full submission (all tracts with predictions)
    submission = pd.DataFrame({
        'GEOID': all_geoids[:len(predictions)],
        'coverage_gap_score': predictions,
    })
    
    # Also create OOF submission (only valid tracts with CV predictions)
    oof_submission = submission.copy()
    
    # Save
    submission_path = OUTPUT_DIR / "submission.csv"
    oof_path = OUTPUT_DIR / "oof_submission.csv"
    
    submission.to_csv(submission_path, index=False)
    oof_submission.to_csv(oof_path, index=False)
    
    logger.info(f"  Submission: {len(submission)} tracts")
    logger.info(f"  Saved to {submission_path}")
    logger.info(f"  Prediction stats: mean={predictions.mean():.6f}, std={predictions.std():.6f}")
    logger.info(f"  Min={predictions.min():.6f}, Max={predictions.max():.6f}")
    
    return submission


def main():
    """Master pipeline."""
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("BIAS BOUNTY MAPPING EQUITY - MASTER PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load features
    features = load_features()
    
    # Step 2: Prepare features for building_gap
    logger.info("\n--- TARGET: building_gap ---")
    X_b, y_b, geoids_b, valid_b = prepare_features(features, 'building_gap')
    
    # Step 3: Self-evolving pipeline - Iteration 1 (baseline)
    iter1 = run_self_evoving_iteration(X_b, y_b, geoids_b, iteration=1)
    best_rmse = iter1['blended_rmse']
    best_predictions = iter1['blended_oof']
    
    # Step 4: Auto-tune XGBoost
    best_xgb_params, best_xgb_score = auto_tune_xgboost(X_b, y_b, geoids_b, n_trials=12)
    
    # Step 5: Auto-tune LightGBM
    best_lgb_params, best_lgb_score = auto_tune_lightgbm(X_b, y_b, geoids_b, n_trials=12)
    
    # Step 6: Iteration 2 with tuned models
    logger.info("\n--- RETRAINING WITH TUNED HYPERPARAMETERS ---")
    tuned_results = {}
    tuned_oof = {}
    
    # Tuned XGBoost
    tuned_xgb = xgb.XGBRegressor(**best_xgb_params)
    xgb_res = train_with_spatial_cv(tuned_xgb, X_b, y_b, geoids_b, n_folds=5)
    tuned_results['xgboost_tuned'] = xgb_res
    tuned_oof['xgboost_tuned'] = xgb_res['oof_predictions']
    xgb_bias = compute_bias_score(y_b.values, xgb_res['oof_predictions'], geoids_b)
    logger.info(f"  Tuned XGBoost: RMSE={xgb_res['cv_rmse']:.6f}, R²={xgb_res['cv_r2']:.4f}, Bias={xgb_bias:.6f}")
    
    # Tuned LightGBM
    tuned_lgb = lgb.LGBMRegressor(**best_lgb_params)
    lgb_res = train_with_spatial_cv(tuned_lgb, X_b, y_b, geoids_b, n_folds=5)
    tuned_results['lightgbm_tuned'] = lgb_res
    tuned_oof['lightgbm_tuned'] = lgb_res['oof_predictions']
    lgb_bias = compute_bias_score(y_b.values, lgb_res['oof_predictions'], geoids_b)
    logger.info(f"  Tuned LightGBM: RMSE={lgb_res['cv_rmse']:.6f}, R²={lgb_res['cv_r2']:.4f}, Bias={lgb_bias:.6f}")
    
    # Blend tuned models
    tuned_blended, tuned_weights, tuned_rmse = optimal_blend(tuned_oof, y_b.values)
    tuned_r2 = r2_score(y_b.values[~np.isnan(tuned_blended)], tuned_blended[~np.isnan(tuned_blended)])
    tuned_bias = compute_bias_score(y_b.values, tuned_blended, geoids_b)
    logger.info(f"\n  TUNED ENSEMBLE: RMSE={tuned_rmse:.6f}, R²={tuned_r2:.4f}, Bias={tuned_bias:.6f}")
    logger.info(f"  Weights: {tuned_weights}")
    
    # Use whichever blend is better
    if tuned_rmse < best_rmse:
        best_predictions = tuned_blended
        best_rmse = tuned_rmse
        logger.info(f"  TUNED IS BETTER! Using tuned predictions.")
    else:
        logger.info(f"  Baseline is better. Keeping iteration 1 predictions.")
    
    # Step 7: Also train road_gap model
    logger.info("\n--- TARGET: road_gap ---")
    X_r, y_r, geoids_r, valid_r = prepare_features(features, 'road_gap')
    
    road_xgb = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', random_state=42,
    )
    road_res = train_with_spatial_cv(road_xgb, X_r, y_r, geoids_r, n_folds=5)
    logger.info(f"  Road XGBoost: RMSE={road_res['cv_rmse']:.6f}, R²={road_res['cv_r2']:.4f}")
    
    # Step 8: Bias discovery
    full_predictions = np.full(len(features), np.nan)
    full_predictions[valid_b.values[:len(best_predictions)]] = best_predictions
    
    findings = run_bias_discovery(features, best_predictions)
    
    # Step 9: Generate submission CSV
    submission = generate_submission_csv(features, best_predictions, valid_b, 'building_gap')
    
    # Step 10: Save all results
    elapsed = time.time() - start_time
    
    # Save feature importance
    for name in ['xgboost', 'lightgbm']:
        if name in iter1['all_results'] and iter1['all_results'][name]['feature_importance'] is not None:
            fi = iter1['all_results'][name]['feature_importance']
            fi.head(50).to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
            logger.info(f"\nTop 20 features ({name}):")
            for _, row in fi.head(20).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save tuned feature importance
    for name, res in [('xgboost_tuned', xgb_res), ('lightgbm_tuned', lgb_res)]:
        if res['feature_importance'] is not None:
            res['feature_importance'].head(50).to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
    
    # Save model comparison
    comparison_data = {}
    for name, res in iter1['all_results'].items():
        comparison_data[name] = {
            'CV RMSE': res['cv_rmse'],
            'CV R2': res['cv_r2'],
            'Bias Score': res['bias_score'],
        }
    comparison_data['tuned_xgboost'] = {
        'CV RMSE': xgb_res['cv_rmse'],
        'CV R2': xgb_res['cv_r2'],
        'Bias Score': xgb_bias,
    }
    comparison_data['tuned_lightgbm'] = {
        'CV RMSE': lgb_res['cv_rmse'],
        'CV R2': lgb_res['cv_r2'],
        'Bias Score': lgb_bias,
    }
    comparison_data['best_blend'] = {
        'CV RMSE': best_rmse,
        'CV R2': tuned_r2 if tuned_rmse < iter1['blended_rmse'] else iter1['blended_r2'],
        'Bias Score': tuned_bias if tuned_rmse < iter1['blended_rmse'] else iter1['blended_bias'],
    }
    comparison_data['road_xgboost'] = {
        'CV RMSE': road_res['cv_rmse'],
        'CV R2': road_res['cv_r2'],
        'Bias Score': compute_bias_score(y_r.values, road_res['oof_predictions'], geoids_r),
    }
    
    comparison_df = pd.DataFrame(comparison_data).T.sort_values('CV RMSE')
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
    logger.info(f"\nMODEL COMPARISON:\n{comparison_df.to_string()}")
    
    # Save pipeline state
    state = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'best_blend_rmse': best_rmse,
        'best_xgb_params': best_xgb_params,
        'best_lgb_params': best_lgb_params,
        'tuned_blend_weights': {k: float(v) for k, v in tuned_weights.items()} if tuned_rmse < iter1['blended_rmse'] else {k: float(v) for k, v in iter1['blend_weights'].items()},
        'n_features': X_b.shape[1],
        'n_tracts': X_b.shape[0],
        'n_bias_findings': len(findings),
    }
    with open(OUTPUT_DIR / "pipeline_state.json", 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    # Save predictions parquet
    pred_df = pd.DataFrame({
        'GEOID': geoids_b.values,
        'building_gap_true': y_b.values,
        'building_gap_pred': best_predictions[:len(y_b)],
        'residual': y_b.values - best_predictions[:len(y_b)],
    })
    pred_df.to_parquet(OUTPUT_DIR / "predictions.parquet")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"Best CV RMSE: {best_rmse:.6f}")
    logger.info(f"Submission saved: {OUTPUT_DIR / 'submission.csv'}")
    logger.info(f"{'='*60}")
    
    return state


if __name__ == "__main__":
    main()
