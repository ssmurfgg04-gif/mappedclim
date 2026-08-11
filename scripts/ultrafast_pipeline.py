"""Ultra-fast pipeline: Train baseline models, blend, generate submission CSV."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import json, time, logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path("/home/z/my-project/bias-bounty-map")
OUTPUT_DIR = PROJECT_ROOT / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = Path("/home/z/my-project/download")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    for p in [PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet",
              PROJECT_ROOT / "kaggle_dataset/all_regions_enhanced_features.parquet"]:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info(f"Loaded {p.name}: {df.shape}")
            return df
    raise FileNotFoundError("No features found!")

def prepare_features(df, target_col='building_gap'):
    drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
            'building_gap','road_gap','building_ratio','road_ratio',
            'building_count_ratio','building_count_gap','road_count_ratio','road_count_gap',
            'road_length_ratio','road_length_gap','poi_facility_gap','poi_to_facility_ratio']
    fcols = [c for c in df.columns if c not in drop
             and df[c].dtype in [np.float64,np.float32,np.int64,np.int32,np.bool_]]
    X = df[fcols].copy()
    y = df[target_col].copy()
    geoids = df['GEOID'].copy()
    valid = y.notna()
    X, y, geoids = X[valid], y[valid], geoids[valid]
    X = X.fillna(-999)
    std = X.std()
    X = X[std[std > 0].index]
    logger.info(f"  {X.shape[1]} features, {X.shape[0]} tracts | target mean={y.mean():.4f} std={y.std():.4f}")
    return X, y, geoids, valid

def train_cv(model, X, y, geoids, n_folds=5):
    groups = geoids.str[:5]
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    importances = []
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        m = type(model)(**model.get_params())
        if isinstance(m, xgb.XGBRegressor):
            m.fit(X.iloc[ti], y.iloc[ti], eval_set=[(X.iloc[vi], y.iloc[vi])], verbose=False)
        elif isinstance(m, lgb.LGBMRegressor):
            m.fit(X.iloc[ti], y.iloc[ti], eval_set=[(X.iloc[vi], y.iloc[vi])])
        else:
            m.fit(X.iloc[ti], y.iloc[ti])
        pred = m.predict(X.iloc[vi])
        oof[vi] = pred
        rmse = np.sqrt(mean_squared_error(y.iloc[vi], pred))
        r2 = r2_score(y.iloc[vi], pred)
        fold_scores.append({'rmse': rmse, 'r2': r2})
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
        logger.info(f"    Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
    fi_df = None
    if importances:
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': np.mean(importances, axis=0)}).sort_values('importance', ascending=False)
    return {'cv_rmse': np.mean([s['rmse'] for s in fold_scores]), 'cv_r2': np.mean([s['r2'] for s in fold_scores]), 'oof': oof, 'fi': fi_df}

def blend(oofs, y):
    names = list(oofs.keys())
    mat = np.column_stack([oofs[n] for n in names])
    valid = ~np.any(np.isnan(mat), axis=1)
    mv, yv = mat[valid], y[valid]
    res = minimize(lambda w: np.sqrt(mean_squared_error(yv, mv@w)),
                   np.ones(len(names))/len(names), method='SLSQP',
                   bounds=[(0,1)]*len(names), constraints={'type':'eq','fun':lambda w:sum(w)-1})
    weights = {n: float(w) for n, w in zip(names, res.x)}
    return mat @ res.x, weights, res.fun

def main():
    t0 = time.time()
    logger.info("="*60)
    logger.info("BIAS BOUNTY - ULTRA-FAST PIPELINE")
    logger.info("="*60)
    features = load_features()
    
    # building_gap
    logger.info("\n=== TARGET: building_gap ===")
    X, y, geo, valid = prepare_features(features, 'building_gap')
    
    # Select top features by correlation with target to speed up
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(80).index.tolist()
    X = X[top_features]
    logger.info(f"  Selected top 80 features by target correlation")
    
    models = {
        'xgb': xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', random_state=42),
        'lgb': lgb.LGBMRegressor(n_estimators=1000, max_depth=6, num_leaves=31, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42),
    }
    
    results, oofs = {}, {}
    for name, m in models.items():
        logger.info(f"\n  Training {name}...")
        r = train_cv(m, X, y, geo, n_folds=5)
        results[name] = r
        oofs[name] = r['oof']
        logger.info(f"  {name}: RMSE={r['cv_rmse']:.6f} R2={r['cv_r2']:.4f}")
    
    # Blend
    best_pred, weights, best_rmse = blend(oofs, y.values)
    best_r2 = r2_score(y.values, best_pred)
    logger.info(f"\n  ENSEMBLE: RMSE={best_rmse:.6f} R2={best_r2:.4f}")
    logger.info(f"  Weights: {weights}")
    
    # road_gap
    logger.info("\n=== TARGET: road_gap ===")
    Xr, yr, geor, validr = prepare_features(features, 'road_gap')
    corr_r = Xr.corrwith(yr).abs().sort_values(ascending=False)
    Xr = Xr[corr_r.head(80).index.tolist()]
    road_m = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', random_state=42)
    road_r = train_cv(road_m, Xr, yr, geor, n_folds=5)
    logger.info(f"  Road XGBoost: RMSE={road_r['cv_rmse']:.6f} R2={road_r['cv_r2']:.4f}")
    
    # BIAS DISCOVERY
    logger.info("\n=== BIAS DISCOVERY ===")
    counties = geo.str[:5]
    county_resid = pd.Series(best_pred - y.values, index=geo.index).groupby(counties).mean()
    worst = county_resid.abs().nlargest(10)
    logger.info(f"  Worst biased counties: {worst.to_dict()}")
    
    if 'region' in features.columns:
        for region in features['region'].unique():
            mask = (features['region'] == region) & valid
            idx = np.where(mask.values)[0]
            idx = idx[idx < len(best_pred)]
            if len(idx) > 0:
                r_rmse = np.sqrt(mean_squared_error(y.values[idx], best_pred[idx]))
                logger.info(f"  Region {region}: RMSE={r_rmse:.6f} n={mask.sum()}")
    
    # SUBMISSION CSV
    logger.info("\n=== GENERATING SUBMISSION CSV ===")
    submission = pd.DataFrame({'GEOID': geo.values, 'coverage_gap_score': best_pred})
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    submission.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    submission.to_csv(DOWNLOAD_DIR / "submission.csv", index=False)
    logger.info(f"  Submission: {len(submission)} tracts")
    logger.info(f"  Stats: mean={best_pred.mean():.6f} std={best_pred.std():.6f} min={best_pred.min():.6f} max={best_pred.max():.6f}")
    
    # Save feature importance
    for name in ['xgb', 'lgb']:
        if results[name]['fi'] is not None:
            results[name]['fi'].head(50).to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
            logger.info(f"\nTop 15 features ({name}):")
            for _, row in results[name]['fi'].head(15).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Model comparison
    comp = pd.DataFrame({
        'xgb': {'RMSE': results['xgb']['cv_rmse'], 'R2': results['xgb']['cv_r2']},
        'lgb': {'RMSE': results['lgb']['cv_rmse'], 'R2': results['lgb']['cv_r2']},
        'blend': {'RMSE': best_rmse, 'R2': best_r2},
        'road_xgb': {'RMSE': road_r['cv_rmse'], 'R2': road_r['cv_r2']},
    }).T.sort_values('RMSE')
    comp.to_csv(OUTPUT_DIR / "model_comparison.csv")
    logger.info(f"\nMODEL COMPARISON:\n{comp.to_string()}")
    
    # State
    state = {'timestamp': datetime.now().isoformat(), 'elapsed_minutes': (time.time()-t0)/60,
             'best_rmse': best_rmse, 'best_r2': best_r2,
             'blend_weights': weights, 'n_features': X.shape[1], 'n_tracts': X.shape[0]}
    with open(OUTPUT_DIR / "pipeline_state.json", 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    # Predictions parquet
    pred_df = pd.DataFrame({'GEOID': geo.values, 'true': y.values, 'pred': best_pred, 'residual': y.values - best_pred})
    pred_df.to_parquet(OUTPUT_DIR / "predictions.parquet")
    
    elapsed = (time.time()-t0)/60
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f} min | Best RMSE={best_rmse:.6f} R2={best_r2:.4f}")
    logger.info(f"Submission CSV: {DOWNLOAD_DIR / 'submission.csv'}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
