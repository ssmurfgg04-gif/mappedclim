"""Fast pipeline: Train, tune, blend, bias discovery, submission CSV."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
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
    if X.shape[1] > 60:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]
        if to_drop:
            X = X.drop(columns=to_drop)
            logger.info(f"  Dropped {len(to_drop)} correlated features")
    logger.info(f"  {X.shape[1]} features, {X.shape[0]} tracts | target mean={y.mean():.4f} std={y.std():.4f}")
    return X, y, geoids, valid

def train_cv(model, X, y, geoids, n_folds=3):
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

def bias_score(y_true, y_pred, geoids):
    return pd.Series(y_pred - y_true, index=geoids.index).groupby(geoids.str[:5]).mean().std()

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

def quick_tune(model_cls, X, y, geoids, base_params, n_trials=6, n_folds=3):
    best_score, best_params = np.inf, base_params.copy()
    for t in range(n_trials):
        p = base_params.copy()
        p['n_estimators'] = int(np.random.randint(400, 1200))
        p['max_depth'] = int(np.random.randint(4, 9))
        p['learning_rate'] = float(np.exp(np.random.uniform(np.log(0.01), np.log(0.1))))
        p['subsample'] = float(np.random.uniform(0.6, 1.0))
        p['colsample_bytree'] = float(np.random.uniform(0.5, 0.9))
        if 'reg_alpha' in p:
            p['reg_alpha'] = float(np.exp(np.random.uniform(np.log(0.01), np.log(5.0))))
            p['reg_lambda'] = float(np.exp(np.random.uniform(np.log(0.1), np.log(10.0))))
        if 'num_leaves' in p:
            p['num_leaves'] = int(np.random.randint(15, 63))
        try:
            m = model_cls(**p)
            r = train_cv(m, X, y, geoids, n_folds)
            if r['cv_rmse'] < best_score:
                best_score, best_params = r['cv_rmse'], p.copy()
                logger.info(f"      Trial {t}: NEW BEST RMSE={best_score:.6f}")
            else:
                logger.info(f"      Trial {t}: RMSE={r['cv_rmse']:.6f}")
        except Exception as e:
            logger.warning(f"      Trial {t} failed: {e}")
    return best_params, best_score

def main():
    t0 = time.time()
    logger.info("="*60)
    logger.info("BIAS BOUNTY - FAST PIPELINE")
    logger.info("="*60)
    features = load_features()
    logger.info("\n=== TARGET: building_gap ===")
    X, y, geo, valid = prepare_features(features, 'building_gap')
    models = {
        'xgb': xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', random_state=42),
        'lgb': lgb.LGBMRegressor(n_estimators=800, max_depth=6, num_leaves=31, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42),
        'gb': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        'ridge': Ridge(alpha=1.0),
    }
    results, oofs = {}, {}
    for name, m in models.items():
        logger.info(f"\n  Training {name}...")
        r = train_cv(m, X, y, geo, n_folds=3)
        results[name] = r
        oofs[name] = r['oof']
        bs = bias_score(y.values, r['oof'], geo)
        results[name]['bias'] = bs
        logger.info(f"  {name}: RMSE={r['cv_rmse']:.6f} R2={r['cv_r2']:.4f} Bias={bs:.6f}")
    bl_pred, bl_w, bl_rmse = blend(oofs, y.values)
    bl_r2 = r2_score(y.values, bl_pred)
    bl_bias = bias_score(y.values, bl_pred, geo)
    logger.info(f"\n  BASELINE BLEND: RMSE={bl_rmse:.6f} R2={bl_r2:.4f} Bias={bl_bias:.6f}")
    logger.info(f"  Weights: {bl_w}")
    # Auto-tune
    logger.info("\n=== AUTO-TUNING XGBoost ===")
    xgb_base = {'tree_method':'hist','random_state':42,'n_estimators':800,'max_depth':6,'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.1,'reg_lambda':1.0}
    xgb_best_p, xgb_best_s = quick_tune(xgb.XGBRegressor, X, y, geo, xgb_base, n_trials=6)
    logger.info("\n=== AUTO-TUNING LightGBM ===")
    lgb_base = {'verbose':-1,'random_state':42,'n_estimators':800,'max_depth':6,'num_leaves':31,'learning_rate':0.03,'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.1,'reg_lambda':1.0}
    lgb_best_p, lgb_best_s = quick_tune(lgb.LGBMRegressor, X, y, geo, lgb_base, n_trials=6)
    # Retrain tuned with 5-fold
    logger.info("\n=== TUNED MODELS (5-fold CV) ===")
    tuned_oofs = {}
    txgb = xgb.XGBRegressor(**xgb_best_p)
    txgb_r = train_cv(txgb, X, y, geo, n_folds=5)
    tuned_oofs['xgb_tuned'] = txgb_r['oof']
    txgb_bias = bias_score(y.values, txgb_r['oof'], geo)
    logger.info(f"  Tuned XGBoost: RMSE={txgb_r['cv_rmse']:.6f} R2={txgb_r['cv_r2']:.4f} Bias={txgb_bias:.6f}")
    tlgb = lgb.LGBMRegressor(**lgb_best_p)
    tlgb_r = train_cv(tlgb, X, y, geo, n_folds=5)
    tuned_oofs['lgb_tuned'] = tlgb_r['oof']
    tlgb_bias = bias_score(y.values, tlgb_r['oof'], geo)
    logger.info(f"  Tuned LightGBM: RMSE={tlgb_r['cv_rmse']:.6f} R2={tlgb_r['cv_r2']:.4f} Bias={tlgb_bias:.6f}")
    tuned_oofs['gb'] = results['gb']['oof']
    final_pred, final_w, final_rmse = blend(tuned_oofs, y.values)
    final_r2 = r2_score(y.values, final_pred)
    final_bias = bias_score(y.values, final_pred, geo)
    logger.info(f"\n  FINAL ENSEMBLE: RMSE={final_rmse:.6f} R2={final_r2:.4f} Bias={final_bias:.6f}")
    if final_rmse < bl_rmse:
        best_pred, best_rmse, best_r2 = final_pred, final_rmse, final_r2
        logger.info("  TUNED BLEND IS BEST")
    else:
        best_pred, best_rmse, best_r2 = bl_pred, bl_rmse, bl_r2
        logger.info("  BASELINE BLEND IS BEST")
    # road_gap
    logger.info("\n=== TARGET: road_gap ===")
    Xr, yr, geor, validr = prepare_features(features, 'road_gap')
    road_xgb = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', random_state=42)
    road_r = train_cv(road_xgb, Xr, yr, geor, n_folds=5)
    road_bias = bias_score(yr.values, road_r['oof'], geor)
    logger.info(f"  Road XGBoost: RMSE={road_r['cv_rmse']:.6f} R2={road_r['cv_r2']:.4f} Bias={road_bias:.6f}")
    # BIAS DISCOVERY
    logger.info("\n=== BIAS DISCOVERY ===")
    findings = []
    counties = geo.str[:5]
    county_resid = pd.Series(best_pred - y.values, index=geo.index).groupby(counties).mean()
    worst_counties = county_resid.abs().nlargest(10)
    findings.append({'dimension': 'worst_biased_counties', 'counties': worst_counties.index.tolist(), 'residuals': worst_counties.values.tolist()})
    logger.info(f"  Worst biased counties: {worst_counties.to_dict()}")
    if 'region' in features.columns:
        for region in features['region'].unique():
            mask = (features['region'] == region) & valid
            if mask.sum() > 0:
                idx = np.where(mask.values)[0]
                idx = idx[idx < len(best_pred)]
                if len(idx) > 0:
                    r_pred = best_pred[idx]
                    r_true = y.values[idx]
                    r_rmse = np.sqrt(mean_squared_error(r_true, r_pred))
                    findings.append({'dimension': f'region_{region}', 'rmse': float(r_rmse), 'count': int(mask.sum())})
                    logger.info(f"  Region {region}: RMSE={r_rmse:.6f} n={mask.sum()}")
    strata_path = None
    for base in [PROJECT_ROOT / "data/features", PROJECT_ROOT / "kaggle_dataset"]:
        p = base / "national-strata-tract-table.parquet"
        if p.exists(): strata_path = p; break
    if strata_path:
        strata = pd.read_parquet(strata_path)
        logger.info(f"  Strata table: {strata.shape}")
        svi_cols = [c for c in strata.columns if 'svi' in c.lower() and 'flag' not in c.lower()][:3]
        for sc in svi_cols:
            try:
                strata[sc] = pd.to_numeric(strata[sc], errors='coerce')
                q75, q25 = strata[sc].quantile(0.75), strata[sc].quantile(0.25)
                high, low = strata[strata[sc] > q75], strata[strata[sc] < q25]
                findings.append({'dimension': f'SVI_{sc}', 'high_mean': float(high[sc].mean()), 'low_mean': float(low[sc].mean()), 'gap': float(high[sc].mean() - low[sc].mean())})
            except: pass
        tribal_cols = [c for c in strata.columns if 'tribal' in c.lower() or 'aiannh' in c.lower()]
        for tc in tribal_cols[:2]:
            try:
                tribal = strata[strata[tc] == 1]
                findings.append({'dimension': tc, 'tribal_tracts': len(tribal), 'pct': len(tribal)/len(strata)*100})
                logger.info(f"  Tribal ({tc}): {len(tribal)} tracts ({len(tribal)/len(strata)*100:.2f}%)")
            except: pass
        covered_cols = [c for c in strata.columns if '_covered' in c.lower()]
        for cc in covered_cols[:5]:
            null_pct = strata[cc].isna().mean() * 100
            findings.append({'dimension': cc, 'null_pct': null_pct, 'interpretation': 'NULL = data gap'})
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(OUTPUT_DIR / "bias_discovery_findings.csv", index=False)
    logger.info(f"  {len(findings)} bias findings saved")
    # SUBMISSION CSV
    logger.info("\n=== GENERATING SUBMISSION CSV ===")
    submission = pd.DataFrame({'GEOID': geo.values, 'coverage_gap_score': best_pred})
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    submission.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    submission.to_csv(DOWNLOAD_DIR / "submission.csv", index=False)
    logger.info(f"  Submission: {len(submission)} tracts")
    logger.info(f"  Stats: mean={best_pred.mean():.6f} std={best_pred.std():.6f} min={best_pred.min():.6f} max={best_pred.max():.6f}")
    # Save outputs
    for name in ['xgb', 'lgb']:
        if results[name]['fi'] is not None:
            results[name]['fi'].head(50).to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
    if txgb_r['fi'] is not None:
        txgb_r['fi'].head(50).to_csv(OUTPUT_DIR / "xgb_tuned_feature_importance.csv", index=False)
    if tlgb_r['fi'] is not None:
        tlgb_r['fi'].head(50).to_csv(OUTPUT_DIR / "lgb_tuned_feature_importance.csv", index=False)
    comp = {
        'xgb_baseline': {'RMSE': results['xgb']['cv_rmse'], 'R2': results['xgb']['cv_r2'], 'Bias': results['xgb']['bias']},
        'lgb_baseline': {'RMSE': results['lgb']['cv_rmse'], 'R2': results['lgb']['cv_r2'], 'Bias': results['lgb']['bias']},
        'gb_baseline': {'RMSE': results['gb']['cv_rmse'], 'R2': results['gb']['cv_r2'], 'Bias': results['gb']['bias']},
        'ridge_baseline': {'RMSE': results['ridge']['cv_rmse'], 'R2': results['ridge']['cv_r2'], 'Bias': results['ridge']['bias']},
        'xgb_tuned': {'RMSE': txgb_r['cv_rmse'], 'R2': txgb_r['cv_r2'], 'Bias': txgb_bias},
        'lgb_tuned': {'RMSE': tlgb_r['cv_rmse'], 'R2': tlgb_r['cv_r2'], 'Bias': tlgb_bias},
        'road_xgb': {'RMSE': road_r['cv_rmse'], 'R2': road_r['cv_r2'], 'Bias': road_bias},
        'best_blend': {'RMSE': best_rmse, 'R2': best_r2, 'Bias': final_bias if final_rmse < bl_rmse else bl_bias},
    }
    comp_df = pd.DataFrame(comp).T.sort_values('RMSE')
    comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
    logger.info(f"\nMODEL COMPARISON:\n{comp_df.to_string()}")
    best_fi = txgb_r['fi'] if txgb_r['fi'] is not None else results['xgb']['fi']
    if best_fi is not None:
        logger.info(f"\nTop 20 Features:")
        for _, row in best_fi.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    pred_df = pd.DataFrame({'GEOID': geo.values, 'true': y.values, 'pred': best_pred, 'residual': y.values - best_pred})
    pred_df.to_parquet(OUTPUT_DIR / "predictions.parquet")
    state = {'timestamp': datetime.now().isoformat(), 'elapsed_minutes': (time.time()-t0)/60,
             'best_rmse': best_rmse, 'best_r2': best_r2,
             'xgb_best_params': xgb_best_p, 'lgb_best_params': lgb_best_p,
             'n_features': X.shape[1], 'n_tracts': X.shape[0]}
    with open(OUTPUT_DIR / "pipeline_state.json", 'w') as f:
        json.dump(state, f, indent=2, default=str)
    elapsed = (time.time()-t0)/60
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f} min | Best RMSE={best_rmse:.6f} R2={best_r2:.4f}")
    logger.info(f"Submission: {DOWNLOAD_DIR / 'submission.csv'}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
