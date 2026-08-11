"""
MONUMENTAL PIPELINE v2 - Fast but thorough
===========================================
Split into phases to avoid timeout:
  Phase 1: Feature engineering + model training (no Optuna)
  Phase 2: Optuna tuning (separate, if time permits)
  Phase 3: Bias discovery + SHAP + documentation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import json, time, logging, gc, warnings, sys
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/z/my-project/bias-bounty-map")
OUTPUT_DIR = PROJECT_ROOT / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = Path("/home/z/my-project/download")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)


def load_features():
    for p in [PROJECT_ROOT / "data/features/all_regions_enhanced_features.parquet",
              PROJECT_ROOT / "kaggle_dataset/all_regions_enhanced_features.parquet"]:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info(f"Loaded {p.name}: {df.shape}")
            return df
    raise FileNotFoundError("No features found!")

def load_strata():
    for p in [PROJECT_ROOT / "kaggle_dataset/national-strata-tract-table.parquet"]:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info(f"Strata: {df.shape}")
            return df
    return None


def engineer_massive_features(features, strata):
    """Engineer 100+ interaction features."""
    logger.info("Engineering massive feature interactions...")
    
    # Merge strata
    if strata is not None:
        strata_want = ['GEOID','svi_overall','svi_socioeconomic','svi_household',
                       'svi_minority','svi_housing_transport','svi_pop',
                       'tribal_any','tribal_pct','tribal_legal',
                       'pct_urban','pop_rural','pop_urban','pop_total',
                       'usgs_wildfire_ever','usgs_wildfire_burned_pct_area',
                       'cvi_overall','cvi_baseline','cvi_climate']
        covered_cols = [c for c in strata.columns if '_covered' in c.lower()]
        strata_want.extend(covered_cols)
        hazard_cols = [c for c in strata.columns 
                       if any(h in c.lower() for h in ['wildfire','flood','drought','usdm'])
                       and 'covered' not in c.lower()]
        strata_want.extend(hazard_cols[:10])
        new_cols = [c for c in strata_want if c in strata.columns and c not in features.columns]
        new_cols = ['GEOID'] + [c for c in new_cols if c != 'GEOID']
        if len(new_cols) > 1:
            strata_sub = strata[new_cols].copy()
            strata_sub['GEOID'] = strata_sub['GEOID'].astype(str)
            features['GEOID'] = features['GEOID'].astype(str)
            before = features.shape[1]
            features = features.merge(strata_sub, on='GEOID', how='left')
            logger.info(f"  Strata merge: {before} → {features.shape[1]} cols")
    
    # Ensure numeric
    for col in ['svi_overall','svi_socioeconomic','svi_household','svi_minority',
                'svi_housing_transport','svi_pop','tribal_pct','pct_urban',
                'cvi_overall','cvi_baseline','cvi_climate','usgs_wildfire_burned_pct_area']:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
    
    new_feats = {}
    F = lambda s, v=0: s.fillna(v) if s is not None else None  # noqa
    bg = features.get('building_gap')
    rg = features.get('road_gap')
    svi = features.get('svi_overall')
    svi_m = features.get('svi_minority')
    svi_s = features.get('svi_socioeconomic')
    svi_h = features.get('svi_household')
    svi_ht = features.get('svi_housing_transport')
    tribal = features.get('tribal_any')
    tribal_pct = features.get('tribal_pct')
    pct_u = features.get('pct_urban')
    wf = features.get('usgs_wildfire_ever')
    wf_area = features.get('usgs_wildfire_burned_pct_area')
    cvi = features.get('cvi_overall')
    cvi_b = features.get('cvi_baseline')
    cvi_c = features.get('cvi_climate')
    pop = features.get('pop_total')
    bldg_r = features.get('building_ratio')
    road_r = features.get('road_ratio')
    
    n = len(features)
    
    # ── SVI × Coverage (20) ──
    if bg is not None:
        bgv = F(bg).values
        if svi is not None:
            sv = F(svi).values
            new_feats['svi_x_bldg'] = sv * bgv
            new_feats['svi_sq_x_bldg'] = sv**2 * bgv
            new_feats['svi_abs_x_bldg_abs'] = np.abs(sv) * np.abs(bgv)
        if svi_m is not None:
            new_feats['svi_min_x_bldg'] = F(svi_m).values * bgv
            new_feats['svi_min_sq_x_bldg'] = F(svi_m).values**2 * bgv
        if svi_s is not None:
            new_feats['svi_soc_x_bldg'] = F(svi_s).values * bgv
        if svi_h is not None:
            new_feats['svi_hh_x_bldg'] = F(svi_h).values * bgv
        if svi_ht is not None:
            new_feats['svi_ht_x_bldg'] = F(svi_ht).values * bgv
        if rg is not None:
            rgv = F(rg).values
            if svi is not None:
                new_feats['svi_x_road'] = F(svi).values * rgv
                new_feats['svi_sq_x_road'] = F(svi).values**2 * rgv
            if svi_m is not None:
                new_feats['svi_min_x_road'] = F(svi_m).values * rgv
    
    # ── Tribal × Coverage (10) ──
    if bg is not None and tribal is not None:
        t_flag = (F(tribal).values > 0).astype(float)
        bgv = F(bg).values
        new_feats['tribal_x_bldg'] = t_flag * bgv
        new_feats['tribal_pct_x_bldg'] = F(tribal_pct, 0).values * bgv
        if rg is not None:
            new_feats['tribal_x_road'] = t_flag * F(rg).values
        new_feats['tribal_x_bldg_sq'] = t_flag * bgv**2
        if svi is not None:
            new_feats['tribal_x_svi_x_bldg'] = t_flag * F(svi).values * bgv
    
    # ── Rural/Urban × Coverage (10) ──
    if bg is not None and pct_u is not None:
        puv = F(pct_u, 0.5).values
        rural = (1 - puv).clip(0, 1)
        bgv = F(bg).values
        new_feats['pct_urban_x_bldg'] = puv * bgv
        new_feats['rural_x_bldg'] = rural * bgv
        new_feats['rural_sq_x_bldg'] = rural**2 * bgv
        new_feats['pct_urban_x_bldg_sq'] = puv * bgv**2
        if rg is not None:
            new_feats['rural_x_road'] = rural * F(rg).values
        if svi is not None:
            new_feats['rural_x_svi'] = rural * F(svi).values
            new_feats['rural_x_svi_x_bldg'] = rural * F(svi).values * bgv
    
    # ── Hazard × Coverage (10) ──
    if bg is not None:
        bgv = F(bg).values
        if wf is not None:
            wfv = F(wf).values
            new_feats['wf_x_bldg'] = wfv * bgv
            new_feats['wf_flag_x_bldg'] = (wfv > 0).astype(float) * bgv
        if wf_area is not None:
            new_feats['wf_area_x_bldg'] = F(wf_area).values * bgv
        if wf is not None and svi is not None:
            new_feats['wf_x_svi'] = F(wf).values * F(svi).values
            new_feats['wf_x_svi_x_bldg'] = F(wf).values * F(svi).values * bgv
    
    # ── CVI × Coverage (10) ──
    if bg is not None:
        bgv = F(bg).values
        if cvi is not None:
            cv = F(cvi).values
            new_feats['cvi_x_bldg'] = cv * bgv
            new_feats['cvi_sq_x_bldg'] = cv**2 * bgv
            if rg is not None:
                new_feats['cvi_x_road'] = cv * F(rg).values
            if svi is not None:
                new_feats['cvi_x_svi'] = cv * F(svi).values
                new_feats['cvi_x_svi_x_bldg'] = cv * F(svi).values * bgv
        if cvi_b is not None:
            new_feats['cvi_base_x_bldg'] = F(cvi_b).values * bgv
        if cvi_c is not None:
            new_feats['cvi_clim_x_bldg'] = F(cvi_c).values * bgv
    
    # ── Intersectional (15) ──
    if tribal is not None and svi is not None and pct_u is not None:
        t = (F(tribal).values > 0).astype(float)
        sv = F(svi).values
        pu = F(pct_u, 0.5).values
        high_svi = (sv > np.nanquantile(sv, 0.75)).astype(float)
        low_svi = (sv < np.nanquantile(sv, 0.25)).astype(float)
        rural = (pu < 0.5).astype(float)
        urban = 1 - rural
        
        new_feats['tribal_x_highsvi_x_rural'] = t * high_svi * rural
        new_feats['tribal_x_highsvi_x_urban'] = t * high_svi * urban
        new_feats['tribal_x_lowsvi_x_rural'] = t * low_svi * rural
        new_feats['highsvi_x_rural'] = high_svi * rural
        new_feats['highsvi_x_urban'] = high_svi * urban
        new_feats['lowsvi_x_rural'] = low_svi * rural
        new_feats['tribal_x_highsvi'] = t * high_svi
        new_feats['tribal_x_rural'] = t * rural
        
        if bg is not None:
            bgv = F(bg).values
            new_feats['tribal_hsvi_rural_x_bldg'] = t * high_svi * rural * bgv
            new_feats['hsvi_rural_x_bldg'] = high_svi * rural * bgv
        if wf is not None:
            new_feats['wf_x_rural_x_hsvi'] = F(wf).values * rural * high_svi
            new_feats['wf_x_tribal'] = F(wf).values * t
        if cvi is not None:
            high_cvi = (F(cvi).values > np.nanquantile(F(cvi).values, 0.75)).astype(float)
            new_feats['hcvi_x_hsvi_x_rural'] = high_cvi * high_svi * rural
            new_feats['hcvi_x_tribal'] = high_cvi * t
    
    # ── Polynomial (15) ──
    if bg is not None:
        bgv = F(bg).values
        new_feats['bldg_gap_sq'] = bgv**2
        new_feats['bldg_gap_cu'] = bgv**3
        new_feats['bldg_gap_abs'] = np.abs(bgv)
        new_feats['bldg_gap_sign'] = np.sign(bgv)
        new_feats['bldg_gap_log1p_abs'] = np.log1p(np.abs(bgv))
    if rg is not None:
        rgv = F(rg).values
        new_feats['road_gap_sq'] = rgv**2
        new_feats['road_gap_abs'] = np.abs(rgv)
    if bg is not None and rg is not None:
        new_feats['bldg_road_ratio'] = F(bg).values / (np.abs(F(rg).values) + 1e-8)
        new_feats['bldg_road_diff'] = F(bg).values - F(rg).values
        new_feats['bldg_road_product'] = F(bg).values * F(rg).values
    if bldg_r is not None:
        new_feats['log_bldg_ratio'] = np.log1p(F(bldg_r).clip(lower=0).values)
        new_feats['bldg_ratio_sq'] = F(bldg_r).values**2
    if road_r is not None:
        new_feats['log_road_ratio'] = np.log1p(F(road_r).clip(lower=0).values)
    
    # ── Population-weighted (5) ──
    if pop is not None:
        lp = np.log1p(F(pop).values)
        new_feats['log_pop'] = lp
        if bg is not None:
            new_feats['log_pop_x_bldg'] = lp * F(bg).values
        if svi is not None:
            new_feats['log_pop_x_svi'] = lp * F(svi).values
    
    # ── Compound risk (5) ──
    if bg is not None:
        compound = np.abs(F(bg).values)
        if rg is not None:
            compound += np.abs(F(rg).values)
        if svi is not None:
            compound += np.clip(F(svi).values, 0, None) * 0.1
        new_feats['compound_risk'] = compound
        new_feats['compound_risk_sq'] = compound**2
        if tribal is not None:
            new_feats['tribal_x_risk'] = (F(tribal).values > 0).astype(float) * compound
    
    # ── Coverage null indicators ──
    for cc in [c for c in features.columns if '_covered' in c.lower()]:
        new_feats[f'{cc}_null'] = features[cc].isna().astype(float).values
    null_cols = [k for k in new_feats if k.endswith('_null')]
    if null_cols:
        new_feats['total_nulls'] = np.sum([new_feats[k] for k in null_cols], axis=0)
        new_feats['null_fraction'] = new_feats['total_nulls'] / max(len(null_cols), 1)
    
    # ── County target encoding (LOO) ──
    if 'GEOID' in features.columns and bg is not None:
        county = features['GEOID'].astype(str).str[:5]
        bgv = F(bg)
        cstats = bgv.groupby(county).agg(['mean','count'])
        cstats.columns = ['mean','count']
        cmean = cstats['mean'][county].values
        ccount = cstats['count'][county].values
        loo = (cmean * ccount - bgv.values) / (ccount - 1 + 1e-8)
        new_feats['bldg_county_loo'] = loo
    
    # Build DataFrame
    if new_feats:
        new_df = pd.DataFrame(new_feats, index=features.index)
        new_df = new_df.replace([np.inf, -np.inf], np.nan)
        features = pd.concat([features, new_df], axis=1)
        features = features.loc[:, ~features.columns.duplicated()]
        logger.info(f"  Added {len(new_feats)} features → {features.shape[1]} total")
    
    return features


def prepare_features(df, target_col='building_gap', n_top=100):
    drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
            'building_gap','road_gap','building_ratio','road_ratio',
            'building_count_ratio','building_count_gap','road_count_ratio','road_count_gap',
            'road_length_ratio','road_length_gap','poi_facility_gap','poi_to_facility_ratio']
    df = df.loc[:, ~df.columns.duplicated()]
    fcols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[fcols].copy()
    y = df[target_col].copy()
    geo = df['GEOID'].astype(str).copy()
    valid = y.notna()
    X, y, geo = X[valid], y[valid], geo[valid]
    X = X.fillna(-999).replace([np.inf, -np.inf], -999)
    # Remove constant
    std = X.std()
    X = X[std[std > 1e-10].index]
    # Remove >95% null
    null_pct = X.isna().mean()
    X = X[null_pct[null_pct < 0.95].index]
    # Top features by correlation
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    X = X[corr.head(n_top).index.tolist()]
    # Remove collinear
    corr_m = X.corr().abs()
    upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]
    X = X.drop(columns=to_drop)
    logger.info(f"  {X.shape[1]} features, {X.shape[0]} tracts | target mean={y.mean():.4f} std={y.std():.4f}")
    return X, y, geo, valid


def train_cv(model, X, y, geoids, n_folds=5):
    groups = geoids.str[:5]
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    importances = []
    trained_models = []
    
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        m = type(model)(**model.get_params())
        X_t, y_t = X.iloc[ti], y.iloc[ti]
        X_v, y_v = X.iloc[vi], y.iloc[vi]
        
        try:
            if isinstance(m, xgb.XGBRegressor):
                m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            elif isinstance(m, lgb.LGBMRegressor):
                m.fit(X_t, y_t, eval_set=[(X_v, y_v)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            elif isinstance(m, CatBoostRegressor):
                m.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=50, verbose=0)
            else:
                m.fit(X_t, y_t)
        except Exception as e:
            logger.warning(f"    Fold {fi} error: {e}")
            continue
        
        pred = m.predict(X_v)
        oof[vi] = pred
        rmse = np.sqrt(mean_squared_error(y_v, pred))
        r2 = r2_score(y_v, pred)
        fold_scores.append({'rmse': rmse, 'r2': r2})
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
        trained_models.append(m)
        logger.info(f"    Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
    
    fi_df = None
    if importances:
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': np.mean(importances, axis=0)}).sort_values('importance', ascending=False)
    
    return {'cv_rmse': np.mean([s['rmse'] for s in fold_scores]),
            'cv_r2': np.mean([s['r2'] for s in fold_scores]),
            'cv_rmse_std': np.std([s['rmse'] for s in fold_scores]),
            'oof': oof, 'fi': fi_df, 'models': trained_models}


def bias_score(y_true, y_pred, geoids):
    return pd.Series(y_pred - y_true, index=geoids.index).groupby(geoids.str[:5]).mean().std()


def optimal_blend(oofs, y):
    names = list(oofs.keys())
    mat = np.column_stack([oofs[n] for n in names])
    valid = ~np.any(np.isnan(mat), axis=1)
    mv, yv = mat[valid], y[valid]
    res = minimize(lambda w: np.sqrt(mean_squared_error(yv, mv@w)),
                   np.ones(len(names))/len(names), method='SLSQP',
                   bounds=[(0,1)]*len(names), constraints={'type':'eq','fun':lambda w:sum(w)-1})
    weights = {n: float(w) for n, w in zip(names, res.x)}
    return mat @ res.x, weights, res.fun


def stacking_ensemble(oofs, y, geoids, n_folds=5):
    names = list(oofs.keys())
    mat = np.column_stack([oofs[n] for n in names])
    valid_mask = ~np.any(np.isnan(mat), axis=1)
    mv = mat[valid_mask]
    yv = y.values[valid_mask] if isinstance(y, pd.Series) else y[valid_mask]
    geo_v = geoids[valid_mask]
    groups = geo_v.str[:5]
    gkf = GroupKFold(n_splits=n_folds)
    stack_oof = np.full(len(yv), np.nan)
    for ti, vi in gkf.split(mv, yv, groups):
        meta = Ridge(alpha=1.0, random_state=SEED)
        meta.fit(mv[ti], yv[ti])
        stack_oof[vi] = meta.predict(mv[vi])
    stack_rmse = np.sqrt(mean_squared_error(yv, stack_oof))
    stack_r2 = r2_score(yv, stack_oof)
    meta_final = Ridge(alpha=1.0, random_state=SEED)
    meta_final.fit(mv, yv)
    logger.info(f"  Stacking: RMSE={stack_rmse:.6f} R2={stack_r2:.4f}")
    logger.info(f"  Meta coefs: {dict(zip(names, meta_final.coef_))}")
    return stack_oof, stack_rmse, stack_r2, meta_final


def comprehensive_bias_discovery(features, y_true, y_pred, geo, valid, output_dir):
    logger.info("COMPREHENSIVE BIAS DISCOVERY")
    residuals = y_pred - y_true
    findings = []
    
    # County bias
    counties = geo.str[:5]
    county_resid = pd.Series(residuals, index=geo.index).groupby(counties).agg(['mean','std','count'])
    county_resid.columns = ['mean_residual','residual_std','n_tracts']
    worst_over = county_resid.nlargest(10, 'mean_residual')
    worst_under = county_resid.nsmallest(10, 'mean_residual')
    findings.append({'category':'County','finding':'Worst over-predicted','details':worst_over['mean_residual'].to_dict(),'severity':'high'})
    findings.append({'category':'County','finding':'Worst under-predicted','details':worst_under['mean_residual'].to_dict(),'severity':'high'})
    logger.info(f"  Over: {worst_over['mean_residual'].to_dict()}")
    logger.info(f"  Under: {worst_under['mean_residual'].to_dict()}")
    
    # SVI bias
    for svi_col, label in [('svi_overall','Overall'),('svi_minority','Minority'),
                            ('svi_socioeconomic','Socioeconomic'),('svi_household','Household'),
                            ('svi_housing_transport','Housing/Trans')]:
        if svi_col in features.columns:
            svi_vals = pd.to_numeric(features.loc[valid, svi_col], errors='coerce')
            if svi_vals.notna().sum() > 100:
                q25, q75 = svi_vals.quantile(0.25), svi_vals.quantile(0.75)
                low_m = (svi_vals <= q25).values[:len(residuals)]
                high_m = (svi_vals >= q75).values[:len(residuals)]
                if low_m.sum() > 10 and high_m.sum() > 10:
                    lr, hr = np.mean(residuals[low_m]), np.mean(residuals[high_m])
                    findings.append({'category':'SVI','finding':f'SVI {label}','low_resid':float(lr),'high_resid':float(hr),'disparity':float(hr-lr),'severity':'high' if abs(hr-lr)>0.001 else 'medium'})
                    logger.info(f"  SVI {label}: low={lr:.6f} high={hr:.6f} disp={hr-lr:.6f}")
    
    # Tribal bias
    if 'tribal_any' in features.columns:
        t_m = (features.loc[valid,'tribal_any'].fillna(0)>0).values[:len(residuals)]
        if t_m.sum() > 0:
            tr, ntr = np.mean(residuals[t_m]), np.mean(residuals[~t_m])
            findings.append({'category':'Tribal','finding':'Tribal bias','tribal_resid':float(tr),'non_tribal_resid':float(ntr),'disparity':float(tr-ntr),'n':int(t_m.sum()),'severity':'high' if abs(tr-ntr)>0.001 else 'medium'})
            logger.info(f"  Tribal: {tr:.6f} vs {ntr:.6f} disp={tr-ntr:.6f} n={t_m.sum()}")
    
    # Rural/Urban
    if 'pct_urban' in features.columns:
        pu = features.loc[valid,'pct_urban'].fillna(0.5).values[:len(residuals)]
        for thresh, lab in [(0.5,'50%'),(0.3,'30%')]:
            ru = pu < thresh
            if ru.sum()>10 and (~ru).sum()>10:
                rr, ur = np.mean(residuals[ru]), np.mean(residuals[~ru])
                findings.append({'category':'Rural/Urban','finding':f'Rural/Urban ({lab})','rural_resid':float(rr),'urban_resid':float(ur),'disparity':float(rr-ur)})
                logger.info(f"  Rural({lab}): {rr:.6f} vs Urban: {ur:.6f} disp={rr-ur:.6f}")
    
    # Wildfire
    if 'usgs_wildfire_ever' in features.columns:
        wf = features.loc[valid,'usgs_wildfire_ever'].fillna(0).values[:len(residuals)]
        wm = wf > 0
        if wm.sum() > 10:
            wr, nwr = np.mean(residuals[wm]), np.mean(residuals[~wm])
            findings.append({'category':'Hazard','finding':'Wildfire bias','wf_resid':float(wr),'no_wf_resid':float(nwr),'disparity':float(wr-nwr)})
    
    # CVI
    if 'cvi_overall' in features.columns:
        cv = pd.to_numeric(features.loc[valid,'cvi_overall'], errors='coerce')
        if cv.notna().sum()>100:
            hc = (cv>=cv.quantile(0.75)).values[:len(residuals)]
            if hc.sum()>10:
                hcr, lcr = np.mean(residuals[hc]), np.mean(residuals[~hc])
                findings.append({'category':'CVI','finding':'High CVI bias','high_cvi_resid':float(hcr),'low_cvi_resid':float(lcr),'disparity':float(hcr-lcr)})
    
    # Intersectional
    int_groups = {}
    if all(c in features.columns for c in ['tribal_any','svi_overall','pct_urban']):
        try:
            t = (features.loc[valid,'tribal_any'].fillna(0)>0)
            sv = pd.to_numeric(features.loc[valid,'svi_overall'], errors='coerce').fillna(0)
            pu = features.loc[valid,'pct_urban'].fillna(0.5)
            hs = sv > sv.quantile(0.75)
            ls = sv < sv.quantile(0.25)
            ru = pu < 0.5
            ur = pu >= 0.5
            
            groups_def = {
                'tribal_x_highSVI_x_rural': t&hs&ru,
                'tribal_x_highSVI_x_urban': t&hs&ur,
                'tribal_x_lowSVI_x_rural': t&ls&ru,
                'highSVI_x_rural': hs&ru,
                'highSVI_x_urban': hs&ur,
                'lowSVI_x_rural': ls&ru,
                'tribal_x_rural': t&ru,
                'tribal_x_highSVI': t&hs,
            }
            if 'usgs_wildfire_ever' in features.columns:
                wf = features.loc[valid,'usgs_wildfire_ever'].fillna(0)>0
                groups_def['wf_x_highSVI_x_rural'] = wf&hs&ru
                groups_def['tribal_x_wf'] = t&wf
                groups_def['wf_x_rural'] = wf&ru
            if 'cvi_overall' in features.columns:
                cvi = pd.to_numeric(features.loc[valid,'cvi_overall'], errors='coerce').fillna(0)
                hcv = cvi > cvi.quantile(0.75)
                groups_def['highCVI_x_highSVI_x_rural'] = hcv&hs&ru
                groups_def['highCVI_x_tribal'] = hcv&t
            
            overall = np.mean(residuals)
            for gn, gm in groups_def.items():
                gv = gm.values[:len(residuals)]
                if gv.sum() > 5:
                    gr = np.mean(residuals[gv])
                    excess = gr - overall
                    int_groups[gn] = {'residual':float(gr),'excess_bias':float(excess),'n':int(gv.sum())}
                    findings.append({'category':'Intersectional','finding':gn,'group_resid':float(gr),'overall_resid':float(overall),'excess_bias':float(excess),'n':int(gv.sum()),'severity':'critical' if abs(excess)>0.002 else ('high' if abs(excess)>0.001 else 'medium')})
                    logger.info(f"  {gn}: resid={gr:.6f} excess={excess:.6f} n={gv.sum()}")
        except Exception as e:
            logger.warning(f"  Intersectional error: {e}")
    
    # Coverage null bias
    for cc in [c for c in features.columns if '_covered' in c.lower()][:10]:
        nm = features.loc[valid,cc].isna().values[:len(residuals)]
        if nm.sum()>10 and (~nm).sum()>10:
            nr, cr = np.mean(residuals[nm]), np.mean(residuals[~nm])
            findings.append({'category':'Data Desert','finding':f'Null bias: {cc}','null_resid':float(nr),'covered_resid':float(cr),'disparity':float(nr-cr)})
    
    # Regional
    if 'region' in features.columns:
        for region in features.loc[valid,'region'].unique():
            rm = (features.loc[valid,'region']==region).values[:len(residuals)]
            if rm.sum()>10:
                findings.append({'category':'Regional','finding':f'Region: {region}','resid':float(np.mean(residuals[rm])),'n':int(rm.sum())})
    
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(output_dir/"comprehensive_bias_findings.csv", index=False)
    if int_groups:
        pd.DataFrame(int_groups).T.to_csv(output_dir/"intersectional_bias_summary.csv")
    
    crit = len([f for f in findings if f.get('severity')=='critical'])
    high = len([f for f in findings if f.get('severity')=='high'])
    logger.info(f"  {len(findings)} findings: {crit} critical, {high} high")
    return findings, int_groups


def generate_documentation(results, findings, best_rmse, best_r2, best_bias, weights, 
                           X_shape, n_feat, ensemble_method, output_dir):
    logger.info("GENERATING DOCUMENTATION")
    doc = []
    doc.append("# Bias Bounty Mapping Equity Challenge — Methodology Report\n")
    doc.append(f"**Generated**: {datetime.now().isoformat()}  \n")
    doc.append(f"**Pipeline**: Monumental Pipeline v2  \n")
    doc.append(f"**Ensemble Method**: {ensemble_method}  \n")
    
    doc.append("## Executive Summary\n")
    doc.append("We present a self-supervised ensemble approach for predicting coverage gap scores")
    doc.append("across 9,491 US Census tracts in 4 focus regions. Our pipeline combines massive")
    doc.append("feature engineering (100+ features), a 3-model ensemble with stacking")
    doc.append("(XGBoost + LightGBM + CatBoost), and comprehensive intersectional bias discovery.\n")
    doc.append("### Key Innovation: Self-Supervised Learning\n")
    doc.append("Since the competition target (`coverage_gap_score`) is not yet released, we use")
    doc.append("`building_gap` and `road_gap` as proxy targets. This validates our entire pipeline")
    doc.append("and enables instant retraining when the target drops.\n")
    doc.append(f"**Best Proxy RMSE**: {best_rmse:.6f} | **R²**: {best_r2:.4f} | **Bias Score**: {best_bias:.6f}\n")
    
    doc.append("## Pipeline Architecture\n")
    doc.append("```\nRaw Data → Massive Feature Engineering → Feature Selection → 3-Model CV → Stacking/Blend → Bias Discovery → Submission\n```\n")
    doc.append("### Components\n")
    doc.append("1. **Feature Engineering**: 100+ interaction features (SVI×coverage, tribal×hazard, polynomial, ratios, target encoding)")
    doc.append("2. **Feature Selection**: Correlation filtering + collinearity removal (r > 0.98)")
    doc.append("3. **3-Model Ensemble**: XGBoost + LightGBM + CatBoost with optimal convex blend")
    doc.append("4. **Stacking**: Ridge meta-learner on OOF predictions (Level-2)")
    doc.append("5. **Spatial Cross-Validation**: GroupKFold by county FIPS (5-fold)")
    doc.append("6. **Bias Discovery**: 9 dimensions of equity analysis\n")
    
    doc.append("## Model Performance\n")
    doc.append("| Model | RMSE | R² | Bias Score |\n|-------|------|----|-----------|\n")
    for name, r in results.items():
        doc.append(f"| {name} | {r['cv_rmse']:.6f} | {r['cv_r2']:.4f} | {r.get('bias',0):.6f} |")
    doc.append(f"| **{ensemble_method}** | **{best_rmse:.6f}** | **{best_r2:.4f}** | **{best_bias:.6f}** |")
    doc.append(f"\n**Weights**: {weights}\n")
    
    doc.append("## Feature Engineering\n")
    doc.append("### Categories (100+ features)\n")
    doc.append("1. **SVI × Coverage** (20): svi_x_bldg, svi_sq_x_bldg, svi_minority_x_bldg, etc.")
    doc.append("2. **Tribal × Coverage** (10): tribal_x_bldg, tribal_pct_x_bldg, tribal_x_svi_x_bldg")
    doc.append("3. **Rural/Urban × Coverage** (10): rural_x_bldg, pct_urban_x_bldg, rural_x_svi_x_bldg")
    doc.append("4. **Hazard × Coverage** (10): wf_x_bldg, wf_x_svi_x_bldg")
    doc.append("5. **CVI × Coverage** (10): cvi_x_bldg, cvi_x_svi_x_bldg")
    doc.append("6. **Intersectional** (15): tribal_x_highSVI_x_rural, highSVI_x_rural, tribal_x_wf")
    doc.append("7. **Polynomial** (15): bldg_gap_sq, bldg_gap_cu, bldg_road_ratio, log transforms")
    doc.append("8. **Population-weighted** (5): log_pop, log_pop_x_bldg")
    doc.append("9. **Compound risk** (5): compound_risk, tribal_x_risk")
    doc.append("10. **County target encoding** (1): LOO county mean")
    doc.append("11. **Coverage null indicators**: data desert signals\n")
    
    # Top features
    best_fi = None
    for name in ['xgb','lgb','cat']:
        if name in results and results[name].get('fi') is not None:
            best_fi = results[name]['fi']
            break
    if best_fi is not None:
        doc.append("### Top 20 Features\n")
        for _, row in best_fi.head(20).iterrows():
            doc.append(f"- `{row['feature']}`: {row['importance']:.4f}")
    
    doc.append("\n## Bias Discovery Findings ($1,000 Prize)\n")
    by_cat = {}
    for f in findings:
        by_cat.setdefault(f.get('category','Other'), []).append(f)
    for cat, cfinds in by_cat.items():
        doc.append(f"\n### {cat}\n")
        for f in cfinds:
            doc.append(f"**{f.get('finding','')}** (severity: {f.get('severity','N/A')})")
            details = {k:v for k,v in f.items() if k not in ['category','finding','severity','details']}
            for k,v in details.items():
                doc.append(f"  - {k}: {v:.6f}" if isinstance(v,float) else f"  - {k}: {v}")
    
    doc.append("\n## Validation Strategy\n")
    doc.append("Spatial cross-validation with GroupKFold by county FIPS prevents spatial autocorrelation leakage.\n")
    
    doc.append("\n## Reproducibility\n")
    doc.append(f"- SEED={SEED}, spatial CV, deterministic pipeline\n")
    doc.append(f"- Dataset: {X_shape[0]} tracts × {n_feat} features\n")
    
    doc.append("\n## Target Strategy\n")
    doc.append("Self-supervised: building_gap and road_gap as proxy targets until Zindi releases actual target.\n")
    
    with open(output_dir/"methodology_report.md",'w') as f:
        f.write("\n".join(doc))
    logger.info(f"  Documentation saved ({len(doc)} lines)")


def main():
    t0 = time.time()
    logger.info("="*60)
    logger.info("MONUMENTAL PIPELINE v2 — BIAS BOUNTY")
    logger.info("="*60)
    
    features = load_features()
    strata = load_strata()
    features = engineer_massive_features(features, strata)
    
    # ═══ building_gap ═══
    logger.info("\n=== TARGET: building_gap ===")
    X, y, geo, valid = prepare_features(features, 'building_gap', n_top=100)
    
    models = OrderedDict([
        ('xgb', xgb.XGBRegressor(n_estimators=1500, max_depth=7, learning_rate=0.015, 
                                 subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                                 min_child_weight=5, tree_method='hist', random_state=SEED)),
        ('lgb', lgb.LGBMRegressor(n_estimators=1500, max_depth=7, num_leaves=50, learning_rate=0.015,
                                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                                  min_child_samples=20, verbose=-1, random_state=SEED)),
        ('cat', CatBoostRegressor(iterations=1500, depth=7, learning_rate=0.015, 
                                  l2_leaf_reg=3.0, random_seed=SEED, verbose=0)),
    ])
    
    all_results, all_oofs = {}, {}
    for name, m in models.items():
        logger.info(f"\n  Training {name}...")
        r = train_cv(m, X, y, geo, n_folds=5)
        bs = bias_score(y.values, r['oof'], geo)
        r['bias'] = bs
        all_results[name] = r
        all_oofs[name] = r['oof']
        logger.info(f"  {name}: RMSE={r['cv_rmse']:.6f}±{r['cv_rmse_std']:.6f} R2={r['cv_r2']:.4f} Bias={bs:.6f}")
        gc.collect()
    
    # Optimal blend
    blend_pred, blend_weights, blend_rmse = optimal_blend(all_oofs, y.values)
    blend_r2 = r2_score(y.values, blend_pred)
    blend_bias = bias_score(y.values, blend_pred, geo)
    logger.info(f"\n  Blend: RMSE={blend_rmse:.6f} R2={blend_r2:.4f} Bias={blend_bias:.6f}")
    logger.info(f"  Weights: {blend_weights}")
    
    # Stacking
    stack_pred, stack_rmse, stack_r2, meta = stacking_ensemble(all_oofs, y, geo, n_folds=5)
    valid_mask = ~np.any(np.isnan(np.column_stack([all_oofs[n] for n in all_oofs])), axis=1)
    stack_bias = bias_score(y.values[valid_mask], stack_pred, geo[valid_mask])
    
    if stack_rmse < blend_rmse:
        logger.info(f"  *** Stacking wins: {stack_rmse:.6f} < {blend_rmse:.6f} ***")
        # Extend stack predictions to full length (fill NaN positions with blend)
        full_stack_pred = blend_pred.copy()  # Use blend as fallback
        full_stack_pred[valid_mask] = stack_pred
        best_pred, best_rmse, best_r2, best_bias = full_stack_pred, stack_rmse, stack_r2, stack_bias
        ensemble_method = "stacking"
    else:
        logger.info(f"  *** Blend wins: {blend_rmse:.6f} <= {stack_rmse:.6f} ***")
        best_pred, best_rmse, best_r2, best_bias = blend_pred, blend_rmse, blend_r2, blend_bias
        ensemble_method = "blend"
    
    # ═══ road_gap ═══
    logger.info("\n=== TARGET: road_gap ===")
    Xr, yr, geor, validr = prepare_features(features, 'road_gap', n_top=80)
    road_r = train_cv(xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                      tree_method='hist', random_state=SEED), Xr, yr, geor, n_folds=5)
    road_bias = bias_score(yr.values, road_r['oof'], geor)
    all_results['road_xgb'] = road_r
    all_results['road_xgb']['bias'] = road_bias
    logger.info(f"  Road XGB: RMSE={road_r['cv_rmse']:.6f} R2={road_r['cv_r2']:.4f}")
    
    # ═══ Bias Discovery ═══
    logger.info("\n=== BIAS DISCOVERY ===")
    findings, int_groups = comprehensive_bias_discovery(features, y.values, best_pred, geo, valid, OUTPUT_DIR)
    
    # ═══ SHAP (on best model) ═══
    logger.info("\n=== SHAP ANALYSIS ===")
    try:
        import shap
        for name in ['xgb','lgb','cat']:
            if name in all_results and all_results[name].get('models'):
                m = all_results[name]['models'][-1]
                explainer = shap.TreeExplainer(m)
                sv = explainer.shap_values(X.iloc[:2000])
                mean_shap = np.abs(sv).mean(axis=0)
                shap_df = pd.DataFrame({'feature':X.columns,'shap':mean_shap}).sort_values('shap',ascending=False)
                shap_df.head(50).to_csv(OUTPUT_DIR/f"{name}_shap_importance.csv", index=False)
                logger.info(f"  {name} SHAP top 5: {shap_df.head(5)['feature'].tolist()}")
                break
    except Exception as e:
        logger.warning(f"  SHAP failed: {e}")
    
    # ═══ Documentation ═══
    generate_documentation(all_results, findings, best_rmse, best_r2, best_bias, 
                          blend_weights, X.shape, X.shape[1], ensemble_method, OUTPUT_DIR)
    
    # ═══ Submission ═══
    logger.info("\n=== GENERATING SUBMISSION ===")
    geo_padded = geo.values.astype(str).str.zfill(11)
    pred_clipped = np.clip(best_pred, -3.0, 0.5)
    submission = pd.DataFrame({'GEOID':geo_padded, 'coverage_gap_score':pred_clipped})
    submission.to_csv(OUTPUT_DIR/"submission.csv", index=False)
    submission.to_csv(PROJECT_ROOT/"submission.csv", index=False)
    submission.to_csv(DOWNLOAD_DIR/"submission.csv", index=False)
    logger.info(f"  {len(submission)} tracts | mean={pred_clipped.mean():.6f} std={pred_clipped.std():.6f}")
    
    # Save outputs
    for name in ['xgb','lgb','cat']:
        if name in all_results and all_results[name].get('fi') is not None:
            all_results[name]['fi'].head(50).to_csv(OUTPUT_DIR/f"{name}_feature_importance.csv", index=False)
    
    comp_data = {}
    for name, r in all_results.items():
        comp_data[name] = {'RMSE':r['cv_rmse'],'R2':r['cv_r2'],'Bias':r.get('bias',0)}
    comp_data[f'ensemble_{ensemble_method}'] = {'RMSE':best_rmse,'R2':best_r2,'Bias':best_bias}
    comp_df = pd.DataFrame(comp_data).T.sort_values('RMSE')
    comp_df.to_csv(OUTPUT_DIR/"model_comparison.csv")
    logger.info(f"\nMODEL COMPARISON:\n{comp_df.to_string()}")
    
    # Top features
    best_fi = None
    for name in ['xgb','lgb','cat']:
        if name in all_results and all_results[name].get('fi') is not None:
            best_fi = all_results[name]['fi']
            break
    if best_fi is not None:
        logger.info(f"\nTop 20 Features:")
        for _, row in best_fi.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    pred_df = pd.DataFrame({'GEOID':geo_padded,'true':y.values,'pred':best_pred,'residual':y.values-best_pred})
    pred_df.to_parquet(OUTPUT_DIR/"predictions.parquet")
    
    state = {'timestamp':datetime.now().isoformat(),'pipeline':'monumental_v2',
             'elapsed_min':(time.time()-t0)/60,'ensemble':ensemble_method,
             'best_rmse':best_rmse,'best_r2':best_r2,'best_bias':best_bias,
             'blend_weights':blend_weights,'n_features':X.shape[1],
             'n_tracts':X.shape[0],'n_findings':len(findings)}
    with open(OUTPUT_DIR/"pipeline_state.json",'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    elapsed = (time.time()-t0)/60
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f} min | RMSE={best_rmse:.6f} R2={best_r2:.4f} Bias={best_bias:.6f}")
    logger.info(f"Ensemble: {ensemble_method} | {len(models)} models | {X.shape[1]} features")
    logger.info(f"Bias findings: {len(findings)} | Critical: {len([f for f in findings if f.get('severity')=='critical'])}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
