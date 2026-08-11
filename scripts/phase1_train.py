"""
Phase 1: Feature Engineering + Train all models + Save intermediate results
"""
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb, time, gc, json, logging, warnings
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent  # project root from scripts/
OUT = ROOT / "data/output"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

def load():
    for p in [ROOT/"data/features/all_regions_enhanced_features.parquet",
              ROOT/"kaggle_dataset/all_regions_enhanced_features.parquet"]:
        if p.exists(): return pd.read_parquet(p)
    raise FileNotFoundError()

def load_strata():
    for p in [ROOT/"kaggle_dataset/national-strata-tract-table.parquet"]:
        if p.exists(): return pd.read_parquet(p)
    return None

def engineer_features(features, strata):
    log.info("Engineering features...")
    if strata is not None:
        want = ['GEOID','svi_overall','svi_socioeconomic','svi_household','svi_minority',
                'svi_housing_transport','svi_pop','tribal_any','tribal_pct','tribal_legal',
                'pct_urban','pop_rural','pop_urban','pop_total',
                'usgs_wildfire_ever','usgs_wildfire_burned_pct_area',
                'cvi_overall','cvi_baseline','cvi_climate']
        want += [c for c in strata.columns if '_covered' in c.lower()]
        want += [c for c in strata.columns if any(h in c.lower() for h in ['wildfire','flood','drought','usdm']) and 'covered' not in c.lower()][:10]
        new_cols = [c for c in want if c in strata.columns and c not in features.columns]
        new_cols = ['GEOID'] + [c for c in new_cols if c != 'GEOID']
        if len(new_cols) > 1:
            sub = strata[new_cols].copy()
            sub['GEOID'] = sub['GEOID'].astype(str)
            features['GEOID'] = features['GEOID'].astype(str)
            features = features.merge(sub, on='GEOID', how='left')
    
    for c in ['svi_overall','svi_socioeconomic','svi_household','svi_minority',
              'svi_housing_transport','svi_pop','tribal_pct','pct_urban',
              'cvi_overall','cvi_baseline','cvi_climate','usgs_wildfire_burned_pct_area']:
        if c in features.columns: features[c] = pd.to_numeric(features[c], errors='coerce')
    
    nf = {}
    F = lambda s, v=0: s.fillna(v) if s is not None else None
    bg, rg = features.get('building_gap'), features.get('road_gap')
    svi = features.get('svi_overall')
    svi_m = features.get('svi_minority')
    svi_s = features.get('svi_socioeconomic')
    tribal = features.get('tribal_any')
    tp = features.get('tribal_pct')
    pu = features.get('pct_urban')
    wf = features.get('usgs_wildfire_ever')
    cvi = features.get('cvi_overall')
    cvi_b = features.get('cvi_baseline')
    pop = features.get('pop_total')
    
    if bg is not None:
        bgv = F(bg).values
        # SVI interactions
        if svi is not None:
            sv = F(svi).values
            nf['svi_x_bldg'] = sv * bgv
            nf['svi_sq_x_bldg'] = sv**2 * bgv
            nf['svi_abs_x_bldg_abs'] = np.abs(sv) * np.abs(bgv)
        if svi_m is not None: nf['svi_min_x_bldg'] = F(svi_m).values * bgv
        if svi_s is not None: nf['svi_soc_x_bldg'] = F(svi_s).values * bgv
        # Tribal
        if tribal is not None:
            tf = (F(tribal).values > 0).astype(float)
            nf['tribal_x_bldg'] = tf * bgv
            nf['tribal_x_bldg_sq'] = tf * bgv**2
            if tp is not None: nf['tribal_pct_x_bldg'] = F(tp, 0).values * bgv
            if svi is not None: nf['tribal_x_svi_x_bldg'] = tf * F(svi).values * bgv
        # Rural
        if pu is not None:
            puv = F(pu, 0.5).values
            rural = (1 - puv).clip(0, 1)
            nf['pct_urban_x_bldg'] = puv * bgv
            nf['rural_x_bldg'] = rural * bgv
            nf['rural_sq_x_bldg'] = rural**2 * bgv
            if svi is not None: nf['rural_x_svi_x_bldg'] = rural * F(svi).values * bgv
        # Hazard
        if wf is not None:
            nf['wf_x_bldg'] = F(wf).values * bgv
            nf['wf_flag_x_bldg'] = (F(wf).values > 0).astype(float) * bgv
            if svi is not None: nf['wf_x_svi_x_bldg'] = F(wf).values * F(svi).values * bgv
        # CVI
        if cvi is not None:
            nf['cvi_x_bldg'] = F(cvi).values * bgv
            nf['cvi_sq_x_bldg'] = F(cvi).values**2 * bgv
            if svi is not None: nf['cvi_x_svi_x_bldg'] = F(cvi).values * F(svi).values * bgv
        if cvi_b is not None: nf['cvi_base_x_bldg'] = F(cvi_b).values * bgv
        # Polynomial
        nf['bldg_gap_sq'] = bgv**2
        nf['bldg_gap_cu'] = bgv**3
        nf['bldg_gap_abs'] = np.abs(bgv)
        nf['bldg_gap_log1p'] = np.log1p(np.abs(bgv))
        if rg is not None:
            rgv = F(rg).values
            nf['road_gap_sq'] = rgv**2
            nf['bldg_road_ratio'] = bgv / (np.abs(rgv) + 1e-8)
            nf['bldg_road_diff'] = bgv - rgv
            nf['bldg_road_product'] = bgv * rgv
        # Compound risk
        compound = np.abs(bgv)
        if rg is not None: compound += np.abs(F(rg).values)
        if svi is not None: compound += np.clip(F(svi).values, 0, None) * 0.1
        nf['compound_risk'] = compound
        nf['compound_risk_sq'] = compound**2
        if tribal is not None: nf['tribal_x_risk'] = (F(tribal).values > 0).astype(float) * compound
        # Pop weighted
        if pop is not None:
            lp = np.log1p(F(pop).values)
            nf['log_pop_x_bldg'] = lp * bgv
            if svi is not None: nf['log_pop_x_svi'] = lp * F(svi).values
    
    if rg is not None and svi is not None:
        nf['svi_x_road'] = F(svi).values * F(rg).values
    
    # Intersectional
    if tribal is not None and svi is not None and pu is not None:
        t = (F(tribal).values > 0).astype(float)
        sv = F(svi).values
        puv = F(pu, 0.5).values
        hs = (sv > np.nanquantile(sv, 0.75)).astype(float)
        ls = (sv < np.nanquantile(sv, 0.25)).astype(float)
        rural = (puv < 0.5).astype(float)
        urban = 1 - rural
        nf['tribal_x_highsvi_x_rural'] = t * hs * rural
        nf['tribal_x_highsvi_x_urban'] = t * hs * urban
        nf['tribal_x_lowsvi_x_rural'] = t * ls * rural
        nf['highsvi_x_rural'] = hs * rural
        nf['highsvi_x_urban'] = hs * urban
        nf['lowsvi_x_rural'] = ls * rural
        nf['tribal_x_highsvi'] = t * hs
        nf['tribal_x_rural'] = t * rural
        if bg is not None:
            nf['tribal_hsvi_rural_x_bldg'] = t * hs * rural * F(bg).values
            nf['hsvi_rural_x_bldg'] = hs * rural * F(bg).values
        if wf is not None:
            nf['wf_x_rural_x_hsvi'] = F(wf).values * rural * hs
            nf['wf_x_tribal'] = F(wf).values * t
        if cvi is not None:
            hc = (F(cvi).values > np.nanquantile(F(cvi).values, 0.75)).astype(float)
            nf['hcvi_x_hsvi_x_rural'] = hc * hs * rural
            nf['hcvi_x_tribal'] = hc * t
    
    # Coverage null
    for cc in [c for c in features.columns if '_covered' in c.lower()]:
        nf[f'{cc}_null'] = features[cc].isna().astype(float).values
    null_cols = [k for k in nf if k.endswith('_null')]
    if null_cols:
        nf['total_nulls'] = np.sum([nf[k] for k in null_cols], axis=0)
    
    # County LOO
    if 'GEOID' in features.columns and bg is not None:
        county = features['GEOID'].astype(str).str[:5]
        bgv = F(bg)
        cs = bgv.groupby(county).agg(['mean','count'])
        cs.columns = ['mean','count']
        cm, cc = cs['mean'][county].values, cs['count'][county].values
        nf['bldg_county_loo'] = (cm * cc - bgv.values) / (cc - 1 + 1e-8)
    
    if nf:
        ndf = pd.DataFrame(nf, index=features.index).replace([np.inf, -np.inf], np.nan)
        features = pd.concat([features, ndf], axis=1).loc[:, ~pd.concat([features, ndf], axis=1).columns.duplicated()]
        log.info(f"  Added {len(nf)} features → {features.shape[1]} total")
    return features

def prepare(df, target, n_top=80):
    drop = ['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
            'building_gap','road_gap','building_ratio','road_ratio',
            'building_count_ratio','building_count_gap','road_count_ratio','road_count_gap',
            'road_length_ratio','road_length_gap','poi_facility_gap','poi_to_facility_ratio']
    df = df.loc[:, ~df.columns.duplicated()]
    fc = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[fc].fillna(-999).replace([np.inf,-np.inf],-999)
    y = df[target]; geo = df['GEOID'].astype(str)
    v = y.notna(); X,y,geo = X[v],y[v],geo[v]
    s = X.std(); X = X[s[s>1e-10].index]
    np_ = X.isna().mean(); X = X[np_[np_<0.95].index]
    c = X.corrwith(y).abs().sort_values(ascending=False)
    X = X[c.head(n_top).index.tolist()]
    # Collinearity
    cm = X.corr().abs()
    u = cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool))
    td = [c for c in u.columns if any(u[c]>0.98)]
    X = X.drop(columns=td)
    log.info(f"  {X.shape[1]} features, {X.shape[0]} tracts")
    return X, y, geo, v

def train_cv(model, X, y, geo, nf=5):
    gkf = GroupKFold(n_splits=nf)
    oof = np.full(len(y), np.nan)
    scores, imps = [], []
    for fi,(ti,vi) in enumerate(gkf.split(X,y,geo.str[:5])):
        m = type(model)(**model.get_params())
        try:
            if isinstance(m, xgb.XGBRegressor):
                m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],verbose=False)
            elif isinstance(m, lgb.LGBMRegressor):
                m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],callbacks=[lgb.early_stopping(50,verbose=False)])
            elif isinstance(m, CatBoostRegressor):
                m.fit(X.iloc[ti],y.iloc[ti],eval_set=(X.iloc[vi],y.iloc[vi]),early_stopping_rounds=50,verbose=0)
            else: m.fit(X.iloc[ti],y.iloc[ti])
        except Exception as e:
            log.warning(f"Fold {fi} error: {e}"); continue
        p = m.predict(X.iloc[vi]); oof[vi] = p
        rmse = np.sqrt(mean_squared_error(y.iloc[vi],p))
        r2 = r2_score(y.iloc[vi],p)
        scores.append({'rmse':rmse,'r2':r2})
        if hasattr(m,'feature_importances_'): imps.append(m.feature_importances_)
        log.info(f"    Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
    fi_df = None
    if imps:
        fi_df = pd.DataFrame({'feature':X.columns,'importance':np.mean(imps,axis=0)}).sort_values('importance',ascending=False)
    return {'cv_rmse':np.mean([s['rmse'] for s in scores]),'cv_r2':np.mean([s['r2'] for s in scores]),
            'cv_rmse_std':np.std([s['rmse'] for s in scores]),'oof':oof,'fi':fi_df}

def bias_score(yt,yp,geo):
    return pd.Series(yp-yt,index=geo.index).groupby(geo.str[:5]).mean().std()

def blend(oofs,y):
    names = list(oofs.keys())
    mat = np.column_stack([oofs[n] for n in names])
    v = ~np.any(np.isnan(mat),axis=1); mv,yv = mat[v],y[v]
    r = minimize(lambda w: np.sqrt(mean_squared_error(yv,mv@w)),np.ones(len(names))/len(names),method='SLSQP',bounds=[(0,1)]*len(names),constraints={'type':'eq','fun':lambda w:sum(w)-1})
    return mat@r.x, {n:float(w) for n,w in zip(names,r.x)}, r.fun

def main():
    t0 = time.time()
    log.info("PHASE 1: Feature Engineering + Model Training")
    
    features = load()
    strata = load_strata()
    features = engineer_features(features, strata)
    
    # building_gap
    log.info("\n=== building_gap ===")
    X, y, geo, valid = prepare(features, 'building_gap', n_top=80)
    
    models = OrderedDict([
        ('xgb', xgb.XGBRegressor(n_estimators=1500, max_depth=7, learning_rate=0.015, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, tree_method='hist', random_state=SEED)),
        ('lgb', lgb.LGBMRegressor(n_estimators=1500, max_depth=7, num_leaves=50, learning_rate=0.015, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20, verbose=-1, random_state=SEED)),
    ])
    
    res, oofs = {}, {}
    for name, m in models.items():
        log.info(f"\n  Training {name}...")
        r = train_cv(m, X, y, geo, nf=5)
        bs = bias_score(y.values, r['oof'], geo)
        r['bias'] = bs
        res[name] = r; oofs[name] = r['oof']
        log.info(f"  {name}: RMSE={r['cv_rmse']:.6f}±{r['cv_rmse_std']:.6f} R2={r['cv_r2']:.4f} Bias={bs:.6f}")
        gc.collect()
    
    # CatBoost (fewer iters for speed)
    log.info(f"\n  Training cat...")
    cat_m = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0, random_seed=SEED, verbose=0)
    r = train_cv(cat_m, X, y, geo, nf=5)
    bs = bias_score(y.values, r['oof'], geo)
    r['bias'] = bs; res['cat'] = r; oofs['cat'] = r['oof']
    log.info(f"  cat: RMSE={r['cv_rmse']:.6f}±{r['cv_rmse_std']:.6f} R2={r['cv_r2']:.4f} Bias={bs:.6f}")
    gc.collect()
    
    # Blend
    bp, bw, br = blend(oofs, y.values)
    br2 = r2_score(y.values, bp)
    bb = bias_score(y.values, bp, geo)
    log.info(f"\n  Blend: RMSE={br:.6f} R2={br2:.4f} Bias={bb:.6f}")
    log.info(f"  Weights: {bw}")
    
    # road_gap
    log.info("\n=== road_gap ===")
    Xr, yr, gr, vr = prepare(features, 'road_gap', n_top=80)
    road_r = train_cv(xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, tree_method='hist', random_state=SEED), Xr, yr, gr, nf=5)
    road_bs = bias_score(yr.values, road_r['oof'], gr)
    res['road_xgb'] = road_r; res['road_xgb']['bias'] = road_bs
    log.info(f"  Road: RMSE={road_r['cv_rmse']:.6f} R2={road_r['cv_r2']:.4f}")
    
    # Save intermediates
    log.info("\nSaving results...")
    # Save OOF predictions
    oof_df = pd.DataFrame(oofs)
    oof_df['y_true'] = y.values
    oof_df['GEOID'] = geo.values
    oof_df.to_parquet(OUT/"oof_predictions.parquet")
    
    # Save blend
    np.save(OUT/"blend_pred.npy", bp)
    
    # Save feature importance
    for name in ['xgb','lgb','cat']:
        if name in res and res[name].get('fi') is not None:
            res[name]['fi'].head(50).to_csv(OUT/f"{name}_feature_importance.csv", index=False)
    
    # Model comparison
    comp = {}
    for name, r in res.items():
        comp[name] = {'RMSE':r['cv_rmse'],'R2':r['cv_r2'],'Bias':r.get('bias',0)}
    comp['blend'] = {'RMSE':br,'R2':br2,'Bias':bb}
    comp_df = pd.DataFrame(comp).T.sort_values('RMSE')
    comp_df.to_csv(OUT/"model_comparison.csv")
    log.info(f"\n{comp_df.to_string()}")
    
    # Submission
    geo_padded = np.array([str(g).zfill(11) for g in geo.values])
    pred_clipped = np.clip(bp, -3.0, 0.5)
    sub = pd.DataFrame({'GEOID':geo_padded,'coverage_gap_score':pred_clipped})
    sub.to_csv(OUT/"submission.csv",index=False)
    sub.to_csv(ROOT/"submission.csv",index=False)
    sub.to_csv((ROOT.parent / "download") / "submission.csv",index=False)  # sibling of project root
    log.info(f"  Submission: {len(sub)} tracts mean={pred_clipped.mean():.6f} std={pred_clipped.std():.6f}")
    
    # Predictions parquet
    pred_df = pd.DataFrame({'GEOID':geo_padded,'true':y.values,'pred':bp,'residual':y.values-bp})
    pred_df.to_parquet(OUT/"predictions.parquet")
    
    # State
    state = {'timestamp':str(datetime.now()),'pipeline':'monumental_v2_phase1',
             'elapsed_min':(time.time()-t0)/60,'best_rmse':br,'best_r2':br2,'best_bias':bb,
             'blend_weights':bw,'n_features':X.shape[1],'n_tracts':X.shape[0]}
    with open(OUT/"pipeline_state.json",'w') as f: json.dump(state,f,indent=2,default=str)
    
    log.info(f"\nDONE in {(time.time()-t0)/60:.1f} min | RMSE={br:.6f} R2={br2:.4f} Bias={bb:.6f}")

if __name__ == "__main__":
    main()
