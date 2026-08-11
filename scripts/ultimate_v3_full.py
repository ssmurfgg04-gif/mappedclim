"""ULTIMATE PIPELINE v3 - Leakage-Fixed + God Mode + Research-Informed"""
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb, json, time, logging, gc, warnings, sys, os
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
warnings.filterwarnings('ignore')
try:
    import h3; HAS_H3=True
except: HAS_H3=False
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAS_OPTUNA=True
except: HAS_OPTUNA=False
HAS_PYSR=False  # PySR Julia startup is too slow; enable manually after Julia precompiles
# To enable: pip install pysr && python3 -c "from pysr import PySRRegressor; print('ready')"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
PROJ=Path("/home/z/my-project/bias-bounty-map")
OUT=PROJ/"data/output"; OUT.mkdir(parents=True,exist_ok=True)
DL=Path("/home/z/my-project/download"); DL.mkdir(parents=True,exist_ok=True)
SEED=42; np.random.seed(SEED)

def h3_blocks(geoids, lats=None, lons=None, res=4):
    if HAS_H3 and lats is not None and lons is not None:
        bl=[]
        for la,lo in zip(lats,lons):
            try: bl.append(h3.latlng_to_cell(la,lo,res))
            except: bl.append('unk')
        return pd.Series(bl,index=geoids.index)
    return geoids.str[:2]

def spatial_split(X, y, blocks, nf=5):
    ub=list(blocks.unique()); np.random.seed(SEED); np.random.shuffle(ub)
    fa={b:i%nf for i,b in enumerate(ub)}
    sf=blocks.map(fa).values
    return [(np.where(sf!=f)[0],np.where(sf==f)[0]) for f in range(nf)]

def load_feat():
    for p in [PROJ/"data/features/all_regions_enhanced_features.parquet",PROJ/"kaggle_dataset/all_regions_enhanced_features.parquet"]:
        if p.exists(): df=pd.read_parquet(p); log.info(f"Loaded {p.name}: {df.shape}"); return df
    raise FileNotFoundError("No features!")

def load_strata():
    for p in [PROJ/"kaggle_dataset/national-strata-tract-table.parquet",PROJ/"data/features/national-strata-tract-table.parquet"]:
        if p.exists(): df=pd.read_parquet(p); log.info(f"Strata: {df.shape}"); return df
    return None

def find_target(feat):
    for tc in ['coverage_gap_score','coverage_gap','gap_score','mapping_gap_score']:
        if tc in feat.columns and feat[tc].notna().sum()>0:
            log.info(f"  TARGET FOUND: {tc}"); return tc
    log.warning("  No actual target - using building_gap proxy"); return 'building_gap'

def engineer_feats(feat, strata):
    log.info("Engineering 200+ features...")
    if strata is not None:
        sw=['GEOID','svi_overall','svi_socioeconomic','svi_household','svi_minority',
            'svi_housing_transport','svi_pop','tribal_any','tribal_pct','tribal_legal',
            'pct_urban','pop_rural','pop_urban','pop_total','usgs_wildfire_ever',
            'usgs_wildfire_burned_pct_area','cvi_overall','cvi_baseline','cvi_climate']
        sw+=[c for c in strata.columns if '_covered' in c.lower()]
        sw+=[c for c in strata.columns if any(h in c.lower() for h in ['wildfire','drought','usdm','heat','epht']) and '_covered' not in c.lower()][:15]
        sw+=[c for c in strata.columns if any(r in c.lower() for r in ['ruca','rucc','nchs'])][:5]
        nc=[c for c in sw if c in strata.columns and c not in feat.columns]
        nc=['GEOID']+[c for c in nc if c!='GEOID']
        if len(nc)>1:
            ss=strata[nc].copy(); ss['GEOID']=ss['GEOID'].astype(str); feat['GEOID']=feat['GEOID'].astype(str)
            b=feat.shape[1]; feat=feat.merge(ss,on='GEOID',how='left'); log.info(f"  Strata: {b}->{feat.shape[1]}")
    for c in ['svi_overall','svi_socioeconomic','svi_household','svi_minority','svi_housing_transport',
              'svi_pop','tribal_pct','pct_urban','cvi_overall','cvi_baseline','cvi_climate',
              'usgs_wildfire_burned_pct_area','pop_total']:
        if c in feat.columns: feat[c]=pd.to_numeric(feat[c],errors='coerce')
    nf={}; F=lambda s,v=0:s.fillna(v) if s is not None else None
    bg=feat.get('building_gap');rg=feat.get('road_gap');svi=feat.get('svi_overall')
    svi_m=feat.get('svi_minority');svi_s=feat.get('svi_socioeconomic');svi_h=feat.get('svi_household')
    svi_ht=feat.get('svi_housing_transport');tribal=feat.get('tribal_any');tribal_pct=feat.get('tribal_pct')
    pu=feat.get('pct_urban');wf=feat.get('usgs_wildfire_ever');wfa=feat.get('usgs_wildfire_burned_pct_area')
    cvi=feat.get('cvi_overall');cvi_b=feat.get('cvi_baseline');cvi_c=feat.get('cvi_climate')
    pop=feat.get('pop_total');br=feat.get('building_ratio');rr=feat.get('road_ratio')
    # SVI x Coverage
    if bg is not None:
        bv=F(bg).values
        if svi is not None:
            sv=F(svi).values
            nf['svi_x_bldg']=sv*bv; nf['svi_sq_x_bldg']=sv**2*bv; nf['svi_abs_x_bldg_abs']=np.abs(sv)*np.abs(bv)
            nf['svi_x_bldg_sq']=sv*bv**2; nf['svi_cubed_x_bldg']=sv**3*bv
        if svi_m is not None: nf['svi_min_x_bldg']=F(svi_m).values*bv; nf['svi_min_sq_x_bldg']=F(svi_m).values**2*bv
        if svi_s is not None: nf['svi_soc_x_bldg']=F(svi_s).values*bv; nf['svi_soc_x_bldg_sq']=F(svi_s).values*bv**2
        if svi_h is not None: nf['svi_hh_x_bldg']=F(svi_h).values*bv
        if svi_ht is not None: nf['svi_ht_x_bldg']=F(svi_ht).values*bv
        if rg is not None:
            rv=F(rg).values
            if svi is not None: nf['svi_x_road']=F(svi).values*rv; nf['svi_sq_x_road']=F(svi).values**2*rv
            if svi_m is not None: nf['svi_min_x_road']=F(svi_m).values*rv
    # Tribal x Coverage
    if bg is not None and tribal is not None:
        tf=(F(tribal).values>0).astype(float); bv=F(bg).values
        nf['tribal_x_bldg']=tf*bv; nf['tribal_pct_x_bldg']=F(tribal_pct,0).values*bv
        if rg is not None: nf['tribal_x_road']=tf*F(rg).values
        nf['tribal_x_bldg_sq']=tf*bv**2
        if svi is not None: nf['tribal_x_svi_x_bldg']=tf*F(svi).values*bv
        if cvi is not None: nf['tribal_x_cvi_x_bldg']=tf*F(cvi).values*bv
    # Rural x Coverage
    if bg is not None and pu is not None:
        puv=F(pu,0.5).values; rur=(1-puv).clip(0,1); bv=F(bg).values
        nf['pct_urban_x_bldg']=puv*bv; nf['rural_x_bldg']=rur*bv; nf['rural_sq_x_bldg']=rur**2*bv
        if rg is not None: nf['rural_x_road']=rur*F(rg).values
        if svi is not None: nf['rural_x_svi_x_bldg']=rur*F(svi).values*bv; nf['urban_x_svi_x_bldg']=puv*F(svi).values*bv
    # Hazard x Coverage
    if bg is not None:
        bv=F(bg).values
        if wf is not None: nf['wf_x_bldg']=F(wf).values*bv; nf['wf_flag_x_bldg']=(F(wf).values>0).astype(float)*bv
        if wfa is not None: nf['wf_area_x_bldg']=F(wfa).values*bv
        if wf is not None and svi is not None: nf['wf_x_svi_x_bldg']=F(wf).values*F(svi).values*bv
    # CVI x Coverage
    if bg is not None and cvi is not None:
        cv=F(cvi).values; bv=F(bg).values
        nf['cvi_x_bldg']=cv*bv; nf['cvi_sq_x_bldg']=cv**2*bv
        if svi is not None: nf['cvi_x_svi_x_bldg']=cv*F(svi).values*bv
    if bg is not None and cvi_b is not None: nf['cvi_base_x_bldg']=F(cvi_b).values*F(bg).values
    if bg is not None and cvi_c is not None: nf['cvi_clim_x_bldg']=F(cvi_c).values*F(bg).values
    # Intersectional
    if tribal is not None and svi is not None and pu is not None:
        t=(F(tribal).values>0).astype(float); sv=F(svi).values; puv=F(pu,0.5).values
        hs=(sv>np.nanquantile(sv,.75)).astype(float); ls=(sv<np.nanquantile(sv,.25)).astype(float)
        rur=(puv<.5).astype(float); urb=1-rur
        nf['tribal_x_highsvi_x_rural']=t*hs*rur; nf['tribal_x_lowsvi_x_rural']=t*ls*rur
        nf['highsvi_x_rural']=hs*rur; nf['tribal_x_rural']=t*rur
        if bg is not None:
            bv=F(bg).values; nf['tribal_hsvi_rural_x_bldg']=t*hs*rur*bv; nf['hsvi_rural_x_bldg']=hs*rur*bv
        if wf is not None: nf['wf_x_rural_x_hsvi']=F(wf).values*rur*hs; nf['wf_x_tribal']=F(wf).values*t
        if cvi is not None:
            hc=(F(cvi).values>np.nanquantile(F(cvi).values,.75)).astype(float)
            nf['hcvi_x_hsvi_x_rural']=hc*hs*rur; nf['hcvi_x_tribal']=hc*t
    # Polynomial
    if bg is not None:
        bv=F(bg).values; nf['bldg_gap_sq']=bv**2; nf['bldg_gap_cu']=bv**3; nf['bldg_gap_abs']=np.abs(bv); nf['bldg_gap_log1p_abs']=np.log1p(np.abs(bv))
    if rg is not None:
        rv=F(rg).values; nf['road_gap_sq']=rv**2; nf['road_gap_abs']=np.abs(rv)
    if bg is not None and rg is not None:
        nf['bldg_road_ratio']=F(bg).values/(np.abs(F(rg).values)+1e-8); nf['bldg_road_diff']=F(bg).values-F(rg).values; nf['bldg_road_product']=F(bg).values*F(rg).values
    if br is not None: nf['log_bldg_ratio']=np.log1p(F(br).clip(lower=0).values); nf['bldg_ratio_sq']=F(br).values**2
    if rr is not None: nf['log_road_ratio']=np.log1p(F(rr).clip(lower=0).values)
    # Population
    if pop is not None:
        lp=np.log1p(F(pop).values); nf['log_pop']=lp
        if bg is not None: nf['log_pop_x_bldg']=lp*F(bg).values
        if svi is not None: nf['log_pop_x_svi']=lp*F(svi).values
    # Compound risk
    if bg is not None:
        comp=np.abs(F(bg).values)
        if rg is not None: comp+=np.abs(F(rg).values)
        if svi is not None: comp+=np.clip(F(svi).values,0,None)*0.1
        nf['compound_risk']=comp; nf['compound_risk_sq']=comp**2
        if tribal is not None: nf['tribal_x_risk']=(F(tribal).values>0).astype(float)*comp
    # Coverage nulls
    for cc in [c for c in feat.columns if '_covered' in c.lower()]: nf[f'{cc}_null']=feat[cc].isna().astype(float).values
    nulc=[k for k in nf if k.endswith('_null')]
    if nulc: nf['total_nulls']=np.sum([nf[k] for k in nulc],axis=0); nf['null_fraction']=nf['total_nulls']/max(len(nulc),1)
    # County LOO
    if 'GEOID' in feat.columns and bg is not None:
        county=feat['GEOID'].astype(str).str[:5]; bgv=F(bg)
        cs=bgv.groupby(county).agg(['mean','count','std']); cs.columns=['mean','count','std']
        gm=bgv.mean(); sm=10
        cms=(cs['mean']*cs['count']+gm*sm)/(cs['count']+sm)
        nf['bldg_county_loo_smooth']=cms[county].values; nf['bldg_county_count']=cs['count'][county].values
        cm=cs['mean'][county].values; cc=cs['count'][county].values
        nf['bldg_county_loo']=(cm*cc-bgv.values)/(cc-1+1e-8)
    if 'region' in feat.columns:
        for r in feat['region'].unique(): nf[f'region_{r}']=(feat['region']==r).astype(float).values
    if nf:
        nd=pd.DataFrame(nf,index=feat.index); nd=nd.replace([np.inf,-np.inf],np.nan)
        feat=pd.concat([feat,nd],axis=1); feat=feat.loc[:,~feat.columns.duplicated()]
        log.info(f"  +{len(nf)} features -> {feat.shape[1]} total")
    return feat

def adv_feat_sel(X, y, ntop=120):
    log.info("  Adv feature selection (corr+MI+var)...")
    cs=X.corrwith(y).abs().fillna(0)
    nmi=min(3000,len(X)); mi_idx=np.random.choice(len(X),nmi,replace=False)
    try:
        mi=mutual_info_regression(X.iloc[mi_idx].fillna(-999),y.iloc[mi_idx],n_neighbors=5,random_state=SEED)
        ms=pd.Series(mi,index=X.columns); ms=ms/(ms.max()+1e-10)
    except: ms=pd.Series(0,index=X.columns)
    vs=X.var().fillna(0); vs=vs/(vs.max()+1e-10)
    comb=0.5*cs+0.35*ms+0.15*vs
    sel=comb.sort_values(ascending=False).head(ntop).index.tolist()
    log.info(f"  Selected {len(sel)} features"); return sel

def prep_feat(df, tcol='building_gap', ntop=120):
    drop=['GEOID','region','county_fips','state_fips','centroid_lat','centroid_lon',
          'building_gap','road_gap','building_ratio','road_ratio','building_count_ratio',
          'building_count_gap','road_count_ratio','road_count_gap','road_length_ratio',
          'road_length_gap','poi_facility_gap','poi_to_facility_ratio',
          'coverage_gap_score','coverage_gap','gap_score','coverage_score']
    df=df.loc[:,~df.columns.duplicated()]
    fc=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X=df[fc].copy(); y=df[tcol].copy(); geo=df['GEOID'].astype(str).copy()
    v=y.notna(); X,y,geo=X[v],y[v],geo[v]
    X=X.fillna(-999).replace([np.inf,-np.inf],-999)
    s=X.std(); X=X[s[s>1e-10].index]
    nlp=X.isna().mean(); X=X[nlp[nlp<0.95].index]
    sel=adv_feat_sel(X,y,ntop); X=X[sel]
    cm=X.corr().abs(); up=cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool))
    td=[c for c in up.columns if any(up[c]>0.98)]; X=X.drop(columns=td)
    log.info(f"  {X.shape[1]} feats, {X.shape[0]} tracts | mean={y.mean():.4f} std={y.std():.4f}")
    return X,y,geo,v

def train_cv(model, X, y, blocks, cv_type='h3', nf=5):
    if cv_type=='h3':
        splits=spatial_split(X,y,blocks,nf)
    else:
        gkf=GroupKFold(n_splits=nf); splits=list(gkf.split(X,y,blocks))
    oof=np.full(len(y),np.nan); scores=[]; imps=[]; models=[]
    for fi,(ti,vi) in enumerate(splits):
        m=type(model)(**model.get_params())
        Xt,yt=X.iloc[ti],y.iloc[ti]; Xv,yv=X.iloc[vi],y.iloc[vi]
        try:
            if isinstance(m,xgb.XGBRegressor): m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
            elif isinstance(m,lgb.LGBMRegressor): m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(50,verbose=False)])
            elif isinstance(m,CatBoostRegressor): m.fit(Xt,yt,eval_set=(Xv,yv),early_stopping_rounds=50,verbose=0)
            else: m.fit(Xt,yt)
        except Exception as e: log.warning(f"    Fold {fi} err: {e}"); continue
        p=m.predict(Xv); oof[vi]=p
        rmse=np.sqrt(mean_squared_error(yv,p)); r2=r2_score(yv,p)
        scores.append({'rmse':rmse,'r2':r2})
        if hasattr(m,'feature_importances_'): imps.append(m.feature_importances_)
        models.append(m); log.info(f"    Fold {fi}: RMSE={rmse:.6f} R2={r2:.4f}")
    fid=None
    if imps: fid=pd.DataFrame({'feature':X.columns,'importance':np.mean(imps,axis=0)}).sort_values('importance',ascending=False)
    return {'cv_rmse':np.mean([s['rmse'] for s in scores]),'cv_r2':np.mean([s['r2'] for s in scores]),
            'cv_rmse_std':np.std([s['rmse'] for s in scores]),'oof':oof,'fi':fid,'models':models}

def bias_sc(yt,yp,geo): return pd.Series(yp-yt,index=geo.index).groupby(geo.str[:5]).mean().std()

def opt_blend(oofs, y):
    ns=list(oofs.keys()); mat=np.column_stack([oofs[n] for n in ns])
    v=~np.any(np.isnan(mat),axis=1); mv,yv=mat[v],y[v]
    res=minimize(lambda w:np.sqrt(mean_squared_error(yv,mv@w)),np.ones(len(ns))/len(ns),method='SLSQP',
                 bounds=[(0,1)]*len(ns),constraints={'type':'eq','fun':lambda w:sum(w)-1})
    wt={n:float(w) for n,w in zip(ns,res.x)}; return mat@res.x,wt,res.fun

def geo_blend(a,b):
    sh=min(a.min(),b.min())
    if sh<0: a_s=a-sh+1e-8; b_s=b-sh+1e-8
    else: a_s=a+1e-8; b_s=b+1e-8
    g=np.sqrt(a_s*b_s)
    if sh<0: g=g+sh-1e-8
    return g

def hybrid_blend(oofs, y, gr=0.70, ar=0.30):
    ns=list(oofs.keys())
    if len(ns)<2: return oofs[ns[0]],{ns[0]:1.0},np.sqrt(mean_squared_error(y,oofs[ns[0]]))
    cp,cw,cr=opt_blend(oofs,y)
    sn=sorted(cw.items(),key=lambda x:-x[1]); t1,t2=sn[0][0],sn[1][0]
    gp=geo_blend(oofs[t1],oofs[t2]); ap=(oofs[t1]+oofs[t2])/2
    hp=gr*gp+ar*ap; v=~np.isnan(hp); hr=np.sqrt(mean_squared_error(y[v],hp[v]))
    if hr<cr: log.info(f"  Hybrid wins: {hr:.6f} vs convex {cr:.6f}"); return hp,{**cw,'_type':'hybrid_70/30'},hr
    else: log.info(f"  Convex wins: {cr:.6f} vs hybrid {hr:.6f}"); return cp,{**cw,'_type':'convex'},cr

def stack_ens(oofs, y, blocks, nf=5):
    ns=list(oofs.keys()); mat=np.column_stack([oofs[n] for n in ns])
    vm=~np.any(np.isnan(mat),axis=1); mv=mat[vm]; yv=y.values[vm] if isinstance(y,pd.Series) else y[vm]
    bv=blocks[vm]; splits=spatial_split(pd.DataFrame(mv),pd.Series(yv),bv,nf)
    # Ridge
    so_r=np.full(len(yv),np.nan)
    for fi,(ti,vi) in enumerate(splits):
        meta=Ridge(alpha=1.0,random_state=SEED); meta.fit(mv[ti],yv[ti]); so_r[vi]=meta.predict(mv[vi])
    v2=~np.isnan(so_r); sr_r=np.sqrt(mean_squared_error(yv[v2],so_r[v2]))
    # ElasticNet
    so_e=np.full(len(yv),np.nan)
    for fi,(ti,vi) in enumerate(splits):
        meta=ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=SEED,max_iter=5000); meta.fit(mv[ti],yv[ti]); so_e[vi]=meta.predict(mv[vi])
    v3=~np.isnan(so_e); sr_e=np.sqrt(mean_squared_error(yv[v3],so_e[v3]))
    if sr_e<sr_r:
        log.info(f"  Stack(EN): RMSE={sr_e:.6f}")
        mf=ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=SEED,max_iter=5000); mf.fit(mv,yv)
        return so_e,sr_e,r2_score(yv[v3],so_e[v3]),mf
    else:
        log.info(f"  Stack(Ridge): RMSE={sr_r:.6f}")
        mf=Ridge(alpha=1.0,random_state=SEED); mf.fit(mv,yv)
        return so_r,sr_r,r2_score(yv[v2],so_r[v2]),mf

def run_pysr(X, y, n_samples=2000, timeout=3):
    if not HAS_PYSR: log.warning("PySR not installed"); return None
    log.info("GOD MODE: PySR Symbolic Regression")
    if n_samples<len(X): idx=np.random.choice(len(X),n_samples,replace=False); Xs,y_s=X.iloc[idx],y.iloc[idx]
    else: Xs,y_s=X,y
    log.info(f"  PySR on {len(Xs)} samples x {Xs.shape[1]} features")
    model=PySRRegressor(niterations=40,binary_operators=["+","-","*","/"],unary_operators=["exp","log","sqrt","abs"],
                        populations=15,population_size=30,maxsize=20,seed=SEED,
                        timeout_in_seconds=timeout*60,temp_equation_file=True,progress=False,verbosity=0,parsimony=0.01)
    try:
        model.fit(Xs,y_s)
        if hasattr(model,'equations_') and model.equations_ is not None:
            best=model.equations_.iloc[0]
            eq=str(best.get('equation','')); r2=float(best.get('r2_score',0)); rmse=float(best.get('rmse',0))
            log.info(f"  Best eq: {eq[:80]}..."); log.info(f"  R2={r2:.6f} RMSE={rmse:.6f}")
            if r2>0.999: log.info("  *** EXACT FORMULA FOUND! R2>0.999! ***")
            return {'equation':eq,'r2':r2,'rmse':rmse,'complexity':int(best.get('complexity',0))}
    except Exception as e: log.warning(f"  PySR err: {e}")
    return None

def optuna_xgb(trial,X,y,blocks,nf=5):
    p={'n_estimators':trial.suggest_int('n',800,2500),'max_depth':trial.suggest_int('d',4,10),
       'learning_rate':trial.suggest_float('lr',0.005,0.05,log=True),'subsample':trial.suggest_float('ss',0.6,1.0),
       'colsample_bytree':trial.suggest_float('cb',0.4,1.0),'reg_alpha':trial.suggest_float('ra',1e-8,10,log=True),
       'reg_lambda':trial.suggest_float('rl',1e-8,10,log=True),'min_child_weight':trial.suggest_int('mcw',1,20),
       'tree_method':'hist','random_state':SEED}
    sp=spatial_split(X,y,blocks,nf); rmses=[]
    for ti,vi in sp:
        m=xgb.XGBRegressor(**p); m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],verbose=False)
        rmses.append(np.sqrt(mean_squared_error(y.iloc[vi],m.predict(X.iloc[vi]))))
    return np.mean(rmses)

def optuna_lgb(trial,X,y,blocks,nf=5):
    bt=trial.suggest_categorical('bt',['gbdt','dart'])
    p={'n_estimators':trial.suggest_int('n',800,2500),'max_depth':trial.suggest_int('d',4,10),
       'learning_rate':trial.suggest_float('lr',0.005,0.05,log=True),'subsample':trial.suggest_float('ss',0.6,1.0),
       'colsample_bytree':trial.suggest_float('cb',0.4,1.0),'reg_alpha':trial.suggest_float('ra',1e-8,10,log=True),
       'reg_lambda':trial.suggest_float('rl',1e-8,10,log=True),'min_child_samples':trial.suggest_int('mcs',5,50),
       'boosting_type':bt,'random_state':SEED,'verbose':-1}
    sp=spatial_split(X,y,blocks,nf); rmses=[]
    for ti,vi in sp:
        m=lgb.LGBMRegressor(**p); m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],callbacks=[lgb.early_stopping(50,verbose=False)])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[vi],m.predict(X.iloc[vi]))))
    return np.mean(rmses)

def run_opt(name,fn,X,y,blocks,nt=40):
    if not HAS_OPTUNA: return None
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED),pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lambda t:fn(t,X,y,blocks),n_trials=nt,show_progress_bar=False)
    log.info(f"  Optuna {name}: Best={study.best_value:.6f}"); return study.best_params

def bias_disc(yt,yp,fd,geo):
    log.info("Bias discovery (5 API dimensions)..."); res=yt-yp; out=[]
    svi=fd.get('svi_overall');tribal=fd.get('tribal_any');pu=fd.get('pct_urban')
    cvi=fd.get('cvi_overall');wf=fd.get('usgs_wildfire_ever');bg=fd.get('building_gap');rg=fd.get('road_gap')
    def add(d,s,m,hm,lm,hn,ln):
        sv='CRITICAL' if m>1.5 else ('HIGH' if m>1.2 else ('MODERATE' if m>1.05 else 'LOW'))
        out.append({'dimension':d,'stratum':s,'metric':m,'high_mean':hm,'low_mean':lm,'high_n':int(hn),'low_n':int(ln),'severity':sv})
    if svi is not None:
        sv=svi.fillna(0.5); hs=sv>sv.quantile(.75); ls=sv<sv.quantile(.25)
        hm,lm=np.abs(res[hs]).mean(),np.abs(res[ls]).mean(); add('Cov Disparity','HighSVI vs LowSVI',hm/(lm+1e-10),hm,lm,hs.sum(),ls.sum())
    if tribal is not None:
        t=(tribal.fillna(0)>0); hm,lm=np.abs(res[t]).mean(),np.abs(res[~t]).mean(); add('Cov Disparity','Tribal vs Non',hm/(lm+1e-10),hm,lm,t.sum(),(~t).sum())
    if pu is not None:
        r=pu.fillna(.5)<.5; hm,lm=np.abs(res[r]).mean(),np.abs(res[~r]).mean(); add('Cov Disparity','Rural vs Urban',hm/(lm+1e-10),hm,lm,r.sum(),(~r).sum())
    if bg is not None:
        bv=bg.fillna(0); d=bv<bv.quantile(.25); hm,lm=np.abs(res[d]).mean(),np.abs(res[~d]).mean(); add('POI Desert','Low vs High Cov',hm/(lm+1e-10),hm,lm,d.sum(),(~d).sum())
    if tribal is not None and pu is not None:
        t=(tribal.fillna(0)>0); r=pu.fillna(.5)<.5; v=t|r; hm,lm=np.abs(res[v]).mean(),np.abs(res[~v]).mean(); add('Emerg Access','Tribal/Rural vs Rest',hm/(lm+1e-10),hm,lm,v.sum(),(~v).sum())
    if rg is not None:
        rv=rg.fillna(0); lr=rv<rv.quantile(.25); hr=rv>rv.quantile(.75); hm,lm=np.abs(res[lr]).mean(),np.abs(res[hr]).mean(); add('Road Equity','Low vs High Road',hm/(lm+1e-10),hm,lm,lr.sum(),hr.sum())
    if cvi is not None and svi is not None and bg is not None:
        cj=(cvi.fillna(0)>cvi.fillna(0).quantile(.75))&(svi.fillna(.5)>svi.fillna(.5).quantile(.75))&(bg.fillna(0)<bg.fillna(0).quantile(.5))
        hm,lm=np.abs(res[cj]).mean(),np.abs(res[~cj]).mean(); add('Climate-Justice','HighCVI×HighSVI×LowCov',hm/(lm+1e-10),hm,lm,cj.sum(),(~cj).sum())
    if svi is not None and tribal is not None and pu is not None:
        sv=svi.fillna(.5); t=(tribal.fillna(0)>0); r=pu.fillna(.5)<.5; hs=sv>sv.quantile(.75); om=np.abs(res).mean()
        for nm,ms in [('tribal×highSVI×rural',t&hs&r),('highSVI×rural',hs&r),('tribal×rural',t&r)]:
            if ms.sum()>=5: gm=np.abs(res[ms]).mean(); add('Intersectional',nm,gm/(om+1e-10),gm,om,ms.sum(),len(res)-ms.sum())
        if wf is not None:
            for nm,ms in [('wf×highSVI×rural',(wf.fillna(0)>0)&hs&r),('wf×tribal',(wf.fillna(0)>0)&t)]:
                if ms.sum()>=5: gm=np.abs(res[ms]).mean(); add('Intersectional',nm,gm/(om+1e-10),gm,om,ms.sum(),len(res)-ms.sum())
    bdf=pd.DataFrame(out); log.info(f"  {len(out)} bias findings"); return bdf

def main():
    t0=time.time()
    log.info("="*70); log.info("ULTIMATE PIPELINE v3 - Leakage-Fixed + God Mode"); log.info("="*70)
    feat=load_feat(); strata=load_strata(); tcol=find_target(feat)
    feat=engineer_feats(feat,strata); X,y,geo,valid=prep_feat(feat,tcol=tcol,ntop=120)
    fns=list(X.columns)
    # H3 blocks
    log.info("Computing H3 spatial blocks...")
    lats=feat.loc[valid,'centroid_lat'] if 'centroid_lat' in feat.columns else None
    lons=feat.loc[valid,'centroid_lon'] if 'centroid_lon' in feat.columns else None
    blocks=h3_blocks(geo,lats=lats,lons=lons,res=4)
    nb=blocks.nunique(); log.info(f"  {nb} spatial blocks")
    # Phase 0: PySR
    log.info("\nPHASE 0: PySR Symbolic Regression (GOD MODE)")
    corr=X.corrwith(y).abs().sort_values(ascending=False); sr_feat=corr.head(10).index.tolist()
    sr_result=run_pysr(X[sr_feat],y,n_samples=1500,timeout=3)
    if sr_result and sr_result.get('r2',0)>0.999:
        log.info("*** EXACT FORMULA FOUND! R2>0.999! Submit for RMSE=0! ***")
    # Phase 1: Train with H3-CV
    log.info("\nPHASE 1: Training with H3 Spatial Block CV (leakage-fixed)")
    oofs=OrderedDict(); mres=OrderedDict()
    # XGBoost
    log.info("  XGBoost..."); xm=xgb.XGBRegressor(n_estimators=1500,max_depth=7,learning_rate=0.015,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,tree_method='hist',random_state=SEED)
    xr=train_cv(xm,X,y,blocks,'h3',5); oofs['xgb']=xr['oof']; mres['xgb']=xr; log.info(f"  XGB(H3): RMSE={xr['cv_rmse']:.6f} R2={xr['cv_r2']:.4f}")
    # County CV comparison
    cb=geo.str[:5]; xrc=train_cv(xm,X,y,cb,'county',5); ld=xrc['cv_r2']-xr['cv_r2']
    log.info(f"  XGB(County): RMSE={xrc['cv_rmse']:.6f} R2={xrc['cv_r2']:.4f}")
    if ld>0.05: log.warning(f"  *** R2 DROP {ld:.4f} = SPATIAL LEAKAGE CONFIRMED! ***")
    else: log.info(f"  R2 drop only {ld:.4f} - minimal leakage")
    # LightGBM
    log.info("  LightGBM..."); lm=lgb.LGBMRegressor(n_estimators=1500,max_depth=7,learning_rate=0.015,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_samples=10,boosting_type='gbdt',random_state=SEED,verbose=-1)
    lr_=train_cv(lm,X,y,blocks,'h3',5); oofs['lgb']=lr_['oof']; mres['lgb']=lr_; log.info(f"  LGB(H3): RMSE={lr_['cv_rmse']:.6f} R2={lr_['cv_r2']:.4f}")
    # LGB DART
    log.info("  LightGBM-DART..."); dm=lgb.LGBMRegressor(n_estimators=800,max_depth=7,learning_rate=0.05,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_samples=10,boosting_type='dart',random_state=SEED,verbose=-1,drop_rate=0.1,max_drop=50)
    dr=train_cv(dm,X,y,blocks,'h3',5); oofs['lgb_dart']=dr['oof']; mres['lgb_dart']=dr; log.info(f"  DART(H3): RMSE={dr['cv_rmse']:.6f} R2={dr['cv_r2']:.4f}")
    # CatBoost
    log.info("  CatBoost..."); cm=CatBoostRegressor(iterations=1500,depth=8,learning_rate=0.015,l2_leaf_reg=3.0,random_strength=1.0,bagging_temperature=0.5,random_seed=SEED,verbose=0)
    cr=train_cv(cm,X,y,blocks,'h3',5); oofs['cat']=cr['oof']; mres['cat']=cr; log.info(f"  Cat(H3): RMSE={cr['cv_rmse']:.6f} R2={cr['cv_r2']:.4f}")
    # ExtraTrees
    log.info("  ExtraTrees..."); em=ExtraTreesRegressor(n_estimators=300,max_depth=15,min_samples_split=5,random_state=SEED,n_jobs=-1)
    er=train_cv(em,X,y,blocks,'h3',5); oofs['et']=er['oof']; mres['et']=er; log.info(f"  ET(H3): RMSE={er['cv_rmse']:.6f} R2={er['cv_r2']:.4f}")
    # Phase 2: Optuna
    if HAS_OPTUNA:
        log.info("\nPHASE 2: Optuna optimization")
        xb=run_opt('xgb',optuna_xgb,X,y,blocks,15)
        if xb: xb.update({'tree_method':'hist','random_state':SEED}); xo=xgb.XGBRegressor(**xb); xor=train_cv(xo,X,y,blocks,'h3',5)
        if xb and xor['cv_rmse']<xr['cv_rmse']: log.info(f"  Optuna XGB: {xr['cv_rmse']:.6f}->{xor['cv_rmse']:.6f}"); oofs['xgb_opt']=xor['oof']; mres['xgb_opt']=xor
        lb=run_opt('lgb',optuna_lgb,X,y,blocks,15)
        if lb: lb.update({'random_state':SEED,'verbose':-1}); lo=lgb.LGBMRegressor(**lb); lor=train_cv(lo,X,y,blocks,'h3',5)
        if lb and lor['cv_rmse']<lr_['cv_rmse']: log.info(f"  Optuna LGB: {lr_['cv_rmse']:.6f}->{lor['cv_rmse']:.6f}"); oofs['lgb_opt']=lor['oof']; mres['lgb_opt']=lor
    # Phase 3: Ensemble
    log.info("\nPHASE 3: Ensemble (Convex + Hybrid 70/30 + Stacking)")
    bp,bw,br_=opt_blend(oofs,y); br2=r2_score(y[~np.isnan(bp)],bp[~np.isnan(bp)]); log.info(f"  Convex: RMSE={br_:.6f} R2={br2:.4f}")
    hp,hw,hr_=hybrid_blend(oofs,y,0.70,0.30); hr2=r2_score(y[~np.isnan(hp)],hp[~np.isnan(hp)]); log.info(f"  Hybrid: RMSE={hr_:.6f} R2={hr2:.4f}")
    sp_,sr_,sr2,meta=stack_ens(oofs,y,blocks,5); log.info(f"  Stack: RMSE={sr_:.6f} R2={sr2:.4f}")
    opts=[('convex',bp,br_,br2,bw),('hybrid_70/30',hp,hr_,hr2,hw),('stacking',sp_,sr_,sr2,None)]
    bn,bpred,brm,br2_,bwt=min(opts,key=lambda x:x[2]); bbs=bias_sc(y,bpred,geo)
    log.info(f"\n  BEST: {bn} RMSE={brm:.6f} R2={br2_:.4f} Bias={bbs:.6f}")
    # Phase 4: Bias
    log.info("\nPHASE 4: Bias Discovery"); bdf=bias_disc(y.values,bpred,feat[valid],geo)
    # Phase 5: Road gap
    log.info("\nPHASE 5: Road gap")
    if 'road_gap' in feat.columns:
        Xr,yr,gr_,vr=prep_feat(feat,tcol='road_gap',ntop=80)
        rb=h3_blocks(gr_,lats=feat.loc[vr,'centroid_lat'] if 'centroid_lat' in feat.columns else None,
                     lons=feat.loc[vr,'centroid_lon'] if 'centroid_lon' in feat.columns else None,res=4)
        rm_=xgb.XGBRegressor(n_estimators=1000,max_depth=7,learning_rate=0.02,subsample=0.8,colsample_bytree=0.7,tree_method='hist',random_state=SEED)
        rr_=train_cv(rm_,Xr,yr,rb,'h3',5); log.info(f"  Road: RMSE={rr_['cv_rmse']:.6f} R2={rr_['cv_r2']:.4f}")
    # Phase 6: Submission
    log.info("\nPHASE 6: Submission")
    tp=np.full(len(feat),np.nan)
    if bn=='stacking':
        mat=np.column_stack([oofs[n] for n in oofs.keys()]); tp[valid]=meta.predict(mat)
    else: tp[valid]=bpred
    tp=np.clip(tp,-3.0,0.5)
    sub=pd.DataFrame({'GEOID':feat['GEOID'].astype(str),'coverage_gap_score':tp}).dropna(subset=['coverage_gap_score'])
    sub.to_csv(OUT/'submission.csv',index=False); sub.to_csv(DL/'submission.csv',index=False); log.info(f"  {len(sub)} tracts")
    # Save results
    res={'timestamp':datetime.now().isoformat(),'pipeline':'ultimate_v3','target':tcol,'cv_type':'H3_spatial_block',
         'n_blocks':int(nb),'n_features':len(fns),'n_tracts':len(sub),'best_ensemble':bn,
         'best_rmse':float(brm),'best_r2':float(br2_),'best_bias':float(bbs),
         'best_weights':{k:float(v) if isinstance(v,(np.floating,float)) else str(v) for k,v in (bwt or {}).items()},
         'models':{k:{'rmse':float(v['cv_rmse']),'r2':float(v['cv_r2'])} for k,v in mres.items()},
         'leakage_check':{'county_r2':float(xrc['cv_r2']),'h3_r2':float(xr['cv_r2']),'r2_drop':float(ld),'leakage':bool(ld>0.05)},
         'symbolic_regression':sr_result,'elapsed_sec':round(time.time()-t0,1)}
    with open(OUT/'pipeline_state.json','w') as f: json.dump(res,f,indent=2,default=str)
    if bdf is not None and len(bdf)>0: bdf.to_csv(OUT/'comprehensive_bias_findings.csv',index=False)
    comp=pd.DataFrame([{'model':k,'rmse':v['cv_rmse'],'r2':v['cv_r2']} for k,v in mres.items()])
    comp.to_csv(OUT/'model_comparison.csv',index=False)
    for n,r in mres.items():
        if r.get('fi') is not None: r['fi'].to_csv(OUT/f'{n}_feature_importance.csv',index=False)
    el=time.time()-t0
    log.info(f"\n{'='*70}"); log.info(f"DONE in {el:.0f}s")
    log.info(f"  Best: {bn} RMSE={brm:.6f} R2={br2_:.4f}")
    log.info(f"  Leakage: County R2={xrc['cv_r2']:.4f} -> H3 R2={xr['cv_r2']:.4f} (drop={ld:.4f})")
    if sr_result and sr_result.get('r2',0)>0.999: log.info(f"  GOD MODE: Exact formula R2={sr_result['r2']:.6f}")
    log.info(f"  Submission: {len(sub)} tracts"); log.info(f"{'='*70}")
    return res

if __name__=='__main__': main()
