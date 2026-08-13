#!/usr/bin/env python3
"""
Weather Forecast Feature Extraction for US Census Tracts
=========================================================
Extracts 15-day forecast features from OpenMeteo API for priority tracts,
then interpolates remaining tracts using KNN (scipy.spatial.cKDTree).

Priority tracts = focus region counties + 3000 national stratified sample.
Processes in chunks of 200 with concurrent API calls within each chunk.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = Path('/home/z/my-project/bias-bounty-map')
DATA_DIR = BASE_DIR / 'kaggle_dataset'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

STRATA_PATH = DATA_DIR / 'national-strata-tract-table.parquet'
OUTPUT_PATH = DATA_DIR / 'weather_forecast_features.parquet'
CHECKPOINT_PATH = RESULTS_DIR / 'weather_checkpoint.parquet'

# --- API Config ---
FORECAST_DAYS = 15
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20
RATE_LIMIT_SEC = 0.05  # 50ms between requests
N_WORKERS = 8

DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,precipitation_probability_max,"
    "wind_speed_10m_max,wind_speed_10m_mean,"
    "wind_speed_100m_max,wind_speed_100m_mean,"
    "uv_index_max"
)

HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,"
    "wind_speed_10m,surface_pressure,"
    "vapour_pressure_deficit"
)

# --- Thresholds ---
HEAT_ALERT_TEMP = 38.0
EXTREME_HEAT_TEMP = 43.0
HEAVY_PRECIP_MM = 25.0
VERY_HEAVY_PRECIP_MM = 50.0
HIGH_WIND_KMH = 50.0

# --- Focus Region County FIPS ---
FOCUS_COUNTIES = {'04013'}

NORCAL = [
    '06003','06007','06009','06015','06021','06023','06033','06035',
    '06041','06045','06049','06051','06057','06061','06067','06073',
    '06075','06089','06091','06093','06095','06097','06099','06101',
    '06103','06105','06107','06109','06111','06113'
]
FOCUS_COUNTIES.update(NORCAL)

for _i in range(1, 48):
    FOCUS_COUNTIES.add(f'40{_i:03d}')

TX_COUNTIES = [
    '48013','48021','48029','48037','48049','48055','48057','48061',
    '48065','48071','48079','48087','48091','48093','48131','48141',
    '48157','48187','48201','48205','48211','48221','48235','48245',
    '48255','48261','48265','48275','48283','48287'
]
FOCUS_COUNTIES.update(TX_COUNTIES)

NATIONAL_SAMPLE_SIZE = 3000
CHUNK_SIZE = 200  # Process & checkpoint every 200 tracts


# ==============================================================
# 1. Load & Clean Tract Centroids
# ==============================================================
def load_tract_centroids():
    log.info("Loading tract centroids from parquet...")
    df = pd.read_parquet(
        STRATA_PATH,
        columns=['GEOID', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP']
    )
    df['lat'] = df['INTPTLAT'].astype(str).str.lstrip('+').astype(float)
    df['lon'] = df['INTPTLON'].astype(str).str.lstrip('+').astype(float)
    valid = (
        (df['lat'] >= 17.0) & (df['lat'] <= 72.0) &
        (df['lon'] >= -180.0) & (df['lon'] <= -65.0)
    )
    df = df[valid].copy()
    df['county_fips'] = df['STATEFP'] + df['COUNTYFP']
    log.info(f"  Loaded {len(df)} valid tracts across {df['county_fips'].nunique()} counties")
    return df


def select_priority_tracts(df):
    """Select focus region counties + 3000 national stratified sample."""
    focus_mask = df['county_fips'].isin(FOCUS_COUNTIES)
    focus_df = df[focus_mask].copy()
    log.info(f"  Focus region tracts: {len(focus_df)} across {focus_df['county_fips'].nunique()} counties")

    non_focus = df[~focus_mask].copy()
    n_states = non_focus['STATEFP'].nunique()
    per_state = max(1, NATIONAL_SAMPLE_SIZE // n_states)
    remainder = NATIONAL_SAMPLE_SIZE - per_state * n_states

    sample_parts = []
    states = non_focus['STATEFP'].unique()
    for i, st in enumerate(states):
        st_df = non_focus[non_focus['STATEFP'] == st]
        n_pick = per_state + (1 if i < remainder else 0)
        n_pick = min(n_pick, len(st_df))
        if n_pick > 0:
            sample_parts.append(st_df.sample(n=n_pick, random_state=42))

    sample_df = pd.concat(sample_parts, ignore_index=False)
    log.info(f"  National stratified sample: {len(sample_df)} tracts across {sample_df['STATEFP'].nunique()} states")

    priority_df = pd.concat([focus_df, sample_df], ignore_index=False)
    priority_df = priority_df.drop_duplicates(subset='GEOID')
    log.info(f"  Total priority tracts: {len(priority_df)}")
    return priority_df


# ==============================================================
# 2. OpenMeteo API Call
# ==============================================================
_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def fetch_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": DAILY_VARS,
        "hourly": HOURLY_VARS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "auto",
    }
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_and_compute(geoid, statefp, countyfp, lat, lon, county_fips):
    """Fetch forecast and compute features for a single tract."""
    time.sleep(RATE_LIMIT_SEC)  # Rate limit per thread
    data = fetch_forecast(lat, lon)
    base = {
        'GEOID': str(geoid),
        'STATEFP': str(statefp),
        'COUNTYFP': str(countyfp),
        'lat': lat,
        'lon': lon,
        'county_fips': county_fips,
    }
    if data:
        feat = compute_features(data)
        if feat:
            return {**base, **feat}, False
    return base, True


# ==============================================================
# 3. Feature Computation
# ==============================================================
def _safe(vals):
    return [v for v in vals if v is not None]


def compute_features(data):
    if not data or 'daily' not in data:
        return None

    features = {}
    daily = data.get('daily', {})
    hourly = data.get('hourly', {})

    # --- Temperature ---
    tmax = _safe(daily.get('temperature_2m_max', []))
    tmin = _safe(daily.get('temperature_2m_min', []))
    if tmax:
        features['wf_temp_max'] = max(tmax)
        features['wf_temp_min'] = min(tmin) if tmin else np.nan
        features['wf_temp_range'] = features['wf_temp_max'] - features['wf_temp_min']
        features['wf_hot_days'] = sum(1 for v in tmax if v > 35.0)
        features['wf_very_hot_days'] = sum(1 for v in tmax if v > HEAT_ALERT_TEMP)
        features['wf_extreme_hot_days'] = sum(1 for v in tmax if v > EXTREME_HEAT_TEMP)
        features['wf_cold_days'] = sum(1 for v in tmin if v < -10.0) if tmin else 0
        features['wf_freeze_days'] = sum(1 for v in tmin if v < 0.0) if tmin else 0

    # --- Precipitation ---
    precip = _safe(daily.get('precipitation_sum', []))
    if precip:
        features['wf_precip_total'] = sum(precip)
        features['wf_precip_max_day'] = max(precip)
        features['wf_heavy_precip_days'] = sum(1 for v in precip if v > HEAVY_PRECIP_MM)
        features['wf_very_heavy_precip_days'] = sum(1 for v in precip if v > VERY_HEAVY_PRECIP_MM)
        features['wf_dry_days'] = sum(1 for v in precip if v < 0.1)

    pprob = _safe(daily.get('precipitation_probability_max', []))
    if pprob:
        features['wf_precip_prob_mean'] = np.mean(pprob)
        features['wf_precip_prob_max'] = max(pprob)

    # --- Wind ---
    w10max = _safe(daily.get('wind_speed_10m_max', []))
    w10mean = _safe(daily.get('wind_speed_10m_mean', []))
    w100max = _safe(daily.get('wind_speed_100m_max', []))
    w100mean = _safe(daily.get('wind_speed_100m_mean', []))

    if w10max:
        features['wf_wind_10m_max'] = max(w10max)
        features['wf_wind_10m_mean'] = np.mean(w10mean) if w10mean else np.nan
        features['wf_high_wind_days'] = sum(1 for v in w10max if v > HIGH_WIND_KMH)
    if w100max:
        features['wf_wind_100m_max'] = max(w100max)
        features['wf_wind_100m_mean'] = np.mean(w100mean) if w100mean else np.nan
        if w10max:
            features['wf_wind_shear'] = features['wf_wind_100m_max'] - features['wf_wind_10m_max']

    # --- UV Index ---
    uv = _safe(daily.get('uv_index_max', []))
    if uv:
        features['wf_uv_max'] = max(uv)
        features['wf_uv_mean'] = np.mean(uv)

    # --- Hourly Features ---
    if hourly:
        sp = _safe(hourly.get('surface_pressure', []))
        rh = _safe(hourly.get('relative_humidity_2m', []))
        vpd = _safe(hourly.get('vapour_pressure_deficit', []))
        htemp = _safe(hourly.get('temperature_2m', []))
        hwind = _safe(hourly.get('wind_speed_10m', []))

        if sp:
            features['wf_sfc_pressure_min'] = min(sp)
            features['wf_sfc_pressure_mean'] = np.mean(sp)
        if rh:
            features['wf_humidity_min'] = min(rh)
            features['wf_humidity_mean'] = np.mean(rh)
            features['wf_low_humidity_hours'] = sum(1 for v in rh if v < 20)
        if vpd:
            features['wf_vpd_max'] = max(vpd)
            features['wf_vpd_mean'] = np.mean(vpd)
        if htemp:
            features['wf_hourly_temp_max'] = max(htemp)
            features['wf_hourly_temp_min'] = min(htemp)
        if hwind:
            features['wf_hourly_wind_max'] = max(hwind)
            features['wf_hourly_wind_mean'] = np.mean(hwind)

    # --- Derived Risk Indices ---

    # Fire weather risk
    if 'wf_temp_max' in features and 'wf_wind_10m_max' in features:
        temp_norm = max(0, features['wf_temp_max'] - 25) / 20
        wind_norm = max(0, features['wf_wind_10m_max'] - 20) / 60
        dry_norm = features.get('wf_dry_days', 0) / FORECAST_DAYS
        hum_min_val = features.get('wf_humidity_min', np.nan)
        if isinstance(hum_min_val, (int, float)) and not np.isnan(hum_min_val):
            hum_norm = max(0, 30 - hum_min_val) / 30
            features['wf_fire_weather_risk'] = (
                temp_norm * 0.3 + wind_norm * 0.3 + dry_norm * 0.15 + hum_norm * 0.25
            )
        else:
            features['wf_fire_weather_risk'] = temp_norm * 0.4 + wind_norm * 0.4 + dry_norm * 0.2
    else:
        features['wf_fire_weather_risk'] = 0.0

    # Flood risk
    features['wf_flood_risk'] = min(1.0, features.get('wf_heavy_precip_days', 0) / 3.0)
    features['wf_extreme_flood_risk'] = min(1.0, features.get('wf_very_heavy_precip_days', 0) / 2.0)

    # Storm risk
    if 'wf_sfc_pressure_min' in features and 'wf_wind_10m_max' in features:
        sp_anomaly = max(0, (100800 - features['wf_sfc_pressure_min']) / 2000)
        wind_factor = max(0, (features['wf_wind_10m_max'] - HIGH_WIND_KMH) / 50.0)
        features['wf_storm_risk'] = min(1.0, sp_anomaly + wind_factor)
    else:
        features['wf_storm_risk'] = 0.0

    # Freeze risk
    features['wf_freeze_risk'] = min(1.0, features.get('wf_freeze_days', 0) / 5.0)

    # Compound hazard
    hazard_count = (
        (1 if features.get('wf_very_hot_days', 0) > 0 else 0) +
        (1 if features.get('wf_fire_weather_risk', 0) > 0.3 else 0) +
        (1 if features.get('wf_flood_risk', 0) > 0.2 else 0) +
        (1 if features.get('wf_storm_risk', 0) > 0.2 else 0) +
        (1 if features.get('wf_freeze_risk', 0) > 0.2 else 0)
    )
    features['wf_compound_hazard'] = 1 if hazard_count >= 2 else 0
    features['wf_hazard_count'] = hazard_count
    features['wf_max_hazard_score'] = max(
        features.get('wf_very_hot_days', 0) / FORECAST_DAYS,
        features.get('wf_fire_weather_risk', 0),
        features.get('wf_flood_risk', 0),
        features.get('wf_storm_risk', 0),
        features.get('wf_freeze_risk', 0)
    )

    return features if features else None


# ==============================================================
# 4. KNN Interpolation
# ==============================================================
def knn_interpolate(priority_df, all_df, feature_cols, k=5):
    """Interpolate features for non-priority tracts using KNN with inverse-distance weighting."""
    log.info(f"KNN interpolation (k={k}) for {len(feature_cols)} features...")

    coords_known = np.column_stack([
        priority_df['lat'].values,
        priority_df['lon'].values
    ])
    tree = cKDTree(coords_known)

    known_geoids = set(priority_df['GEOID'].astype(str))
    interp_mask = ~all_df['GEOID'].astype(str).isin(known_geoids)
    interp_df = all_df[interp_mask].copy()

    if len(interp_df) == 0:
        log.info("  No tracts to interpolate.")
        return all_df

    log.info(f"  Interpolating {len(interp_df)} tracts...")

    for col in feature_cols:
        if col not in interp_df.columns:
            interp_df[col] = np.nan

    coords_query = np.column_stack([
        interp_df['lat'].values,
        interp_df['lon'].values
    ])

    distances, indices = tree.query(coords_query, k=k)

    for col in feature_cols:
        known_vals = priority_df[col].values
        nn_vals = known_vals[indices]
        valid_mask = ~np.isnan(nn_vals)
        dists_safe = distances + 1e-10
        weights = 1.0 / dists_safe
        weights = weights * valid_mask
        weight_sums = weights.sum(axis=1)
        has_valid = weight_sums > 0
        result = np.full(len(interp_df), np.nan)
        if has_valid.any():
            weighted_sums = np.nansum(nn_vals * weights, axis=1)
            result[has_valid] = weighted_sums[has_valid] / weight_sums[has_valid]
        interp_df[col] = result

    result_df = pd.concat([priority_df, interp_df], ignore_index=True)
    log.info(f"  Interpolation complete: {len(result_df)} total tracts")
    return result_df


# ==============================================================
# 5. Main
# ==============================================================
def main():
    log.info("=" * 70)
    log.info("Weather Forecast Feature Extraction - OpenMeteo API")
    log.info("=" * 70)

    all_tracts = load_tract_centroids()
    n_total = len(all_tracts)
    log.info(f"Total tracts in dataset: {n_total}")

    priority_df = select_priority_tracts(all_tracts)
    n_priority = len(priority_df)
    log.info(f"Priority tracts to query: {n_priority}")

    # -- Resume from checkpoint --
    done_geoids = set()

    if CHECKPOINT_PATH.exists():
        existing = pd.read_parquet(CHECKPOINT_PATH)
        done_geoids = set(existing['GEOID'].astype(str))
        log.info(f"Checkpoint: {len(done_geoids)} tracts already done")

    todo_df = priority_df[~priority_df['GEOID'].astype(str).isin(done_geoids)].copy()
    todo_df = todo_df.reset_index(drop=True)
    n_todo = len(todo_df)
    log.info(f"Remaining to query: {n_todo} tracts")

    # -- Fetch forecasts in chunks --
    if n_todo > 0:
        total_failed = 0
        total_done = 0
        t0 = time.time()

        todo_records = todo_df[['GEOID','STATEFP','COUNTYFP','lat','lon','county_fips']].to_dict('records')
        n_chunks = (n_todo + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_todo)
            chunk = todo_records[start:end]

            chunk_results = []
            chunk_failed = 0

            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = []
                for rec in chunk:
                    fut = executor.submit(
                        fetch_and_compute,
                        rec['GEOID'], rec['STATEFP'], rec['COUNTYFP'],
                        rec['lat'], rec['lon'], rec['county_fips']
                    )
                    futures.append(fut)

                for fut in as_completed(futures):
                    try:
                        result, is_fail = fut.result()
                        chunk_results.append(result)
                        if is_fail:
                            chunk_failed += 1
                    except Exception:
                        chunk_failed += 1

            total_done += len(chunk)
            total_failed += chunk_failed

            # Save checkpoint after each chunk
            new_df = pd.DataFrame(chunk_results)
            if CHECKPOINT_PATH.exists():
                prev = pd.read_parquet(CHECKPOINT_PATH)
                combined = pd.concat([prev, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(CHECKPOINT_PATH)
            done_geoids.update(new_df['GEOID'].astype(str).tolist())

            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (n_todo - total_done) / rate if rate > 0 else 0
            log.info(
                f"  Chunk {chunk_idx+1}/{n_chunks}: {total_done}/{n_todo} done, "
                f"{total_failed} failed, {rate:.1f} tracts/sec, ETA {eta:.0f}s"
            )

    # -- Load complete priority results --
    priority_results = pd.read_parquet(CHECKPOINT_PATH)
    wf_cols = [col for col in priority_results.columns if col.startswith('wf_')]
    n_have_data = priority_results[wf_cols].notna().any(axis=1).sum()
    log.info(f"Priority results: {len(priority_results)} tracts, "
             f"{n_have_data} with data ({n_have_data/len(priority_results)*100:.1f}%)")

    # -- KNN Interpolation for remaining tracts --
    feature_cols = [c for c in priority_results.columns if c.startswith('wf_')]
    final_df = knn_interpolate(priority_results, all_tracts, feature_cols, k=5)

    # -- Save final output --
    final_df.to_parquet(OUTPUT_PATH, index=False)

    # -- Report --
    log.info("=" * 70)
    log.info("FINAL REPORT")
    log.info("=" * 70)
    log.info(f"Total tracts: {len(final_df)}")
    log.info(f"Weather features: {len(feature_cols)}")

    for col in sorted(feature_cols):
        cov = final_df[col].notna().mean() * 100
        mean_val = final_df[col].mean()
        log.info(f"  {col}: {cov:.1f}% coverage, mean={mean_val:.4f}")

    known_set = set(priority_results['GEOID'].astype(str))
    pri_mask = final_df['GEOID'].astype(str).isin(known_set)
    interp_mask = ~pri_mask
    log.info(f"\n  Priority tracts (API): {pri_mask.sum()}")
    log.info(f"  Interpolated tracts: {interp_mask.sum()}")
    if interp_mask.sum() > 0:
        interp_cov = final_df.loc[interp_mask, feature_cols].notna().mean().mean() * 100
        log.info(f"  Interpolated feature coverage: {interp_cov:.1f}%")

    log.info(f"\nSaved to: {OUTPUT_PATH}")
    return final_df


if __name__ == '__main__':
    main()
