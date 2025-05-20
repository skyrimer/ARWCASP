import pandas as pd
import numpy as np
import pymc as pm
import geopandas as gpd

def load_and_prepare_data(parquet_path: str, n_lsoas: int = 20):
    # 1) Load & force 'period' into monthly Period dtype
    crimes = gpd.read_parquet(parquet_path)
    crimes['period'] = (
        pd.to_datetime(crimes['period'].astype(str))
          .dt.to_period('M')
    )

    # 2) Count by (period, LSOA), silencing the FutureWarning
    counts = (
        crimes
        .groupby(['period', 'LSOA code'], observed=False)
        .size()
        .reset_index(name='counts')
        .sort_values(['LSOA code', 'period'])
    )

    # 3) Seasonal & lag features
    counts['month']     = counts['period'].dt.month
    counts['month_sin'] = np.sin(2 * np.pi * counts['month'] / 12)
    counts['month_cos'] = np.cos(2 * np.pi * counts['month'] / 12)
    counts['lag_1m']    = counts.groupby('LSOA code', observed=False)['counts'].shift(1)
    counts['lag_12m']   = counts.groupby('LSOA code', observed=False)['counts'].shift(12)

    # 4) Drop any rows that lack full lags
    counts = counts.dropna(subset=['lag_1m', 'lag_12m']).reset_index(drop=True)

    # 5) Keep only the top-n LSOAs by frequency
    top_lsoas = counts['LSOA code'].value_counts().index[:n_lsoas]
    df_small  = counts[counts['LSOA code'].isin(top_lsoas)].copy()
    df_small['lsoa_idx'], lsoa_categories = pd.factorize(df_small['LSOA code'])

    return df_small, lsoa_categories

def build_and_fit_model(df_model: pd.DataFrame, n_lsoa: int, seed: int = 42):
    with pm.Model() as model:
        # Data containers
        lsoa_idx_dc  = pm.Data('lsoa_idx',  df_model['lsoa_idx'].values)
        lag1_dc      = pm.Data('lag_1m',    df_model['lag_1m'].values)
        lag12_dc     = pm.Data('lag_12m',   df_model['lag_12m'].values)
        sin_dc       = pm.Data('month_sin', df_model['month_sin'].values)
        cos_dc       = pm.Data('month_cos', df_model['month_cos'].values)

        # Hierarchical intercepts
        mu_a    = pm.Normal('mu_a',    mu=0, sigma=10)
        sigma_a = pm.HalfNormal('sigma_a', sigma=10)
        a       = pm.Normal('a',       mu=mu_a, sigma=sigma_a, shape=n_lsoa)

        # Fixed-effect slopes
        b1  = pm.Normal('beta_lag1',  mu=0, sigma=10)
        b12 = pm.Normal('beta_lag12', mu=0, sigma=10)
        bs  = pm.Normal('beta_sin',   mu=0, sigma=10)
        bc  = pm.Normal('beta_cos',   mu=0, sigma=10)

        # Linear predictor & Poisson likelihood
        log_mu = (
            a[lsoa_idx_dc] +
            b1  * lag1_dc +
            b12 * lag12_dc +
            bs  * sin_dc +
            bc  * cos_dc
        )
        mu = pm.Deterministic('mu', pm.math.exp(log_mu))
        pm.Poisson('y_obs', mu=mu, observed=df_model['counts'].values)

        # ADVI + posterior samples (no InferenceData)
        approx = pm.fit(10_000, method='advi', random_seed=seed)
        trace  = approx.sample(1_000, random_seed=seed, return_inferencedata=False)

    return model, trace

def forecast_next_month(df_model: pd.DataFrame,
                        lsoa_categories: np.ndarray,
                        trace,
                        seed: int = 42):
    # 1) Grab each LSOA’s final row by dropping duplicates
    last = (
        df_model
        .sort_values('period')
        .drop_duplicates('LSOA code', keep='last')
        .reset_index(drop=True)
    )

    # 2) Build next-month covariates
    nxt = last.copy()
    nxt['period']    = nxt['period'] + 1
    nxt['month']     = nxt['period'].dt.month
    nxt['month_sin'] = np.sin(2 * np.pi * nxt['month'] / 12)
    nxt['month_cos'] = np.cos(2 * np.pi * nxt['month'] / 12)
    nxt['lag_1m']    = last['counts'].values
    nxt['lag_12m']   = last['lag_12m'].values

    # 3) Preserve the exact integer indices
    idx_new = last['lsoa_idx'].values.astype(int)

    # 4) Extract posterior arrays
    a_samps   = trace.get_values('a')             # shape (nsamps, n_lsoa)
    b1_samps  = trace.get_values('beta_lag1')     # shape (nsamps,)
    b12_samps = trace.get_values('beta_lag12')
    bs_samps  = trace.get_values('beta_sin')
    bc_samps  = trace.get_values('beta_cos')

    # 5) Compute μ for each posterior sample & each LSOA
    nsamps = a_samps.shape[0]
    mu_samps = np.exp(
        a_samps[:, idx_new] +
        b1_samps[:, None]  * nxt['lag_1m'].values[None, :] +
        b12_samps[:, None] * nxt['lag_12m'].values[None, :] +
        bs_samps[:, None]  * nxt['month_sin'].values[None, :] +
        bc_samps[:, None]  * nxt['month_cos'].values[None, :]
    )

    # 6) Draw Poisson forecasts
    rng   = np.random.default_rng(seed)
    draws = rng.poisson(mu_samps)

    # 7) Summarize into a DataFrame aligned by row
    return pd.DataFrame({
        'LSOA code':   nxt['LSOA code'].values,
        'period':      nxt['period'].values,
        'pred_mean':   draws.mean(axis=0),
        'pred_hdi_3%': np.percentile(draws, 3,  axis=0),
        'pred_hdi_97%':np.percentile(draws, 97, axis=0)
    })

def main():
    PARQUET_PATH = 'processed_data/street.parquet'
    N_LSOAS      = 20
    SEED         = 42

    df_model, cats = load_and_prepare_data(PARQUET_PATH, n_lsoas=N_LSOAS)
    model, trace   = build_and_fit_model(df_model,
                                         n_lsoa=df_model['lsoa_idx'].nunique(),
                                         seed=SEED)
    preds = forecast_next_month(df_model, cats, trace, seed=SEED)

    print(preds)

if __name__ == '__main__':
    main()
