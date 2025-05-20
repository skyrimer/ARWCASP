import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random, device_put

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

# 1) Load & prep only 5 LSOAs
def load_and_prepare(parquet_path: str, n_lsoas: int = 5):
    df = (pd.read_parquet(parquet_path)
          .assign(period=lambda d: pd.to_datetime(d['period'].astype(str)).dt.to_period('M'))
          .groupby(['period','LSOA code'], observed=False)
          .size().reset_index(name='counts'))
    df['month'] = df['period'].dt.month
    df['sin']   = np.sin(2*np.pi*df['month']/12)
    df['cos']   = np.cos(2*np.pi*df['month']/12)
    df['lag1']  = df.groupby('LSOA code', observed=False)['counts'].shift(1)
    df['lag12'] = df.groupby('LSOA code', observed=False)['counts'].shift(12)
    df = df.dropna(subset=['lag1','lag12'])
    top = df['LSOA code'].value_counts().index[:n_lsoas]
    df = df[df['LSOA code'].isin(top)].sort_values(['LSOA code','period']).reset_index(drop=True)
    df['lsoa_idx'], cats = pd.factorize(df['LSOA code'])
    return df, cats

# 2) NumPyro model
def hierarchical_poisson(lsoa_idx, lag1, lag12, sin, cos, counts=None, n_lsoa=None):
    mu_a    = numpyro.sample("mu_a", dist.Normal(0., 10.))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(10.))
    a       = numpyro.sample("a", dist.Normal(mu_a, sigma_a).expand([n_lsoa]))
    b1      = numpyro.sample("beta_lag1",  dist.Normal(0.,10.))
    b12     = numpyro.sample("beta_lag12", dist.Normal(0.,10.))
    bs      = numpyro.sample("beta_sin",   dist.Normal(0.,10.))
    bc      = numpyro.sample("beta_cos",   dist.Normal(0.,10.))
    log_mu  = a[lsoa_idx] + b1*lag1 + b12*lag12 + bs*sin + bc*cos
    numpyro.sample("y_obs", dist.Poisson(jnp.exp(log_mu)), obs=counts)


# 3) Fit with only 500 ADVI steps
def run_svi(df, num_steps=500, lr=1e-2, seed=0):
    data = {k: device_put(jnp.array(df[k].values))
            for k in ['lsoa_idx','lag1','lag12','sin','cos','counts']}
    data['n_lsoa'] = df['lsoa_idx'].nunique()
    guide    = AutoNormal(hierarchical_poisson)
    optimizer= Adam(lr)
    svi      = SVI(hierarchical_poisson, guide, optimizer, loss=Trace_ELBO())
    rng_key  = random.PRNGKey(seed)
    state    = svi.init(rng_key, **data)
    for i in range(num_steps):
        rng_key, _ = random.split(rng_key)
        state, loss = svi.update(state, **data)
        if i % 100 == 0:
            print(f"Step {i:>4d} loss = {loss:.1f}")
    params = svi.get_params(state)
    return params, guide

# 4) Forecast with 100 posterior draws
def forecast_next_month(df, params, guide, cats, seed=1, num_samples=100):
    last = (df.sort_values('period')
              .drop_duplicates('LSOA code', keep='last')
              .reset_index(drop=True))
    nxt = last.copy()
    nxt['period'] = nxt['period'] + 1
    for col in ['lag1','lag12','sin','cos','lsoa_idx']:
        nxt[col] = last[col].values
    fdata = {k: device_put(jnp.array(nxt[k].values))
             for k in ['lsoa_idx','lag1','lag12','sin','cos']}
    fdata['n_lsoa'] = cats.shape[0]
    predictive = Predictive(hierarchical_poisson, guide=guide,
                            params=params, num_samples=num_samples)
    post = predictive(random.PRNGKey(seed), **fdata)
    draws = np.array(post['y_obs'])
    return pd.DataFrame({
        'LSOA code':    nxt['LSOA code'],
        'period_start': nxt['period'].dt.to_timestamp(),
        'pred_mean':    draws.mean(axis=0),
        'pred_hdi_3%':  np.percentile(draws, 3, axis=0),
        'pred_hdi_97%': np.percentile(draws,97,axis=0)
    })

# 5) Run it
def main():
    df, cats     = load_and_prepare("processed_data/street.parquet", n_lsoas=5_000)
    params, guide= run_svi(df, num_steps=500, lr=1e-2, seed=0)
    preds        = forecast_next_month(df, params, guide, cats,
                                       seed=1, num_samples=100)
    print(preds)
    preds.to_parquet("processed_data/street_forecast2.parquet",
                   index=False, compression='gzip')

if __name__=="__main__":
    main()
