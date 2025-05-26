import pandas as pd
import numpy as np
import geopandas as gpd
import jax.numpy as jnp
from jax import random, device_put

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_feasible
from numpyro.optim import Adam

# 1) Load & prepare data with geometry and spatial lag
def load_and_prepare(parquet_path: str, n_lsoas: int = 5):
    # Read GeoDataFrame
    gdf = gpd.read_parquet(parquet_path)
    # Clean and rename
    gdf = gdf.rename(columns={
        'LSOA code (2021)': 'LSOA code',
        'date': 'period',
        'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)': 'imd_rank',
        'Burglaries amount': 'burglaries'
    })
    # Unique geometry per LSOA
    geo_df = gdf[['LSOA code','geometry']].drop_duplicates('LSOA code').reset_index(drop=True)
    # Build adjacency mapping via touches
    geo_df['geometry'] = geo_df['geometry'].buffer(0)
    neighbors = {}
    for i, row in geo_df.iterrows():
        code = row['LSOA code']
        geom = row['geometry']
        touches = geo_df[geo_df.geometry.touches(geom)]['LSOA code'].tolist()
        neighbors[code] = touches
    # Prepare raw DataFrame
    df = (gdf.assign(period=lambda d: pd.to_datetime(d['period'].astype(str)).dt.to_period('M'))
            [['period','LSOA code','imd_rank','burglaries'] +
             [c for c in gdf.columns if c.startswith('poi_')]]
    )
    # Aggregate per LSOA×month
    poi_cols = [c for c in df.columns if c.startswith('poi_')]
    agg_def = {'counts':('burglaries','sum'), 'imd_rank':('imd_rank','first')}
    for col in poi_cols:
        agg_def[col] = (col,'first')
    grouped = df.groupby(['period','LSOA code'], as_index=False).agg(**agg_def)
    # Seasonal features
    grouped['month'] = grouped['period'].dt.month
    grouped['sin']   = np.sin(2*np.pi*grouped['month']/12)
    grouped['cos']   = np.cos(2*np.pi*grouped['month']/12)
    # Lag features
    grouped['lag1']  = grouped.groupby('LSOA code')['counts'].shift(1)
    grouped['lag12'] = grouped.groupby('LSOA code')['counts'].shift(12)
    grouped = grouped.dropna(subset=['lag1','lag12']).reset_index(drop=True)
    # Spatial lag: neighbor average lag1
    adj_rows = []
    for code, neighs in neighbors.items():
        for nbr in neighs:
            adj_rows.append({'LSOA code':code, 'neighbor':nbr})
    adj_df = pd.DataFrame(adj_rows)
    lag_df = grouped[['period','LSOA code','lag1']].rename(columns={'LSOA code':'neighbor','lag1':'lag1_nbr'})
    spatial = (adj_df.merge(lag_df, on='neighbor')
                     .groupby(['period','LSOA code'])['lag1_nbr']
                     .mean().reset_index())
    grouped = grouped.merge(spatial, on=['period','LSOA code'], how='left')
    grouped['lag1_nbr'] = grouped['lag1_nbr'].fillna(grouped['lag1'])
    # Standardize IMD and counts
    grouped['imd_s']  = (grouped['imd_rank'] - grouped['imd_rank'].mean())/grouped['imd_rank'].std()
    grouped['burg_s'] = np.log1p(grouped['counts'])
    grouped['burg_s'] = (grouped['burg_s'] - grouped['burg_s'].mean())/grouped['burg_s'].std()
    # Standardize POIs across LSOAs
    poi_s_cols = []
    for col in poi_cols:
        s_col = f"{col}_s"
        grouped[s_col] = np.log1p(grouped[col])
        grouped[s_col] = (grouped[s_col] - grouped[s_col].mean())/grouped[s_col].std()
        poi_s_cols.append(s_col)
    # Select top N LSOAs by total burglaries
    totals = grouped.groupby('LSOA code')['counts'].sum()
    top = totals.nlargest(n_lsoas).index
    df_sel = (grouped[grouped['LSOA code'].isin(top)]
                    .sort_values(['LSOA code','period'])
                    .reset_index(drop=True))
    # Encode indices
    df_sel['period'] = df_sel['period'].dt.to_timestamp()
    df_sel['lsoa_idx'], cats = pd.factorize(df_sel['LSOA code'])
    return df_sel, cats, poi_s_cols

# 2) Hierarchical NB with tightened priors + spatial lag
def hierarchical_nb(lsoa_idx, lag1, lag12, sin, cos,
                    imd_s, burg_s, lag1_nbr, poi_matrix,
                    counts=None, n_lsoas=None, n_pois=None):
    # Intercept ~ N(log mean, 0.25)
    mu_init = np.log(12.8)
    mu_a    = numpyro.sample('mu_a',    dist.Normal(mu_init, 0.25))
    sigma_a = numpyro.sample('sigma_a', dist.HalfNormal(0.25))
    a       = numpyro.sample('a',       dist.Normal(mu_a, sigma_a).expand([n_lsoas]))
    # Slopes ~ N(0, 0.25)
    b1    = numpyro.sample('beta_lag1',  dist.Normal(0., 0.25))
    b12   = numpyro.sample('beta_lag12', dist.Normal(0., 0.25))
    bs    = numpyro.sample('beta_sin',   dist.Normal(0., 0.25))
    bc    = numpyro.sample('beta_cos',   dist.Normal(0., 0.25))
    bim   = numpyro.sample('beta_imd',   dist.Normal(0., 0.25))
    bbu   = numpyro.sample('beta_burg',  dist.Normal(0., 0.25))
    bnbr  = numpyro.sample('beta_neigh', dist.Normal(0., 0.25))
    # POI shrinkage ~ HalfNormal(0.25)
    tau_poi   = numpyro.sample('tau_poi',  dist.HalfNormal(0.25))
    beta_poi  = numpyro.sample('beta_poi',  dist.Normal(0., tau_poi).expand([n_pois]))
    # Overdispersion ~ Gamma(2,1)
    phi = numpyro.sample('phi', dist.Gamma(2., 1.))
    # Linear predictor
    log_mu = (
        a[lsoa_idx]
      + b1    * lag1
      + b12   * lag12
      + bs    * sin
      + bc    * cos
      + bim   * imd_s
      + bbu   * burg_s
      + bnbr  * lag1_nbr
      + jnp.dot(poi_matrix, beta_poi)
    )
    mu = jnp.exp(log_mu)
    numpyro.sample('y_obs', dist.NegativeBinomial2(phi, mu), obs=counts)

# 3) SVI fit with lowered LR only + prior check
def run_svi(df, poi_s_cols, num_steps=3000, lr=5e-5, seed=0):
    data = dict(
        lsoa_idx    = device_put(jnp.array(df['lsoa_idx'])),
        lag1        = device_put(jnp.array(df['lag1'])),
        lag12       = device_put(jnp.array(df['lag12'])),
        sin         = device_put(jnp.array(df['sin'])),
        cos         = device_put(jnp.array(df['cos'])),
        imd_s       = device_put(jnp.array(df['imd_s'])),
        burg_s      = device_put(jnp.array(df['burg_s'])),
        lag1_nbr    = device_put(jnp.array(df['lag1_nbr'])),
        poi_matrix  = device_put(jnp.array(df[poi_s_cols].values)),
        counts      = device_put(jnp.array(df['counts'])),
        n_lsoas     = df['lsoa_idx'].nunique(),
        n_pois      = len(poi_s_cols)
    )
    # Prior predictive
    prior = Predictive(hierarchical_nb, params=None, num_samples=200)
    pd_draws = prior(random.PRNGKey(seed), **data)['y_obs']
    pd_cov   = ((data['counts'] >= np.percentile(pd_draws,3,axis=0)) &
                (data['counts'] <= np.percentile(pd_draws,97,axis=0))).mean()
    print(f"Prior coverage (94% CI): {pd_cov:.2%}")
    # Inference
    guide = AutoNormal(hierarchical_nb, init_loc_fn=init_to_feasible)
    svi   = SVI(hierarchical_nb, guide, Adam(lr), Trace_ELBO())
    rng   = random.PRNGKey(seed)
    state = svi.init(rng, **data)
    for i in range(num_steps):
        rng, _ = random.split(rng)
        state, loss = svi.update(state, **data)
        if i % 500 == 0:
            print(f"Step {i:>4d} loss = {loss:.1f}")
    params = svi.get_params(state)
    return params, guide, data

# 4) Forecast & posterior checks
def forecast_next_month(df, poi_s_cols, params, guide, cats, seed=1, num_samples=100):
    last = df.drop_duplicates('LSOA code', keep='last').reset_index(drop=True)
    # compute next period
    nxt = last.copy()
    nxt['period'] = nxt['period'] + pd.offsets.MonthBegin(1)
    # compute neighbor lag1 for next
    # static neighbor map from load... use cats.index to map
    neighbors = {}
    # (reconstruct or reuse mapping) -- here assume global 'neighbors' dict available
    for code, neighs in neighbors.items():
        pass  # placeholder
    # copy features
    for col in ['lag1','lag12','sin','cos','lsoa_idx','imd_s','burg_s','lag1_nbr'] + poi_s_cols:
        nxt[col] = last[col]
    fdata = dict(
        lsoa_idx   = device_put(jnp.array(nxt['lsoa_idx'])),
        lag1       = device_put(jnp.array(nxt['lag1'])),
        lag12      = device_put(jnp.array(nxt['lag12'])),
        sin        = device_put(jnp.array(nxt['sin'])),
        cos        = device_put(jnp.array(nxt['cos'])),
        imd_s      = device_put(jnp.array(nxt['imd_s'])),
        burg_s     = device_put(jnp.array(nxt['burg_s'])),
        lag1_nbr   = device_put(jnp.array(nxt['lag1_nbr'])),
        poi_matrix = device_put(jnp.array(nxt[poi_s_cols].values)),
        n_lsoas    = cats.shape[0],
        n_pois     = len(poi_s_cols)
    )
    post = Predictive(hierarchical_nb, guide=guide, params=params,
                      num_samples=num_samples)(random.PRNGKey(seed), **fdata)['y_obs']
    return pd.DataFrame({
        'LSOA code':    nxt['LSOA code'],
        'period_start': nxt['period'],
        'pred_mean':    post.mean(axis=0),
        'pred_hdi_3%':  np.percentile(post,3,axis=0),
        'pred_hdi_97%': np.percentile(post,97,axis=0)
    })

# 5) Main entry
if __name__ == "__main__":
    df, cats, poi_s_cols = load_and_prepare("merged_data.parquet", n_lsoas=5)
    print("Feature summary:")
    print(df[['counts','imd_s','burg_s','lag1_nbr'] + poi_s_cols].describe())
    params, guide, data_train = run_svi(df, poi_s_cols)
    print("Forecast next month:")
    print(forecast_next_month(df, poi_s_cols, params, guide, cats))
    post = Predictive(hierarchical_nb, guide=guide, params=params,
                      num_samples=200)(random.PRNGKey(1), **data_train)['y_obs']
    lower = np.percentile(post,3,axis=0)
    upper = np.percentile(post,97,axis=0)
    cov   = ((data_train['counts']>=lower)&(data_train['counts']<=upper)).mean()
    print(f"Posterior coverage (94% CI): {cov:.2%}")
