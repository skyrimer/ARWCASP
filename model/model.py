import pyro
import torch
import pyro.distributions as dist


def burglary_model(
    occupation_idx,
    ward_idx_map,
    X_static,
    X_dynamic,
    X_seasonal,
    X_time_trend,
    X_temporal,
    X_spatial,
    y=None,
    use_subsample=True,
):
    """
    Hierarchical Poisson model with five distinct groups of covariates:
      - X_static:   covariates constant per LSOA
      - X_dynamic:  covariates varying per LSOA & time
      - X_seasonal: seasonal covariates (e.g., month dummies)
      - X_time_trend: global time-trend covariates (e.g., linear time index)
      - X_temporal: other temporal covariates (e.g., day-of-week)
      - X_spatial:  spatial covariates (e.g., neighbors' lagged values)
      - y: observed target values (optional, for training)
      - use_subsample: if True, use a subsample of 2048 data points for training
    This model uses a hierarchical structure with random intercepts for each LSOA,
    and coefficients for each group of covariates. The model predicts the number of burglaries
    using a Poisson distribution, with the linear predictor being the sum of the LSOA intercept
    and the weighted contributions of each covariate group.
    """

    # Dimensions
    n_lsoas    = ward_idx_map.shape[0]  # total number of LSOAs
    assert ward_idx_map.max().item() < n_lsoas, (
        "ward_idx_map must include an entry for each LSOA index")
    # Ensure feature tensors are 2D
    if X_static.dim() == 1:
        X_static = X_static.unsqueeze(1)
    if X_dynamic.dim() == 1:
        X_dynamic = X_dynamic.unsqueeze(1)
    if X_seasonal.dim() == 1:
        X_seasonal = X_seasonal.unsqueeze(1)
    if X_time_trend.dim() == 1:
        X_time_trend = X_time_trend.unsqueeze(1)
    if X_temporal.dim() == 1:
        X_temporal = X_temporal.unsqueeze(1)
    if X_spatial.dim() == 1:
        X_spatial = X_spatial.unsqueeze(1)

    n_wards    = ward_idx_map.max().item() + 1     # total number of wards
    n_static   = X_static.shape[1]                 # number of static covariates 
    n_dynamic  = X_dynamic.shape[1]                # number of dynamic covariates 
    n_seasonal = X_seasonal.shape[1]               # number of seasonal covariates 
    n_time_tr  = X_time_trend.shape[1]             # number of time-trend covariates 
    n_temporal = X_temporal.shape[1]               # number of other temporal covariates
    n_spatial  = X_spatial.shape[1]                # number of spatial covariates 

    # Hierarchical intercept (random effect by LSOA)
    mu_w = pyro.sample("mu_w", dist.Normal(0.0, 1.0))
    sigma_w = pyro.sample("sigma_w", dist.HalfNormal(1.0))
    with pyro.plate("wards", n_wards):
        w = pyro.sample("w", dist.Normal(mu_w, sigma_w))

    sigma_a = pyro.sample("sigma_a", dist.HalfNormal(1.0))
    with pyro.plate("ls", n_lsoas):
        a = pyro.sample("a", dist.Normal(w[ward_idx_map], sigma_a))
    # Coefficients for each covariate group
    if n_static:
        b_static = pyro.sample(
            "b_static", dist.Normal(0.0, 1.0).expand([n_static]).to_event(1)
        )
    else:
        b_static = torch.tensor([], device=X_static.device)

    if n_dynamic:
        b_dynamic = pyro.sample(
            "b_dynamic", dist.Normal(0.0, 1.0).expand([n_dynamic]).to_event(1)
        )
    else:
        b_dynamic = torch.tensor([], device=X_dynamic.device)

    if n_seasonal:
        b_seasonal = pyro.sample(
            "b_seasonal", dist.Normal(0.0, 1.0).expand([n_seasonal]).to_event(1)
        )
    else:
        b_seasonal = torch.tensor([], device=X_seasonal.device)

    if n_time_tr:
        b_time_tr = pyro.sample(
            "b_time_tr", dist.Normal(0.0, 1.0).expand([n_time_tr]).to_event(1)
        )
    else:
        b_time_tr = torch.tensor([], device=X_time_trend.device)

    if n_temporal:
        b_temporal = pyro.sample(
            "b_temporal", dist.Normal(0.0, 1.0).expand([n_temporal]).to_event(1)
        )
    else:
        b_temporal = torch.tensor([], device=X_temporal.device)

    if n_spatial:
        b_spatial = pyro.sample(
            "b_spatial", dist.Normal(0.0, 1.0).expand([n_spatial]).to_event(1)
        )
    else:
        b_spatial = torch.tensor([], device=X_spatial.device)


    N = occupation_idx.shape[0]

    # Subsampled inference
    # If use_subsample is True, we will use a subsample of 2048 data points
    if use_subsample and y is not None:
        with pyro.plate("data", size=N, subsample_size=2048) as i:
            # create linear predictor eta for each data point
            # using the LSOA intercept and covariate coefficients
            eta = a[occupation_idx[i]] \
                + (X_static[i] * b_static).sum(-1) \
                + (X_dynamic[i] * b_dynamic).sum(-1) \
                + (X_seasonal[i] * b_seasonal).sum(-1) \
                + (X_time_trend[i] * b_time_tr).sum(-1) \
                + (X_temporal[i] * b_temporal).sum(-1) \
                + (X_spatial[i] * b_spatial).sum(-1) 
            # Ensure numerical stability by clamping eta
            # to a reasonable range before exponentiation
            mu  = torch.exp(eta.clamp(-4, 4))
            # Sample from Poisson distribution with observed data
            pyro.deterministic("lam", mu) 
            pyro.sample("obs", dist.Poisson(mu), obs=y[i])
    else:
        with pyro.plate("data", N):
            eta = a[occupation_idx] \
                + (X_static     * b_static).sum(-1) \
                + (X_dynamic    * b_dynamic).sum(-1) \
                + (X_seasonal   * b_seasonal).sum(-1) \
                + (X_time_trend * b_time_tr).sum(-1) \
                + (X_temporal * b_temporal).sum(-1) \
                + (X_spatial * b_spatial).sum(-1)  
            mu  = torch.exp(eta.clamp(-4, 4))
            pyro.deterministic("lam", mu) 
            pyro.sample("obs", dist.Poisson(mu), obs=y if y is not None else None)