import pyro
import torch
import pyro.distributions as dist


def burglary_model(
    occupation_idx,
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
    n_lsoas    = occupation_idx.max().item() + 1   # total number of LSOAs 
    n_static   = X_static.shape[1]                 # number of static covariates 
    n_dynamic  = X_dynamic.shape[1]                # number of dynamic covariates 
    n_seasonal = X_seasonal.shape[1]               # number of seasonal covariates 
    n_time_tr  = X_time_trend.shape[1]             # number of time-trend covariates 
    n_temporal = X_temporal.shape[1]               # number of other temporal covariates
    n_spatial  = X_spatial.shape[1]                # number of spatial covariates 

    # Hierarchical intercept (random effect by LSOA)
    mu_a    = pyro.sample("mu_a",    dist.Normal(0., 1.0)) # overall mean of intercepts 
    sigma_a = pyro.sample("sigma_a", dist.HalfNormal(1.0)) # shared SD of LSOA intercepts 
    with pyro.plate("ls", n_lsoas): # plate over LSOAs 
        a = pyro.sample("a", dist.Normal(mu_a, sigma_a)) # random intercept for each LSOA 

    # Coefficients for each covariate group
    b_static    = pyro.sample("b_static",    dist.Normal(0., 1.0)
                              .expand([n_static]).to_event(1))
    b_dynamic   = pyro.sample("b_dynamic",   dist.Normal(0., 1.0)
                              .expand([n_dynamic]).to_event(1)) 
    b_seasonal  = pyro.sample("b_seasonal",  dist.Normal(0., 1.0)
                              .expand([n_seasonal]).to_event(1))
    b_time_tr   = pyro.sample("b_time_tr",   dist.Normal(0., 1.0)
                              .expand([n_time_tr]).to_event(1))
    b_temporal  = pyro.sample("b_temporal",  dist.Normal(0., 1.0)
                              .expand([n_temporal]).to_event(1))
    b_spatial  = pyro.sample("b_spatial",  dist.Normal(0., 1.0)
                              .expand([n_spatial]).to_event(1))


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
            mu  = torch.exp(eta.clamp(-10, 10))
            # Sample from Poisson distribution with observed data
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
            mu  = torch.exp(eta.clamp(-10, 10))
            pyro.sample("obs", dist.Poisson(mu), obs=y if y is not None else None)