import numpy as np
import torch
import pandas as pd
from itertools import product
import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    RenyiELBO,
    Trace_ELBO,
    TraceGraph_ELBO,
)
from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLowRankMultivariateNormal,
)
from pyro.optim import ClippedAdam
from tqdm import tqdm

def prepare_model_data(
    df,
    static_cols,
    dynamic_cols,
    seasonal_cols,
    time_trend_cols,
    temporal_cols,
    spatial_cols,
    device,
    means = None,
    stds = None,
    ward_idx_map=None
    ):
    """
    Prepares and standardizes model input data from a DataFrame for training.
    This function extracts features and target variables, checks column validity,
    standardizes features, and returns tensors for model consumption.

    Args:
        df: Input pandas DataFrame containing all features and target columns.
        static_cols: List of column names for static covariates.
        dynamic_cols: List of column names for dynamic covariates.
        seasonal_cols: List of column names for seasonal covariates.
        time_trend_cols: List of column names for time-trend covariates.
        temporal_cols: List of column names for other temporal covariates.
        spatial_cols: List of column names for spatial covariates.
        device: PyTorch device to place tensors on.
        means: Optional precomputed means for standardization.
        stds: Optional precomputed standard deviations for standardization.

    Returns:
        dict: Dictionary containing tensors for each covariate group, the target, and the means and stds used for scaling.

    Raises:
        ValueError: If any provided column names are missing from the DataFrame.
    """
    # 1) Extract target & LSOA index
    idx = df["occupation_idx"].astype(np.int16).values
    if ward_idx_map is None:
        ward_idx_map = (
            df[["occupation_idx", "ward_idx"]]
            .drop_duplicates("occupation_idx")
            .sort_values("occupation_idx")
            .set_index("occupation_idx")
            ["ward_idx"]
            .astype(np.int16)
            .values
        )
    max_idx = idx.max()
    if ward_idx_map.shape[0] <= max_idx:
        raise ValueError(
            "ward_idx_map too short for occupation indices: "
            f"len={ward_idx_map.shape[0]}, max_idx={max_idx}"
        )
    y = df["burglaries"].astype(np.int16).values

    # 2) Full list of feature columns (exclude target & index)
    feat_cols = [c for c in df.columns if c not in (
       "burglaries", "occupation_idx", "ward_idx")]

    # 3) Sanity check: ensure all provided col names appear in feat_cols
    for col_list, name in [
        (static_cols, "static_cols"),
        (dynamic_cols, "dynamic_cols"),
        (seasonal_cols, "seasonal_cols"),
        (time_trend_cols, "time_trend_cols"),
        (temporal_cols, "temporal_cols"),
        (spatial_cols, "spatial_cols"),
    ]:
        if missing := set(col_list) - set(feat_cols):
            raise ValueError(
                f"{name} contains columns not in DataFrame: {missing}")

    # 4) Map each feature name to its column index
    col_to_idx = {col: i for i, col in enumerate(feat_cols)}

    # 5) Compute integer index lists for each group
    static_idx = [col_to_idx[c] for c in static_cols]
    dynamic_idx = [col_to_idx[c] for c in dynamic_cols]
    seasonal_idx = [col_to_idx[c] for c in seasonal_cols]
    time_trend_idx = [col_to_idx[c] for c in time_trend_cols]
    temporal_idx = [col_to_idx[c] for c in temporal_cols]
    spatial_idx = [col_to_idx[c] for c in spatial_cols]

    # TODO: this assumes that we standardize all features together
    # FIXME: the scalling that we want to apply
    X = df[feat_cols].values.astype(np.float32)
    means = X.mean(axis=0) if means is None else means
    stds = X.std(axis=0) if stds is None else stds
    X = (X - means) / stds

    return {
        "occupation_idx": torch.tensor(idx, dtype=torch.long, device=device),
        "ward_idx": torch.tensor(ward_idx_map, dtype=torch.long, device=device),
        "X_static": torch.tensor(X[:, static_idx], dtype=torch.float32, device=device),
        "X_dynamic": torch.tensor(X[:, dynamic_idx], dtype=torch.float32, device=device),
        "X_seasonal": torch.tensor(X[:, seasonal_idx], dtype=torch.float32, device=device),
        "X_time_trend": torch.tensor(X[:, time_trend_idx], dtype=torch.float32, device=device),
        "X_temporal": torch.tensor(X[:, temporal_idx], dtype=torch.float32, device=device),
        "X_spatial": torch.tensor(X[:, spatial_idx], dtype=torch.float32, device=device),
        "y": torch.tensor(y, dtype=torch.int16, device=device),
        "means": means,
        "stds": stds,
    }


def create_learner(
    model_function,
    guide_type: str = "diag",       # “diag”, “lowrank”, or “iaf”
    guide_rank: int = 10,  # rank for low-rank guide
    lr: float = 1e-3,
    elbo_type: str = "trace",       # “trace”, “graph”, “renyi”, or “jit”
    renyi_alpha: float = 0.5,
    num_particles: int = 1
):
    # 1) Choose guide
    if guide_type == "diag":
        guide = AutoDiagonalNormal(model_function)
    elif guide_type == "lowrank":
        guide = AutoLowRankMultivariateNormal(
            model_function, rank=guide_rank)  # e.g. rank=10
    elif guide_type == "iaf":
        guide = AutoIAFNormal(model_function)
    else:
        raise ValueError("Unknown guide_type")

    # 2) Choose optimizer
    optimizer = ClippedAdam({"lr": lr})

    # 3) Choose loss (ELBO variant)
    if elbo_type == "trace":
        loss = Trace_ELBO(num_particles=num_particles)
    elif elbo_type == "graph":
        loss = TraceGraph_ELBO(num_particles=num_particles)
    elif elbo_type == "renyi":
        loss = RenyiELBO(alpha=renyi_alpha, num_particles=num_particles)
    elif elbo_type == "jit":
        loss = JitTrace_ELBO(num_particles=num_particles)
    else:
        raise ValueError("Unknown elbo_type")

    return SVI(model_function, guide, optimizer, loss=loss)

def train_model(data, svi, num_steps):
    losses = []
    occupation_idx = data["occupation_idx"]        # (N,)
    ward_idx = data["ward_idx"]
    X_static = data["X_static"]        # (N, n_static)
    X_dynamic = data["X_dynamic"]       # (N, n_dynamic)
    X_seasonal = data["X_seasonal"]      # (N, n_seasonal)
    X_time_trend = data["X_time_trend"]    # (N, n_time_trend)
    X_temporal = data["X_temporal"]      # (N, n_temporal)
    X_spatial = data["X_spatial"]       # (N, n_spatial)
    y = data["y"]               # (N,)

    # Training loop
    for _ in tqdm(range(num_steps), desc="Training SVI"):
        loss = svi.step(
            occupation_idx,
            ward_idx,
            X_static,
            X_dynamic,
            X_seasonal,
            X_time_trend,
            X_temporal,
            X_spatial,
            y
        )
        losses.append(loss)
    return svi, losses, data["means"], data["stds"]

def grid_search(
    model_function,
    train_df,
    val_df,
    static_cols,
    dynamic_cols,
    seasonal_cols,
    time_trend_cols,
    temporal_cols,
    spatial_cols,
    device,
    param_grid,
    ward_idx_map=None,
    num_steps: int = 200,
):
    """Simple hyperparameter grid search.

    Args:
        model_function: Pyro model to train.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        *_cols: Feature column names for ``prepare_model_data``.
        device: Torch device.
        param_grid: Dict with keys ``"lr"`` and ``"guide_type"``.
        ward_idx_map: Optional LSOA-to-ward mapping. If ``None``, a mapping is
            built from ``train_df`` and reused for validation.
        num_steps: Number of SVI training steps per run.

    Returns:
        pandas.DataFrame summarising the validation loss for each setting.
    """

    results = []
    for lr, guide_type in product(param_grid["lr"], param_grid["guide_type"]):
        pyro.clear_param_store()
        svi = create_learner(model_function, lr=lr, guide_type=guide_type)

        train_ds = prepare_model_data(
            train_df,
            static_cols,
            dynamic_cols,
            seasonal_cols,
            time_trend_cols,
            temporal_cols,
            spatial_cols,
            device,
            ward_idx_map=ward_idx_map,
        )
        ward_idx_current = train_ds["ward_idx"].cpu().numpy()
        svi, _, means, stds = train_model(train_ds, svi, num_steps=num_steps)

        val_ds = prepare_model_data(
            val_df,
            static_cols,
            dynamic_cols,
            seasonal_cols,
            time_trend_cols,
            temporal_cols,
            spatial_cols,
            device,
            means,
            stds,
            ward_idx_current,
        )

        val_loss = svi.evaluate_loss(
            val_ds["occupation_idx"],
            val_ds["ward_idx"],
            val_ds["X_static"],
            val_ds["X_dynamic"],
            val_ds["X_seasonal"],
            val_ds["X_time_trend"],
            val_ds["X_temporal"],
            val_ds["X_spatial"],
            val_ds["y"],
        )
        results.append({"lr": lr, "guide_type": guide_type, "val_loss": val_loss})

    return pd.DataFrame(results).sort_values("val_loss").reset_index(drop=True)


