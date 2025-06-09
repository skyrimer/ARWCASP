import copy
from typing import Dict, List

import pyro
from pyro.infer import Predictive
import torch
from .training import create_learner, prepare_model_data, train_model


def forward_feature_selection(
    model_function,
    train_df,
    val_df,
    candidate_features: Dict[str, List[str]],
    device,
    num_steps: int = 200,
    lr: float = 1e-3,
    guide_type: str = "diag",
    verbose: bool = False,
    max_features: int | None = None,
    print_progress: bool = True,
):
    """Greedy forward selection of features using validation loss.

    Parameters
    ----------
    model_function : callable
        Pyro model.
    train_df : pandas.DataFrame
        Training data.
    val_df : pandas.DataFrame
        Validation data.
    candidate_features : dict
        Mapping of feature group names ("static", "dynamic", "seasonal",
        "time_trend", "temporal", "spatial") to lists of candidate columns.
    device : torch.device
        Device used for tensors.
    num_steps : int, optional
        Number of SVI steps for each evaluation, by default 200.
    lr : float, optional
        Learning rate passed to :func:`create_learner`, by default 1e-3.
    guide_type : str, optional
        Guide type for :func:`create_learner`, by default "diag".
    verbose : bool, optional
        If True, prints progress information.
    max_features : int or None, optional
        Maximum total number of features to select. If ``None`` all
        improving features are added. Use this to trade accuracy for
        speed.

    Returns
    -------
    dict
        Selected features for each group.
    """
    torch.set_default_device(device)

    groups = [
        "static",
        "dynamic",
        "seasonal",
        "time_trend",
        "temporal",
        "spatial",
    ]

    selected = {g: [] for g in groups}
    remaining = {g: candidate_features.get(g, []).copy() for g in groups}

    def evaluate(current: Dict[str, List[str]]) -> float:
        pyro.clear_param_store()
        svi = create_learner(model_function, lr=lr, guide_type=guide_type)
        train_ds = prepare_model_data(
            train_df,
            current["static"],
            current["dynamic"],
            current["seasonal"],
            current["time_trend"],
            current["temporal"],
            current["spatial"],
            device,
        )
        svi, _, means, stds = train_model(train_ds, svi, num_steps=num_steps)
        val_ds = prepare_model_data(
            val_df,
            current["static"],
            current["dynamic"],
            current["seasonal"],
            current["time_trend"],
            current["temporal"],
            current["spatial"],
            device,
            means,
            stds,
            train_ds["ward_idx"].cpu().numpy(),
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
        return float(val_loss)

    best_loss = evaluate(selected)
    improved = True

    def total_selected() -> int:
        return sum(len(v) for v in selected.values())

    while improved:
        improved = False
        current_best = best_loss
        best_group = None
        best_feat = None

        for group in groups:
            for feat in list(remaining[group]):
                trial = {k: v.copy() for k, v in selected.items()}
                trial[group].append(feat)
                loss = evaluate(trial)
                if verbose:
                    print(f"{group}:{feat} -> {loss:.2f}")
                if loss < current_best:
                    current_best = loss
                    best_group = group
                    best_feat = feat

        if best_feat is not None:
            selected[best_group].append(best_feat)
            remaining[best_group].remove(best_feat)
            best_loss = current_best
            improved = True
        if print_progress:
            print(f"Selected {best_group}:{best_feat} -> {best_loss:.2f}")

        if max_features is not None and total_selected() >= max_features:
            break

    return selected

def correlation_feature_selection(
    df,
    candidate_features: Dict[str, List[str]],
    max_features: int | None = None,
    print_progress: bool = True,
):
    """Select features by ranking absolute correlation with the target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing training data including the ``burglaries`` column.
    candidate_features : dict
        Mapping of feature group names ("static", "dynamic", "seasonal",
        "time_trend", "temporal", "spatial") to lists of candidate columns.
    max_features : int or None, optional
        Maximum total number of features to retain. If ``None`` all candidates
        are returned ordered by correlation.
    print_progress : bool, optional
        If True, prints each selected feature with its correlation score.

    Returns
    -------
    dict
        Selected features for each group.
    """

    groups = [
        "static",
        "dynamic",
        "seasonal",
        "time_trend",
        "temporal",
        "spatial",
    ]

    scores: list[tuple[float, str, str]] = []
    for group in groups:
        feats = candidate_features.get(group, [])
        if not feats:
            continue
        sub = df[feats + ["burglaries"]]
        corr = sub.corr()["burglaries"].drop("burglaries").abs()
        corr = corr.dropna()
        for feat, val in corr.items():
            scores.append((float(val), group, feat))

    scores.sort(key=lambda x: x[0], reverse=True)

    selected = {g: [] for g in groups}
    count = 0
    for val, group, feat in scores:
        if max_features is not None and count >= max_features:
            break
        selected[group].append(feat)
        count += 1
        if print_progress:
            print(f"Selected {group}:{feat} corr={val:.2f}")

    return selected

def projection_predictive_selection(
    model_function,
    train_df,
    val_df,
    candidate_features: Dict[str, List[str]],
    device,
    num_steps: int = 200,
    lr: float = 1e-3,
    guide_type: str = "diag",
    verbose: bool = False,
    max_features: int | None = None,
    num_samples: int = 100,
    projection_steps: int = 100,
    print_progress: bool = True,
):
    """Projection predictive variable selection.

    A reference model is first trained using all candidate features. The
    algorithm then greedily selects features whose projected submodels
    best match the reference model's predictive distribution.
    """

    torch.set_default_device(device)

    groups = [
        "static",
        "dynamic",
        "seasonal",
        "time_trend",
        "temporal",
        "spatial",
    ]

    # Fit full reference model
    pyro.clear_param_store()
    ref_svi = create_learner(model_function, lr=lr, guide_type=guide_type)
    ref_train = prepare_model_data(
        train_df,
        candidate_features.get("static", []),
        candidate_features.get("dynamic", []),
        candidate_features.get("seasonal", []),
        candidate_features.get("time_trend", []),
        candidate_features.get("temporal", []),
        candidate_features.get("spatial", []),
        device,
    )
    ref_svi, _, ref_means, ref_stds = train_model(ref_train, ref_svi, num_steps=num_steps)
    ref_val = prepare_model_data(
        val_df,
        candidate_features.get("static", []),
        candidate_features.get("dynamic", []),
        candidate_features.get("seasonal", []),
        candidate_features.get("time_trend", []),
        candidate_features.get("temporal", []),
        candidate_features.get("spatial", []),
        device,
        ref_means,
        ref_stds,
        ref_train["ward_idx"].cpu().numpy(),
    )
    predictive = Predictive(
        model_function,
        guide=ref_svi.guide,
        num_samples=num_samples,
        return_sites=["obs"],
    )
    ref_pred = predictive(
        ref_val["occupation_idx"],
        ref_val["ward_idx"],
        ref_val["X_static"],
        ref_val["X_dynamic"],
        ref_val["X_seasonal"],
        ref_val["X_time_trend"],
        ref_val["X_temporal"],
        ref_val["X_spatial"],
    )["obs"].float().mean(0)

    selected = {g: [] for g in groups}
    remaining = {g: candidate_features.get(g, []).copy() for g in groups}

    def evaluate(current: Dict[str, List[str]]) -> float:
        pyro.clear_param_store()
        svi = create_learner(model_function, lr=lr, guide_type=guide_type)
        train_ds = prepare_model_data(
            train_df,
            current["static"],
            current["dynamic"],
            current["seasonal"],
            current["time_trend"],
            current["temporal"],
            current["spatial"],
            device,
        )
        svi, _, means, stds = train_model(train_ds, svi, num_steps=projection_steps)
        val_ds = prepare_model_data(
            val_df,
            current["static"],
            current["dynamic"],
            current["seasonal"],
            current["time_trend"],
            current["temporal"],
            current["spatial"],
            device,
            means,
            stds,
            train_ds["ward_idx"].cpu().numpy(),
        )
        pred = Predictive(
            model_function,
            guide=svi.guide,
            num_samples=num_samples,
            return_sites=["obs"],
        )(
            val_ds["occupation_idx"],
            val_ds["ward_idx"],
            val_ds["X_static"],
            val_ds["X_dynamic"],
            val_ds["X_seasonal"],
            val_ds["X_time_trend"],
            val_ds["X_temporal"],
            val_ds["X_spatial"],
        )["obs"].float().mean(0)
        return float(torch.mean((pred - ref_pred) ** 2).item())

    best_loss = evaluate(selected)
    improved = True

    def total_selected() -> int:
        return sum(len(v) for v in selected.values())

    while improved:
        improved = False
        current_best = best_loss
        best_group = None
        best_feat = None

        for group in groups:
            for feat in list(remaining[group]):
                trial = {k: v.copy() for k, v in selected.items()}
                trial[group].append(feat)
                loss = evaluate(trial)
                if verbose:
                    print(f"{group}:{feat} -> {loss:.2f}")
                if loss < current_best:
                    current_best = loss
                    best_group = group
                    best_feat = feat

        if best_feat is not None:
            selected[best_group].append(best_feat)
            remaining[best_group].remove(best_feat)
            best_loss = current_best
            improved = True
        if print_progress:
            print(f"Selected {best_group}:{best_feat} -> {best_loss:.2f}")

        if max_features is not None and total_selected() >= max_features:
            break

    return selected