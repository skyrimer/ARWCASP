"""Hyperparameter search utilities for burglary model."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
import torch

from model import burglary_model
from training.training import create_learner, prepare_model_data, train_model
from testing.testing import PredictionTester
from evaluation_metric.evaluation_functions import (
    Full_Data_In_Merged_RMSE,
    rmse_score,
)


def evaluate_predictions(pred_df: pd.DataFrame, obs_df: pd.DataFrame) -> Dict[str, float]:
    """Return simple evaluation metrics for predictions."""
    merged = Full_Data_In_Merged_RMSE(pred_df, obs_df)
    return {"rmse": rmse_score(merged)}


def _prepare(df: pd.DataFrame, feature_cols: Dict[str, List[str]], device: torch.device, *, means=None, stds=None):
    data = prepare_model_data(
        df,
        static_cols=feature_cols.get("static", []),
        dynamic_cols=feature_cols.get("dynamic", []),
        seasonal_cols=feature_cols.get("seasonal", []),
        time_trend_cols=feature_cols.get("time_trend", []),
        temporal_cols=feature_cols.get("temporal", []),
        spatial_cols=feature_cols.get("spatial", []),
        device=device,
        means=means,
        stds=stds,
    )
    return data


def hyperparam_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    occupation_mapping: Dict[int, str],
    feature_cols: Dict[str, List[str]],
    param_grid: Iterable[Dict[str, object]],
    num_steps: int = 100,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Train and evaluate models for each set of parameters."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = _prepare(train_df, feature_cols, device)
    test_data = _prepare(
        test_df,
        feature_cols,
        device,
        means=train_data["means"],
        stds=train_data["stds"],
    )

    results = []

    for params in param_grid:
        svi = create_learner(burglary_model, **params)
        svi, _, _, _ = train_model(train_data, svi, num_steps)

        tester = PredictionTester(test_data, burglary_model, svi.guide, occupation_mapping)
        tester.predict(num_samples=num_samples)

        # Obtain mean predictions and prepare for evaluation
        pred_df = tester.get_mean_predictions()
        pred_df = pred_df.rename(columns={"mean": "count"})
        pred_df.index.name = "LSOA"
        pred_df = pred_df.reset_index()

        obs_df = test_df[["LSOA", "burglaries"]].rename(columns={"burglaries": "count"})

        metrics = evaluate_predictions(pred_df, obs_df)
        results.append({"params": params, **metrics})

    return pd.DataFrame(results)