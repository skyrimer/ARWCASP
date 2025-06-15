from typing import Any, Dict

import pandas as pd
import pyro
from testing.testing import PredictionTester
from training.training import create_learner, prepare_model_data, train_model
from utils.utils import last_n_time_splits

from model import burglary_model


def train_and_evaluate_model(training_data, testing_data,
                             model_function,
                             inx_to_occupation_map,
                             guide_type: str = "lowrank",       # “diag”, “lowrank”, or “iaf”
                             guide_rank: int = 14,  # rank for low-rank guide
                             lr: float = 0.007,
                             elbo_type: str = "jit",       # “trace”, “graph”, “renyi”, or “jit”
                             renyi_alpha: float = 0.5,
                             num_particles: int = 9,
                             training_steps=971,
                             testing_steps=1000):
    pyro.clear_param_store()

    svi = create_learner(model_function,
                         guide_type=guide_type,
                         guide_rank=guide_rank,
                         lr=lr,
                         elbo_type=elbo_type,
                         renyi_alpha=renyi_alpha,
                         num_particles=num_particles)

    _ = train_model(training_data, svi, num_steps=training_steps)
    prediction_tester = PredictionTester(
        testing_data, burglary_model, svi.guide, inx_to_occupation_map)
    prediction_tester.predict(testing_steps)
    evaluation_metrics = {
        'rmse': prediction_tester.get_rmse(),
        'mae': prediction_tester.get_mae(),
        'crps': prediction_tester.get_crps()
    }
    return evaluation_metrics, svi, svi.guide, prediction_tester


def cross_validate_time_splits(
    model_tuple: Any,
    time_col: str,
    n_splits: int,
    burglary_model: Any,
    occupation_map: Dict[int, str],
    device,
    ward_idx_map: Dict[Any, Any],
) -> pd.DataFrame:
    """
    Perform rolling‐window cross‐validation over the last `n_splits` time points.
    For each split, train on the immediately preceding time_s value and test on that time_s.
    Returns a DataFrame summarizing the evaluation metrics for each split.
    """
    splits = last_n_time_splits(model_tuple[0], time_col=time_col, n_splits=n_splits)
    records = []

    for train_df, test_df in splits:
        train_time = train_df[time_col].iloc[0]
        test_time = test_df[time_col].iloc[0]

        # Prepare data
        train_data = prepare_model_data(
            train_df,
            *model_tuple[1:],
            device,
            ward_idx_map=ward_idx_map
        )
        test_data = prepare_model_data(
            test_df,
            *model_tuple[1:],
            device,
            train_data["means"],
            train_data["stds"],
            ward_idx_map=ward_idx_map
        )

        # Train & evaluate
        evaluation_metrics, _, _, _ = train_and_evaluate_model(
            train_data,
            test_data,
            burglary_model,
            occupation_map
        )

        # Record metrics along with split info
        record = {
            "train_time": train_time,
            "test_time":  test_time,
            **evaluation_metrics
        }
        records.append(record)

    return pd.DataFrame(records)
