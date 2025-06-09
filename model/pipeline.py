from testing.testing import PredictionTester
from training.training import create_learner, train_model

from model import burglary_model


def train_and_evaluate_model(training_data, testing_data,
                             model_function,
                             inx_to_occupation_map,
                             guide_type: str = "diag",       # “diag”, “lowrank”, or “iaf”
                             guide_rank: int = 10,  # rank for low-rank guide
                             lr: float = 1e-3,
                             elbo_type: str = "trace",       # “trace”, “graph”, “renyi”, or “jit”
                             renyi_alpha: float = 0.5,
                             num_particles: int = 1,
                             training_steps=500,
                             testing_steps=1000):

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
