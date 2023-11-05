import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback

MLFLOW_URI = "http://127.0.0.1:9911"

mlflc = MLflowCallback(
    tracking_uri=MLFLOW_URI,
    metric_name="my metric score",
)


@mlflc.track_in_mlflow()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    mlflow.log_param("power", 2)
    mlflow.log_metric("base of metric", x - 2)

    return (x - 2) ** 2


study = optuna.create_study(study_name="my_other_study")
study.optimize(objective, n_trials=10, callbacks=[mlflc])
