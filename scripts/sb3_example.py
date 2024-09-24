import sys
from typing import Any

import mlflow
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, str | tuple[str, ...]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


total_timesteps: int = 100

experiment_name = "SB3 Example"
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment(experiment_name)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


with mlflow.start_run(log_system_metrics=True):
    mlflow.log_params({"total_timesteps": total_timesteps})

    model = SAC("MlpPolicy", "Pendulum-v1", verbose=2)
    # Set custom logger
    model.set_logger(loggers)
    model.learn(total_timesteps=1000, log_interval=1)

    # Save the model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ModelWrapper(model),
    )
