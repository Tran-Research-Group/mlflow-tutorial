import os
import sys

import mlflow
from mlflow_rl_tools.sb3.log import MLflowOutputFormat
from mlflow_rl_tools.sb3.wrapper import ModelWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.logger import HumanOutputFormat, Logger
import imageio
import gymnasium as gym

total_timesteps: int = 100

experiment_name = "SB3 Example"
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment(experiment_name)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


with mlflow.start_run(log_system_metrics=True) as run:

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

    # Create animation
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    images = [env.render()]
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        images.append(env.render())
        if terminated or truncated:
            break

    image_save_dir: str = os.path.join("tmp", run.info.run_id, "plots")
    os.makedirs(image_save_dir, exist_ok=True)
    imageio.mimsave(os.path.join(image_save_dir, "animation.gif"), images)
    mlflow.log_artifact(os.path.join(image_save_dir, "animation.gif"), "plots")
