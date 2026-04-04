import os

from src.server_simulator.envs.cluster_simulator.base.internal.dilation import AbstractDilationParams
from src.server_simulator.envs.cluster_simulator.metric_based.internal.dilation import MetricBasedDilator
from src.server_simulator.wrappers.cluster_simulator.dilation_wrapper import DilatorWrapper

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import logging
logging.basicConfig(level="INFO")

from src import server_simulator
from src.experiment.common.wrappers import FlattenActionWrapper, FlattenActionWrapperDilation
from src import server_simulator
from src.server_simulator.envs import MetricBasedEnvCreator, DifferentInPendingJobsRewardCaculator, \
    MetricBasedCreatorParameters
from src.server_simulator.envs.cluster_simulator.metric_based.renderer import ClusterMetricRenderer
from src.server_simulator.wrappers.cluster_simulator.render_wrapper import ClusterGameRendererWrapper

# TODO: Understand what happen when I activate zoom action and then skip time

def main():
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 25_000,
        "env_name": "ClusterScheduling-metric-online-v1",
    }

    run = wandb.init(
        project="cluster-scheduling",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    def make_env():
        env = gym.make(config["env_name"], render_mode="rgb_array", n_machines=10)
        # env = DilatorWrapper(env, dilator_cls=MetricBasedDilator, kernel=(3,3), operation=np.max)
        # env = FlattenActionWrapperDilation(env)
        env = FlattenActionWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )


if __name__ == '__main__':
    main()