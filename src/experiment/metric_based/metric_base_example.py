import os

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import NatureCNN

from src.experiment.callbacks.metrics import CustomMetricsCallback
from src.experiment.common.wrappers_new.machine_selection import AutoSelectJobWrapper
from src.server_simulator.envs.cluster_simulator.base.extractors.reward import AverageSlowDownReward
from src.server_simulator.envs.cluster_simulator.base.internal.dilation import AbstractDilationParams
from src.server_simulator.envs.cluster_simulator.base.internal.job import Status
from src.server_simulator.envs.cluster_simulator.metric_based.internal.dilation import MetricBasedDilator
from src.server_simulator.wrappers.cluster_simulator.dilation_wrapper import DilatorWrapper
import logging
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import logging
logging.basicConfig(level=logging.ERROR)

from src import server_simulator
from src.experiment.common.wrappers import FlattenActionWrapper, FlattenActionWrapperDilation, TimeLimitPenaltyWrapper
from src import server_simulator
from src.server_simulator.envs import MetricBasedEnvCreator, DifferentInPendingJobsRewardCaculator, \
    MetricBasedCreatorParameters
from src.server_simulator.envs.cluster_simulator.metric_based.renderer import ClusterMetricRenderer
from src.server_simulator.wrappers.cluster_simulator.render_wrapper import ClusterGameRendererWrapper

# TODO: Understand what happen when I activate zoom action and then skip time

def main():
    policy_kwargs = dict(
        # features_extractor_class=NatureCNN,  # built-in CNN
        # features_extractor_kwargs=dict(features_dim=256),
    )

    config = {
        "policy_type": "MultiInputPolicy", # MultiInputPolicy
        "total_timesteps": 500_000,
        "env_name": "ClusterScheduling-metric-offline-v1",
    }

    run = wandb.init(
        project="cluster-scheduling",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    run_id = "example-local"


    def make_env():
        n_jobs = 10
        n_machines = 5
        n_resources = 1
        n_ticks = 2
        max_episode_steps = 10
        penalty = -1e4
        # TODO: ADD PICKER FOR BEST JOB (WRapper)
        # TODO: ADD New State that caculate all of the possible assignment of job to node
        reward_caculator=AverageSlowDownReward(n_jobs)
        env = gym.make(
            config["env_name"],
            render_mode="rgb_array",
            n_jobs=n_jobs,
            n_machines=n_machines,
            n_resources=n_resources,
            n_ticks=n_ticks,
            reward_caculator=reward_caculator
        )
        # # env = DilatorWrapper(env, dilator_cls=MetricBasedDilator, kernel=(3,3), operation=np.max)
        # # env = FlattenActionWrapperDilation(env)
        # print(type(env.action_space))  # ← add this
        # print(env.action_space)  # ← and this
        # # env = FlattenMultiDiscreteWrapper(env)

        env = TimeLimitPenaltyWrapper(env, max_episode_steps=max_episode_steps, penalty=penalty)
        env = AutoSelectJobWrapper(env)
        # env = FlattenTupleActionWrapper(env)
        # env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        f"videos/{run_id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    model = PPO(
        config["policy_type"],
        env,
        policy_kwargs=policy_kwargs,
        # learning_rate=5e-5,
        verbose=1,
        tensorboard_log=f"runs/{run_id}"
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=500,
        model_save_path=f"models/{run_id}",
        verbose=2,
    )
    metric_callback = CustomMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=CallbackList([
            metric_callback,
            wandb_callback
        ]),
    )


if __name__ == '__main__':
    main()