import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src import envs # noqa: F401

env = gym.make("ClusterScheduling-metric-online-v1").unwrapped
check_env(env)
env = gym.make("ClusterScheduling-metric-offline-v1").unwrapped
check_env(env)
