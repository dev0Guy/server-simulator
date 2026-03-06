import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src import envs # noqa: F401

env = gym.make("ClusterScheduling-single-slot-v1").unwrapped
check_env(env)
