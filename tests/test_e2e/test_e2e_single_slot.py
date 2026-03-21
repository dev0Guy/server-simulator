import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import src.server_simulator # noqa: F401

env = gym.make("ClusterScheduling-single-slot-v1").unwrapped
check_env(env)
