import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src.envs import *

env = gym.make("ClusterScheduling-single-slot-v1").unwrapped
check_env(env)
