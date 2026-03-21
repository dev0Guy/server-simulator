import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import src.server_simulator.envs  # noqa: F401


env = gym.make("ClusterScheduling-deeprm-v1").unwrapped
check_env(env)
