from typing import NamedTuple, Tuple
import gymnasium as gym
import abc

from src.cluster.core.cluster import ClusterAction
from src.envs.utils.common_types import Cluster


class EnvironmentAction(NamedTuple):
    should_schedule: bool
    schedule: Tuple[int, int]


class ActionConvertor:

    @staticmethod
    def convert(original: EnvironmentAction) -> ClusterAction:
        if original.should_schedule:
            return ClusterAction.SkipTime()

        assert all(idx >= 0 for idx in original.schedule)
        return ClusterAction.Schedule(*original.schedule)

    @staticmethod
    def create_space(cluster: Cluster) -> gym.Space:
        return gym.spaces.Tuple((
            gym.spaces.Discrete(2),
            gym.spaces.Tuple((
                gym.spaces.Discrete(cluster.n_machines), # type: ignore
                gym.spaces.Discrete(cluster.n_jobs) # type: ignore
            ))
        ))