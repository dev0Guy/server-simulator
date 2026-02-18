from typing import NamedTuple, Tuple
import gymnasium as gym

from src.envs.cluster_simulator.core.cluster import ClusterAction
from src.envs.utils.common_types import Cluster


class EnvironmentAction(NamedTuple):
    should_schedule: bool
    schedule: Tuple[int, int]


class DilationEnvironmentAction(NamedTuple):
    selected_machine_cell: Tuple[int, int]
    selected_job: int
    execute_schedule_command: bool  # a.k.a skip time
    contract: bool

    @classmethod
    def into_action_space(
        cls, kernel_shape: Tuple[int, int], n_jobs: int
    ) -> gym.Space[tuple]:
        return gym.spaces.Tuple(
            spaces=(
                gym.spaces.MultiDiscrete(kernel_shape),
                gym.spaces.Discrete(n_jobs),
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(2),
            )
        )


class ActionConvertor:
    @staticmethod
    def convert(original: EnvironmentAction) -> ClusterAction:
        if original.should_schedule:
            return ClusterAction.SkipTime()

        assert all(idx >= 0 for idx in original.schedule)
        return ClusterAction.Schedule(*original.schedule)

    @staticmethod
    def create_space(cluster: Cluster) -> gym.Space:
        return gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(cluster.n_machines),  # type: ignore
                        gym.spaces.Discrete(cluster.n_jobs),  # type: ignore
                    )
                ),
            )
        )
