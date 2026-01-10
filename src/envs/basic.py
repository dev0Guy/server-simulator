import typing as tp

import numpy.typing as npt
import numpy as np
from rust_enum import enum, Case

from src.cluster.core.cluster import ClusterABC, Machines, Jobs, ClusterObservation, ClusterAction

InputActType = np.int64
InfoType = tp.TypeVar("InfoType", bound=dict)
T = tp.TypeVar("T", bound=type)

import gymnasium as gym

EnvAction = tp.Tuple[
    int,
    tp.Tuple[int, int]
]



class BasicClusterEnv(gym.Env[ClusterObservation, EnvAction], tp.Generic[T, InfoType]):

    def __init__(
        self,
        cluster: ClusterABC[Machines, Jobs],
        reward_func: tp.Callable[[InfoType, InfoType], tp.SupportsFloat],
        info_func: tp.Callable[[ClusterABC[Machines, Jobs]], InfoType],
    ):
        self._cluster = cluster
        self._reward_func = reward_func
        self._info_func = info_func

        self.observation_space = self._get_observation_space()

        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2), # 0=NOOP, 1=APPLY
            gym.spaces.Tuple((
                gym.spaces.Discrete(self._cluster.n_machines),
                gym.spaces.Discrete(self._cluster.n_jobs)
            )),
        ))

    def _get_observation_space(self) -> gym.spaces.Dict:
        machines_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._cluster._machines.get_representation().shape,
            dtype=self._cluster._machines.get_representation().dtype
        )
        jobs_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._cluster._jobs.get_representation().shape,
            dtype=self._cluster._jobs.get_representation().dtype
        )
        return gym.spaces.Dict(ClusterObservation(
            machines=machines_space,
            jobs=jobs_space
        ))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, tp.Any] | None = None,
    ) -> tuple[ClusterObservation, dict[str, tp.Any]]:
        super().reset(seed=seed)
        self._cluster.reset(seed)

        observation = self._cluster.get_representation()
        info = self._info_func(self._cluster)

        return observation, info

    def step(
        self, action: EnvAction
    ) -> tuple[ClusterObservation, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        prev_info = self._info_func(self._cluster)

        self._cluster.execute(
            self.cast_action_to_cluster_action(action)
        )

        observation = self._cluster.get_representation()
        info = self._info_func(self._cluster)
        terminated = self._cluster.is_finished()
        reward = self._reward_func(prev_info, info)
        truncated = False

        return observation, reward, terminated, truncated, info


    @staticmethod
    def cast_action_to_cluster_action(action: EnvAction) -> ClusterAction:
        should_skip_time, (m_idx, j_idx) = action
        if should_skip_time:
            return ClusterAction.SkipTime()

        return ClusterAction.Schedule(m_idx, j_idx)

