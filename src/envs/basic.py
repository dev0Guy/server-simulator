import typing as tp

import gymnasium as gym
import numpy.typing as npt
import numpy as np
from src.cluster.core.cluster import ClusterABC, Machines, Jobs, ClusterObservation

InputActType = np.int64
InfoType = tp.TypeVar("InfoType", bound=dict)
T = tp.TypeVar("T", bound=type)



class BasicClusterEnv(gym.Env[ClusterObservation, InputActType], tp.Generic[T, InfoType]):

    def __init__(
        self,
        cluster: ClusterABC[Machines, Jobs],
        reward_func: tp.Callable[[InfoType, InfoType], tp.SupportsFloat],
        info_func: tp.Callable[[ClusterABC[Machines, Jobs]], InfoType],
    ):
        self._cluster = cluster
        self._reward_func = reward_func
        self._info_func = info_func

        self.observation_space = self._get_observation_space

        self._action_combination = (self._cluster.n_machines * self._cluster.n_jobs) + 1
        self.action_space = gym.spaces.Discrete(self._action_combination)

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
        self, action: int
    ) -> tuple[ClusterObservation, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        prev_info = self._info_func(self._cluster)

        match self.cast_action(action):
            case None:
                self._cluster.execute_clock_tick()
            case (m_idx, j_idx):
                self._cluster.schedule(m_idx, j_idx)
            case _:
                raise AssertionError("Expected code to be unreachable")

        observation = self._cluster.get_representation()
        info = self._info_func(self._cluster)
        terminated = self._cluster.is_finished()
        reward = self._reward_func(prev_info, info)
        truncated = False

        return observation, reward, terminated, truncated, info

    def cast_action(self, action: int) -> tp.Optional[tuple[int, int]]:
        if not (0 <= action <= self._action_combination):
            raise ValueError(
                f"Received action should be in range [{0},{self._action_combination}], which {action} don't fulfill."
            )

        if action == 0:
            return None

        adapted_action = action - 1
        m_idx = adapted_action % self._cluster.n_machines
        j_idx = adapted_action // self._cluster.n_machines

        return m_idx, j_idx

    def create_action_from(self, m_idx: int, j_idx: int) -> int:
        return 1 + (m_idx + j_idx * self._cluster.n_machines)
