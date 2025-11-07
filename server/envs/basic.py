import gymnasium as gym
import typing as tp

from gymnasium.core import ObsType

from server.envs.core.cluster import Cluster, T

InfoType = tp.TypeVar("InfoType", bound=dict)


class BasicClusterEnv(gym.Env, tp.Generic[T, InfoType]):

    def __init__(
        self,
        cluster: Cluster[T],
        reward_func: tp.Callable[[InfoType, InfoType], tp.SupportsFloat],
        info_func: tp.Callable[[Cluster[T]], InfoType],
    ):
        self._cluster = cluster
        self._reward_func = reward_func
        self._info_func = info_func

        self.observation_space = self._cluster.get_observation_space()

        self._action_combination = (self._cluster.n_machines * self._cluster.n_jobs) + 1
        self.action_space = gym.spaces.Discrete(self._action_combination)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, tp.Any] | None = None,
    ) -> tuple[ObsType, dict[str, tp.Any]]:
        super().reset(seed=seed)
        self._cluster.reset(seed)

        observation = self._cluster.get_observation()
        info = self._info_func(self._cluster)

        return observation, info

    def step(
        self, action: int
    ) -> tuple[ObsType, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        prev_info = self._info_func(self._cluster)

        match self.cast_action(action):
            case None:
                self._cluster.execute_clock_tick()
            case (m_idx, j_idx):
                self._cluster.schedule(m_idx, j_idx)
            case _:
                raise AssertionError("Expected code to be unreachable")

        observation = self._cluster.get_observation()
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
