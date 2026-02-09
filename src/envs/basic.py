import gymnasium as gym
import typing as tp
import numpy as np

from src.cluster.core.cluster import ClusterABC, Machines, Jobs, ClusterObservation, ClusterAction

InputActType = np.int64
InfoType = tp.TypeVar("InfoType", bound=dict)
T = tp.TypeVar("T", bound=type)


class EnvAction(tp.NamedTuple):
    should_schedule: bool
    schedule: tp.Tuple[int, int]

    @staticmethod
    def generate_space(n_machines: int, n_jobs: int) -> gym.spaces.Space['EnvAction']:
        return gym.spaces.Tuple((  # type: ignore
            gym.spaces.Discrete(2),
            gym.spaces.Tuple((
                gym.spaces.Discrete(n_machines),
                gym.spaces.Discrete(n_jobs)
            ))
        ))

    def to_cluster_action(self) -> ClusterAction:
        if self.should_schedule:
            return ClusterAction.SkipTime()

        assert all(idx >= 0 for idx in self.schedule)
        return ClusterAction.Schedule(*self.schedule)


class BasicClusterEnv(gym.Env[ClusterObservation, EnvAction], tp.Generic[T, InfoType]):

    @staticmethod
    def generate_observation_space(cluster: 'ClusterABC') -> gym.spaces.Dict:
        machines_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=cluster._machines.get_representation().shape,
            dtype=cluster._machines.get_representation().dtype
        )
        jobs_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=cluster._jobs.get_representation().shape,
            dtype=cluster._jobs.get_representation().dtype
        )
        return gym.spaces.Dict(ClusterObservation(  # type: ignore
            machines=machines_space,  # type: ignore
            jobs=jobs_space  # type: ignore
        ))

    def __init__(
        self,
        cluster: ClusterABC[Machines, Jobs],
        reward_func: tp.Callable[[InfoType, InfoType], tp.SupportsFloat],
        info_func: tp.Callable[[ClusterABC[Machines, Jobs]], InfoType],
    ):
        self._cluster = cluster
        self._reward_func = reward_func
        self._info_func = info_func

        self.observation_space = self.generate_observation_space(self._cluster)

        self.action_space = EnvAction.generate_space(
            self._cluster.n_machines, self._cluster.n_jobs)

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
        assert isinstance(action, EnvAction)
        prev_info = self._info_func(self._cluster)
        cluster_action = action.to_cluster_action()
        self._cluster.execute(cluster_action)

        observation = self._cluster.get_representation()
        info = self._info_func(self._cluster)
        terminated = self._cluster.is_finished()
        reward = self._reward_func(prev_info, info)
        truncated = False

        return observation, reward, terminated, truncated, info
