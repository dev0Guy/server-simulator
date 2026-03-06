import gymnasium as gym
import typing as tp
import numpy as np

from src.envs.cluster_simulator.actions import EnvironmentAction, ActionConvertor
from src.envs.cluster_simulator.base.extractors.reward import RewardCaculator
from src.envs.cluster_simulator.base.extractors.information import (
    ClusterInformation,
    BaceClusterInformationExtractor,
)
from src.envs.cluster_simulator.base.extractors.observation import (
    ClusterObservation,
    BaseObservationCreatorProtocol,
)
from src.envs.cluster_simulator.base.internal.cluster import ClusterABC

InputActType = np.int64
T = tp.TypeVar("T", bound=type)
Cluster = tp.TypeVar("Cluster", bound=ClusterABC)


class BasicClusterEnv(
    gym.Env[ClusterObservation, EnvironmentAction],
    tp.Generic[T, ClusterInformation, ClusterObservation],
):
    def __init__(
        self,
        cluster: Cluster,
        reward_caculator: RewardCaculator[ClusterInformation],
        info_builder: BaceClusterInformationExtractor[
            ClusterObservation, ClusterInformation
        ],
        obs_extractor: BaseObservationCreatorProtocol[Cluster, ClusterObservation],
    ):
        self._cluster = cluster
        self._reward_caculator = reward_caculator
        self._info_builder = info_builder
        self._obs_creator = obs_extractor
        self.observation_space = self._obs_creator.create_space(self._cluster)
        self.action_space = ActionConvertor.create_space(self._cluster)
        self._seed = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, tp.Any] | None = None,
    ) -> tuple[ClusterObservation, ClusterInformation]:
        if seed is not None:
            self._seed = seed
        super().reset(seed=self._seed)
        self._cluster.reset(self._seed)

        observation = self._obs_creator.create(self._cluster)
        info = self._info_builder(observation)

        return observation, info

    def step(
        self, action: EnvironmentAction
    ) -> tuple[ClusterObservation, tp.SupportsFloat, bool, bool, ClusterInformation]:
        if isinstance(action, tuple) and not isinstance(action, EnvironmentAction):
            action = EnvironmentAction(*action)
        assert isinstance(action, EnvironmentAction)
        prev_observation = self._obs_creator.create(self._cluster)
        prev_info = self._info_builder(prev_observation)
        cluster_action = ActionConvertor.convert(action)
        self._cluster.execute(cluster_action)
        observation = self._obs_creator.create(self._cluster)
        info = self._info_builder(observation)
        terminated = self._cluster.has_completed()
        reward = self._reward_caculator(prev_info, info)
        truncated = self._cluster.are_all_jobs_executed()
        return observation, reward, terminated, truncated, info
