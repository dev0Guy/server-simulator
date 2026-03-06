from gymnasium.envs.registration import EnvCreator

from src.envs.cluster_simulator.base.extractors.information import (
    BaceClusterInformationExtractor,
)
from src.envs.cluster_simulator.base.extractors.reward import RewardCaculator
from src.envs.cluster_simulator.basic import BasicClusterEnv
from typing import TypedDict, Optional
from typing_extensions import Unpack

from src.envs.cluster_simulator.metric_based import MetricCluster, MetricClusterCreator
from src.envs.cluster_simulator.metric_based.observation import (
    MetricClusterObservationCreator,
)

__all__ = ["MetricBasedCreatorParameters", "MetricBasedEnvCreator"]


class MetricBasedCreatorParameters(TypedDict):
    n_jobs: int
    n_machines: int
    n_resources: int
    n_ticks: int
    poisson_lambda: float
    offline: bool
    reward_caculator: RewardCaculator
    seed: Optional[int]


class MetricBasedEnvCreator(EnvCreator):
    def __call__(
        self, **kwargs: Unpack[MetricBasedCreatorParameters]
    ) -> BasicClusterEnv:
        cluster = MetricCluster(
            workload_creator=MetricClusterCreator.generate_workload(
                kwargs["n_jobs"],
                kwargs["n_resources"],
                kwargs["n_ticks"],
                kwargs["poisson_lambda"],
                offline=kwargs["offline"],
            ),
            machine_creator=MetricClusterCreator.generate_homogeneous_machines(
                kwargs["n_machines"], kwargs["n_resources"], kwargs["n_ticks"]
            ),
            seed=kwargs["seed"],
        )
        return BasicClusterEnv(
            cluster=cluster,
            reward_caculator=kwargs["reward_caculator"],
            info_builder=BaceClusterInformationExtractor(),
            obs_extractor=MetricClusterObservationCreator(),
        )
