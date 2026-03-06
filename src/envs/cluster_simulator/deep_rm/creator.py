from gymnasium.envs.registration import EnvCreator

from src.envs.cluster_simulator.base.extractors.information import (
    BaceClusterInformationExtractor,
)
from src.envs.cluster_simulator.base.extractors.reward import RewardCaculator
from src.envs.cluster_simulator.basic import BasicClusterEnv
from typing import TypedDict, Optional
from typing_extensions import Unpack
from src.envs.cluster_simulator.deep_rm import DeepRMCluster, DeepRMCreators
from src.envs.cluster_simulator.deep_rm.observation import DeepRMObservationCreator

__all__ = ["DeepRMCreatorParameters", "DeepRMEnvCreator"]


class DeepRMCreatorParameters(TypedDict):
    n_jobs: int
    n_machines: int
    n_resources: int
    n_resources_unit: int
    n_ticks: int
    reward_caculator: RewardCaculator
    seed: Optional[int]


class DeepRMEnvCreator(EnvCreator):
    def __call__(self, **kwargs: Unpack[DeepRMCreatorParameters]) -> BasicClusterEnv:
        cluster = DeepRMCluster(
            workload_creator=DeepRMCreators.generate_random_workload(
                kwargs["n_jobs"],
                kwargs["n_resources"],
                kwargs["n_resources_unit"],
                kwargs["n_ticks"],
            ),
            machine_creator=DeepRMCreators.generate_homogeneous_machines(
                kwargs["n_machines"],
                kwargs["n_resources"],
                kwargs["n_resources_unit"],
                kwargs["n_ticks"],
            ),
            seed=kwargs["seed"],
        )
        return BasicClusterEnv(
            cluster=cluster,
            reward_caculator=kwargs["reward_caculator"],
            info_builder=BaceClusterInformationExtractor(),
            obs_extractor=DeepRMObservationCreator(),
        )
