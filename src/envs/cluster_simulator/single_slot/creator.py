from gymnasium.envs.registration import EnvCreator

from src.envs.cluster_simulator.base.extractors.information import BaceClusterInformationExtractor
from src.envs.cluster_simulator.base.extractors.reward import RewardCaculator
from src.envs.cluster_simulator.basic import BasicClusterEnv
from typing import Unpack, TypedDict, Optional

from src.envs.cluster_simulator.single_slot import SingleSlotCluster, SingleSlotClusterCreators
from src.envs.cluster_simulator.single_slot.observation import SingleSlotObservationCreator

__all__ = [
    'SingleSlotEnvCreator',
    'SingleSlotCreatorParameters'
]

class SingleSlotCreatorParameters(TypedDict):
    n_jobs: int
    n_machines: int
    reward_caculator: RewardCaculator
    seed: Optional[int]


class SingleSlotEnvCreator(EnvCreator):


    def __call__(self, **kwargs: Unpack[SingleSlotCreatorParameters]) -> BasicClusterEnv:
        cluster = SingleSlotCluster(
            workload_creator=SingleSlotClusterCreators.random_workload_creator(kwargs["n_jobs"]),
            machine_creator=SingleSlotClusterCreators.static_machine_creator(n_machines=kwargs["n_machines"], value=1.0),
            seed=kwargs["seed"]
        )
        return BasicClusterEnv(
            cluster=cluster,
            reward_caculator=kwargs["reward_caculator"],
            info_builder=BaceClusterInformationExtractor(),
            obs_extractor=SingleSlotObservationCreator(),
        )