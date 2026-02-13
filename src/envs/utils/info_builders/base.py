import abc
from typing import Protocol, TypeVar, TypedDict, Generic
import numpy.typing as npt

from src.envs.utils.observation_extractors.proto import ClusterObservation


class ClusterBaseInformation(TypedDict):
    n_machines: int
    n_jobs: int
    jobs_status: npt.ArrayLike
    current_tick: int

ClusterInformation = TypeVar('ClusterInformation', bound=ClusterBaseInformation)

class BaceClusterInformationExtractor(Generic[ClusterObservation, ClusterInformation]):

    @abc.abstractmethod
    def __call__(self, obs: ClusterObservation) -> ClusterInformation:
        return ClusterBaseInformation(
            n_machines=obs["machines"].shape[0],
            n_jobs=obs["jobs_usage"].shape[0],
            jobs_status=obs["jobs_status"],
            current_tick=obs["current_tick"]
        )