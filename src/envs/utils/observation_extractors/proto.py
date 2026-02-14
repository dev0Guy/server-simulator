import abc
from typing import Protocol, TypeVar, runtime_checkable, overload, Literal
import gymnasium as gym
import numpy.typing as npt

from src.cluster.core.job import JobCollectionConvertor
from src.cluster.core.machine import MachinesCollectionConvertor
from src.envs.utils.common_types import Cluster

MachinesRepresentation = TypeVar("MachinesRepresentation")
JobsRepresentation = TypeVar("JobsRepresentation")


@runtime_checkable
class BaseClusterObservation(Protocol[MachinesRepresentation, JobsRepresentation]):
    """Protocol ensuring dict-like access with specific required keys"""

    @overload
    def __getitem__(self, key: Literal["machines"]) -> MachinesRepresentation: ...
    @overload
    def __getitem__(self, key: Literal["jobs_usage"]) -> JobsRepresentation: ...
    @overload
    def __getitem__(self, key: Literal["jobs_status"]) -> npt.ArrayLike: ...
    @overload
    def __getitem__(self, key: Literal["current_tick"]) -> npt.ArrayLike: ...
    def __getitem__(self, key: str) -> object: ...


ClusterObservation = TypeVar("ClusterObservation", bound=BaseClusterObservation)


class BaseObservationCreatorProtocol(Protocol[Cluster, ClusterObservation]):
    _machines_convertor: MachinesCollectionConvertor
    _jobs_convertor: JobCollectionConvertor

    @abc.abstractmethod
    def create(self, cluster: Cluster) -> ClusterObservation: ...

    @abc.abstractmethod
    def create_space(self, cluster: Cluster) -> gym.Space: ...
