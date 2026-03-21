from typing import Protocol, TypeVar, runtime_checkable
import abc

from hypothesis.strategies import SearchStrategy

from src.server_simulator.envs.cluster_simulator.base.extractors.observation import (
    BaseObservationCreatorProtocol,
)
from src.server_simulator.envs.cluster_simulator.base.internal.cluster import ClusterABC

Cluster = TypeVar("Cluster", bound=ClusterABC)
Creator = TypeVar("Creator", bound=BaseObservationCreatorProtocol)


@runtime_checkable
class ClusterStrategies(Protocol[Cluster]):
    @abc.abstractstaticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]: ...

    @abc.abstractstaticmethod
    def creation() -> SearchStrategy[Cluster]: ...
