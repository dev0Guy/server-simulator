from typing import Protocol, TypeVar, runtime_checkable
import abc

from hypothesis.strategies import SearchStrategy

from src.envs.cluster_simulator.core.cluster import ClusterABC
from src.envs.cluster_simulator.core.extractors.observation import BaseObservationCreatorProtocol

Cluster = TypeVar("Cluster", bound=ClusterABC)
Creator = TypeVar("Creator", bound=BaseObservationCreatorProtocol)


@runtime_checkable
class ClusterStrategies(Protocol[Cluster]):
    @abc.abstractstaticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]: ...

    @abc.abstractstaticmethod
    def creation() -> SearchStrategy[Cluster]: ...
