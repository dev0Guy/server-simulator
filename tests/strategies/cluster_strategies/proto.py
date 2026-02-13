from typing import Protocol, TypeVar, runtime_checkable
import abc

from hypothesis.strategies import SearchStrategy

from src.cluster.core.cluster import ClusterABC
from src.envs.utils.observation_extractors.proto import BaseObservationCreatorProtocol

Cluster = TypeVar('Cluster', bound=ClusterABC)
Creator = TypeVar('Creator', bound=BaseObservationCreatorProtocol)

@runtime_checkable
class ClusterStrategies(Protocol[Cluster]):

    @abc.abstractstaticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]: ...

    @abc.abstractstaticmethod
    def creation() -> SearchStrategy[Cluster]: ...
