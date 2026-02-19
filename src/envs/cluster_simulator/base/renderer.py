from types import TracebackType
from typing import TypeVar, Generic, TypeAlias
from abc import abstractmethod
from src.envs.cluster_simulator.base.extractors.information import (
    ClusterBaseInformation,
)
from src.envs.cluster_simulator.base.extractors.observation import (
    BaseClusterObservation,
)
from typing_extensions import Self, NamedTuple, Literal

ClusterInformation = TypeVar("ClusterInformation", bound=ClusterBaseInformation)
ClusterObservation = TypeVar("ClusterObservation", bound=BaseClusterObservation)


class AbstractClusterGameRenderer(Generic[ClusterObservation, ClusterInformation]):
    @abstractmethod
    def render(
        self, new_info: ClusterInformation, new_observation: ClusterObservation
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
