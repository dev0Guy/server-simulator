from typing import Protocol, TypeVar, runtime_checkable
import abc

from hypothesis.strategies import SearchStrategy

from src.cluster.core.dilation import AbstractDilation

Dilator = TypeVar("Dilator", bound=AbstractDilation)


@runtime_checkable
class DilationStrategies(Protocol[Dilator]):
    @abc.abstractstaticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]: ...

    @abc.abstractstaticmethod
    def creation(draw) -> SearchStrategy[Dilator]: ...
