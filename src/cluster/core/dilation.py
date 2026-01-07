import typing as tp
import numpy.typing as npt
import abc
from rust_enum import enum, Case

K = tp.TypeVar("K")
State = tp.TypeVar("State", bound=npt.NDArray)
Action = tp.TypeVar("Action", bound=int)

@enum
class DilationSteps(tp.Generic[State]):
    Initial = Case(value=State, level=int)
    Expanded = Case(prev=tp.Union[Initial, 'Expanded'], value=State, level=int)
    FullyExpanded = Case(prev=Expanded, value=State, level=int)


class DilationProtocol(tp.Protocol[K, State]):

    @abc.abstractmethod
    def generate_dilation_expansion(self, original: State) -> DilationSteps.Initial: ...

    @abc.abstractmethod
    def expand(self, cell: tp.Tuple[int, int]) -> tp.Union[DilationSteps.Expanded, DilationSteps.FullyExpanded]: ...

    @abc.abstractmethod
    def contract(self) -> tp.Union[DilationSteps.Initial, DilationSteps.Expanded]: ...

    @abc.abstractproperty
    def kernel(self) -> tp.Tuple[int, int]: ...