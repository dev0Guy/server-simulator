import typing as tp
import numpy.typing as npt
import abc
from rust_enum import enum, Case

K = tp.TypeVar("K")
State = tp.TypeVar("State", bound=npt.NDArray)
SelectCellAction = tp.Tuple[int, int]

# TODO: need to add prev selection action to the enum, for calculating the global selected cell
@enum
class DilationState(tp.Generic[State]):
    Initial = Case(value=State, level=int)
    Expanded = Case(prev_action=SelectCellAction, prev_value=tp.Union[Initial, 'Expanded'], value=State, level=int)
    FullyExpanded = Case(prev_action=SelectCellAction, prev_value=Expanded, value=State, level=int)


class DilationProtocol(tp.Protocol[K, State]):
    state: DilationState[State]
    _kernel: tp.Tuple[int, int]

    @abc.abstractmethod
    def generate_dilation_expansion(self, original: State) -> DilationState.Initial: ...

    @abc.abstractmethod
    def expand(self, cell: SelectCellAction) -> tp.Union[DilationState.Expanded, DilationState.FullyExpanded]: ...

    @property
    def kernel(self) -> tp.Tuple[int, int]: ...


    @abc.abstractmethod
    def contract(self) -> tp.Union[DilationState.Initial, DilationState.Expanded]:
        match self.state:
            case DilationState.Initial(_):
                return self.state
            case DilationState.Expanded(prev, _, _, _) | DilationState.FullyExpanded(prev, _, _, _):
                return prev
            case _:
                raise ValueError("Unreachable code")