import copy
import typing as tp
import numpy.typing as npt
import abc
from rust_enum import enum, Case

K = tp.TypeVar("K")
State = tp.TypeVar("State", bound=npt.NDArray)
SelectCellAction = tp.Tuple[int, int]

@enum
class DilationState(tp.Generic[State]):
    Initial = Case(value=State, level=int)
    Expanded = Case(prev_action=SelectCellAction, prev_value=tp.Union[Initial, 'Expanded'], value=State, level=int)
    FullyExpanded = Case(prev_action=SelectCellAction, prev_value=Expanded, value=State, level=int)


class AbstractDilation(abc.ABC):
    state: DilationState[State]
    _kernel: tp.Tuple[int, int]
    _dilation_levels: tp.List[State]
    _n_levels: int

    @abc.abstractmethod
    def get_window_from_cell(self, cell: SelectCellAction, level:int) -> State: ...

    @abc.abstractmethod
    def generate_dilation_levels(self, original: State) -> tp.List[State]: ...

    def __init__(
        self,
        kernel: tp.Tuple[int, int],
        array: State,
    ) -> None:
        self._kernel = kernel
        self._dilation_levels = None
        self._n_levels = None
        self.state = self.generate_dilation_expansion(array)
        assert self._n_levels >= 1, "Dilation can't be called on two small values"

    def expand(self, cell: tp.Tuple[int, int]) -> tp.Union[DilationState.Expanded, DilationState.FullyExpanded]:

            match self.state:
                case DilationState.FullyExpanded(_, _, _):
                    raise ValueError("Cannot expand in fully expanded mode")
                case DilationState.Initial(_, level) | DilationState.Expanded(_, _, _, level) if level == 1:
                    value = self.get_window_from_cell(level=1, cell=cell)
                    self.state = DilationState.FullyExpanded(prev_action=cell, prev_value=self.state, value=value,level=0)
                case DilationState.Initial(_, level) | DilationState.Expanded(_, _, _, level):
                    value = self.get_window_from_cell(level=level, cell=cell)
                    self.state = DilationState.Expanded(prev_action=cell, prev_value=self.state, value=value, level=level - 1)

            return self.state

    def contract(self) -> tp.Union[DilationState.Initial, DilationState.Expanded]:
        match self.state:
            case DilationState.Initial(_):
                return self.state
            case DilationState.Expanded(_, prev, _, _) | DilationState.FullyExpanded(_, prev, _, _):
                return prev
            case _:
                raise ValueError("Unreachable code")

    def generate_dilation_expansion(self, original: State) -> DilationState.Initial:
        self._dilation_levels = self.generate_dilation_levels(original)
        self._n_levels = len(self._dilation_levels)

        if self._n_levels < 1:
            raise ValueError("Cannot call dilation on input with same size as kernel")

        level = self._n_levels-1
        self.state = DilationState.Initial(value=self._dilation_levels[level], level=level)

        return self.state

    def get_selected_machine_idx_in_original(self, action: tp.Tuple[int, int]) -> tp.Tuple[int, int]:
        """This function need to return the original action to execute (machine) given all the dilation option"""

        assert isinstance(self.state, DilationState.FullyExpanded), f"When running get_selected_machine_idx should be fully expanded {self.state}"
        assert action[0] < self._kernel[0] and action[1] < self._kernel[1]
        return self._calculate_original_cell_recursive(self.state, action, self._kernel)

    @classmethod
    def _calculate_original_cell_recursive(cls, current_state: DilationState.FullyExpanded, action: tp.Tuple[int, int], kernel: tp.Tuple[int, int]) -> tp.Tuple[int, int]:
        final_action = [action[0], action[1]]
        while True:
            match current_state:
                case DilationState.FullyExpanded(prev_action, prev_state, _, _) | DilationState.Expanded(prev_action,prev_state, _, _):
                    final_action[0] += prev_action[0] * kernel[0]
                    final_action[1] += prev_action[1] * kernel[1]
                    current_state = prev_state
                case DilationState.Initial(_, _):
                    break
                case _:
                    raise ValueError
        return final_action[0], final_action[1]