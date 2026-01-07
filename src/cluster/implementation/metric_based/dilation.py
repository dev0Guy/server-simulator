import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import DilationProtocol, DilationState
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell, global_cell_from_local

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = int

class MetricBasedDilator(DilationProtocol[Kernel, State]):

    def expand(self, cell: tp.Tuple[int, int]) -> tp.Union[DilationState.Expanded, DilationState.FullyExpanded]:

        match self.state:
            case DilationState.FullyExpanded(_, _, _):
                raise ValueError("Cannot expand in fully expanded mode")
            case DilationState.Initial(_, _, level) | DilationState.Expanded(_, _, _, level) if level == 1:
                value = get_window_from_cell(self._dilation_levels, level=1, cell=cell, kernel=self._kernel)
                self.state = DilationState.FullyExpanded(prev_action=cell,prev_value=self.state, value=value, level=0)
            case DilationState.Initial(_, level) | DilationState.Expanded(_, _, _, level):
                value = get_window_from_cell(self._dilation_levels, level=level, cell=cell, kernel=self._kernel)
                self.state = DilationState.Expanded(prev_action=cell,prev_value=self.state, value=value, level=level - 1)

        return self.state

    def generate_dilation_expansion(self, original: State) -> DilationState.Initial:
        self._dilation_levels = hierarchical_pooling(original, self._kernel, fill_value=self._fill_value,operation=self._operation)
        self._n_levels = len(self._dilation_levels)

        if self._n_levels < 1:
            raise ValueError("Cannot call dilation on input with same size as kernel")

        level = self._n_levels-1
        self.state = DilationState.Initial(value=self._dilation_levels[level], level=level)

        return self.state

    def __init__(
            self,
            *,
            kernel: tp.Tuple[int, int],
            array: State,
            fill_value: float = 0.0,
            operation: tp.Callable
    ) -> None:
        self._kernel = kernel
        self._fill_value = fill_value
        self._operation = operation
        self.state = None
        self._dilation_levels = None
        self._n_levels = None
        self.generate_dilation_expansion(array)
        assert self._n_levels >= 1, "Dilation can't be called on two small values"

    # TODO: add testing to this function
    def get_selected_machine_idx_in_original(self) -> tp.Optional[Action]:
        """This function need to return the original action to execute (machine) given all the dilation option"""

        assert isinstance(self.state, DilationState.FullyExpanded), f"When running get_selected_machine_idx should be fully expanded {self.state}"
        self._calculate_original_cell_recursive(self.state)

    def _calculate_original_cell_recursive(self) -> Action:
        assert isinstance(self.state,DilationState.FullyExpanded), f"When running get_selected_machine_idx should be fully expanded {self.state}"
        match self.state:
            case DilationState.FullyExpanded(prev, value, level):
                pass
            case _:
                raise AssertionError("This code should be unreachable")

        # assert self._current_dilation_level_ptr != 0,  f"When running get_selected_machine_idx level pointer should be equal to 0 and not {self._current_dilation_level_ptr}"
        #
        # # TODO: think about what happen if kernel is size of the input, in other word there is no real dilation, +1 out of bound
        # prev_selected = self._prev_selected_cell[self._current_dilation_level_ptr]
        # current_selected = self._prev_selected_cell[self._current_dilation_level_ptr + 1]
        #
        # if prev_selected is None or current_selected is None:
        #     raise ValueError("Selected cells are not properly initialized")
        #
        # global_cell = global_cell_from_local(prev_selected, current_selected, self._kernel)
        #
        # return global_cell[0] * self._kernel[0] + global_cell[1]


