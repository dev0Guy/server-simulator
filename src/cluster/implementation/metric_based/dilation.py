import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import AbstractDilation, DilationState, SelectCellAction
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell, global_cell_from_local

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = int

class MetricBasedDilator(AbstractDilation[Kernel, State]):

    def get_window_from_cell(self, cell: SelectCellAction, level: int) -> State:
        return get_window_from_cell(self._dilation_levels, level=1, cell=cell, kernel=self._kernel)

    def generate_dilation_levels(self, original: State) -> tp.List[State]:
        return hierarchical_pooling(original, self._kernel, fill_value=self._fill_value, operation=self._operation)

    def __init__(
            self,
            *,
            kernel: tp.Tuple[int, int],
            array: State,
            fill_value: float = 0.0,
            operation: tp.Callable
    ) -> None:
        self._fill_value = fill_value
        self._operation = operation
        super().__init__(kernel, array)

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


