import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import DilationProtocol, DilationSteps
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell, global_cell_from_local

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = int

class MetricBasedDilator(DilationProtocol[Kernel, State]):

    def expand(self, cell: tp.Tuple[int, int]) -> tp.Union[DilationSteps.Expanded, DilationSteps.FullyExpanded]:

        match self.state:
            case DilationSteps.FullyExpanded(_, _, _):
                raise ValueError("Cannot expand in fully expanded mode")
            case DilationSteps.Initial(_, level) | DilationSteps.Expanded(_, _, level) if level==1:
                value = get_window_from_cell(self._dilation_levels, level=1, cell=cell, kernel=self._kernel)
                self.state = DilationSteps.FullyExpanded(prev=self.state, value=value, level=0)
            case DilationSteps.Initial(_, level) | DilationSteps.Expanded(_,_, level):
                value = get_window_from_cell(self._dilation_levels, level=level, cell=cell, kernel=self._kernel)
                self.state = DilationSteps.Expanded(prev=self.state, value=value, level=level - 1)

        return self.state

    def contract(self) -> tp.Union[DilationSteps.Initial, DilationSteps.Expanded]:
        match self.state:
            case DilationSteps.Initial(_):
                return self.state
            case DilationSteps.Expanded(prev, _, _) | DilationSteps.FullyExpanded(prev, _, _):
                return prev
            case _:
                raise ValueError("Unreachable code")

    def generate_dilation_expansion(self, original: State) -> DilationSteps.Initial:
        self._dilation_levels = hierarchical_pooling(original, self._kernel, fill_value=self._fill_value,operation=self._operation)
        self._n_levels = len(self._dilation_levels)

        if self._n_levels < 1:
            raise ValueError("Cannot call dilation on input with same size as kernel")

        level = self._n_levels-1
        self.state = DilationSteps.Initial(value=self._dilation_levels[level], level=level)

        return self.state

    def __init__(
            self,
            *,
            kernel: tp.Tuple[int, int],
            state: State,
            fill_value: float = 0.0,
            operation: tp.Callable
    ) -> None:
        self._kernel = kernel
        self._fill_value = fill_value
        self._operation = operation
        self.generate_dilation_expansion(state)
        assert self._n_levels >= 1, "Dilation can't be called on two small values"

    def get_selected_machine_idx_in_original(self) -> tp.Optional[Action]:
        # TODO: add testing to this function

        assert self._current_dilation_level_ptr != 0,  f"When running get_selected_machine_idx level pointer should be equal to 0 and not {self._current_dilation_level_ptr}"

        # TODO: think about what happen if kernel is size of the input, in other word there is no real dilation, +1 out of bound
        prev_selected = self._prev_selected_cell[self._current_dilation_level_ptr]
        current_selected = self._prev_selected_cell[self._current_dilation_level_ptr + 1]

        if prev_selected is None or current_selected is None:
            raise ValueError("Selected cells are not properly initialized")

        global_cell = global_cell_from_local(prev_selected, current_selected, self._kernel)

        return global_cell[0] * self._kernel[0] + global_cell[1]

    @property
    def kernel(self) -> tp.Tuple[int, int]:
        return copy.copy(self._kernel)


