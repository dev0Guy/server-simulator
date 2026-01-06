import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import DilationProtocol, DilationOperation, DilationOperationReturnType
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell, global_cell_from_local

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = int

class MetricBasedDilator(DilationProtocol[Kernel, State]):

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
        self._n_levels = None
        self._dilation_levels = None
        self._current_dilation_level_ptr = None
        self._prev_selected_cell = None
        self.update(state)
        assert self._n_levels >= 1, "Dilation can't be called on two small values"

    def zoom_out(self) -> DilationOperationReturnType:
        if self._current_dilation_level_ptr == self._n_levels -1:
            return DilationOperationReturnType(DilationOperation.Error, None)

        self._current_dilation_level_ptr += 1
        selected_cell = self._prev_selected_cell[self._current_dilation_level_ptr]

        if selected_cell is None:
            raise ValueError("No cell selected at this level")

        self._prev_selected_cell[self._current_dilation_level_ptr] = None
        if not selected_cell:
            raise ValueError("Can't select cell if not exist")

        current_window = get_window_from_cell(
            outputs=self._dilation_levels,
            level=self._current_dilation_level_ptr+1,
            cell=selected_cell,
            kernel=self._kernel
        )

        return DilationOperationReturnType(DilationOperation.ZoomOut, current_window)

    def zoom_in(self, selected_cell: tp.Tuple[int, int]) -> DilationOperationReturnType:
        current_window = get_window_from_cell(
            outputs=self._dilation_levels,
            level=self._current_dilation_level_ptr,
            cell=selected_cell,
            kernel=self._kernel
        )
        self._prev_selected_cell[self._current_dilation_level_ptr] = selected_cell
        self._current_dilation_level_ptr -= 1
        if self._current_dilation_level_ptr < 0:
            raise ValueError(f"{self._current_dilation_level_ptr=} can't be negative")
        return DilationOperationReturnType(DilationOperation.ZoomIn, current_window)

    def select_real_machine(self, selected_cell: tp.Tuple[int, int]) -> DilationOperationReturnType:
        self._prev_selected_cell[self._current_dilation_level_ptr] = selected_cell
        current_view = self._dilation_levels[self._current_dilation_level_ptr]
        real_machine = current_view[selected_cell[0], selected_cell[1]]
        return DilationOperationReturnType(DilationOperation.Execute, real_machine)

    def execute_action(self, action: Action) -> DilationOperationReturnType:
        """Execute action where action is a multiplication of `(x_index * kernel[1] ,y_index * kernel[1]) and negative value is zoom out action`

        :param action: operation to achieve, zoom-out/zoom-in/select machine.
        :return: the operationEnum with its state
        """
        action_upper_bound = self._kernel[0] * self._kernel[1]
        assert action >= -1, "Action can't be negative value unless zoom out action (which is '-1')"
        assert action <= self._kernel[0] * self._kernel[1], f"Action can't be bigger than kernel. where limit is '{action_upper_bound}' and action '{action}'"

        is_zoom_in_action = action > 0
        if not is_zoom_in_action:
            return self.zoom_out()

        is_real_machine_select_action = self._current_dilation_level_ptr == 1
        selected_cell = (action // self._kernel[1], action % self._kernel[1])

        if is_real_machine_select_action:
           return self.select_real_machine(selected_cell)

        return self.zoom_in(selected_cell)

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

    def update(self, original: State) -> State:
        self._dilation_levels = hierarchical_pooling(original, self._kernel, fill_value=self._fill_value, operation=self._operation)
        self._n_levels = len(self._dilation_levels)
        if self._n_levels < 1:
            raise ValueError("Cannot call dilation on input with same size as kernel")
        self._current_dilation_level_ptr = self._n_levels - 1
        self._prev_selected_cell = [None for _ in range(self._n_levels)]
        return self._dilation_levels[self._current_dilation_level_ptr]

    def kernel_shape(self) -> tp.Tuple[int, int]:
        return copy.copy(self._kernel)


