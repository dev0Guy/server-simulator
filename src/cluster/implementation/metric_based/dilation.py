import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import AbstractDilation, SelectCellAction
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = tp.Tuple[int, int]

class MetricBasedDilator(AbstractDilation[State]):

    def get_window_from_cell(self, cell: SelectCellAction, level: int) -> State:
        return get_window_from_cell(self._dilation_levels, level=1, cell=cell, kernel=self._kernel)

    def generate_dilation_levels(self, original: State) -> tp.List[State]:
        return hierarchical_pooling(original, self._kernel, fill_value=self._fill_value, operation=self._operation)

    def cast_into_dilation_format(self, array: State, grid_shape: tp.Tuple[int, int]) -> State:
        n_machines, n_resources, n_ticks =  array.shape
        grid_size = grid_shape[0] * grid_shape[1]
        if grid_size > n_machines:
            pad = grid_size - n_machines
            padding = ((0, pad), (0, 0), (0, 0))
            array = np.pad(array, padding, mode="constant", constant_values=self._fill_value)

        return array.reshape(
            *grid_shape,
            n_resources,
            n_ticks,
        )

    def __init__(
            self,
            kernel: tp.Tuple[int, int],
            array: State,
            *,
            operation: tp.Callable,
            fill_value: float = 0.0
    ) -> None:
        self._fill_value = fill_value
        self._operation = operation
        self._n_machines = array.shape[0]
        window_x = int(np.ceil(np.sqrt(self._n_machines)))
        window_y = int(np.ceil(self._n_machines / window_x))
        self._original_2d_shape = (window_x, window_y)
        grid = self.cast_into_dilation_format(array, grid_shape=self._original_2d_shape)
        super().__init__(kernel, grid)

    def get_selected_machine(self, action: SelectCellAction) -> tp.Optional[int]:
        un_dilated_action = self.get_selected_initialize_cell(action)
        return self._calculate_original_machine_index(un_dilated_action, self._original_2d_shape, self._n_machines)

    @staticmethod
    def _calculate_original_machine_index(
        un_dilated_action: SelectCellAction,
        original_2d_shape: tp.Tuple[int,int],
        number_of_machines: int
    ) -> tp.Optional[int]:
        machine_index = un_dilated_action[0] * original_2d_shape[1] + un_dilated_action[1]
        if machine_index >= number_of_machines:
            return None
        return machine_index