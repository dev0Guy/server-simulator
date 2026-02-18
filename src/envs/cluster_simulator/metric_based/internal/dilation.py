import typing as tp
import numpy.typing as npt
import numpy as np

from src.envs.cluster_simulator.base.internal.dilation import AbstractDilation, SelectCellAction
from src.envs.cluster_simulator.utils.array_operations import hierarchical_pooling, get_window_from_cell

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = tp.Tuple[int, int]


class MetricBasedDilator(AbstractDilation[State]):
    def get_window_from_cell(self, cell: SelectCellAction, level: int) -> State:
        return get_window_from_cell(
            self._dilation_levels, level=1, cell=cell, kernel=self._kernel
        )

    def generate_dilation_levels(self, original: State) -> tp.List[State]:
        return hierarchical_pooling(
            original,
            self._kernel,
            fill_value=self._fill_value,
            operation=self._operation,
        )

    @classmethod
    def cast_into_dilation_format(
        cls, array: State, *, fill_value: float = 0.0
    ) -> State:
        n_machines, n_resources, n_ticks = array.shape
        reorganize_shape = cls._reorganize_array_shape(array)
        grid_size = reorganize_shape[0] * reorganize_shape[1]

        if grid_size > n_machines:
            pad = grid_size - n_machines
            padding = ((0, pad), (0, 0), (0, 0))
            array = np.pad(array, padding, mode="constant", constant_values=fill_value)

        return array.reshape(
            *reorganize_shape,
            n_resources,
            n_ticks,
        )

    @staticmethod
    def _reorganize_array_shape(array: State) -> tp.Tuple[int, int]:
        n_machines = array.shape[0]
        window_x = int(np.ceil(np.sqrt(n_machines)))
        window_y = int(np.ceil(n_machines / window_x))
        return window_x, window_y

    def __init__(
        self,
        kernel: tp.Tuple[int, int],
        array: State,
        *,
        operation: tp.Callable,
        fill_value: float = 0.0,
    ) -> None:
        self._fill_value = fill_value
        self._operation = operation
        self._original_grid_shape: tp.Tuple[int, int] = array.shape[:2]  # type: ignore
        super().__init__(kernel, array)

    def get_selected_machine(self, action: SelectCellAction) -> int:
        un_dilated_action = self.get_selected_initialize_cell(action)
        return self._calculate_original_machine_index(
            un_dilated_action, self._original_grid_shape
        )

    @staticmethod
    def _calculate_original_machine_index(
        un_dilated_action: SelectCellAction,
        original_2d_shape: tp.Tuple[int, int],
    ) -> int:
        return un_dilated_action[0] * original_2d_shape[1] + un_dilated_action[1]
