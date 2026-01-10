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
        super().__init__(kernel, array)



