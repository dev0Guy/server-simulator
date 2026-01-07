import copy
import typing as tp
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import AbstractDilation, DilationState, SelectCellAction
from src.utils.array_operations import hierarchical_pooling, get_window_from_cell, global_cell_from_local

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = tp.Tuple[int, int]

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

    # TODO: write TEST
    def get_selected_machine_idx_in_original(self) -> tp.Optional[Action]:
        """This function need to return the original action to execute (machine) given all the dilation option"""

        assert isinstance(self.state, DilationState.FullyExpanded), f"When running get_selected_machine_idx should be fully expanded {self.state}"
        self._calculate_original_cell_recursive()

    # TODO: write TEST
    def _calculate_original_cell_recursive(self) -> Action:
        current_state = self.state
        final_action = (0, 0)
        while True:
            match current_state:
                case DilationState.FullyExpanded(prev_action, _, _, _) | DilationState.Expanded(prev_action, _, _, _):
                    final_action[0] += prev_action[0] * self._kernel[0]
                    final_action[1] += prev_action[1] * self._kernel[1]
                case DilationState.Initial:
                    break
                case _:
                    raise ValueError
        return final_action


