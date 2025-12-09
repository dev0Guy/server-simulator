import typing as tp

import gymnasium as gym
import numpy.typing as npt
import numpy as np

from src.cluster.core.dilation import DilationProtocol, DilationOperation, DilationOperationReturnType

Kernel = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Action = np.int64

class MetricBasedDilator(DilationProtocol[Kernel, State]):



    def __init__(
            self,
            kernel: npt.NDArray[np.float64],
            observation_space: gym.Space[State]
    ) -> None:
        self._kernel = kernel
        # TODO: extract the new state with history
        self._current_dilation_level_ptr = 0
        self._state_history = None # TODO

    def execute_action(self, action: Action) -> DilationOperationReturnType:
        pass

    def get_selected_machine_idx(self) -> Action:
        pass

    def update(self, original: State) -> None:
        self._state_history[:] = 0
        self._current_dilation_level_ptr = 0

    def get_observation_space_from(self, space: gym.Space[State]):
        pass

    def get_action_space_from(self, space: gym.Space[Action]) -> gym.Space[Action]:
        pass
