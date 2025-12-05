import typing as tp

import gymnasium as gym
import numpy as np

from src.cluster.core.machine import Machine, MachineCollection
import numpy.typing as npt

class SingleSlotMachine(Machine[np.float64]):

    def __init__(self, free_space: np.float64) -> None:
        self.free_space = free_space


class SingleSlotMachines(MachineCollection[npt.NDArray[np.float64]]):
    MAX_FREE_SPACE = 1.0

    def __init__(
        self,
        machine_usage: np.ndarray,
    ):
        self.machine_usage = machine_usage
        self._machines = [SingleSlotMachine(v) for v in self.machine_usage]

    def __len__(self) -> int:
        return len(self._machines)

    def __getitem__(self, item: int) -> Machine[np.float64]:
        return self._machines[item]

    def clean_and_reset(self, seed: tp.Optional[int]) -> None:
        for idx, machine in enumerate(self._machines):
            machine.free_space = self.MAX_FREE_SPACE

    def get_representation(self) -> np.array:
        return np.array([m.free_space for m in self._machines])

    def execute_clock_tick(self) -> None:
        for machine in self._machines:
            machine.free_space = self.MAX_FREE_SPACE
