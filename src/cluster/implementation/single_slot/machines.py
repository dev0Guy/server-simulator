import typing as tp
from typing import TypeAlias
from typing_extensions import Unpack
import numpy as np
from src.cluster.core.machine import (
    Machine,
    MachineCollection,
    MachinesCollectionConvertor,
)

SingleSlotMachinesArgs: TypeAlias = np.ndarray


class SingleSlotMachine(Machine[float]):
    free_space: float

    def __init__(self, free_space: float) -> None:
        self.free_space = free_space


class SingleSlotMachines(MachineCollection[float]):
    MAX_FREE_SPACE = 1.0

    def __init__(
        self,
        *args: Unpack[SingleSlotMachinesArgs],
    ):
        self._machines = [SingleSlotMachine(v) for v in args[0]]

    def __len__(self) -> int:
        return len(self._machines)

    def __getitem__(self, item: int) -> Machine[float]:
        return self._machines[item]

    def clean_and_reset(self, seed: tp.Optional[int]) -> None:
        for idx, machine in enumerate(self._machines):
            machine.free_space = self.MAX_FREE_SPACE

    def execute_clock_tick(self) -> None:
        for machine in self._machines:
            machine.free_space = self.MAX_FREE_SPACE


class SingleSlotMachinesConvertor(
    MachinesCollectionConvertor[float, SingleSlotMachinesArgs]
):
    def to_representation(self, value: SingleSlotMachines) -> SingleSlotMachinesArgs:
        return np.array([machine.free_space for machine in value._machines])
