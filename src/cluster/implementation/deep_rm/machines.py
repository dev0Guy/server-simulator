import typing as tp
import numpy.typing as npt

from src.cluster.core.machine import Machine, MachineCollection, MachineCollectionConvertor
from src.cluster.implementation.deep_rm.custom_type import _MACHINE_TYPE, _MACHINES_TYPE
from typing import TypeAlias
from typing_extensions import Unpack

DeepRMMachinesArgs: TypeAlias = npt.NDArray[_MACHINE_TYPE]

class DeepRMMachine(Machine[_MACHINE_TYPE]):

    def __init__(self, free_space: _MACHINE_TYPE) -> None:
        self.free_space = free_space


class DeepRMMachines(MachineCollection[npt.NDArray[_MACHINE_TYPE]]):

    def __init__(self, *args: Unpack[DeepRMMachinesArgs]) -> None:
        self._machines_usage = args[0]
        assert (
            len(self._machines_usage.shape) == 4
        ), "Machine shape should be 4 dim (n.machines, n.resource, n.resource_units, n.ticks)."
        assert (
            self._machines_usage.shape[3] > 1
        ), "Machine should've more than single time slot (a.k.a time tick)."
        self._machines = [
            DeepRMMachine(self._machines_usage[idx, :])
            for idx in range(self._machines_usage.shape[0])
        ]

    def __len__(self) -> int:
        return len(self._machines)

    def __getitem__(self, item: int) -> DeepRMMachine:
        return self._machines[item]

    def clean_and_reset(self, seed: tp.Optional[int]) -> None:
        self._machines_usage[:] = True

    def execute_clock_tick(self) -> None:
        self._machines_usage[:, :, :, :-1] = self._machines_usage[:, :, :, 1:]
        self._machines_usage[:, :, :, -1] = True


class DeepRMMachinesConvertor(MachineCollectionConvertor[_MACHINES_TYPE, DeepRMMachinesArgs]):

    def to_representation(self, value: DeepRMMachines) -> DeepRMMachinesArgs:
        return value._machines_usage[:]