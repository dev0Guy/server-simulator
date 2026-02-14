import typing as tp
from typing import TypeAlias
from typing_extensions import Unpack
import numpy.typing as npt

from src.cluster.core.machine import (
    Machine,
    MachineCollection,
    MachinesCollectionConvertor,
)
from src.cluster.implementation.metric_based.custom_type import (
    _MACHINE_TYPE,
    _MACHINES_TYPE,
)

MetricsMachinesArgs: TypeAlias = _MACHINES_TYPE


class MetricMachine(Machine[_MACHINE_TYPE]):
    def __init__(self, free_space: _MACHINE_TYPE) -> None:
        self.free_space = free_space


class MetricMachines(MachineCollection[npt.NDArray[_MACHINE_TYPE]]):
    def __init__(self, *args: Unpack[MetricsMachinesArgs]) -> None:
        self._machines_usage = args[0]
        assert len(self._machines_usage.shape) == 3, (
            "Machine shape should be 3 dim (n.machines, n.resource, n.ticks)."
        )
        assert self._machines_usage.shape[2] > 1, (
            "Machine should've more than single time slot (a.k.a time tick)."
        )
        self._machines = [
            MetricMachine(self._machines_usage[idx, :])
            for idx in range(self._machines_usage.shape[0])
        ]

    def __len__(self) -> int:
        return len(self._machines)

    def __getitem__(self, item: int) -> Machine[_MACHINE_TYPE]:
        return self._machines[item]

    def clean_and_reset(self, seed: tp.Optional[int]) -> None:
        self._machines_usage[:] = 1.0

    def execute_clock_tick(self) -> None:
        self._machines_usage[:, :-1] = self._machines_usage[:, 1:]
        self._machines_usage[:, -1] = 1.0


class MetricMachinesConvertor(
    MachinesCollectionConvertor[_MACHINES_TYPE, MetricsMachinesArgs]
):
    def to_representation(self, value: MetricMachines) -> MetricsMachinesArgs:
        return value._machines_usage[:]
