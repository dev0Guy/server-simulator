import typing as tp

import gymnasium as gym

from src.envs.core.proto.machine import Machine, MachineCollection
from src.envs.deep_rm.custom_type import _MACHINE_TYPE, _MACHINES_TYPE


class DeepRMMachine(Machine[_MACHINE_TYPE]):

    def __init__(self, free_space: _MACHINE_TYPE) -> None:
        self.free_space = free_space


class DeepRMMachines(MachineCollection[_MACHINE_TYPE]):

    def __init__(self, machines_usage: _MACHINES_TYPE) -> None:
        assert (
            len(machines_usage.shape) == 4
        ), "Machine shape should be 4 dim (n.machines, n.resource, n.resource_units, n.ticks)."
        assert (
            machines_usage.shape[3] > 1
        ), "Machine should've more than single time slot (a.k.a time tick)."
        self._machines_usage = machines_usage
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

    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._machines_usage.shape,
            dtype=self._machines_usage.dtype,
        )

    def get_observation(self) -> gym.spaces.Space[gym.spaces.Box]:
        return self._machines_usage[:]
