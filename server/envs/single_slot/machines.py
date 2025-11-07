from server.envs.core.proto.machine import Machine, MachineCollection
import gymnasium as gym
import typing as tp
import numpy as np


class SingleSlotMachine(Machine[np.float64]):

    def __init__(self, free_space: np.float64) -> None:
        self.free_space = free_space


class SingleSlotMachines(MachineCollection[np.float64]):

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
            machine.free_space = 1

    def execute_clock_tick(self) -> None:
        self.clean_and_reset(None)

    def observation_space(self) -> "ObsType":
        return gym.spaces.Box(low=0.0, high=0.0, shape=self.machine_usage.shape)

    def get_observation(self) -> "ObsType":
        return np.array([m.free_space for m in self._machines])
