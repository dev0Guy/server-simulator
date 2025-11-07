from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import typing as tp
import numpy as np
from gymnasium.core import ObsType, ActType


@dataclass
class ClusterOrchestration(gym.Env):
    n_machines: int
    n_jobs: int
    n_resources: int
    number_of_global_ticks: int

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __post_init__(self) -> None:
        assert self.n_resources > 0, "Can't create server farm with less than 1 resource types (cpu, ram, etc..)."
        assert self.number_of_global_ticks > 0, "Can't create server farm with tick counter less than 1."
        assert self.n_jobs > 0, "Can't create server farm with no jobs."
        assert self.n_machines > 0, "Can't create server farm with no machines."

        self._jobs_shape = (
            self.n_jobs,
            self.n_resources,
            self.number_of_global_ticks
        )

        self._machines_shape = (
            self.n_machines,
            self.n_resources,
            self.number_of_global_ticks
        )

        self.observation_space = gym.spaces.Dict({
            "machines": gym.spaces.Box(0, 1, self._machines_shape, dtype=np.double),
            "jobs": gym.spaces.Box(0, 1, self._jobs_shape, dtype=np.double)
        })

        _scheduling_combination = self.n_jobs * self.n_machines
        _null_action = 1

        self.action_space = gym.spaces.Discrete(_scheduling_combination + _null_action)

        self.reset()


    @staticmethod
    def generate_workload(job_shape: tuple[int, int, int]) -> np.array:
        jobs = np.zeros(job_shape, dtype=np.double)
        jobs[:, :, 0] = np.random.rand(job_shape[0], job_shape[1])
        return jobs

    def _get_obs(self) -> dict:
        return {
            "machines": self._machines,
            "jobs": self._jobs,
        }

    def _get_info(self) -> dict:
        return {}

    def _try_assigning_job_to_machine(self, m_idx, j_idx) -> bool:
        machine = self._machines[m_idx]
        job = self._jobs[j_idx]
        can_run = np.all(machine >= job)
        if not can_run:
            return False

        self._machines[m_idx] -= job
        return True


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self._machines = np.ones(self._machines_shape, dtype=np.double)
        self._jobs = self.generate_workload(self._jobs_shape)
        self._time_left = self.number_of_global_ticks

        return self._get_obs(), self._get_info()


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if is_null_action := action == 0:
            self._time_left -= 1
            self._machines = np.roll(self._machines, shift=-1, axis=-2)
            is_terminated = self._time_left == 0
            return self._get_obs(), 0, is_terminated, False, self._get_info()

        selected_machine_idx = (action - 1) % self.n_machines
        selected_job_idx = (action - 1) // self.n_machines

        if self._try_assigning_job_to_machine(selected_machine_idx, selected_job_idx):
            self._jobs[selected_job_idx, :, :] = 0
            # can run
            return self._get_obs(), 1, False, False, self._get_info()

        return self._get_obs(), 0, False, False, self._get_info()