import typing as tp

from server.envs.core.proto.job import JobCollection, Status as JobStatus
from server.envs.core.proto.machine import MachineCollection
from server.envs.core.types import SupportsSubBool
import gymnasium as gym


T = tp.TypeVar('T', bound=SupportsSubBool)


class Cluster(tp.Generic[T]):

    def __init__(
        self,
        workload_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], JobCollection[T]],
        cluster_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], MachineCollection[T]],
        seed: tp.Optional[tp.SupportsFloat] = None
    ) -> None:
        self._current_tick = 0
        self._workload_creator = workload_creator

        self._machines = cluster_creator(seed)
        self._jobs = self._workload_creator(seed)
        self._jobs.execute_clock_tick(self._current_tick)

    @property
    def n_jobs(self) -> int:
        return len(self._jobs)

    @property
    def n_machines(self) -> int:
        return len(self._machines)

    def is_finished(self) -> bool:
        return all(job.status == JobStatus.Completed for job in self._jobs)

    def get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "machines": self._machines.observation_space(),
            "jobs": self._jobs.observation_space()
        })

    def get_observation(self) -> dict: ## TODO :fix typehint
        return {
            "machines": self._machines.get_observation(),
            "jobs": self._jobs.get_observation()
        }

    def schedule(self, m_idx: int, j_idx: int) -> bool:
        job = self._jobs[j_idx]
        machine = self._machines[m_idx]

        if job.status != JobStatus.Pending:
            return False

        can_run = bool(machine.free_space - job.usage)
        if not can_run:
            return False

        machine.free_space -= job.usage
        job.status = JobStatus.Running
        return True

    def execute_clock_tick(self) -> None:
        self._current_tick += 1
        self._jobs.execute_clock_tick(self._current_tick)
        self._machines.execute_clock_tick()

    def reset(self, seed: tp.Optional[tp.SupportsFloat]) -> None:
        self._current_tick = 0
        self._jobs = self._workload_creator(seed)
        self._machines.clean_and_reset(seed)

