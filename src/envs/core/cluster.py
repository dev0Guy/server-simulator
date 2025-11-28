import typing as tp

import gymnasium as gym
import abc

from src.envs.core.proto.job import Job, JobCollection
from src.envs.core.proto.job import Status as JobStatus
from src.envs.core.proto.machine import Machine, MachineCollection
from src.envs.core.types import SupportsSub

T = tp.TypeVar("T", bound=SupportsSub)



class ClusterABC(tp.Generic[T], abc.ABC):

    @abc.abstractmethod
    def workload_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> JobCollection[T]: ...

    @abc.abstractmethod
    def machine_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> MachineCollection[T]: ...

    @abc.abstractmethod
    def is_allocation_possible(self, machine: Machine[T], job: Job[T]) -> bool: ...

    @abc.abstractmethod
    def allocation(self, machine: Machine[T], job: Job[T]) -> None: ...

    def __init__(self, seed: tp.Optional[tp.SupportsFloat] = None):
        self._current_tick = 0
        self._machines = self.machine_creator(seed)
        self._jobs = self.workload_creator(seed)
        self._jobs.execute_clock_tick(self._current_tick)
        self._running_job_to_machine: dict[int, int] = {}

    @property
    def n_jobs(self) -> int:
        return len(self._jobs)

    @property
    def n_machines(self) -> int:
        return len(self._machines)

    def is_finished(self) -> bool:
        return all(job.status == JobStatus.Completed for job in self._jobs)

    def get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "machines": self._machines.observation_space(),
                "jobs": self._jobs.observation_space(),
            }
        )

    def get_observation(self) -> dict:
        return {
            "machines": self._machines.get_observation(),
            "jobs": self._jobs.get_observation(),
        }

    def schedule(self, m_idx: int, j_idx: int) -> bool:
        job = self._jobs[j_idx]
        machine = self._machines[m_idx]

        if job.status != JobStatus.Pending:
            return False

        if not self.is_allocation_possible(machine, job):
            return False

        self.allocation(machine, job)
        job.status = JobStatus.Running
        self._jobs.execute_clock_tick(self._current_tick)
        self._running_job_to_machine[m_idx] = j_idx
        return True

    def execute_clock_tick(self) -> None:
        self._current_tick += 1
        self._jobs.execute_clock_tick(self._current_tick)
        running_jobs = {
            j_idx
            for j_idx, job in enumerate(iter(self._jobs))
            if job.status != JobStatus.Running
        }
        self._running_job_to_machine = {k: v for k,v in self._running_job_to_machine.items() if k in running_jobs}
        self._machines.execute_clock_tick()

    def reset(self, seed: tp.Optional[tp.SupportsFloat]) -> None:
        self._current_tick = 0
        self._jobs = self.workload_creator(seed)
        self._machines.clean_and_reset(seed)



def default_allocation(m: Machine[T], j: Job[T]) -> None:
    m.free_space -= j.usage


class Cluster(tp.Generic[T]):

    def __init__(
        self,
        workload_creator: tp.Callable[
            [tp.Optional[tp.SupportsFloat]], JobCollection[T]
        ],
        machine_creator: tp.Callable[
            [tp.Optional[tp.SupportsFloat]], MachineCollection[T]
        ],
        can_run: tp.Callable[[Machine[T], Job[T]], bool],
        allocate: tp.Callable[[Machine[T], Job[T]], None] = default_allocation,
        seed: tp.Optional[tp.SupportsFloat] = None,
    ) -> None:
        self._current_tick = 0
        self._workload_creator = workload_creator
        self._can_run = can_run
        self._allocate = allocate

        self._machines = machine_creator(seed)
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
        return gym.spaces.Dict(
            {
                "machines": self._machines.observation_space(),
                "jobs": self._jobs.observation_space(),
            }
        )

    def get_observation(self) -> dict:  ## TODO :fix typehint
        return {
            "machines": self._machines.get_observation(),
            "jobs": self._jobs.get_observation(),
        }

    def schedule(self, m_idx: int, j_idx: int) -> bool:
        job = self._jobs[j_idx]
        machine = self._machines[m_idx]

        if job.status != JobStatus.Pending:
            return False

        if not self._can_run(machine, job):
            return False

        self._allocate(machine, job)
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
