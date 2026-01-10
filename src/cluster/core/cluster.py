import typing as tp
import abc

from src.cluster.core.job import Job, JobCollection, JobsRepresentation
from src.cluster.core.job import Status as JobStatus
from src.cluster.core.machine import Machine, MachineCollection, MachinesRepresentation

T = tp.TypeVar("T")

Machines = tp.TypeVar("Machines", bound=MachineCollection[T])
Jobs = tp.TypeVar("Jobs", bound=JobCollection[T])

class ClusterObservation(tp.TypedDict):
    machines: MachinesRepresentation
    jobs: JobsRepresentation

class ClusterABC(tp.Generic[Machines, Jobs], abc.ABC):

    @abc.abstractmethod
    def workload_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> Jobs: ...

    @abc.abstractmethod
    def machine_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> Machines: ...

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

    def get_representation(self) -> dict:
        return ClusterObservation(
            machines= self._machines.get_representation(),
            jobs=self._jobs.get_representation()
        )

    def schedule(self, m_idx: int, j_idx: int) -> bool:
        job = self._jobs[j_idx]
        machine = self._machines[m_idx]

        if job.status != JobStatus.Pending:
            return False

        if not self.is_allocation_possible(machine, job):
            return False

        self.allocation(machine, job)
        job.status = JobStatus.Running
        job.run_time = 1 # Assume that if start running the in next one will finish
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
