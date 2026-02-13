import typing as tp
import abc

from rust_enum import enum, Case

from src.cluster.core.job import Job, JobCollection
from src.cluster.core.job import Status as JobStatus
from src.cluster.core.machine import Machine, MachineCollection
import logging

T = tp.TypeVar("T")

Machines = tp.TypeVar("Machines", bound=MachineCollection)
Jobs = tp.TypeVar("Jobs", bound=JobCollection)


@enum
class ClusterAction:
    SkipTime = Case()
    Schedule = Case(machine=int, job=int)


class ClusterABC(tp.Generic[Machines, Jobs], abc.ABC):

    @abc.abstractmethod
    def workload_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> Jobs:
        ...

    @abc.abstractmethod
    def machine_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> Machines: ...

    @abc.abstractmethod
    def is_allocation_possible(self, machine: Machine[T], job: Job[T]) -> bool: ...

    @abc.abstractmethod
    def allocation(self, machine: Machine[T], job: Job[T]) -> None: ...

    def __init__(self, seed: tp.Optional[tp.SupportsFloat]):
        self._current_tick = 0
        self._machines = self.machine_creator(seed)
        self._jobs = self.workload_creator(seed)
        self._jobs.execute_clock_tick(self._current_tick)
        self._running_job_to_machine: dict[int, int] = {}
        self.logger = logging.getLogger(type(self).__name__)

    @property
    def n_jobs(self) -> int:
        return len(self._jobs)

    @property
    def n_machines(self) -> int:
        return len(self._machines)

    def is_finished(self) -> bool:
        n_none_finished_jobs = sum(job.status != JobStatus.Completed for job in self._jobs)
        self.logger.debug("Remaining jobs: %d", n_none_finished_jobs)
        return n_none_finished_jobs == 0

    def schedule(self, m_idx: int, j_idx: int) -> bool:
        job = self._jobs[j_idx]
        machine = self._machines[m_idx]

        if job.status != JobStatus.Pending:
            self.logger.warning(
                "Invalid action: job %d has status %s (expected: Pending)",
                j_idx,
                job.status,
            )
            return False

        if not self.is_allocation_possible(machine, job):
            self.logger.warning(
                "Schedule rejected: insufficient capacity | machine=%d job=%d",
                m_idx,
                j_idx,
            )
            return False

        self.allocation(machine, job)
        self.logger.info(
            "Scheduling job %d on machine %d",
            j_idx,
            m_idx,
        )
        job.status = JobStatus.Running
        job.run_time = 1  # Assume that if start running the in next one will finish
        self._running_job_to_machine[m_idx] = j_idx
        self.logger.debug(
            "Running job %d on machine %d",
            j_idx,
            m_idx,
        )
        return True

    def execute_clock_tick(self) -> None:
        self.logger.info(
            "Executing clock tick: %d → %d",
            self._current_tick,
            self._current_tick + 1,
        )
        self._current_tick += 1
        self._jobs.execute_clock_tick(self._current_tick)
        running_jobs = {
            j_idx
            for j_idx, job in enumerate(iter(self._jobs))
            if job.status != JobStatus.Running
        }
        self._running_job_to_machine = {
            k: v for k, v in self._running_job_to_machine.items() if k in running_jobs}
        self._machines.execute_clock_tick()

    def reset(self, seed: tp.Optional[tp.SupportsFloat]) -> None:
        self._current_tick = 0
        self._jobs = self.workload_creator(seed)
        self._machines.clean_and_reset(seed)

    def execute(self, action: ClusterAction) -> tp.Optional[bool]:
        match action:
            case ClusterAction.SkipTime():
                return self.execute_clock_tick()
            case ClusterAction.Schedule(machine_idx, job_idx):
                return self.schedule(machine_idx, job_idx)
            case _action:
                raise RuntimeError("Provided command should be %s or %s and not %s".format(
                    ClusterAction.SkipTime.__class__, ClusterAction.Schedule.__class__, type(_action).__class__))
