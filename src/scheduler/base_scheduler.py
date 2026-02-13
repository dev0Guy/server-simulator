import logging
import typing as tp
import abc

from src.cluster.core.job import JobCollection, Job, Status
from src.cluster.core.machine import MachineCollection, Machine

T = tp.TypeVar("T")
MachineT = tp.TypeVar("MachineT", bound=Machine)
JobT = tp.TypeVar("JobT", bound=Job)


class ABCScheduler(abc.ABC, tp.Generic[T]):

    def __init__(self, can_run_func: tp.Callable[[MachineT, JobT], bool]):
        self._can_run_func = can_run_func
        self.logger = logging.getLogger(type(self).__name__)


    @staticmethod
    def pending_jobs(jobs: JobCollection[T]) -> tp.List[int]:
        return [
            j_idx
            for j_idx, job in enumerate(iter(jobs))
            if job.status == Status.Pending
        ]

    def possible_machines(self, job: Job[T], machines: MachineCollection[T]) -> tp.List[int]:
        return [
            m_idx
            for m_idx, machine in enumerate(iter(machines))
            if self._can_run_func(machine, job)
        ]

    @abc.abstractmethod
    def schedule(
        self,
        machines: MachineCollection[T],
        jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]: ...
