import random
import typing as tp
import abc

from server.envs.core.proto.job import JobCollection, Status, Job
from server.envs.core.proto.machine import MachineCollection, Machine

T = tp.TypeVar("T")
MachineT = tp.TypeVar("MachineT", bound=Machine)
JobT = tp.TypeVar("JobT", bound=Job)


class ABCScheduler(abc.ABC, tp.Generic[T]):

    def __init__(self, can_run_func: tp.Callable[[MachineT, JobT], bool]):
        self._can_run_func = can_run_func

    @abc.abstractmethod
    def schedule(
            self,
            machines: MachineCollection[T],
            jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]: ...


class RandomScheduler(ABCScheduler[T]):

    def schedule(
        self,
        machines: MachineCollection[T],
        jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:
        pending_jobs = [
            j_idx
            for j_idx, job in enumerate(iter(jobs))
            if job.status == Status.Pending
        ]

        if not pending_jobs:
            return None

        selected_job = random.choice(pending_jobs)
        possible_machines = [
            m_idx
            for m_idx, machine in enumerate(iter(machines))
            if self._can_run_func(machine, jobs[selected_job])
        ]

        if not possible_machines:
            return None

        selected_machine = random.choice(possible_machines)

        return selected_machine, selected_job
