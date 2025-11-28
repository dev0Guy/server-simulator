import typing as tp
import random

from src.cluster.core.job import JobCollection, Status
from src.cluster.core.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler, T


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
