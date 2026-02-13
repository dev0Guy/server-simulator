import typing as tp
import random
import numpy as np

from src.cluster.core.job import JobCollection, Status
from src.cluster.core.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler, T


class RandomScheduler(ABCScheduler[T]):

    def schedule(
        self,
        machines: MachineCollection[T],
        jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:

        pending_jobs = self.pending_jobs(jobs)

        if not pending_jobs:
            return None

        selected_job = random.choice(pending_jobs)

        possible_machines = self.possible_machines(jobs[selected_job], machines)
        self.logger.debug("Possible Options: %s", [(float(np.max(m.free_space)), float(np.min(m.free_space))) for m in machines])
        self.logger.debug("possible machines: %d", len(possible_machines))
        self.logger.debug("pending jobs: %d", len(pending_jobs))
        if not possible_machines:
            return None

        selected_machine = random.choice(possible_machines)

        return selected_machine, selected_job
