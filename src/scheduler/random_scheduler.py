import typing as tp
import random
import numpy as np

from src.envs.cluster_simulator.base.internal.job import JobCollection
from src.envs.cluster_simulator.base.internal.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler, T


class RandomScheduler(ABCScheduler[T]):
    def schedule(
        self, machines: MachineCollection[T], jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:
        pending_jobs = self.pending_jobs(jobs)

        if not pending_jobs:
            return None

        selected_job = random.choice(pending_jobs)
        self.logger.debug(
            "Selected job usage: (%f, %f)",
            float(np.max(jobs[selected_job].usage)),
            float(np.min(jobs[selected_job].usage)),
        )

        possible_machines = self.possible_machines(jobs[selected_job], machines)
        self.logger.debug(
            "Machines: %s",
            [
                (float(np.max(m.free_space)), float(np.min(m.free_space)))
                for m in machines
            ],
        )
        self.logger.debug("possible machines: %s", possible_machines)
        self.logger.debug("pending jobs: %d", len(pending_jobs))
        if not possible_machines:
            return None

        selected_machine = random.choice(possible_machines)

        return selected_machine, selected_job
