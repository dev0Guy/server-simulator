import typing as tp

from src.envs.cluster_simulator.base.internal.job import JobCollection
from src.envs.cluster_simulator.base.internal.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler

T = tp.TypeVar("T")


class SJFScheduler(ABCScheduler[T]):
    """
    Shortest Job First scheduler.
    Among all pending jobs that have at least one available machine,
    picks the one with the smallest expected duration.
    """

    def schedule(
        self, machines: MachineCollection[T], jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:
        pending = self.pending_jobs(jobs)

        if not pending:
            self.logger.debug("No pending jobs.")
            return None

        # Filter to only jobs that can actually run on at least one machine
        schedulable = [
            (job_idx, self.possible_machines(jobs[job_idx], machines))
            for job_idx in pending
            if self.possible_machines(jobs[job_idx], machines)
        ]

        if not schedulable:
            self.logger.debug("No available machines for any pending job.")
            return None

        # SJF: pick the job with the smallest duration
        job_idx, available_machines = min(
            schedulable, key=lambda x: jobs[x[0]].length
        )

        machine_idx = available_machines[0]
        self.logger.debug(
            "Scheduling job %d (duration=%s) on machine %d",
            job_idx, jobs[job_idx].length, machine_idx
        )
        return machine_idx, job_idx