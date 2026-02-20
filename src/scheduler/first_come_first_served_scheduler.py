import typing as tp

from src.envs.cluster_simulator.base.internal.job import JobCollection
from src.envs.cluster_simulator.base.internal.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler

T = tp.TypeVar("T")


class FCFSScheduler(ABCScheduler[T]):
    """
    First Come, First Served scheduler.
    Assigns the earliest submitted pending job to the first available machine.
    """

    def schedule(
        self, machines: MachineCollection[T], jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:
        pending = self.pending_jobs(jobs)

        if not pending:
            self.logger.debug("No pending jobs.")
            return None

        for job_idx in pending:
            job = jobs[job_idx]

            if available := self.possible_machines(job, machines):
                machine_idx = available[0]
                self.logger.debug(
                    "Scheduling job %d on machine %d", job_idx, machine_idx
                )
                return machine_idx, job_idx

        self.logger.debug("No available machines for any pending job.")
        return None