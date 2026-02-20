import typing as tp

from src.envs.cluster_simulator.base.internal.job import JobCollection
from src.envs.cluster_simulator.base.internal.machine import MachineCollection
from src.scheduler.base_scheduler import ABCScheduler

T = tp.TypeVar("T")


class RoundRobinScheduler(ABCScheduler[T]):
    """
    Round Robin scheduler.
    Cycles through pending jobs in order, giving each one a turn
    before returning to the beginning of the queue.
    """

    def __init__(self, can_run_func: tp.Callable):
        super().__init__(can_run_func)
        self._last_job_idx: int = -1

    def schedule(
        self, machines: MachineCollection[T], jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]:
        pending = self.pending_jobs(jobs)

        if not pending:
            self.logger.debug("No pending jobs.")
            return None

        # Sort pending jobs to ensure consistent cycling order
        pending = sorted(pending)

        # Find the next job after the last scheduled one (wrap around)
        next_candidates = [i for i in pending if i > self._last_job_idx]
        ordered = next_candidates if next_candidates else pending  # wrap around

        for job_idx in ordered:
            available = self.possible_machines(jobs[job_idx], machines)
            if available:
                machine_idx = available[0]
                self._last_job_idx = job_idx
                self.logger.debug(
                    "Scheduling job %d on machine %d (last_idx=%d)",
                    job_idx, machine_idx, self._last_job_idx
                )
                return machine_idx, job_idx

        self.logger.debug("No available machines for any pending job.")
        return None