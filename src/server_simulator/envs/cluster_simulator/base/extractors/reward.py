from collections import defaultdict
from typing import Generic, Optional
import abc

from src.server_simulator.envs.cluster_simulator.base.internal.job import Status
from src.server_simulator.envs.cluster_simulator.base.extractors.information import (
    ClusterInformation,
)


class RewardCaculator(Generic[ClusterInformation]):
    @abc.abstractmethod
    def __call__(
        self,
        prev_extra_information: ClusterInformation,
        current_extra_information: ClusterInformation,
    ) -> float: ...


    @abc.abstractmethod
    def reset(self) -> None: ...

class DifferentInPendingJobsRewardCaculator(RewardCaculator[ClusterInformation]):
    def reset(self) -> None:
        pass

    def __call__(
        self,
        prev_extra_information: ClusterInformation,
        current_extra_information: ClusterInformation,
    ) -> float:
        prev_not_pending_jobs_count = sum(
            s != Status.Pending for s in prev_extra_information["jobs_status"]
        )
        current_not_pending_jobs_count = sum(
            s != Status.Pending for s in current_extra_information["jobs_status"]
        )
        return current_not_pending_jobs_count - prev_not_pending_jobs_count


class AverageSlowDownReward(RewardCaculator[ClusterInformation]):

    def reset(self) -> None:
        self._job_start_time: list[Optional[int]] = [
            None
            for _ in range(self.n_jobs)
        ]

    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs
        self._job_start_time = []
        self.reset()

    def __call__(  self,
        prev_extra_information: ClusterInformation,
        current_extra_information: ClusterInformation,
    ) -> float:
        reward = 0
        current_time = current_extra_information["current_tick"]
        for idx, status in enumerate(current_extra_information["jobs_status"]):
            status = Status(status)
            match status:
                case Status.Pending | Status.Running if self._job_start_time[idx] is None:
                    self._job_start_time[idx] = current_time
                case Status.Completed | Status.NotCreated | Status.Running:
                    continue
                case Status.Pending:
                    turnaround = current_time - self._job_start_time[idx]
                    if turnaround != 0:
                        reward += -1 / turnaround
                case _:
                    raise ValueError(f"{status=}, {self._job_start_time[idx]=}")
        return reward