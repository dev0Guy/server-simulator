from typing import Generic
import abc

from src.envs.cluster_simulator.base.internal.job import Status
from src.envs.cluster_simulator.base.extractors.information import ClusterInformation


class RewardCaculator(Generic[ClusterInformation]):
    @abc.abstractmethod
    def __call__(
        self,
        prev_extra_information: ClusterInformation,
        current_extra_information: ClusterInformation,
    ) -> float: ...


class DifferentInPendingJobsRewardCaculator(RewardCaculator[ClusterInformation]):
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
