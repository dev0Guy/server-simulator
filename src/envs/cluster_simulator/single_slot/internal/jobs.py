import numpy as np

from src.envs.cluster_simulator.base.internal.job import (
    Job,
    JobCollection,
    Status,
    JobCollectionConvertor,
)
from typing import TypeAlias
from typing_extensions import Unpack


SingleSlotJobsArgs: TypeAlias = tuple[np.ndarray, list[Status]]


class SingleSlotJob(Job[float]):
    def __init__(self, value: float, status: Status) -> None:
        self._value = value
        self.status = status
        self.length = 1
        self.run_time = 0
        self.arrival_time = 0

    @property
    def usage(self) -> float:
        return self._value


class SingleSlotJobs(JobCollection[float]):
    def __init__(self, *args: Unpack[SingleSlotJobsArgs]) -> None:
        job_usage, job_status = args
        assert job_usage.shape[0] == len(job_status)
        self._jobs = [
            SingleSlotJob(job_usage[j_idx], status=job_status[j_idx])
            for j_idx in range(len(job_usage))
        ]

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, item: int) -> Job[float]:
        return self._jobs[item]

    def __iter__(self):
        return iter(self._jobs)


class SingleSlotJobsConvertor(JobCollectionConvertor[float, SingleSlotJobsArgs]):
    def to_representation(self, value: SingleSlotJobs) -> SingleSlotJobsArgs:
        return (np.array([j.usage for j in value]), [j.status for j in value])
