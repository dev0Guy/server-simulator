import typing as tp
from typing import TypeAlias
from typing_extensions import Unpack
import numpy as np
import numpy.typing as npt

from src.envs.cluster_simulator.base.internal.job import (
    Job,
    Status,
    JobCollection,
    JobCollectionConvertor,
)
from src.envs.cluster_simulator.metric_based.internal.custom_type import (
    _JOBS_TYPE,
    _JOB_TYPE,
)

MetricJobsArgs: TypeAlias = tuple[_JOBS_TYPE, npt.NDArray[int], npt.NDArray[int]]


class MetricJobSlot(Job[_JOB_TYPE]):
    def __init__(self, usage: _JOB_TYPE, status: Status, arrival_time: int):
        self._usage = usage
        self.status = status
        self.arrival_time = arrival_time
        self.length = self._calculate_job_length(self._usage)
        self.run_time = self.length

    @classmethod
    def _calculate_job_length(cls, job: _JOB_TYPE) -> int:
        active = np.any(job > 0, axis=1)
        idx = np.where(active)[0]
        return int(idx[-1] - idx[0] + 1) if idx.size > 0 else 0

    @property
    def usage(self) -> _JOB_TYPE:
        return self._usage


class MetricJobs(JobCollection[npt.NDArray[_JOB_TYPE]]):
    def __init__(self, *args: Unpack[MetricJobsArgs]) -> None:
        self._job_slots, self._job_status, self._job_arrivals_time = args

        n_jobs_slot, n_job_status, n_arrival = (
            self._job_slots.shape[0],
            self._job_status.shape[0],
            self._job_arrivals_time.shape[0],
        )

        assert n_jobs_slot == n_job_status, (
            f"Number of jobs slot ({n_jobs_slot}) should be equal to number of job status ({n_job_status})"
        )
        assert n_jobs_slot == n_arrival, (
            f"Number of jobs slot ({n_jobs_slot}) should be equal to number of job arrival array ({n_arrival})"
        )

        self._jobs = self._jobs = [
            MetricJobSlot(slot_usage, status, arrival_time)
            for slot_usage, status, arrival_time in zip(
                self._job_slots[:], self._job_status, self._job_arrivals_time
            )
        ]

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, item: int) -> MetricJobSlot:
        return self._jobs[item]

    def __iter__(self) -> tp.Iterable[MetricJobSlot]:
        return iter(self._jobs)


class MetricJobsConvertor(JobCollectionConvertor[_JOB_TYPE, MetricJobsArgs]):
    def to_representation(self, value: MetricJobs) -> MetricJobsArgs:
        return (  # type: ignore
            value._job_slots,
            np.array([job.status.value for job in value]),
            value._job_arrivals_time,
        )
