import typing as tp

import numpy as np
import numpy.typing as npt

from src.cluster.core.job import Job, JobCollection, Status
from src.cluster.implementation.deep_rm.custom_type import _JOB_TYPE, _JOBS_TYPE


class DeepRMJobSlot(Job[_JOB_TYPE]):

    def __init__(self, usage: _JOB_TYPE, status: Status, arrival_time: int):
        self.status = status
        self._usage = usage
        self.arrival_time = arrival_time
        self.length = self._calculate_job_length(self._usage)
        self.run_time = 0

    @property
    def usage(self) -> _JOB_TYPE:
        return self._usage

    @classmethod
    def _calculate_job_length(cls, job: _JOB_TYPE) -> int:
        active = np.any(job > 0, axis=1)
        idx = np.where(active)[0]
        return int(idx[-1] - idx[0] + 1) if idx.size > 0 else 0


class DeepRMJobs(JobCollection[npt.NDArray[_JOB_TYPE]]):

    def __init__(
        self,
        job_slots: _JOBS_TYPE,
        job_status: npt.NDArray[int],
        job_arrivals_time: npt.NDArray[int],
    ) -> None:
        n_jobs_slot, n_job_status, n_arrival = (
            job_slots.shape[0],
            job_status.shape[0],
            job_arrivals_time.shape[0],
        )

        assert (
            n_jobs_slot == n_job_status
        ), f"Number of jobs slot ({n_jobs_slot}) should be equal to number of job status ({n_job_status})"
        assert (
            n_jobs_slot == n_arrival
        ), f"Number of jobs slot ({n_jobs_slot}) should be equal to number of job arrival array ({n_arrival})"

        self._jobs_slots = job_slots
        self._job_status = job_status
        self._job_arrivals_time = job_arrivals_time

        self._jobs = [
            DeepRMJobSlot(slot_usage, status, arrival_time)
            for slot_usage, status, arrival_time in zip(
                self._jobs_slots[:], self._job_status, self._job_arrivals_time
            )
        ]

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, item: int) -> DeepRMJobSlot:
        return self._jobs[item]

    def __iter__(self) -> tp.Iterable[DeepRMJobSlot]:
        return iter(self._jobs)

    def get_representation(self) -> npt.NDArray[_JOB_TYPE]:
        return self._jobs_slots
