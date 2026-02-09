import numpy.typing as npt
import numpy as np

from src.cluster.core.job import Job, JobCollection, Status


class SingleSlotJob(Job[np.float64]):

    def __init__(self, value: np.float64, status: Status) -> None:
        self._value = value
        self.status = status
        self.length = 1
        self.run_time = 0
        self.arrival_time = 0

    @property
    def usage(self) -> np.float64:
        return self._value


class SingleSlotJobs(JobCollection[npt.NDArray[np.float64]]):

    def __init__(self, job_usage: np.ndarray, job_status: list[Status]) -> None:
        assert job_usage.shape[0] == len(job_status)

        self._jobs = [
            SingleSlotJob(job_usage[j_idx], status=job_status[j_idx])
            for j_idx in range(len(job_usage))
        ]

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, item: int) -> Job[np.float64]:
        return self._jobs[item]

    def __iter__(self):
        return iter(self._jobs)

    def get_representation(self) -> np.array:
        return np.array([j.usage for j in self])
