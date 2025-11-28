import typing as tp

import numpy as np

from src.core.cluster.cluster import ClusterABC
from src.envs.single_slot.jobs import SingleSlotJobs, Status, SingleSlotJob
from src.envs.single_slot.machines import SingleSlotMachines, SingleSlotMachine


class SingleSlotClusterCreators:

    @staticmethod
    def static_workload_creator(
            n_jobs: int, value: tp.SupportsFloat = 1.0
    ) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotJobs]:
        def inner(_: tp.Optional[tp.SupportsFloat]) -> SingleSlotJobs:
            job_usage = np.zeros(shape=(n_jobs,)) + value
            job_status = [Status.Pending for _ in range(n_jobs)]
            return SingleSlotJobs(job_usage, job_status)

        return inner

    @staticmethod
    def random_workload_creator(
            n_jobs: int,
    ) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotJobs]:
        def inner(seed: tp.Optional[tp.SupportsFloat]) -> SingleSlotJobs:
            np.random.seed(seed)
            job_usage = np.random.rand(n_jobs)
            job_status = [Status.Pending for _ in range(n_jobs)]
            return SingleSlotJobs(job_usage, job_status)

        return inner

    @staticmethod
    def static_machine_creator(
            n_machines: int, value: tp.SupportsFloat = 1.0
    ) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotMachines]:
        def inner(_: tp.Optional[tp.SupportsFloat]) -> SingleSlotMachines:
            return SingleSlotMachines(machine_usage=np.zeros((n_machines,)) + value)

        return inner



class SingleSlotCluster(ClusterABC[np.float64]):

    def __init__(
        self,
        workload_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotJobs],
        machine_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotMachines],
        seed: tp.Optional[tp.SupportsFloat] = None,
    ):
        self._workload_creator = workload_creator
        self._machine_creator = machine_creator

        super().__init__(seed)

    def workload_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> SingleSlotJobs:
        return self._workload_creator(seed)

    def machine_creator(self, seed: tp.Optional[tp.SupportsFloat] = None) -> SingleSlotMachines:
        return self._machine_creator(seed)

    def is_allocation_possible(self, machine: SingleSlotMachine, job: SingleSlotJob) -> bool:
        left_space_after_allocation = machine.free_space - job.usage
        return left_space_after_allocation >= 0

    def allocation(self, machine: SingleSlotMachine, job: SingleSlotJob) -> None:
        machine.free_space -= job.usage
