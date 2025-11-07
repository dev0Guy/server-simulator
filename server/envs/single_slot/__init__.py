import typing as tp

import numpy as np

from server.envs.core.cluster import Cluster
from server.envs.core.proto.job import Job
from server.envs.core.proto.machine import Machine
from server.envs.single_slot.jobs import SingleSlotJobs, Status
from server.envs.single_slot.machines import SingleSlotMachines


def can_run(m: Machine, j: Job) -> bool:
    left_space_after_allocation = m.free_space - j.usage
    return left_space_after_allocation >= 0


def static_workload_creator(
    n_jobs: int, value: tp.SupportsFloat = 1.0
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotJobs]:

    def inner(_: tp.Optional[tp.SupportsFloat]) -> SingleSlotJobs:
        job_usage = np.zeros(shape=(n_jobs,)) + value
        job_status = [Status.Pending for _ in range(n_jobs)]
        return SingleSlotJobs(job_usage, job_status)

    return inner


def random_workload_creator(
    n_jobs: int,
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotJobs]:
    def inner(seed: tp.Optional[tp.SupportsFloat]) -> SingleSlotJobs:
        np.random.seed(seed)
        job_usage = np.random.rand(n_jobs)
        job_status = [Status.Pending for _ in range(n_jobs)]
        return SingleSlotJobs(job_usage, job_status)

    return inner


def static_machine_creator(
    n_machines: int, value: tp.SupportsFloat = 1.0
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], SingleSlotMachines]:
    def inner(_: tp.Optional[tp.SupportsFloat]) -> SingleSlotMachines:
        return SingleSlotMachines(machine_usage=np.zeros((n_machines,)) + value)

    return inner


def generate_single_slot_cluster(
    n_machines: int, n_jobs: int, seed: tp.Optional[tp.SupportsFloat] = None
) -> Cluster[tp.SupportsFloat]:
    return Cluster[tp.SupportsFloat](
        random_workload_creator(n_jobs),
        static_machine_creator(n_machines),
        can_run=can_run,
        seed=seed,
    )
