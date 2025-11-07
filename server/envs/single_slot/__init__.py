from server.envs.core.cluster import Cluster
from server.envs.single_slot.jobs import SingleSlotJobs, Status
from server.envs.single_slot.machines import SingleSlotMachines

import typing as tp
import numpy as np


def create_float_cluster(n_machines: int ,n_jobs: int, seed: tp.Optional[tp.SupportsFloat] = None) -> Cluster[tp.SupportsFloat]:

    def workload_creator(seed: tp.Optional[tp.SupportsFloat]) -> SingleSlotJobs:
        np.random.seed(seed)
        job_usage = np.random.rand(n_jobs)
        job_status = [Status.Pending for _ in range(n_jobs)]
        return SingleSlotJobs(
            job_usage,
            job_status
        )

    def cluster_creator(seed: tp.Optional[tp.SupportsFloat]) -> SingleSlotMachines:
        return SingleSlotMachines(
            machine_usage=np.ones((n_machines,))
        )

    return Cluster[float](
        workload_creator,
        cluster_creator,
        seed
    )

