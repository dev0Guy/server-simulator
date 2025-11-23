import typing as tp

import numpy as np

from server.envs.core.cluster import Cluster
from server.envs.core.proto.job import Status
from server.envs.deep_rm.custom_type import _JOBS_TYPE, _MACHINE_TYPE
from server.envs.deep_rm.jobs import DeepRMJobs, DeepRMJobSlot
from server.envs.deep_rm.machines import DeepRMMachine, DeepRMMachines


def can_run(m: DeepRMMachine, j: DeepRMJobSlot) -> bool:
    return np.all(m.free_space | ~j.usage)

def deep_rm_allocation(m: DeepRMMachine, j: DeepRMJobSlot):
    m.free_space &= ~j.usage

def generate_deeprm_random_jobs(
    n_jobs: int,
    n_resources: int,
    n_resource_unit: int,
    n_ticks: int,
    poisson_lambda: float = 5.0,
    offline: bool = True,
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], DeepRMJobs]:

    def inner(seed: tp.Optional[tp.SupportsFloat]) -> DeepRMJobs:
        np.random.seed(seed)
        jobs_slot = np.zeros(
            (n_jobs, n_resources, n_resource_unit, n_ticks), dtype=np.bool_
        )

        main_res = np.random.randint(0, n_resources, size=(n_jobs,))
        long_mask = np.zeros(n_jobs, dtype=bool)
        long_mask[np.random.choice(n_jobs, int(0.2 * n_jobs), replace=False)] = True
        durations = np.where(
            long_mask,
            np.random.randint(10, 15 + 1, size=n_jobs),
            np.random.randint(1, 3 + 1, size=n_jobs),
        )[:, None, None, None]
        job_arrivals_tick = (
            np.zeros(n_jobs, dtype=int)
            if offline
            else np.minimum(
                np.random.poisson(poisson_lambda, size=n_jobs),
                n_ticks - 1,
            )
        )

        start = np.zeros(n_jobs)[:, None, None, None]
        end = np.clip(start + durations, 0, n_ticks)
        main_usage_units = np.ceil(
            np.random.uniform(0.5, 1.0, size=(n_jobs,)) * n_resource_unit
        ).astype(int)
        usage = np.ceil(
            np.random.uniform(0.1, 0.2, size=(n_jobs, n_resources)) * n_resource_unit
        ).astype(int)
        usage[np.arange(n_jobs), main_res] = main_usage_units

        tick_idx = np.arange(n_ticks)[None, None, None, :]
        unit_idx = np.arange(n_resource_unit)[None, None, :, None]
        usage_exp = usage[:, :, None, None]
        unit_mask = unit_idx < usage_exp

        active_mask = (tick_idx >= start) & (tick_idx < end)
        jobs_slot[:] = active_mask & unit_mask
        jobs_status = np.array(
            [
                (
                    Status.Pending.value
                    if job_arrivals_tick[j_idx] == 0
                    else Status.NotCreated.value
                )
                for j_idx in range(n_jobs)
            ]
        )

        return DeepRMJobs(jobs_slot, jobs_status, job_arrivals_tick)

    return inner

def generate_deeprm_homogeneous_jobs() -> None:
    pass

def create_homogeneous_machines(
    n_machines: int, n_resources: int, n_resource_units: int, n_ticks: int
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], DeepRMMachines]:

    def inner(seed: tp.Optional[tp.SupportsFloat]) -> DeepRMMachines:
        np.random.seed(seed)
        machine_usage = np.ones(
            (n_machines, n_resources, n_resource_units, n_ticks), dtype=np.bool_
        )
        return DeepRMMachines(machine_usage)

    return inner



def generate_deeprm_cluster(
    n_machines: int,
    n_jobs: int,
    n_resources: int,
    n_resource_unit: int,
    n_ticks: int,
    is_offline: bool = True,
    poisson_lambda: float = 6.0,
    seed: tp.Optional[tp.SupportsFloat] = None,
) -> Cluster[_MACHINE_TYPE]:
    return Cluster[_MACHINE_TYPE](
        generate_deeprm_random_jobs(
            n_jobs, n_resources, n_resource_unit, n_ticks, poisson_lambda, is_offline
        ),
        create_homogeneous_machines(n_machines, n_resources, n_resource_unit, n_ticks),
        can_run=can_run, # type: ignore
        allocate=deep_rm_allocation, # type: ignore
        seed=seed,
    )

def generate_const_deeprm_cluster() -> Cluster[_MACHINE_TYPE]:
    pass
