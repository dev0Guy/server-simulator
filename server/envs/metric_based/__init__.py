import typing as tp

import numpy as np

from server.envs.core.cluster import Cluster
from server.envs.core.proto.job import Status
from server.envs.metric_based.custom_type import _JOBS_TYPE, _MACHINE_TYPE
from server.envs.metric_based.jobs import MetricJobSlot, MetricJobs
from server.envs.metric_based.machines import MetricMachine, MetricMachines


def can_run(m: MetricMachine, j: MetricJobSlot) -> bool:
    return np.all(m.free_space > j.usage)

def metric_based_allocation(m: MetricMachine, j: MetricJobSlot):
    m.free_space -= j.usage

def create_homogeneous_machines(
    n_machines: int, n_resources: int, n_ticks: int
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], MetricMachines]:

    def inner(seed: tp.Optional[tp.SupportsFloat]) -> MetricMachines:
        np.random.seed(seed)
        machine_usage = np.ones(
            (n_machines, n_resources, n_ticks), dtype=np.float64
        )
        return MetricMachines(machine_usage)

    return inner

def create_heterogeneous_machines(
    n_machines: int, n_resources: int, n_ticks: int
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], MetricMachines]:
     # TODO: decide how to create the machines
    def inner(seed: tp.Optional[tp.SupportsFloat]) -> MetricMachines:
        raise NotImplementedError
        np.random.seed(seed)
        machine_usage = np.ones(
            (n_machines, n_resources, n_ticks), dtype=np.float64
        )
        return MetricMachines(machine_usage)

    return inner


def generate_metric_based_random_jobs(
    n_jobs: int,
    n_resources: int,
    n_ticks: int,
    poisson_lambda: float = 5.0,
    offline: bool = True,
) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], MetricJobs]:

    def inner(seed: tp.Optional[tp.SupportsFloat]) -> MetricJobs:
        np.random.seed(seed)
        jobs_slot = np.zeros(
            (n_jobs, n_resources, n_ticks), dtype=np.float64
        )
        # Pick a main resource per job
        main_res = np.random.randint(0, n_resources, size=(n_jobs,))
        # Long jobs (20%)
        long_mask = np.zeros(n_jobs, dtype=bool)
        long_mask[np.random.choice(n_jobs, int(0.2 * n_jobs), replace=False)] = True
        # Job durations
        durations = np.where(
            long_mask,
            np.random.randint(10, 15 + 1, size=n_jobs),
            np.random.randint(1, 3 + 1, size=n_jobs),
        )[:, None, None]
        # Arrival tick
        job_arrivals_tick = (
            np.zeros(n_jobs, dtype=int)
            if offline
            else np.minimum(
                np.random.poisson(poisson_lambda, size=n_jobs),
                n_ticks - 1,
            )
        )
        # Each job starts at tick 0 (can be adjusted)
        start = np.zeros(n_jobs)[:, None, None]
        end = np.clip(start + durations, 0, n_ticks)
        # --- Compute masks for (job, resource, tick) ---
        tick_idx = np.arange(n_ticks)[None, None, :]
        active_mask = (tick_idx >= start) & (tick_idx < end)
        #  Resource usage mask: each job uses some subset of resources
        usage = np.random.uniform(0.1, 1.0, size=(n_jobs, n_resources))
        usage[np.arange(n_jobs), main_res] = np.random.uniform(0.5, 1.0, size=n_jobs)
        # Expand usage to broadcast over ticks
        resource_mask = usage[:, :, None] > 0
        # Final boolean activity mask
        jobs_slot_bool = active_mask & resource_mask
        # Assign per-job constant float value
        job_value = np.random.uniform(0.1, 1.0, size=(n_jobs, 1, 1))
        # Apply per-job constant into active positions
        jobs_slot[:] = jobs_slot_bool * job_value
        # Status
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

        return MetricJobs(jobs_slot, jobs_status, job_arrivals_tick)

    return inner

def generate_deeprm_homogeneous_jobs() -> None:
    pass


def generate_metric_based_cluster(
    n_machines: int,
    n_jobs: int,
    n_resources: int,
    n_ticks: int,
    is_offline: bool = True,
    poisson_lambda: float = 6.0,
    seed: tp.Optional[tp.SupportsFloat] = None,
) -> Cluster[_MACHINE_TYPE]:
    return Cluster[_MACHINE_TYPE](
        generate_metric_based_random_jobs(
            n_jobs, n_resources, n_ticks, poisson_lambda, is_offline
        ),
        create_homogeneous_machines(n_machines, n_resources, n_ticks),
        can_run=can_run, # type: ignore
        allocate=metric_based_allocation, # type: ignore
        seed=seed,
    )
