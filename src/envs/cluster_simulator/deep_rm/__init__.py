import typing as tp

import numpy as np

from src.envs.cluster_simulator.base.internal.job import Status



from src.envs.cluster_simulator.deep_rm.internal.custom_type import (
    _JOBS_TYPE as _JOBS_TYPE,
    _MACHINE_TYPE as _MACHINE_TYPE,
    _DTYPE as _DTYPE,
)
from src.envs.cluster_simulator.deep_rm.internal.jobs import (
    DeepRMJobs,
    DeepRMJobSlot,
    DeepRMJobsConvertor as DeepRMJobsConvertor,
)
from src.envs.cluster_simulator.deep_rm.internal.machines import (
    DeepRMMachine,
    DeepRMMachines,
    DeepRMMachinesConvertor as DeepRMMachinesConvertor,
)

from src.envs.cluster_simulator.base.internal.cluster import ClusterABC


class DeepRMCluster(ClusterABC[DeepRMMachines, DeepRMJobs]):
    def __init__(
        self,
        workload_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], DeepRMJobs],
        machine_creator: tp.Callable[[tp.Optional[tp.SupportsFloat]], DeepRMMachines],
        seed: tp.Optional[tp.SupportsFloat] = None,
    ):
        self._workload_creator = workload_creator
        self._machine_creator = machine_creator

        super().__init__(seed)

    def workload_creator(
        self, seed: tp.Optional[tp.SupportsFloat] = None
    ) -> DeepRMJobs:
        return self._workload_creator(seed)

    def machine_creator(
        self, seed: tp.Optional[tp.SupportsFloat] = None
    ) -> DeepRMMachines:
        return self._machine_creator(seed)

    def is_allocation_possible(
        self, machine: DeepRMMachine, job: DeepRMJobSlot
    ) -> bool:
        return np.all(machine.free_space | ~job.usage)

    def allocation(self, machine: DeepRMMachine, job: DeepRMJobSlot) -> None:
        machine.free_space &= ~job.usage


class DeepRMCreators:
    @staticmethod
    def generate_random_workload(
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
                np.random.uniform(0.1, 0.2, size=(n_jobs, n_resources))
                * n_resource_unit
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

    @staticmethod
    def generate_homogeneous_machines(
        n_machines: int, n_resources: int, n_resource_units: int, n_ticks: int
    ) -> tp.Callable[[tp.Optional[tp.SupportsFloat]], DeepRMMachines]:
        def inner(seed: tp.Optional[tp.SupportsFloat]) -> DeepRMMachines:
            np.random.seed(seed)
            machine_usage = np.ones(
                (n_machines, n_resources, n_resource_units, n_ticks), dtype=np.bool_
            )
            return DeepRMMachines(
                machine_usage,
            )

        return inner

    @classmethod
    def generate_default_cluster(
        cls,
        n_machines: int,
        n_jobs: int,
        n_resources: int,
        n_resource_unit: int,
        n_ticks: int,
        is_offline: bool = True,
        poisson_lambda: float = 6.0,
        seed: tp.Optional[tp.SupportsFloat] = None,
    ) -> DeepRMCluster:
        return DeepRMCluster(
            cls.generate_random_workload(
                n_jobs,
                n_resources,
                n_resource_unit,
                n_ticks,
                poisson_lambda,
                is_offline,
            ),
            cls.generate_homogeneous_machines(
                n_machines, n_resources, n_resource_unit, n_ticks
            ),
            seed=seed,
        )
