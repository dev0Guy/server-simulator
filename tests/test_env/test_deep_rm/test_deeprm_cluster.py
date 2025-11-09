import pytest

from server.envs.core.proto.job import Status
from server.envs.deep_rm import generate_deeprm_cluster
import numpy as np


def test_float_cluster_creation():
    n_machines = 3
    n_jobs = 5
    n_resources = 3
    n_resource_unit = 5
    n_ticks = 20
    is_offline = True
    poisson_lambda = 6.0

    cluster = generate_deeprm_cluster(
        n_machines,
        n_jobs,
        n_resources,
        n_resource_unit,
        n_ticks,
        is_offline,
        poisson_lambda=poisson_lambda,
        seed=None
    )

    observation = cluster.get_observation()

    assert observation["machines"].shape[0] == n_machines
    assert observation["jobs"].shape[0] == n_jobs
    assert np.all(observation["machines"] == 1.0)
    assert np.all(observation["jobs"] >= 0.0)

    for job in cluster._jobs:
        assert job.status == Status.Pending or job.status == Status.NotCreated

def test_reproducibility():
    cluster1 = generate_deeprm_cluster(1, 3, 2, 4, 5, seed=123)
    cluster2 = generate_deeprm_cluster(1, 3, 2, 4, 5, seed=123)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)

def test_schedule_available(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError

def test_schedule_full_machine(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError

def test_schedule_and_tick(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError

def test_tick_without_schedule(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError

def test_single_machine_multiple_one_size_jobs(n_jobs=20, n_machines=1):
    raise NotImplementedError

def test_single_machine_multiple_half_size_jobs(n_jobs=20, n_machines=1):
    raise NotImplementedError
