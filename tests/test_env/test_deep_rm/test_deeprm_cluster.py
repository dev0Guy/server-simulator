import numpy as np
import pytest

from server.envs.core.proto.job import Status
from server.envs.deep_rm import DeepRMCreators


# TODO: Fix Test to include more

def test_float_cluster_creation():
    n_machines = 3
    n_jobs = 5
    n_resources = 3
    n_resource_unit = 5
    n_ticks = 20
    is_offline = True
    poisson_lambda = 6.0

    cluster = DeepRMCreators.generate_default_cluster(
        n_machines,
        n_jobs,
        n_resources,
        n_resource_unit,
        n_ticks,
        is_offline,
        poisson_lambda=poisson_lambda,
        seed=None,
    )

    observation = cluster.get_observation()

    assert observation["machines"].shape[0] == n_machines
    assert observation["jobs"].shape[0] == n_jobs
    assert np.all(observation["machines"] == 1.0)
    assert np.all(observation["jobs"] >= 0.0)

    for job in cluster._jobs:
        assert job.status == Status.Pending or job.status == Status.NotCreated


def test_reproducibility():
    cluster1 = DeepRMCreators.generate_default_cluster(1, 3, 2, 4, 5, seed=123)
    cluster2 = DeepRMCreators.generate_default_cluster(1, 3, 2, 4, 5, seed=123)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)


def test_schedule_available(
    m_idx: int = 0,
    j_idx: int = 0,
    cluster=DeepRMCreators.generate_default_cluster(1, 2, 1, 2, 2, seed=123)
) -> None:
    before_free_space =  cluster._machines[m_idx].free_space
    assert cluster._jobs[j_idx].status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert cluster._jobs[j_idx].status == Status.Running
    assert np.all(cluster._machines[m_idx].free_space == before_free_space | ~cluster._jobs[j_idx].usage)


def test_schedule_same_machine(
    m_idx: int = 0,
    j_idx: int = 0,
    cluster=DeepRMCreators.generate_default_cluster(1, 2, 1, 2, 2, seed=123)
) -> None:
    before_free_space = cluster._machines[m_idx].free_space
    assert cluster._jobs[j_idx].status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert cluster._jobs[j_idx].status == Status.Running
    assert np.all(cluster._machines[m_idx].free_space == before_free_space | ~cluster._jobs[j_idx].usage)

    assert not cluster.schedule(m_idx, j_idx)
    assert cluster._jobs[j_idx].status == Status.Running
    assert np.all(cluster._machines[m_idx].free_space == before_free_space | ~cluster._jobs[j_idx].usage)


def test_tick_without_schedule(
    tick_num: int = 3,
    machine_idx: int = 1,
    cluster=DeepRMCreators.generate_default_cluster(3, 2, 2, 1, 5, seed=123)
) -> None:
    cluster._machines._machines_usage[machine_idx, :, :, :tick_num] = False
    for _ in range(tick_num):
        cluster.execute_clock_tick()
    assert np.all(cluster._machines[0].free_space)


def test_schedule_and_tick_until_completion(
    m_idx: int = 0,
    j_idx: int = 0,
    cluster=DeepRMCreators.generate_default_cluster(1, 2, 1, 2, 2, seed=123)
) -> None:
    before_free_space = cluster._machines[m_idx].free_space
    assert cluster._jobs[j_idx].status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert cluster._jobs[j_idx].status == Status.Running
    assert np.all(cluster._machines[m_idx].free_space == before_free_space | ~cluster._jobs[j_idx].usage)
    for _ in range(cluster._jobs[j_idx].length):
        assert cluster._jobs[j_idx].status == Status.Running
        cluster.execute_clock_tick()
    assert cluster._jobs[j_idx].status == Status.Completed
