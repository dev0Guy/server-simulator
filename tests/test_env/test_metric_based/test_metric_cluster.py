from server.envs.core.proto.job import Status
from server.envs.metric_based import generate_metric_based_cluster
import numpy as np

def check_creation(
    n_machines: int,
    n_jobs: int,
    n_resources: int,
    n_ticks: int,
    is_offline: bool,
    poisson_lambda: float,
) -> None:
    cluster = generate_metric_based_cluster(
        n_machines,
        n_jobs,
        n_resources,
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


def test_float_cluster_creation_offline(
    n_machines=3,
    n_jobs = 5,
    n_resources = 3,
    n_ticks = 20,
    is_offline = True,
    poisson_lambda = 6.0,
):
    check_creation(n_machines, n_jobs, n_resources, n_ticks, is_offline, poisson_lambda)


def test_float_cluster_creation_online(
    n_machines=3,
    n_jobs = 5,
    n_resources = 3,
    n_ticks = 20,
    is_offline = False,
    poisson_lambda = 6.0,
):
    check_creation(n_machines, n_jobs, n_resources, n_ticks, is_offline, poisson_lambda)


def test_reproducibility():
    cluster1 = generate_metric_based_cluster(1, 3, 2, 4, True, seed=123)
    cluster2 = generate_metric_based_cluster(1, 3, 2, 4, True, seed=123)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)


# Test:
# Test:
# Test:
# Test:
# Test:
# Test:
# Test:


def test_schedule_available() -> None:
    cluster = generate_metric_based_cluster(1, 2, 1, 2, 2, seed=123)
    assert cluster.schedule(0,0)

def test_schedule_full_machine(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError
