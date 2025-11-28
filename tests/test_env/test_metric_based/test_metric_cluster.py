from server.envs.core.proto.job import Status
import numpy as np

from server.envs.metric_based import MetricClusterCreator
from server.envs.metric_based import MetricCluster
from hypothesis import given, strategies as st, assume, reproduce_failure


def cluster_params_strategy():
    return st.fixed_dictionaries({
        "n_machines": st.integers(1, 10),
        "n_jobs": st.integers(1, 20),
        "n_resources": st.integers(1, 5),
        "n_resource_unit": st.integers(1, 20),
        "n_ticks": st.integers(2, 200),
        "is_offline": st.booleans(),
        "poisson_lambda": st.floats(
            min_value=0.1,
            max_value=15.0,
            allow_nan=False,
            allow_infinity=False,
        )
    })

def cluster_strategy():
    params = cluster_params_strategy()
    return params.map(lambda p: MetricCluster.generate_default_cluster(**p))

def check_creation(
    n_machines: int,
    n_jobs: int,
    n_resources: int,
    n_ticks: int,
    is_offline: bool,
    poisson_lambda: float,
) -> None:
    cluster = MetricClusterCreator.generate_default(
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
    cluster1 = MetricClusterCreator.generate_default(1, 3, 2, 4, True, seed=123)
    cluster2 = MetricClusterCreator.generate_default(1, 3, 2, 4, True, seed=123)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)

def test_schedule_available() -> None:
    raise NotImplementedError

def test_schedule_full_machine(n_machines: int, n_jobs: int) -> None:
    raise NotImplementedError

def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick():
    raise NotImplementedError

def test_select_single_job_and_run_until_ticks_equal_to_job_length():
    raise NotImplementedError

def test_cluster_run_with_random_scheduler_until_completion():
    raise NotImplementedError