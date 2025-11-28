import numpy as np
import pytest
from hypothesis.strategies import DataObject

from server.envs.core.proto.job import Status
from server.envs.deep_rm import DeepRMCreators, DeepRMCluster
from hypothesis import given, strategies as st, assume, reproduce_failure

from server.envs.scheduler.basic import RandomScheduler
from tests.test_env.test_deep_rm.utils import get_index_of_min_job_arrival_time
from tests.test_env.test_single_slot.test_cluster import seed_strategy


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
    return params.map(lambda p: DeepRMCreators.generate_default_cluster(**p))

@given(cluster=cluster_strategy())
def test_float_cluster_creation(cluster: DeepRMCluster):
    observation = cluster.get_observation()

    assert observation["machines"].shape[0] == cluster.n_machines
    assert observation["jobs"].shape[0] == cluster.n_jobs
    assert np.all(observation["machines"] == 1.0)
    assert np.all(observation["jobs"] >= 0.0)

    for job in cluster._jobs:
        assert job.status == Status.Pending or job.status == Status.NotCreated

@given(params=cluster_params_strategy(), seed=seed_strategy)
def test_reproducibility(params: dict, seed: int):
    cluster1 = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    cluster2 = DeepRMCreators.generate_default_cluster(**params, seed=seed)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)

@given(
    params=cluster_params_strategy(),
    seed1=seed_strategy,
    seed2=seed_strategy
)
@pytest.mark.xfail(reason="Different seeds can still create same cluster Thus flakiness", strict=False)
def test_different_between_seeds(params: dict, seed1: int, seed2: int):
    assume(seed1 != seed2)

    cluster_1 = DeepRMCreators.generate_default_cluster(**params, seed=seed1)
    cluster_2 = DeepRMCreators.generate_default_cluster(**params, seed=seed2)

    jobs_1 = cluster_1._jobs._jobs_slots.copy()
    jobs_2 = cluster_2._jobs._jobs_slots.copy()

    assert not np.array_equal(jobs_1, jobs_2), \
        "Different seeds should produce different job matrices"

@given(cluster=cluster_strategy())
def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick(cluster: DeepRMCluster) -> None:

    j_idx = get_index_of_min_job_arrival_time(cluster._jobs)

    for _ in range(cluster._jobs[j_idx].arrival_time):
        assert cluster._jobs[j_idx].status == Status.NotCreated
        cluster.execute_clock_tick()

    assert cluster._jobs[j_idx].status == Status.Pending

@given(data=st.data(), params=cluster_params_strategy(), seed=seed_strategy)
def test_select_single_job_and_run_until_ticks_equal_to_job_length(
    data: DataObject,
    params: dict,
    seed: int
) -> None:
    cluster = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    j_idx = get_index_of_min_job_arrival_time(cluster._jobs)

    m_idx = data.draw(st.integers(min_value=0, max_value=params["n_machines"] - 1))
    for _ in range(cluster._jobs[j_idx].arrival_time):
        assert cluster._jobs[j_idx].status == Status.NotCreated
        cluster.execute_clock_tick()

    assert cluster._jobs[j_idx].status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert cluster._jobs[j_idx].status == Status.Running

    for _ in range(cluster._jobs[j_idx].length):
        assert cluster._jobs[j_idx].status == Status.Running
        cluster.execute_clock_tick()

    assert cluster._jobs[j_idx].status == Status.Completed

@given(cluster=cluster_strategy())
def test_cluster_run_with_random_scheduler_until_completion(cluster: DeepRMCluster) -> None:
    scheduler = RandomScheduler(cluster.is_allocation_possible)

    while not cluster.is_finished():
        output = scheduler.schedule(cluster._machines, cluster._jobs)
        if output is None:
            cluster.execute_clock_tick()
        else:
            is_schedule_succeed = cluster.schedule(*output)
            assert is_schedule_succeed

    assert all(
        job.status == Status.Completed
        for job in cluster._jobs

    )
