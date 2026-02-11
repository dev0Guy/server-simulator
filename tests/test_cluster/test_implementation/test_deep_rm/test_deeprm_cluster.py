from src.cluster.implementation.deep_rm.jobs import DeepRMJobsConvertor
from src.cluster.implementation.deep_rm.machines import DeepRMMachinesConvertor
from src.scheduler.random_scheduler import RandomScheduler
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from src.cluster.implementation.deep_rm import DeepRMCreators, DeepRMCluster
from src.cluster.core.job import Status
import numpy as np
import pytest
from src.cluster.implementation.deep_rm import DeepRMCluster
from hypothesis import given, strategies as st, settings, HealthCheck

from tests.test_cluster.test_implementation.test_single_slot.test_single_slot_cluster import seed_strategy
from tests.strategies.cluster_strategies import DeepRMStrategies

@pytest.fixture(scope="module")
def jobs_convertor() -> DeepRMJobsConvertor:
    return DeepRMJobsConvertor()

@pytest.fixture(scope="module")
def machine_convertor() -> DeepRMMachinesConvertor:
    return DeepRMMachinesConvertor()

@pytest.mark.usefixtures("machine_convertor", "jobs_convertor")
@given(cluster=DeepRMStrategies.creation())
def test_float_cluster_creation(cluster: DeepRMCluster, machine_convertor: DeepRMMachinesConvertor, jobs_convertor: DeepRMJobsConvertor):
    initialize_machines_usage = machine_convertor.to_representation(cluster._machines)
    initialize_jobs_usage, initialize_jobs_status, initialize_job_arrivals_time = jobs_convertor.to_representation(cluster._jobs)

    assert initialize_machines_usage.shape[0] == cluster.n_machines
    assert initialize_jobs_usage.shape[0] == cluster.n_jobs
    assert np.all(initialize_machines_usage == 1.0)
    assert np.all(initialize_jobs_usage >= 0.0)
    assert all(status in (Status.Pending, Status.NotCreated) for status in initialize_jobs_status)
    assert np.all(initialize_job_arrivals_time >= 0)

@pytest.mark.usefixtures("jobs_convertor")
@given(params=DeepRMStrategies.initialization_parameters(), seed=seed_strategy)
def test_reproducibility(params: dict, seed: int, jobs_convertor: DeepRMJobsConvertor):
    cluster1 = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    cluster2 = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    initialize_jobs_1_usage, initialize_job_1_status, initialize_job_1_arrivals_time = jobs_convertor.to_representation(cluster1._jobs)
    initialize_jobs_2_usage, initialize_job_2_status, initialize_job_2_arrivals_time = jobs_convertor.to_representation(cluster2._jobs)

    np.testing.assert_array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage)
    np.testing.assert_array_equal(initialize_job_2_status, initialize_job_1_status)
    np.testing.assert_array_equal(initialize_job_1_arrivals_time, initialize_job_2_arrivals_time)

@pytest.mark.usefixtures("jobs_convertor")
@given(
    params=DeepRMStrategies.initialization_parameters(),
    seed1=seed_strategy,
    seed2=seed_strategy
)
@pytest.mark.xfail(reason="Different seeds can still create same cluster Thus flakiness", strict=False)
def test_different_between_seeds(params: dict, seed1: int, seed2: int, jobs_convertor: DeepRMJobsConvertor):
    assume(seed1 != seed2)

    cluster_1 = DeepRMCreators.generate_default_cluster(**params, seed=seed1)
    cluster_2 = DeepRMCreators.generate_default_cluster(**params, seed=seed2)

    initialize_jobs_1_usage, initialize_job_1_status, initialize_job_1_arrivals_time = jobs_convertor.to_representation(cluster_1._jobs)
    initialize_jobs_2_usage, initialize_job_2_status, initialize_job_2_arrivals_time = jobs_convertor.to_representation(cluster_2._jobs)

    assert not np.array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage), "Different seeds should produce different job matrices"
    assert not np.array_equal(initialize_job_1_status, initialize_job_2_status), "Different seeds should produce different job matrices"
    assert not np.array_equal(initialize_job_1_arrivals_time, initialize_job_2_arrivals_time), "Different seeds should produce different job matrices"

@given(
    params=DeepRMStrategies.initialization_parameters(),
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

@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(cluster=DeepRMStrategies.creation(), j_idx=st.integers(0))
def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick(cluster: DeepRMCluster, j_idx: int) -> None:
    assume(j_idx < cluster.n_jobs)
    job = cluster._jobs[j_idx]
    assume(job.status == Status.NotCreated)

    for _ in range(job.arrival_time):
        assert job.status == Status.NotCreated
        cluster.execute_clock_tick()

    assert job.status == Status.Pending

@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    cluster=DeepRMStrategies.creation(),
    m_idx=st.integers(0),
    j_idx=st.integers(0),
)
def test_select_single_job_and_run_until_ticks_equal_to_job_length(
    cluster: DeepRMCluster,
    m_idx: int,
    j_idx: int,
) -> None:
    assume(m_idx < cluster.n_machines)
    assume(j_idx < cluster.n_jobs)
    job = cluster._jobs[j_idx]
    assume(job.status == Status.Pending)

    for _ in range(job.arrival_time):
        assert job.status == Status.NotCreated
        cluster.execute_clock_tick()

    assert job.status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert job.status == Status.Running

    for _ in range(job.length):
        assert job.status == Status.Running
        cluster.execute_clock_tick()

    assert job.status == Status.Completed

@given(cluster=DeepRMStrategies.creation())
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

@pytest.mark.usefixtures("machine_convertor", "jobs_convertor")
@given(cluster=DeepRMStrategies.creation())
def test_float_cluster_creation(cluster: DeepRMCluster, jobs_convertor: DeepRMJobsConvertor, machine_convertor: DeepRMMachinesConvertor):
    initialize_jobs_usage, initialize_jobs_status, initialize_jobs_arrival_time = jobs_convertor.to_representation(cluster._jobs)
    initialize_machines_usage = machine_convertor.to_representation(cluster._machines)

    assert initialize_machines_usage.shape[0] == cluster.n_machines
    assert initialize_jobs_usage.shape[0] == initialize_jobs_status.shape[0] == initialize_jobs_arrival_time.shape[0] == cluster.n_jobs
    assert np.all(initialize_machines_usage == 1.0)
    assert np.all(initialize_jobs_usage >= 0.0)
    assert all(status in (Status.Pending, Status.NotCreated) for status in initialize_jobs_status)

@pytest.mark.usefixtures("jobs_convertor")
@given(params=DeepRMStrategies.initialization_parameters(), seed=seed_strategy)
def test_reproducibility(params: dict, seed: int, jobs_convertor: DeepRMJobsConvertor):
    cluster1 = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    cluster2 = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    initialize_jobs_1_usage, initialize_jobs_1_status, initialize_jobs_1_arrival_time = jobs_convertor.to_representation(cluster1._jobs)
    initialize_jobs_2_usage, initialize_jobs_2_status, initialize_jobs_2_arrival_time = jobs_convertor.to_representation(cluster2._jobs)

    np.testing.assert_array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage)
    np.testing.assert_array_equal(initialize_jobs_1_status, initialize_jobs_2_status)
    np.testing.assert_array_equal(initialize_jobs_1_arrival_time, initialize_jobs_2_arrival_time)

@pytest.mark.usefixtures("jobs_convertor")
@given(
    params=DeepRMStrategies.initialization_parameters(),
    seed1=seed_strategy,
    seed2=seed_strategy
)
@pytest.mark.xfail(reason="Different seeds can still create same cluster Thus flakiness", strict=False)
def test_different_between_seeds(params: dict, seed1: int, seed2: int, jobs_convertor: DeepRMJobsConvertor):
    assume(seed1 != seed2)

    cluster_1 = DeepRMCreators.generate_default_cluster(**params, seed=seed1)
    cluster_2 = DeepRMCreators.generate_default_cluster(**params, seed=seed2)

    initialize_jobs_1_usage, initialize_jobs_1_status, initialize_jobs_1_arrival_time = jobs_convertor.to_representation(cluster_1._jobs)
    initialize_jobs_2_usage, initialize_jobs_2_status, initialize_jobs_2_arrival_time = jobs_convertor.to_representation(cluster_2._jobs)

    assert not np.array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage), "Different seeds should produce different job matrices"
    assert not np.array_equal(initialize_jobs_1_status, initialize_jobs_1_status), "Different seeds should produce different job matrices"
    assert not np.array_equal(initialize_jobs_1_arrival_time, initialize_jobs_2_arrival_time), "Different seeds should produce different job matrices"

@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(cluster=DeepRMStrategies.creation(), j_idx=st.integers(0))
def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick(cluster: DeepRMCluster, j_idx: int) -> None:
    assume(j_idx < cluster.n_jobs)
    job = cluster._jobs[j_idx]
    assume(job.status == Status.NotCreated)

    for _ in range(job.arrival_time):
        assert job.status == Status.NotCreated
        cluster.execute_clock_tick()

    assert job.status == Status.Pending

@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    cluster=DeepRMStrategies.creation(),
    m_idx=st.integers(0),
    j_idx=st.integers(0),
)
def test_select_single_job_and_run_until_ticks_equal_to_job_length(
    cluster: DeepRMCluster,
    m_idx: int,
    j_idx: int,
) -> None:
    assume(m_idx < cluster.n_machines)
    assume(j_idx < cluster.n_jobs)
    job = cluster._jobs[j_idx]
    assume(job.status == Status.Pending)

    for _ in range(job.arrival_time):
        assert job.status == Status.NotCreated
        cluster.execute_clock_tick()

    assert job.status == Status.Pending
    assert cluster.schedule(m_idx, j_idx)
    assert job.status == Status.Running

    for _ in range(job.length):
        assert job.status == Status.Running
        cluster.execute_clock_tick()

    assert job.status == Status.Completed

@given(cluster=DeepRMStrategies.creation())
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

@given(params=DeepRMStrategies.initialization_parameters(), seed=seed_strategy)
def test_cluster_reset_functionality(params: dict, seed: int) -> None:
    cluster = DeepRMCreators.generate_default_cluster(**params, seed=seed)
    prev_jobs_status = cluster._jobs._job_status.copy()

    cluster.execute_clock_tick()
    cluster.reset(seed=seed)

    after_rest_jobs_status = cluster._jobs._job_status.copy()

    assert np.all(prev_jobs_status == after_rest_jobs_status)
    assert cluster._current_tick == 0

@given(cluster=DeepRMStrategies.creation())
def test_cluster_execute_with_none_possible_action(cluster: DeepRMCluster) -> None:
    with pytest.raises(RuntimeError):
        cluster.execute(5)