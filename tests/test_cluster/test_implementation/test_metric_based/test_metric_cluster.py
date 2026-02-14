import pytest

from src.cluster.core.job import Status
import numpy as np

from src.cluster.implementation.metric_based import (
    MetricClusterCreator,
    MetricCluster,
    MetricJobsConvertor,
    MetricMachinesConvertor,
)
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.cluster_strategies import MetricClusterStrategies
from tests.test_cluster.test_implementation.test_single_slot.test_single_slot_cluster import (
    seed_strategy,
)
import typing as tp

EPSILON = np.finfo(float).eps


@pytest.fixture(scope="module")
def machines_convertor() -> MetricMachinesConvertor:
    return MetricMachinesConvertor()


@pytest.fixture(scope="module")
def jobs_convertor() -> MetricJobsConvertor:
    return MetricJobsConvertor()


@pytest.mark.usefixtures("machines_convertor", "jobs_convertor")
@given(cluster=MetricClusterStrategies.creation())
def test_cluster_creation(
    cluster: MetricCluster,
    machines_convertor: MetricMachinesConvertor,
    jobs_convertor: MetricJobsConvertor,
) -> None:
    initialize_machines_usage = machines_convertor.to_representation(cluster._machines)
    initialize_jobs_usage, initialize_jobs_status, initialize_jobs_arrival_time = (
        jobs_convertor.to_representation(cluster._jobs)
    )
    assert initialize_machines_usage.shape[0] == cluster.n_machines
    assert initialize_jobs_usage.shape[0] == cluster.n_jobs
    assert np.all(initialize_machines_usage == 1.0)
    assert np.all(initialize_jobs_usage >= 0.0)
    assert all(
        status in (Status.Pending, Status.NotCreated)
        for status in initialize_jobs_status
    )


@pytest.mark.usefixtures("jobs_convertor")
@given(
    params=MetricClusterStrategies.initialization_parameters(),
    seed=seed_strategy,
)
def test_reproducibility(
    params: dict,
    seed: tp.Optional[tp.SupportsFloat],
    jobs_convertor: MetricJobsConvertor,
):
    cluster1 = MetricClusterCreator.generate_default(**params, seed=seed)
    cluster2 = MetricClusterCreator.generate_default(**params, seed=seed)
    (
        initialize_jobs_1_usage,
        initialize_jobs_1_status,
        initialize_jobs_1_arrival_time,
    ) = jobs_convertor.to_representation(cluster1._jobs)
    (
        initialize_jobs_2_usage,
        initialize_jobs_2_status,
        initialize_jobs_2_arrival_time,
    ) = jobs_convertor.to_representation(cluster2._jobs)

    np.testing.assert_array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage)
    np.testing.assert_array_equal(initialize_jobs_1_status, initialize_jobs_2_status)
    np.testing.assert_array_equal(
        initialize_jobs_1_arrival_time, initialize_jobs_2_arrival_time
    )


@pytest.mark.usefixtures("jobs_convertor")
@given(
    params=MetricClusterStrategies.initialization_parameters(),
    seed1=seed_strategy,
    seed2=seed_strategy,
)
@pytest.mark.xfail(
    reason="Different seeds can still create same cluster Thus flakiness", strict=False
)
def test_different_between_seeds(
    params: dict, seed1: int, seed2: int, jobs_convertor: MetricJobsConvertor
):
    assume(seed1 != seed2)

    cluster1 = MetricClusterCreator.generate_default(**params, seed=seed1)
    cluster2 = MetricClusterCreator.generate_default(**params, seed=seed2)

    (
        initialize_jobs_1_usage,
        initialize_jobs_1_status,
        initialize_jobs_1_arrival_time,
    ) = jobs_convertor.to_representation(cluster1._jobs)
    (
        initialize_jobs_2_usage,
        initialize_jobs_2_status,
        initialize_jobs_2_arrival_time,
    ) = jobs_convertor.to_representation(cluster2._jobs)

    assert not np.array_equal(initialize_jobs_1_usage, initialize_jobs_2_usage), (
        "Different seeds should produce different job matrices"
    )
    assert not np.array_equal(initialize_jobs_1_status, initialize_jobs_2_status), (
        "Different seeds should produce different job matrices"
    )
    assert not np.array_equal(
        initialize_jobs_1_arrival_time, initialize_jobs_2_arrival_time
    ), "Different seeds should produce different job matrices"


@given(
    cluster=MetricClusterStrategies.creation(),
    machine_idx=st.integers(0),
    job_idx=st.integers(0),
)
def test_schedule_available_on_machine_when_job_pending(
    cluster: MetricCluster, machine_idx: int, job_idx: int
) -> None:
    assume(machine_idx < cluster.n_machines)
    assume(job_idx < cluster.n_jobs)
    machine = cluster._machines[machine_idx]
    job = cluster._jobs[job_idx]
    assume(job.status == Status.Pending)
    assume(np.all(machine.free_space >= job.usage))

    assert cluster.is_allocation_possible(machine, job)
    assert cluster.schedule(machine_idx, job_idx), (
        "Schedule should be possible specially when `is_allocation_possible` is true"
    )


@given(
    cluster=MetricClusterStrategies.creation(),
    machine_idx=st.integers(0),
    job_idx=st.integers(0),
)
def test_schedule_full_machine(
    cluster: MetricCluster, machine_idx: int, job_idx: int
) -> None:
    assume(machine_idx < cluster.n_machines)
    assume(job_idx < cluster.n_jobs)
    machine = cluster._machines[machine_idx]
    job = cluster._jobs[job_idx]
    assume(job.status == Status.Pending)

    machine.free_space = job.usage - EPSILON

    assert not cluster.is_allocation_possible(machine, job)
    assert not cluster.schedule(machine_idx, job_idx)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(cluster=MetricClusterStrategies.creation(), job_idx=st.integers(0))
def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick(
    cluster: MetricCluster, job_idx: int
) -> None:
    assume(job_idx < cluster.n_jobs)
    job = cluster._jobs[job_idx]
    assume(job.status != Status.Pending)

    for _ in range(job.arrival_time):
        assert job.status == Status.NotCreated
        cluster.execute_clock_tick()

    assert job.status == Status.Pending


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    cluster=MetricClusterStrategies.creation(),
    machine_idx=st.integers(0),
    job_idx=st.integers(0),
)
def test_select_single_job_and_run_until_ticks_equal_to_job_length(
    cluster: MetricCluster, machine_idx: int, job_idx: int
) -> None:
    assume(machine_idx < cluster.n_machines)
    assume(job_idx < cluster.n_jobs)
    machine = cluster._machines[machine_idx]
    job = cluster._jobs[job_idx]
    assume(job.status == Status.Pending)
    assume(np.all(job.usage <= machine.free_space))

    assert cluster.schedule(machine_idx, job_idx)
    assert job.status == Status.Running

    for _ in range(job.length):
        assert job.status == Status.Running
        cluster.execute_clock_tick()

    assert job.status == Status.Completed
    assert job.tick_left is None


@given(cluster=MetricClusterStrategies.creation())
def test_cluster_run_with_random_scheduler_until_completion_1(
    cluster: MetricCluster,
) -> None:
    scheduler = RandomScheduler(cluster.is_allocation_possible)
    while not cluster.has_completed():
        output = scheduler.schedule(cluster._machines, cluster._jobs)
        if output is None:
            cluster.execute_clock_tick()
        else:
            is_schedule_succeed = cluster.schedule(*output)
            assert is_schedule_succeed
    assert all(job.status == Status.Completed for job in cluster._jobs)
