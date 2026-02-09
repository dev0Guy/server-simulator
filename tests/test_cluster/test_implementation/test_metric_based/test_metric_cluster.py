import pytest

from src.cluster.core.job import Status
import numpy as np

from src.cluster.implementation.metric_based import MetricClusterCreator
from src.cluster.implementation.metric_based import MetricCluster
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.cluster_strategies import MetricClusterStrategies
from tests.test_cluster.test_implementation.test_single_slot.test_single_slot_cluster import seed_strategy
import typing as tp

EPSILON = np.finfo(float).eps


@given(cluster=MetricClusterStrategies.creation())
def test_cluster_creation(
    cluster: MetricCluster
) -> None:
    observation = cluster.get_representation()

    assert observation["machines"].shape[0] == cluster.n_machines
    assert observation["jobs"].shape[0] == cluster.n_jobs
    assert np.all(observation["machines"] == 1.0)
    assert np.all(observation["jobs"] >= 0.0)

    for job in cluster._jobs:
        assert job.status == Status.Pending or job.status == Status.NotCreated


@given(
    params=MetricClusterStrategies.initialization_parameters(),
    seed=seed_strategy,
)
def test_reproducibility(
    params: dict,
    seed:  tp.Optional[tp.SupportsFloat],
):
    cluster1 = MetricClusterCreator.generate_default(**params, seed=seed)
    cluster2 = MetricClusterCreator.generate_default(**params, seed=seed)

    jobs1 = cluster1._jobs._jobs_slots.copy()
    jobs2 = cluster2._jobs._jobs_slots.copy()

    np.testing.assert_array_equal(jobs1, jobs2)


@given(
    params=MetricClusterStrategies.initialization_parameters(),
    seed1=seed_strategy,
    seed2=seed_strategy
)
@pytest.mark.xfail(reason="Different seeds can still create same cluster Thus flakiness", strict=False)
def test_different_between_seeds(params: dict, seed1: int, seed2: int):
    assume(seed1 != seed2)

    cluster_1 = MetricClusterCreator.generate_default(**params, seed=seed1)
    cluster_2 = MetricClusterCreator.generate_default(**params, seed=seed2)

    jobs_1 = cluster_1._jobs._jobs_slots.copy()
    jobs_2 = cluster_2._jobs._jobs_slots.copy()

    assert not np.array_equal(jobs_1, jobs_2), \
        "Different seeds should produce different job matrices"


@given(
    cluster=MetricClusterStrategies.creation(),
    machine_idx=st.integers(0),
    job_idx=st.integers(0)
)
def test_schedule_available_on_machine_when_job_pending(
        cluster: MetricCluster,
        machine_idx: int,
        job_idx: int
) -> None:
    assume(machine_idx < cluster.n_machines)
    assume(job_idx < cluster.n_jobs)
    machine = cluster._machines[machine_idx]
    job = cluster._jobs[job_idx]
    assume(job.status == Status.Pending)
    assume(np.all(machine.free_space >= job.usage))

    assert cluster.is_allocation_possible(machine, job)
    assert cluster.schedule(
        machine_idx, job_idx), "Schedule should be possible specially when `is_allocation_possible` is true"


@given(
    cluster=MetricClusterStrategies.creation(),
    machine_idx=st.integers(0),
    job_idx=st.integers(0)
)
def test_schedule_full_machine(
        cluster: MetricCluster,
        machine_idx: int,
        job_idx: int
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
@given(
    cluster=MetricClusterStrategies.creation(),
    job_idx=st.integers(0)
)
def test_job_status_change_to_pending_when_arrival_time_equal_to_current_tick(
        cluster: MetricCluster,
        job_idx: int
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
        cluster: MetricCluster,
        machine_idx: int,
        job_idx: int
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
def test_cluster_run_with_random_scheduler_until_completion(cluster: MetricCluster) -> None:
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
