import numpy as np
import pytest

from src.envs.cluster_simulator.core.job import Status as JobStatus
from src.envs.cluster_simulator.single_slot.jobs import SingleSlotJobsConvertor
from src.envs.cluster_simulator.single_slot.machines import SingleSlotMachinesConvertor
from tests.test_envs.test_cluster_simulator.test_single_slot.utils import (
    random_machine_with_static_machine,
    static_machine_with_static_machine,
)

from hypothesis import given, strategies as st, settings

JOB_CONVERTOR = SingleSlotJobsConvertor()
MACHINE_CONVERTOR = SingleSlotMachinesConvertor()

machines_strategy = st.integers(min_value=1, max_value=5)
jobs_strategy = st.integers(min_value=1, max_value=30)
seed_strategy = st.integers(0, 10_000)


@pytest.fixture(scope="module")
def jobs_convertor() -> SingleSlotJobsConvertor:
    return SingleSlotJobsConvertor()


@pytest.fixture(scope="module")
def machine_convertor() -> SingleSlotMachinesConvertor:
    return SingleSlotMachinesConvertor()


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=40)
def test_float_cluster_creation(
    n_machines: int,
    n_jobs: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
):
    cluster = random_machine_with_static_machine(n_machines, n_jobs)

    machines_free_space = machine_convertor.to_representation(cluster._machines)
    jobs_usage, jobs_status = jobs_convertor.to_representation(cluster._jobs)

    assert machines_free_space.shape[0] == n_machines
    assert jobs_usage.shape[0] == n_jobs
    assert np.all(jobs_usage >= 0.0) and np.all(jobs_usage <= 1.0), (
        "All cell value should be in range [0,1]"
    )
    assert np.all(machines_free_space) == 1.0
    assert all(status == JobStatus.Pending for status in jobs_status)


@pytest.mark.usefixtures("jobs_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy, seed=seed_strategy)
@settings(max_examples=30)
def test_reproducibility(
    n_machines: int, n_jobs: int, seed: int, jobs_convertor: SingleSlotJobsConvertor
):
    cluster1 = random_machine_with_static_machine(n_machines, n_jobs, seed=seed)
    cluster2 = random_machine_with_static_machine(n_machines, n_jobs, seed=seed)

    jobs_1_usage, jobs_1_status = jobs_convertor.to_representation(cluster1._jobs)
    jobs2_2_usage, jobs_2_status = jobs_convertor.to_representation(cluster2._jobs)

    np.testing.assert_allclose(jobs_1_usage, jobs2_2_usage)
    assert all(
        status1 == status2 for status1, status2 in zip(jobs_1_status, jobs_2_status)
    )


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_available_last_job_to_first_machine(
    n_machines: int,
    n_jobs: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
) -> None:
    m_idx, j_idx = (0, n_jobs - 1)

    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    before_jobs_usage, before_jobs_status = jobs_convertor.to_representation(
        cluster._jobs
    )
    before_machine_usage = machine_convertor.to_representation(cluster._machines)

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_jobs_usage, after_jobs_status = jobs_convertor.to_representation(
        cluster._jobs
    )
    after_machine_usage = machine_convertor.to_representation(cluster._machines)
    assert after_jobs_status[j_idx] == JobStatus.Running

    assert np.equal(
        after_machine_usage[m_idx],
        before_machine_usage[m_idx] - before_jobs_usage[j_idx],
    )


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_twice_the_same(
    n_machines: int,
    n_jobs: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
) -> None:
    m_idx, j_idx = (0, n_jobs - 1)
    cluster = random_machine_with_static_machine(n_machines, n_jobs)

    before_machines = machine_convertor.to_representation(cluster._machines)
    before_jobs_usage, before_jobs_status = jobs_convertor.to_representation(
        cluster._jobs
    )
    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_machines = machine_convertor.to_representation(cluster._machines)
    after_jobs_usage, after_jobs_status = jobs_convertor.to_representation(
        cluster._jobs
    )
    assert after_jobs_status[j_idx] == JobStatus.Running
    assert np.equal(
        after_machines[m_idx],
        before_machines[m_idx] - before_jobs_usage[j_idx],
    )

    assert not cluster.schedule(m_idx, j_idx), "scheduling should not be available"
    after_second_machines = machine_convertor.to_representation(cluster._machines)
    after_second_jobs_usage, after_second_jobs_status = (
        jobs_convertor.to_representation(cluster._jobs)
    )
    assert np.all(after_machines == after_second_machines), (
        "UnSchedule action change machine state, which its shouldn't"
    )
    assert np.all(after_jobs_usage == after_second_jobs_usage), (
        "UnSchedule action change jobs state, which its shouldn't"
    )

    assert after_second_jobs_status == after_jobs_status


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_and_tick(
    n_machines: int,
    n_jobs: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
) -> None:
    m_idx, j_idx = (0, n_jobs - 1)
    cluster = random_machine_with_static_machine(n_machines, n_jobs)

    initialize_machines_usage = machine_convertor.to_representation(cluster._machines)
    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_schedule_machines_usage = machine_convertor.to_representation(
        cluster._machines
    )
    after_schedule_jobs_usage, after_schedule_jobs_status = (
        jobs_convertor.to_representation(cluster._jobs)
    )
    assert np.equal(
        after_schedule_machines_usage[m_idx],
        initialize_machines_usage[m_idx] - after_schedule_jobs_usage[j_idx],
    )
    assert after_schedule_jobs_status[j_idx] == JobStatus.Running

    assert not cluster.schedule(m_idx, j_idx), "scheduling should not be available"
    cluster.execute_clock_tick()
    after_tick_machines_usage = machine_convertor.to_representation(cluster._machines)
    after_tick_jobs_usage, after_tick_jobs_status = jobs_convertor.to_representation(
        cluster._jobs
    )
    assert after_tick_jobs_status[j_idx] == JobStatus.Completed
    assert np.all(after_tick_machines_usage == initialize_machines_usage), (
        "after tick schedule jobs should be finished and clear from machine"
    )


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_tick_without_schedule(
    n_machines: int,
    n_jobs: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
) -> None:
    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    initialize_machines_usage = machine_convertor.to_representation(cluster._machines)
    initialize_jobs_usage, _ = jobs_convertor.to_representation(cluster._jobs)

    cluster.execute_clock_tick()
    after_tick_machines_usage = machine_convertor.to_representation(cluster._machines)
    after_tick_jobs_usage, _ = jobs_convertor.to_representation(cluster._jobs)

    assert np.all(initialize_machines_usage == after_tick_machines_usage)
    assert np.all(initialize_jobs_usage == after_tick_jobs_usage)


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_jobs=st.integers(1, 20), n_machines=st.integers(1, 1))
@settings(max_examples=20)
def test_single_machine_multiple_one_size_jobs(
    n_jobs: int,
    n_machines: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
):
    cluster = static_machine_with_static_machine(n_machines, n_jobs)

    for j_idx in range(cluster.n_jobs):
        assert not cluster.has_completed()
        assert cluster.schedule(m_idx=0, j_idx=j_idx)
        assert not cluster.schedule(m_idx=0, j_idx=j_idx), (
            "After schedule machine should not be allowed to run another jobs."
        )
        after_schedule_jobs_usage, after_schedule_jobs_status = (
            jobs_convertor.to_representation(cluster._jobs)
        )

        assert after_schedule_jobs_status[j_idx] == JobStatus.Running
        cluster.execute_clock_tick()
        after_tick_jobs_usage, after_tick_jobs_status = (
            jobs_convertor.to_representation(cluster._jobs)
        )
        after_tick_machines_usage = machine_convertor.to_representation(
            cluster._machines
        )
        assert after_tick_jobs_status[j_idx] == JobStatus.Completed
        assert after_tick_machines_usage[0] == 1.0

    assert cluster.has_completed()
    assert cluster._current_tick == cluster.n_jobs == n_jobs


@pytest.mark.usefixtures("jobs_convertor", "machine_convertor")
@given(n_jobs=st.integers(1, 10).map(lambda x: x * 2), n_machines=st.integers(1, 1))
@settings(max_examples=20)
def test_single_machine_multiple_half_size_jobs(
    n_jobs: int,
    n_machines: int,
    jobs_convertor: SingleSlotJobsConvertor,
    machine_convertor: SingleSlotMachinesConvertor,
):
    static_value = 0.5
    cluster = static_machine_with_static_machine(
        n_machines, n_jobs, static_value=static_value
    )

    for j_idx in range(0, cluster.n_jobs, 2):
        assert not cluster.has_completed()
        assert cluster.schedule(m_idx=0, j_idx=j_idx)
        after_schedule_machines_usage = machine_convertor.to_representation(
            cluster._machines
        )
        assert after_schedule_machines_usage[0] == 0.5
        assert cluster.schedule(m_idx=0, j_idx=j_idx + 1)
        after_second_schedule_jobs_usage, after_second_schedule_jobs_status = (
            jobs_convertor.to_representation(cluster._jobs)
        )
        after_second_schedule_machines_usage = machine_convertor.to_representation(
            cluster._machines
        )
        assert after_second_schedule_machines_usage[0] == 0.0
        assert after_second_schedule_jobs_status[j_idx] == JobStatus.Running
        assert after_second_schedule_jobs_status[j_idx + 1] == JobStatus.Running
        cluster.execute_clock_tick()
        after_tick_jobs_usage, after_tick_jobs_status = (
            jobs_convertor.to_representation(cluster._jobs)
        )
        after_tick_machines_usage = machine_convertor.to_representation(
            cluster._machines
        )
        assert after_tick_jobs_status[j_idx] == JobStatus.Completed
        assert after_tick_jobs_status[j_idx + 1] == JobStatus.Completed
        assert after_tick_machines_usage[0] == 1.0
    #
    assert cluster.has_completed()
    assert cluster._current_tick == (cluster.n_jobs // 2) == (n_jobs // 2)
