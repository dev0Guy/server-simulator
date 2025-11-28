import numpy as np

from src.cluster.core.job import Status as JobStatus
from tests.test_cluster.test_implementation.test_single_slot.utils import random_machine_with_static_machine, static_machine_with_static_machine

from hypothesis import given, strategies as st, settings

machines_strategy = st.integers(min_value=1, max_value=5)
jobs_strategy = st.integers(min_value=1, max_value=30)
seed_strategy = st.integers(0, 10_000)


@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=40)
def test_float_cluster_creation(n_machines: int, n_jobs: int):
    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    obs = cluster.get_observation()

    machines_obs = obs["machines"]
    jobs_obs = obs["jobs"]

    assert machines_obs.shape[0] == n_machines
    assert jobs_obs.shape[0] == n_jobs
    assert np.all(jobs_obs >= 0.0) and np.all(
        jobs_obs <= 1.0
    ), "All cell value should be in range [0,1]"
    assert np.all(machines_obs) == 1.0

    assert all(job.status == JobStatus.Pending for job in cluster._jobs)


@given(n_machines=machines_strategy, n_jobs=jobs_strategy, seed=seed_strategy)
@settings(max_examples=30)
def test_reproducibility(n_machines: int, n_jobs: int, seed: int):
    cluster1 = random_machine_with_static_machine(n_machines, n_jobs, seed=seed)
    cluster2 = random_machine_with_static_machine(n_machines, n_jobs, seed=seed)

    jobs1 = cluster1.get_observation()["jobs"].copy()
    jobs2 = cluster2.get_observation()["jobs"].copy()

    np.testing.assert_allclose(jobs1, jobs2)


@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_available_last_job_to_first_machine(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, n_jobs-1)

    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"
    assert cluster._jobs[j_idx].status == JobStatus.Running

    after_schedule_observation = cluster.get_observation()
    assert np.equal(
        after_schedule_observation["machines"][m_idx],
        start_observation["machines"][m_idx] - start_observation["jobs"][j_idx],
    )


@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_twice_the_same(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, n_jobs - 1)

    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"
    assert cluster._jobs[j_idx].status == JobStatus.Running

    after_schedule_observation = cluster.get_observation()
    assert np.equal(
        after_schedule_observation["machines"][m_idx],
        start_observation["machines"][m_idx] - start_observation["jobs"][j_idx],
    )

    assert not cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_second_schedule_observation = cluster.get_observation()
    assert np.all(
        after_schedule_observation["machines"]
        == after_second_schedule_observation["machines"]
    ), "UnSchedule action change machine state, which its shouldn't"
    assert np.all(
        after_schedule_observation["jobs"] == after_second_schedule_observation["jobs"]
    ), "UnSchedule action change jobs state, which its shouldn't"


@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_schedule_and_tick(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, n_jobs - 1)
    cluster = random_machine_with_static_machine(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_schedule_observation = cluster.get_observation()
    assert np.equal(
        after_schedule_observation["machines"][m_idx],
        start_observation["machines"][m_idx] - start_observation["jobs"][j_idx],
    )
    assert not cluster.schedule(m_idx, j_idx), "scheduling should be available"

    cluster.execute_clock_tick()
    after_tick_observation = cluster.get_observation()

    assert cluster._jobs[j_idx].status == JobStatus.Completed
    assert np.all(
        after_tick_observation["machines"] == start_observation["machines"]
    ), "after tick schedule jobs should be finished and clear from machine"


@given(n_machines=machines_strategy, n_jobs=jobs_strategy)
@settings(max_examples=30)
def test_tick_without_schedule(n_machines: int, n_jobs: int) -> None:
    cluster = random_machine_with_static_machine(n_machines, n_jobs)

    start_observation = cluster.get_observation()

    cluster.execute_clock_tick()
    after_tick_observation = cluster.get_observation()

    assert np.all(start_observation["machines"] == after_tick_observation["machines"])
    assert np.all(start_observation["jobs"] == after_tick_observation["jobs"])


@given(n_jobs=st.integers(1, 20), n_machines=st.integers(1, 1))
@settings(max_examples=20)
def test_single_machine_multiple_one_size_jobs(n_jobs: int, n_machines: int):
    cluster = static_machine_with_static_machine(n_machines, n_jobs)

    for j_idx in range(cluster.n_jobs):
        assert not cluster.is_finished()
        assert cluster.schedule(m_idx=0, j_idx=j_idx)
        assert not cluster.schedule(
            m_idx=0, j_idx=j_idx
        ), "After schedule machine should not be allowed to run another jobs."
        assert cluster._jobs[j_idx].status == JobStatus.Running
        cluster.execute_clock_tick()
        assert cluster._jobs[j_idx].status == JobStatus.Completed
        assert cluster._machines[0].free_space == 1.0

    assert cluster.is_finished()
    assert cluster._current_tick == cluster.n_jobs


@given(n_jobs=st.integers(1, 10).map(lambda x: x * 2), n_machines=st.integers(1, 1))
@settings(max_examples=20) # TODO: Investegate error
def test_single_machine_multiple_half_size_jobs(n_jobs: int, n_machines: int):
    static_value = 0.5
    cluster = static_machine_with_static_machine(n_machines, n_jobs, static_value=static_value)

    for j_idx in range(0, cluster.n_jobs, 2):
        assert not cluster.is_finished()
        assert cluster.schedule(m_idx=0, j_idx=j_idx)
        assert cluster._machines[0].free_space == 0.5
        assert cluster.schedule(m_idx=0, j_idx=j_idx + 1)
        assert cluster._machines[0].free_space == 0.0
        assert cluster._jobs[j_idx].status == JobStatus.Running
        assert cluster._jobs[j_idx + 1].status == JobStatus.Running
        cluster.execute_clock_tick()
        assert cluster._jobs[j_idx].status == JobStatus.Completed
        assert cluster._jobs[j_idx + 1].status == JobStatus.Completed
        assert cluster._machines[0].free_space == 1.0

    assert cluster.is_finished()
    assert cluster._current_tick == (cluster.n_jobs // 2)
