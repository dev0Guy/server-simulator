import pytest

from server.envs.single_slot import create_float_cluster
from server.envs.core.proto.job import Status as JobStatus
import numpy as np

@pytest.fixture
def n_machines() -> int:
    return 3

@pytest.fixture
def n_jobs() -> int:
    return 5

@pytest.mark.usefixtures
def test_float_cluster_creation(n_machines: int, n_jobs: int):
    cluster = create_float_cluster(n_machines, n_jobs)
    obs = cluster.get_observation()

    machines_obs = obs["machines"]
    jobs_obs = obs["jobs"]

    assert machines_obs.shape[0] == n_machines
    assert jobs_obs.shape[0] == n_jobs
    assert np.all(jobs_obs >= 0.0) and np.all(jobs_obs <= 1.0), "All cell value should be in range [0,1]"
    assert np.all(machines_obs) == 1.0
    assert all(job.status == JobStatus.Pending for job in cluster._jobs)

@pytest.mark.usefixtures
def test_reproducibility(n_machines: int, n_jobs: int):
    cluster1 = create_float_cluster(n_machines, n_jobs, seed=42)
    cluster2 = create_float_cluster(n_machines, n_jobs, seed=42)

    jobs1 = cluster1.get_observation()["jobs"].copy()
    jobs2 = cluster2.get_observation()["jobs"].copy()

    np.testing.assert_allclose(jobs1, jobs2)

@pytest.mark.usefixtures
def test_schedule_available(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, 2)

    cluster = create_float_cluster(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"
    assert cluster._jobs[j_idx].status == JobStatus.Running

    after_schedule_observation = cluster.get_observation()
    assert np.equal(after_schedule_observation["machines"][m_idx], start_observation["machines"][m_idx] - start_observation["jobs"][j_idx])


@pytest.mark.usefixtures
def test_schedule_twice_the_same(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, 2)

    cluster = create_float_cluster(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"
    assert cluster._jobs[j_idx].status == JobStatus.Running

    after_schedule_observation = cluster.get_observation()
    assert np.equal(after_schedule_observation["machines"][m_idx], start_observation["machines"][m_idx] - start_observation["jobs"][j_idx])

    assert not cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_second_schedule_observation = cluster.get_observation()
    assert np.all(after_schedule_observation["machines"] == after_second_schedule_observation["machines"]), "UnSchedule action change machine state, which its shouldn't"
    assert np.all(after_schedule_observation["jobs"] == after_second_schedule_observation["jobs"]), "UnSchedule action change jobs state, which its shouldn't"

@pytest.mark.usefixtures
def test_schedule_and_tick(n_machines: int, n_jobs: int) -> None:
    m_idx, j_idx = (0, 2)
    cluster = create_float_cluster(n_machines, n_jobs)
    start_observation = cluster.get_observation()

    assert cluster.schedule(m_idx, j_idx), "scheduling should be available"

    after_schedule_observation = cluster.get_observation()
    assert np.equal(after_schedule_observation["machines"][m_idx], start_observation["machines"][m_idx] - start_observation["jobs"][j_idx])
    assert not cluster.schedule(m_idx, j_idx), "scheduling should be available"

    cluster.execute_clock_tick()
    after_tick_observation = cluster.get_observation()

    assert cluster._jobs[j_idx].status == JobStatus.Completed
    assert np.all(after_tick_observation["machines"] == start_observation["machines"]), "after tick schedule jobs should be finished and clear from machine"

@pytest.mark.usefixtures
def test_tick_without_schedule(n_machines: int, n_jobs: int) -> None:
    cluster = create_float_cluster(n_machines, n_jobs)

    start_observation = cluster.get_observation()

    cluster.execute_clock_tick()
    after_tick_observation = cluster.get_observation()

    assert np.all(start_observation["machines"] == after_tick_observation["machines"])
    assert np.all(start_observation["jobs"] == after_tick_observation["jobs"])

