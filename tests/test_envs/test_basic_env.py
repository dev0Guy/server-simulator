import numpy as np
import typing as tp

from src.cluster.core.job import Status
from src.envs.basic import BasicClusterEnv, EnvAction
from src.cluster.implementation.single_slot import SingleSlotCluster
from hypothesis import given, strategies as st, assume

from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.cluster_strategies.proto import ClusterStrategies
from tests.strategies.cluster_strategies import MetricClusterStrategies, DeepRMStrategies, SingleSlotClusterStrategies



CLUSTER_CLASS_OPTIONS: tp.Tuple[tp.Type[ClusterStrategies], ...] = (SingleSlotClusterStrategies, DeepRMStrategies, MetricClusterStrategies)


def machine_after_allocating_job(prev_obs: dict, m_idx, j_idx):
    return prev_obs["machines"][m_idx].astype(float) - prev_obs["jobs"][j_idx].astype(float)

class InfoType(tp.TypedDict):
    n_machines: int
    n_jobs: int
    jobs_status: list[Status]
    current_tick: int

def none_pending_job_change_reward(prev_info: InfoType, current_info: InfoType) -> float:
    prev_not_pending_jobs_count = sum(s != Status.Pending for s in prev_info["jobs_status"])
    current_not_pending_jobs_count = sum(s != Status.Pending for s in current_info["jobs_status"])
    return current_not_pending_jobs_count - prev_not_pending_jobs_count

def fixed_info_func(cluster: SingleSlotCluster) -> InfoType:
    representation: dict = cluster.get_representation()
    return InfoType(
        n_machines=representation["machines"].shape[0],
        n_jobs=representation["jobs"].shape[0],
        jobs_status=[j.status for j in iter(cluster._jobs)],
        current_tick=cluster._current_tick
    )

def is_allocation_possible(machine: np.ndarray, job: np.ndarray) -> bool:
    leftover = machine.astype(float) - job.astype(float)
    return np.all(leftover >= 0)

@st.composite
def cluster_env_strategy(draw):
    cluster_class = draw(st.sampled_from(CLUSTER_CLASS_OPTIONS))
    cluster = draw(cluster_class.creation())

    return BasicClusterEnv(
        cluster,
        none_pending_job_change_reward,
        fixed_info_func,
    )

@st.composite
def cluster_env_with_possible_allocation(draw) -> tp.Tuple[BasicClusterEnv, tp.Any, InfoType, int, int]:
    env = draw(cluster_env_strategy())
    obs, info = env.reset()

    possible_pending_jobs = [
        idx
        for idx, status in enumerate(info["jobs_status"])
        if status == Status.Pending
    ]

    assume(len(possible_pending_jobs) > 0)

    j_idx = draw(st.sampled_from(possible_pending_jobs))

    possible_machines = [
        idx
        for idx, machine in enumerate(obs["machines"])
        if is_allocation_possible(machine, obs["jobs"][j_idx])
    ]

    assume(len(possible_machines) > 0)
    m_idx = draw(st.sampled_from(possible_machines))

    return env, obs, info, m_idx, j_idx

@given(env=cluster_env_strategy())
def test_env_reset(
    env: BasicClusterEnv[np.float64, InfoType]
): # READY
   obs, info = env.reset()
   assert env.generate_observation_space(env._cluster).contains(obs)
   assert isinstance(info, dict)
   assert info["n_machines"] == env._cluster.n_machines and info["n_jobs"] == env._cluster.n_jobs
   assert all(status in (Status.Pending, Status.NotCreated) for status in info["jobs_status"])
   assert info["current_tick"] == 0

@given(env=cluster_env_strategy())
def test_step_clock_tick(env: BasicClusterEnv[np.float64, InfoType]):
    acceptable_status_after_time_forward = [(Status.NotCreated, Status.NotCreated), (Status.NotCreated, Status.Pending), (Status.Pending, Status.Pending)]
    _, prev_info = env.reset()
    skip_time_action = EnvAction(1, (-1, -1))
    obs, reward, terminated, truncated, current_info = env.step(skip_time_action)
    assert env.generate_observation_space(env._cluster).contains(obs)
    assert prev_info["current_tick"] + 1 == current_info["current_tick"]
    assert all(
        (prev_status, current_status) in acceptable_status_after_time_forward
        for prev_status, current_status in zip(prev_info["jobs_status"], current_info["jobs_status"])
    )
    assert terminated is False
    assert truncated is False
    assert reward == none_pending_job_change_reward(prev_info, current_info)


@given(params=cluster_env_with_possible_allocation())
def test_step_schedule(
    params: tp.Tuple[BasicClusterEnv, tp.Any, InfoType, int, int]
):
    env, prev_obs, prev_info, m_idx, j_idx = params

    schedule_action = EnvAction(False, (m_idx, j_idx)) # schedule action
    current_obs, reward, terminated, truncated, current_info = env.step(schedule_action)

    assert env.generate_observation_space(env._cluster).contains(current_obs)
    assert np.all(current_obs["machines"][m_idx] >= 0)
    assert current_info["jobs_status"][j_idx] == Status.Running

    all_jobs_status_except_schedule_stay_the_same = all(
        prev_status == current_status
        for idx, (prev_status, current_status) in enumerate(zip(current_info["jobs_status"], prev_info["jobs_status"]))
        if idx != j_idx
    )

    all_machines_free_space_stay_the_same = all(
        np.allclose(prev_machine, current_machine)
        for idx, (prev_machine, current_machine) in enumerate(zip(current_obs["machines"], prev_obs["machines"]))
        if idx != m_idx
    )

    assert all_jobs_status_except_schedule_stay_the_same
    assert all_machines_free_space_stay_the_same

@given(env=cluster_env_strategy())
def test_env_run_with_random_scheduler_until_completion(env: BasicClusterEnv[np.float64, InfoType]) -> None:
    # TODO: Need to update scheduler
    _, prev_info  = env.reset()
    cluster = env._cluster
    scheduler = RandomScheduler(cluster.is_allocation_possible)
    while not cluster.is_finished():
        action = None
        match scheduler.schedule(cluster._machines, cluster._jobs):
            case None:
                action = EnvAction(True, (-1,-1))
            case m_idx, j_idx:
                action = EnvAction(False, (m_idx, j_idx))
        current_obs, reward, terminated, truncated, current_info = env.step(action)
        assert env.generate_observation_space(env._cluster).contains(current_obs)
        assert reward == none_pending_job_change_reward(prev_info, current_info)
        prev_info = current_info
    assert terminated and all(
        job.status == Status.Completed
        for job in cluster._jobs
    )