import numpy as np
import typing as tp

from src.cluster.core.cluster import ClusterAction
from src.cluster.core.job import Status
from src.envs.basic import BasicClusterEnv, EnvAction
from src.cluster.implementation.single_slot import SingleSlotCluster
from hypothesis import given, strategies as st, assume

from src.scheduler.random_scheduler import RandomScheduler
from tests.test_cluster.test_implementation.test_single_slot.utils import random_machine_with_static_machine

machines_strategy = st.integers(min_value=1, max_value=5)
jobs_strategy = st.integers(min_value=1, max_value=30)
seed_strategy = st.integers(0, 10_000)

class InfoType(tp.TypedDict):
    n_machines: int
    n_jobs: int
    jobs_status: list[Status]
    current_tick: int


def cluster_params_strategy():
    return st.fixed_dictionaries({
        "n_machines": machines_strategy,
        "n_jobs": jobs_strategy,
        "seed": seed_strategy,
    })


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

def cluster_env_strategy():
    params = cluster_params_strategy()
    return params.map(
        lambda p: BasicClusterEnv(
            random_machine_with_static_machine(**p),
            none_pending_job_change_reward,
            fixed_info_func
        )
    )


@given(env=cluster_env_strategy())
def test_env_reset(
    env: BasicClusterEnv[np.float64, InfoType]
):
   obs, info = env.reset()
   assert env.generate_observation_space(env._cluster).contains(obs)
   assert isinstance(info, dict)
   assert info["n_machines"] == env._cluster.n_machines and info["n_jobs"] == env._cluster.n_jobs
   assert all(status == Status.Pending for status in info["jobs_status"])
   assert info["current_tick"] == 0


@given(env=cluster_env_strategy())
def test_step_clock_tick(env: BasicClusterEnv[np.float64, InfoType]):
    _, prev_info = env.reset()
    skip_time_action = EnvAction(1, (-1, -1))
    obs, reward, terminated, truncated, current_info = env.step(skip_time_action)
    assert env.generate_observation_space(env._cluster).contains(obs)
    assert prev_info["current_tick"] + 1 == current_info["current_tick"]
    assert all(
        prev_status == current_status
        for prev_status, current_status in zip(prev_info["jobs_status"], current_info["jobs_status"])
    )
    assert terminated is False
    assert truncated is False
    assert reward == none_pending_job_change_reward(prev_info, current_info)


@given(env=cluster_env_strategy(), m_idx=machines_strategy, j_idx=jobs_strategy)
def test_step_schedule(
    env: BasicClusterEnv[np.float64, InfoType],
    m_idx: int,
    j_idx: int
):
    assume(m_idx < env._cluster.n_machines)
    assume(j_idx < env._cluster.n_jobs)

    prev_obs, prev_info = env.reset()
    schedule_action = EnvAction(False, (m_idx, j_idx))
    current_obs, reward, terminated, truncated, current_info = env.step(schedule_action)

    assert env.generate_observation_space(env._cluster).contains(current_obs)
    assert prev_info["jobs_status"][j_idx] == Status.Pending
    assert current_info["jobs_status"][j_idx] == Status.Running
    assert current_obs["machines"][m_idx] == prev_obs["machines"][m_idx] - prev_obs["jobs"][j_idx]
    assert reward == none_pending_job_change_reward(prev_info, current_info)


@given(env=cluster_env_strategy())
def test_env_run_with_random_scheduler_until_completion(env: BasicClusterEnv[np.float64, InfoType]) -> None:
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