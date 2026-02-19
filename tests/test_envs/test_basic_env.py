import logging
from typing import Tuple

import numpy as np

from src.envs.cluster_simulator.base.internal.job import Status
from src.envs.cluster_simulator.basic import BasicClusterEnv
from hypothesis import given, settings

from src.envs.cluster_simulator.base.extractors.information import ClusterInformation
from src.envs.cluster_simulator.base.extractors.observation import ClusterObservation
from src.experiments.schedulers.random_scheduler import RandomScheduler
from tests.strategies.env_strategies.basic_env_st import BasicGymEnvironmentStrategies
from src.envs.cluster_simulator.basic import EnvironmentAction


@given(env=BasicGymEnvironmentStrategies.creation())
def test_env_reset(env: BasicClusterEnv):
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert (
        info["n_machines"] == env._cluster.n_machines
        and info["n_jobs"] == env._cluster.n_jobs
    )
    assert all(
        status in (Status.Pending, Status.NotCreated) for status in info["jobs_status"]
    )
    assert info["current_tick"] == 0


@given(env=BasicGymEnvironmentStrategies.creation())
def test_step_clock_tick(env: BasicClusterEnv):
    acceptable_status_after_time_forward = [
        (Status.NotCreated, Status.NotCreated),
        (Status.NotCreated, Status.Pending),
        (Status.Pending, Status.Pending),
    ]
    _, prev_info = env.reset()
    skip_time_action = EnvironmentAction(True, (-1, -1))
    obs, reward, terminated, truncated, current_info = env.step(skip_time_action)
    assert env.observation_space.contains(obs)
    assert prev_info["current_tick"] + 1 == current_info["current_tick"]
    assert all(
        (prev_status, current_status) in acceptable_status_after_time_forward
        for prev_status, current_status in zip(
            prev_info["jobs_status"], current_info["jobs_status"]
        )
    )
    assert bool(terminated) is False and bool(truncated) is False


@given(params=BasicGymEnvironmentStrategies.creation_with_schedule_option())
def test_step_schedule(
    params: Tuple[BasicClusterEnv, ClusterObservation, ClusterInformation, int, int],
):
    # PROBLEM HERE
    env, prev_obs, prev_info, m_idx, j_idx = params

    schedule_action = EnvironmentAction(False, (m_idx, j_idx))
    current_obs, reward, terminated, truncated, current_info = env.step(schedule_action)

    assert env.observation_space.contains(current_obs)
    assert np.all(current_obs["machines"][m_idx] >= 0)
    assert current_info["jobs_status"][j_idx] == Status.Running

    all_jobs_status_except_schedule_stay_the_same = all(
        prev_status == current_status
        for idx, (prev_status, current_status) in enumerate(
            zip(current_info["jobs_status"], prev_info["jobs_status"])
        )
        if idx != j_idx
    )
    all_machines_free_space_stay_the_same = all(
        np.allclose(prev_machine, current_machine)
        for idx, (prev_machine, current_machine) in enumerate(
            zip(current_obs["machines"], prev_obs["machines"])
        )
        if idx != m_idx
    )
    assert all_jobs_status_except_schedule_stay_the_same
    assert all_machines_free_space_stay_the_same


@given(env=BasicGymEnvironmentStrategies.creation())
@settings(max_examples=1_000)
def test_env_run_with_random_scheduler_until_completion(env: BasicClusterEnv) -> None:
    logging.info("Starting test_env_run_with_random_scheduler_until_completion")
    _, prev_info = env.reset()
    cluster = env._cluster
    terminated = False
    scheduler = RandomScheduler(cluster.is_allocation_possible)
    while not terminated:
        match scheduler.schedule(cluster._machines, cluster._jobs):
            case None:
                action = EnvironmentAction(True, (-1, -1))
            case m_idx, j_idx:
                action = EnvironmentAction(False, (m_idx, j_idx))
            case _:
                raise ValueError
        current_obs, reward, terminated, truncated, current_info = env.step(action)
        assert env.observation_space.contains(current_obs)
    assert terminated and all(job.status == Status.Completed for job in cluster._jobs)
