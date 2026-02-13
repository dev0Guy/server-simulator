import logging

import pytest

from src.cluster.core.job import Status
from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from src.envs.wrappers.dilation_wrapper import DilatorWrapper, DilationEnvironmentAction
from typing import Tuple, Type
from hypothesis import given, settings, HealthCheck, reproduce_failure
from src.cluster.core.dilation import AbstractDilation, DilationState
from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.dilation_strategies.metric_cluster_dilator_st import MetricClusterDilationStrategies
from src.cluster.implementation.metric_based import MetricMachinesConvertor, MetricMachines
import numpy.typing as npt

DILATOR_CLASS_OPTIONS: Tuple[Type[AbstractDilation], ...] = (MetricBasedDilator,)

def create_machines_from_representation(representation: npt.NDArray) -> MetricMachines:
    return MetricMachines(MetricBasedDilator.reshape_machines(representation))

@given(env=MetricClusterDilationStrategies.creation())
@reproduce_failure('6.148.2', b'AXicc+R0ZARCJgYN+w8MYODI5MjiyKjBAAUASdUDWw==')
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=1)
def test_example_simple_run(env: DilatorWrapper):
    logging.info("Starting test_example_simple_run")
    current_obs, current_info = env.reset()
    cluster = env.env._cluster
    incorrect_machines = 0
    while not cluster.is_finished():
        if current_info["current_tick"][0] >= current_obs["jobs_usage"].shape[-1]:
            raise pytest.fail("Should be complete by now")
        scheduler = RandomScheduler(cluster.is_allocation_possible)
        machines = create_machines_from_representation(current_obs["machines"])
        logging.info(f"jobs statues: {[j.status.name for j in cluster._jobs]}")
        match scheduler.schedule(machines, cluster._jobs):
            case None:
                action = DilationEnvironmentAction((-1,-1), -1, True, False)
            case m_idx, j_idx:
                action = DilationEnvironmentAction(
                    env.cell_from_machine_idx(m_idx),
                    j_idx,
                    False,
                    False
                )
                if isinstance(env._dilator.state, DilationState.FullyExpanded) and env._dilator.get_selected_machine(action.selected_machine_cell) >= env._n_machines:
                    action = DilationEnvironmentAction((-1, -1), -1, True, False)
                    logging.warning("Selected unreal machine")

        current_obs, reward, terminated, truncated, current_info = env.step(action)
        assert env.observation_space["machines"].contains(current_obs["machines"])
    assert terminated and all(
        job.status == Status.Completed
        for job in cluster._jobs
    )
    logging.info("All jobs are completed")