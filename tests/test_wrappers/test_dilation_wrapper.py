import logging
from datetime import timedelta

import numpy as np

from src.envs.cluster_simulator.base.internal.job import Status
from src.envs.cluster_simulator.metric_based.internal.dilation import MetricBasedDilator
from src.wrappers.cluster_simulator.dilation_wrapper import DilatorWrapper, DilationEnvironmentAction
from typing import Tuple, Type
from hypothesis import given, settings, HealthCheck, assume
from src.envs.cluster_simulator.base.internal.dilation import AbstractDilation, DilationState
from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.dilation_strategies.metric_cluster_dilator_st import (
    MetricClusterDilationStrategies,
)
from src.envs.cluster_simulator.metric_based import MetricMachines
import numpy.typing as npt

DILATOR_CLASS_OPTIONS: Tuple[Type[AbstractDilation], ...] = (MetricBasedDilator,)


def create_machines_from_representation(
    representation: npt.NDArray,
) -> tuple[MetricMachines, tuple[int, int]]:
    m_x, m_y = representation.shape[:2]  # Store original dimensions
    machines = MetricMachines(MetricBasedDilator.reshape_machines(representation))
    return machines, (m_x, m_y)


def flat_idx_to_cell(m_idx: int, grid_shape: tuple[int, int]) -> tuple[int, int]:
    m_x, m_y = grid_shape
    return m_idx // m_y, m_idx % m_y


@given(env=MetricClusterDilationStrategies.creation())
@settings(
    suppress_health_check=[HealthCheck.filter_too_much], deadline=timedelta(minutes=10)
)
def test_example_simple_run(env: DilatorWrapper):
    logging.info("Starting test_example_simple_run")
    current_obs, current_info = env.reset()
    cluster = env.env._cluster
    assume(env._dilator._operation == np.max)  # NOTICE:
    terminated, truncated = False, False
    while not terminated and not truncated:
        scheduler = RandomScheduler(cluster.is_allocation_possible)
        machines, grid_shape = create_machines_from_representation(
            current_obs["machines"]
        )
        logging.info(f"jobs statues: {[j.status.name for j in cluster._jobs]}")
        match scheduler.schedule(machines, cluster._jobs):
            case None:
                action = DilationEnvironmentAction((-1, -1), -1, True, False)
            case m_idx, j_idx:
                logging.info(
                    "Selected real machine: %d with max %d",
                    m_idx,
                    np.max(machines[m_idx].free_space),
                )
                cell_coords = flat_idx_to_cell(m_idx, grid_shape)
                action = DilationEnvironmentAction(cell_coords, j_idx, False, False)
                if (
                    isinstance(env._dilator.state, DilationState.FullyExpanded)
                    and env._dilator.get_selected_machine(action.selected_machine_cell)
                    >= env._n_machines
                ):
                    action = DilationEnvironmentAction((-1, -1), -1, True, False)
                    logging.warning("Selected unreal machine")

        current_obs, reward, terminated, truncated, current_info = env.step(action)
        assert env.observation_space["machines"].contains(current_obs["machines"])

    assert (terminated or truncated) and all(
        job.status in (Status.Running, Status.Completed) for job in cluster._jobs
    )
    logging.info("All jobs are completed")
