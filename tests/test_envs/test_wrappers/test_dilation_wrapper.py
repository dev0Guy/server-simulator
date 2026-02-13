import logging

logging.basicConfig(
    level=logging.DEBUG,  # minimum level to log
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)



from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from src.envs.wrappers.dilation_wrapper import DilatorWrapper, DilationEnvironmentAction
from typing import Tuple, Type
from hypothesis import given, settings, HealthCheck
from src.cluster.core.dilation import AbstractDilation
from src.scheduler.random_scheduler import RandomScheduler
from tests.strategies.dilation_strategies.metric_cluster_dilator_st import MetricClusterDilationStrategies


DILATOR_CLASS_OPTIONS: Tuple[Type[AbstractDilation], ...] = (MetricBasedDilator,)


@given(env=MetricClusterDilationStrategies.creation())
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_example_simple_run(env: DilatorWrapper):
    action = DilationEnvironmentAction(
        selected_machine_cell=(0, 0),
        selected_job=0,
        execute_schedule_command=False,
        contract=False
    )
    scheduler = RandomScheduler(env.env._cluster.is_allocation_possible)
    _, _ = env.reset()
    obs, *_ = env.step(action)
    obs, *_ = env.step(action)
