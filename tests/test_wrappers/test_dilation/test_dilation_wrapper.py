import numpy as np
import logging

from tests.test_cluster.test_implementation.test_metric_based.tests_metric_dilation import kernel_strategy
from tests.test_cluster.test_utils.test_array_operations import reduction_operation_strategy
from tests.test_envs.test_basic_env import cluster_env_strategy

logging.basicConfig(
    level=logging.DEBUG,  # minimum level to log
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)

from src.cluster.implementation.metric_based import MetricClusterCreator, MetricCluster
from src.envs import BasicClusterEnv
from src.wrappers.dilation.dilation_wrapper import DilatorWrapper, EnvWrapperAction
from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from hypothesis import given, strategies as st, assume, settings, HealthCheck, reproduce_failure

## TODO: Create Composite that create the cluster

# @st.composite
# def dilated_cluster_env_strategy(draw) -> DilatorWrapper:
#     base_env = draw(cluster_env_strategy())
#     kernel = draw(kernel_strategy)
#     operation = draw(reduction_operation_strategy)
#     wrapped_env = DilatorWrapper(
#         base_env,
#         dilator_type=MetricBasedDilator,
#         kernel=kernel,
#         operation=operation,
#         fill_value=0.0,
#     )
#
#     assume(wrapped_env._dilator._n_levels > 1)
#     print(wrapped_env)
#     return wrapped_env
#
# @given(dilated_cluster_env_strategy())
# def test_example(env_wrapped) -> None:
#     print(env_wrapped)


seed = 0
cluster = MetricClusterCreator.generate_default(
    n_machines=20,
    n_jobs=4,
    n_ticks=2,
    n_resources=2,
    is_offline=True,
    seed=seed
)

def reward_func(x, info)-> float:
    return 0.0

original_env = BasicClusterEnv(cluster,reward_func=reward_func, info_func=lambda _: {})
obs, _ = original_env.reset(seed=seed)

env = DilatorWrapper(original_env, dilator_type=MetricBasedDilator, kernel=(5,3), operation=np.max, fill_value=0.0)
action = EnvWrapperAction(selected_machine_cell=(0,0), selected_job=3, execute_schedule_command=False, contract=False)

_, _ = env.reset()

while True:
    obs, *_ = env.step(action)
    obs, *_ = env.step(action)
    break
