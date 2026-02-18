import logging
from typing import TypedDict, Tuple, Callable

from hypothesis.strategies import SearchStrategy
from hypothesis import strategies as st, assume
import numpy as np

from src.envs.cluster_simulator.metric_based.dilation import MetricBasedDilator
from src.envs import BasicClusterEnv
from src.envs.cluster_simulator.core.extractors.information import BaceClusterInformationExtractor
from src.envs.utils.observation_extractors.metric_observation_extractor import (
    MetricClusterObservationCreator,
)
from src.envs.cluster_simulator.core.extractors.reward import DifferentInPendingJobsRewardCaculator
from src.envs.wrappers.dilation_wrapper import DilatorWrapper
from tests.strategies.cluster_strategies import MetricClusterStrategies
from tests.strategies.dilation_strategies.proto import DilationStrategies, Dilator


class MetricBasedDilationKwargs(TypedDict):
    kernel: Tuple[int, int]
    operation: Callable[[np.ndarray], np.ndarray]
    fill_value: float


class MetricClusterDilationStrategies(DilationStrategies[MetricBasedDilator]):
    PossibleOperations = (np.max, np.min, np.mean)

    @staticmethod
    @st.composite
    def initialization_parameters(draw) -> SearchStrategy[dict]:
        operation = draw(
            st.sampled_from(MetricClusterDilationStrategies.PossibleOperations)
        )
        if operation == np.min:
            fill_value = np.inf
        else:
            fill_value = 0.0
        logging.debug(
            "Operation Selected is %s with fill value %s",
            operation.__name__,
            fill_value,
        )

        return MetricBasedDilationKwargs(
            operation=operation, fill_value=fill_value, kernel=(1, 1)
        )

    @staticmethod
    @st.composite
    def creation(draw) -> SearchStrategy[Dilator]:
        cluster = draw(MetricClusterStrategies.creation())
        base_env = BasicClusterEnv(
            cluster,
            reward_caculator=DifferentInPendingJobsRewardCaculator(),
            info_builder=BaceClusterInformationExtractor(),
            obs_extractor=MetricClusterObservationCreator(),
        )
        n_machines = base_env.observation_space["machines"].shape[0]
        kernel_st = st.tuples(
            st.integers(1, max_value=n_machines), st.integers(1, max_value=n_machines)
        )
        kernel = draw(kernel_st)
        kernel_machine_view = kernel[0] * kernel[1]
        assume(1 < kernel_machine_view < n_machines)
        assume(kernel[0] > 1 and kernel[1] > 1)
        params = draw(MetricClusterDilationStrategies.initialization_parameters())
        params["kernel"] = kernel
        return DilatorWrapper(  # type: ignore
            base_env, dilator_cls=MetricBasedDilator, **params
        )
