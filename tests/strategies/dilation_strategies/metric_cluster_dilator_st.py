from typing import TypedDict, Tuple, Callable

from hypothesis.strategies import SearchStrategy
from hypothesis import strategies as st, assume
import numpy as np

from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from src.envs import BasicClusterEnv
from src.envs.utils.info_builders.base import BaceClusterInformationExtractor
from src.envs.utils.observation_extractors.metric_observation_extractor import MetricClusterObservationCreator
from src.envs.utils.reward_caculators.base import DifferentInPendingJobsRewardCaculator
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
    def initialization_parameters() -> SearchStrategy[dict]:
        operation_st = st.sampled_from(
            MetricClusterDilationStrategies.PossibleOperations)
        fill_value_st = st.floats()
        return st.fixed_dictionaries(
            MetricBasedDilationKwargs(  # type: ignore
                operation=operation_st,  # type: ignore
                fill_value=fill_value_st  # type: ignore
            )
        )

    @staticmethod
    @st.composite
    def creation(draw) -> SearchStrategy[Dilator]:
        cluster = draw(MetricClusterStrategies.creation())
        base_env = BasicClusterEnv(
            cluster,
            reward_caculator=DifferentInPendingJobsRewardCaculator(),
            info_builder=BaceClusterInformationExtractor(),
            obs_extractor=MetricClusterObservationCreator()
        )
        n_machines = base_env.observation_space["machines"].shape[0]
        kernel_st = st.tuples(
            st.integers(1, max_value=n_machines),
            st.integers(1, max_value=n_machines)
        )
        kernel = draw(kernel_st)
        kernel_machine_view = kernel[0]*kernel[1]
        assume(1 < kernel_machine_view < n_machines)
        # TODO: understand why the hell
        assume(kernel[0] > 1 and kernel[1] > 1)
        params = draw(MetricClusterDilationStrategies.initialization_parameters())
        params["kernel"] = kernel

        return DilatorWrapper(  # type: ignore
            base_env,
            dilator_cls=MetricBasedDilator,
            **params
        )
