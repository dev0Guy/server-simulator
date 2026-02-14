from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from src.cluster.implementation.metric_based import MetricCluster, MetricClusterCreator
from tests.strategies.cluster_strategies.proto import ClusterStrategies


class MetricClusterStrategies(ClusterStrategies[MetricCluster]):
    @staticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]:
        return st.fixed_dictionaries(
            {
                "n_machines": st.integers(1, 20),
                "n_jobs": st.integers(1, 30),
                "n_resources": st.integers(1, 20),
                "n_ticks": st.integers(2, 5),  # TODO: change back
                "is_offline": st.booleans(),
                "poisson_lambda": st.floats(
                    min_value=0.1,
                    max_value=15.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            }
        )

    @staticmethod
    def creation() -> SearchStrategy[MetricCluster]:
        params = MetricClusterStrategies.initialization_parameters()
        return params.map(lambda p: MetricClusterCreator.generate_default(**p))
