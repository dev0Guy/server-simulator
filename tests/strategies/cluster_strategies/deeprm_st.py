from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from src.envs.cluster_simulator.deep_rm import DeepRMCreators, DeepRMCluster
from tests.strategies.cluster_strategies.proto import ClusterStrategies


class DeepRMStrategies(ClusterStrategies[DeepRMCluster]):
    @staticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]:
        return st.fixed_dictionaries(
            {
                "n_machines": st.integers(1, 10),
                "n_jobs": st.integers(1, 20),
                "n_resources": st.integers(1, 5),
                "n_resource_unit": st.integers(1, 20),
                "n_ticks": st.integers(2, 200),
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
    def creation() -> SearchStrategy[DeepRMCluster]:
        params = DeepRMStrategies.initialization_parameters()
        return params.map(lambda p: DeepRMCreators.generate_default_cluster(**p))
