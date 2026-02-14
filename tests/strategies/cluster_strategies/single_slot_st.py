from hypothesis.strategies import SearchStrategy
from hypothesis import strategies as st

from tests.strategies.cluster_strategies.proto import ClusterStrategies
from src.cluster.implementation.single_slot import (
    SingleSlotCluster,
    SingleSlotClusterCreators,
)


class SingleSlotClusterStrategies(ClusterStrategies[SingleSlotCluster]):
    @staticmethod
    def initialization_parameters() -> SearchStrategy[dict[str, int]]:
        machines_strategy = st.integers(min_value=1, max_value=5)
        jobs_strategy = st.integers(min_value=1, max_value=30)
        seed_strategy = st.integers(0, 10_000)
        return st.fixed_dictionaries(
            {
                "n_machines": machines_strategy,
                "n_jobs": jobs_strategy,
                "seed": seed_strategy,
            }
        )

    @staticmethod
    def creation() -> SearchStrategy[SingleSlotCluster]:
        params = SingleSlotClusterStrategies.initialization_parameters()

        return params.map(
            lambda p: SingleSlotCluster(
                SingleSlotClusterCreators.random_workload_creator(p["n_jobs"]),
                SingleSlotClusterCreators.static_machine_creator(p["n_jobs"]),
                seed=p["seed"],
            )
        )
