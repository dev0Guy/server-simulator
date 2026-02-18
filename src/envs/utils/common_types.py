from typing import TypeVar

from src.envs.cluster_simulator.core.cluster import ClusterABC

Cluster = TypeVar("Cluster", bound=ClusterABC)
