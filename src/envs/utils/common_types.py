from typing import TypeVar

from src.cluster.core.cluster import ClusterABC

Cluster = TypeVar("Cluster", bound=ClusterABC)
