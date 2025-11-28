import typing as tp

import numpy as np

from src.envs.basic import BasicClusterEnv
from src.envs.core.cluster import ClusterABC
from src.envs.core.proto.job import Status


class Information(tp.TypedDict):
    jobs_status: np.ndarray


def extract_information(cluster: ClusterABC[tp.SupportsFloat]) -> Information:
    job_status = np.array([job.status for job in cluster._jobs])
    return Information(jobs_status=job_status)


def extract_reward(prev: Information, current: Information) -> tp.SupportsFloat:
    changed_to_completed = np.sum(
        (prev["jobs_status"] != current["jobs_status"])
        & (current["jobs_status"] == Status.Completed)
    )
    return changed_to_completed






