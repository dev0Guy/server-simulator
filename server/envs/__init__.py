import typing as tp

import gymnasium as gym
import numpy as np

from server.envs.basic import BasicClusterEnv
from server.envs.core.cluster import Cluster
from server.envs.core.proto.job import Status
from server.envs.single_slot import generate_single_slot_cluster


class Information(tp.TypedDict):
    jobs_status: np.ndarray


def extract_information(cluster: Cluster[tp.SupportsFloat]) -> Information:
    job_status = np.array([job.status for job in cluster._jobs])
    return Information(jobs_status=job_status)


def extract_reward(prev: Information, current: Information) -> tp.SupportsFloat:
    changed_to_completed = np.sum(
        (prev["jobs_status"] != current["jobs_status"])
        & (current["jobs_status"] == Status.Completed)
    )
    return changed_to_completed


def create_single_slot_env(
    n_machines: int = 1, n_jobs: int = 5, seed: tp.Optional[tp.SupportsFloat] = None
) -> BasicClusterEnv[tp.SupportsFloat, Information]:
    cluster = generate_single_slot_cluster(n_machines, n_jobs, seed)
    return BasicClusterEnv(
        cluster,
        reward_func=extract_reward,
        info_func=extract_information,
    )


gym.register(
    id="server/SingleSlot-v0",
    entry_point=create_single_slot_env,
    max_episode_steps=300,
)
