from hypothesis import strategies as st, assume
from typing import TYPE_CHECKING, TypedDict

from src.cluster.core.job import Status
from src.cluster.implementation.single_slot import SingleSlotCluster
from src.envs import BasicClusterEnv
from tests.strategies.cluster_strategies import SingleSlotClusterStrategies, DeepRMStrategies, MetricClusterStrategies
import numpy as np

if not TYPE_CHECKING:
    from typing import Tuple, Type, Any


class InfoType(TypedDict):
    n_machines: int
    n_jobs: int
    jobs_status: list[Status]
    current_tick: int


class BasicGymEnvironmentStrategies:
    CLUSTER_CLASS_OPTIONS = (SingleSlotClusterStrategies, DeepRMStrategies, MetricClusterStrategies)

    @staticmethod
    @st.composite
    def creation(draw):
        cluster_class = draw(st.sampled_from(BasicGymEnvironmentStrategies.CLUSTER_CLASS_OPTIONS))
        cluster = draw(cluster_class.creation())

        return BasicClusterEnv(
            cluster,
            BasicGymEnvironmentStrategies.none_pending_job_change_reward, # type: ignore
            BasicGymEnvironmentStrategies.fixed_info_func, # type: ignore
        )

    @staticmethod
    @st.composite
    def creation_with_schedule_option(draw) -> Tuple[BasicClusterEnv, Any, InfoType, int, int]:
        env = draw(BasicGymEnvironmentStrategies.creation())
        seed = draw(st.integers(0, 10_000))
        obs, info = env.reset(seed=seed)

        possible_pending_jobs = [
            idx
            for idx, status in enumerate(info["jobs_status"])
            if status == Status.Pending
        ]

        assume(len(possible_pending_jobs) > 0)

        j_idx = draw(st.sampled_from(possible_pending_jobs))

        possible_machines = [
            idx
            for idx, machine in enumerate(obs["machines"])
            if BasicGymEnvironmentStrategies.is_allocation_possible(machine, obs["jobs"][j_idx])
        ]

        assume(len(possible_machines) > 0)
        m_idx = draw(st.sampled_from(possible_machines))

        return env, obs, info, m_idx, j_idx

    @staticmethod
    def none_pending_job_change_reward(prev_info: InfoType, current_info: InfoType) -> float:
        prev_not_pending_jobs_count = sum(s != Status.Pending for s in prev_info["jobs_status"])
        current_not_pending_jobs_count = sum(s != Status.Pending for s in current_info["jobs_status"])
        return current_not_pending_jobs_count - prev_not_pending_jobs_count

    @staticmethod
    def fixed_info_func(cluster: SingleSlotCluster) -> InfoType:
        representation: dict = cluster.get_representation() # type: ignore
        return InfoType(
            n_machines=representation["machines"].shape[0],
            n_jobs=representation["jobs"].shape[0],
            jobs_status=[j.status for j in iter(cluster._jobs)],
            current_tick=cluster._current_tick
        )

    @staticmethod
    def is_allocation_possible(machine: np.ndarray, job: np.ndarray) -> bool:
        leftover = machine.astype(float) - job.astype(float)
        return np.all(leftover >= 0)
