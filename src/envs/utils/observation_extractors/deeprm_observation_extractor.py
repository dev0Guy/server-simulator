from typing import TypedDict

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from src.cluster.core.job import Status
from src.envs.utils.observation_extractors.proto import BaseObservationCreatorProtocol
from src.cluster.implementation.deep_rm import (
    DeepRMCluster,
    DeepRMMachinesConvertor,
    DeepRMJobsConvertor,
)


class DeepRMClusterObservation(TypedDict):
    machines: npt.NDArray[float]
    jobs_usage: npt.NDArray[float]
    jobs_status: npt.ArrayLike
    current_tick: npt.ArrayLike  # type: int
    arrival_time: npt.NDArray[int]


class DeepRMObservationCreator(
    BaseObservationCreatorProtocol[DeepRMCluster, DeepRMClusterObservation]
):
    _machines_convertor = DeepRMMachinesConvertor()
    _jobs_convertor = DeepRMJobsConvertor()

    def create(self, cluster: DeepRMCluster) -> DeepRMClusterObservation:
        machines_usage = self._machines_convertor.to_representation(cluster._machines)
        jobs_usage, job_status, job_arrival_time = (
            self._jobs_convertor.to_representation(cluster._jobs)
        )
        return DeepRMClusterObservation(
            machines=machines_usage,
            jobs_usage=jobs_usage,
            jobs_status=job_status,
            current_tick=np.array([cluster._current_tick], dtype=np.int64),
            arrival_time=job_arrival_time,
        )

    def create_space(self, cluster: DeepRMCluster) -> gym.Space:
        jobs_usage, job_status, job_arrival_time = (
            self._jobs_convertor.to_representation(cluster._jobs)
        )
        machines_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._machines_convertor.to_representation(cluster._machines).shape,
            dtype=np.float64,
        )
        jobs_usage_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=jobs_usage.shape, dtype=np.float64
        )
        jobs_status_space = gym.spaces.Box(
            low=0.0,
            high=max(s.value for s in Status),
            shape=(len(job_status),),
            dtype=np.float64,
        )
        arrival_time_space = gym.spaces.Box(
            low=0.0,
            high=jobs_usage.shape[-1],
            shape=job_arrival_time.shape,
            dtype=np.int64,
        )
        observation_dict: dict = DeepRMClusterObservation(  # type: ignore
            machines=machines_space,  # type: ignore
            jobs_usage=jobs_usage_space,  # type: ignore
            jobs_status=jobs_status_space,
            current_tick=gym.spaces.Box(0, np.inf, (1,), dtype=np.int64),
            arrival_time=arrival_time_space,  # type: ignore
        )
        return gym.spaces.Dict(
            observation_dict,
        )
