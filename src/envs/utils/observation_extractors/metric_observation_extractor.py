from typing import TypedDict

import gymnasium as gym
import numpy.typing as npt

from src.envs.cluster_simulator.core.job import Status
from src.envs.cluster_simulator.metric_based import (
    MetricCluster,
    MetricJobsConvertor,
    MetricMachinesConvertor,
)
from src.envs.cluster_simulator.core.extractors.observation import BaseObservationCreatorProtocol
import numpy as np


class MetricClusterObservation(TypedDict):
    machines: npt.NDArray[float]
    jobs_usage: npt.NDArray[float]
    jobs_status: npt.ArrayLike
    current_tick: npt.ArrayLike
    arrival_time: npt.NDArray[int]


class MetricClusterObservationCreator(
    BaseObservationCreatorProtocol[MetricCluster, MetricClusterObservation]
):
    _jobs_convertor = MetricJobsConvertor()
    _machines_convertor = MetricMachinesConvertor()

    def create(self, cluster: MetricCluster) -> MetricClusterObservation:
        machine_usage = self._machines_convertor.to_representation(cluster._machines)
        job_usage, job_status, job_arrival_time = (
            self._jobs_convertor.to_representation(cluster._jobs)
        )
        return MetricClusterObservation(
            machines=machine_usage,
            jobs_usage=job_usage,
            jobs_status=job_status,
            arrival_time=job_arrival_time,
            current_tick=np.array([cluster._current_tick], dtype=np.int64),
        )

    def create_space(self, cluster: MetricCluster) -> gym.Space:
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
        observation_dict: dict = MetricClusterObservation(  # type: ignore
            machines=machines_space,  # type: ignore
            jobs_usage=jobs_usage_space,  # type: ignore
            jobs_status=jobs_status_space,
            current_tick=gym.spaces.Box(0, np.inf, (1,), dtype=np.int64),
            arrival_time=arrival_time_space,  # type: ignore
        )
        return gym.spaces.Dict(
            observation_dict,
        )
