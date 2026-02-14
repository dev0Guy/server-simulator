from typing import TypedDict

import gymnasium as gym

from src.cluster.core.job import Status
from .proto import BaseObservationCreatorProtocol
from src.cluster.implementation.single_slot import (
    SingleSlotCluster,
    SingleSlotMachinesConvertor,
    SingleSlotJobsConvertor,
)
import numpy as np
import numpy.typing as npt


class SingleSlotClusterObservation(TypedDict):
    machines: npt.NDArray[float]
    jobs_usage: npt.NDArray[float]
    jobs_status: npt.ArrayLike
    current_tick: npt.ArrayLike


class SingleSlotObservationCreator(
    BaseObservationCreatorProtocol[SingleSlotCluster, SingleSlotClusterObservation]
):
    _machines_convertor = SingleSlotMachinesConvertor()
    _jobs_convertor = SingleSlotJobsConvertor()

    def create(self, cluster: SingleSlotCluster) -> SingleSlotClusterObservation:
        machines_usage = self._machines_convertor.to_representation(cluster._machines)
        jobs_usage, job_status = self._jobs_convertor.to_representation(cluster._jobs)
        return SingleSlotClusterObservation(
            machines=machines_usage,
            jobs_usage=jobs_usage,
            jobs_status=np.array(
                [status.value for status in job_status], dtype=np.float64
            ),
            current_tick=np.array([cluster._current_tick], dtype=np.int64),
        )

    def create_space(self, cluster: SingleSlotCluster) -> gym.Space:
        jobs_usage, job_status = self._jobs_convertor.to_representation(cluster._jobs)
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
        observation_dict: dict = SingleSlotClusterObservation(  # type: ignore
            machines=machines_space,  # type: ignore
            jobs_usage=jobs_usage_space,  # type: ignore
            jobs_status=jobs_status_space,
            current_tick=gym.spaces.Box(0, np.inf, (1,), dtype=np.int64),
        )
        return gym.spaces.Dict(
            observation_dict,
        )
