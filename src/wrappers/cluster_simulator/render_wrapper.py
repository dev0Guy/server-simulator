from typing import TypeVar, Optional, TypeAlias

import gymnasium as gym

from src.envs.cluster_simulator.actions import EnvironmentAction
from src.envs.cluster_simulator.base.extractors.observation import (
    BaseClusterObservation,
)
from src.envs.cluster_simulator.base.renderer import ClusterInformation

EnvironmentObservation = TypeVar("EnvironmentObservation", bound=BaseClusterObservation)
WrapperObservation = TypeVar("WrapperObservation", bound=BaseClusterObservation)
WrapperInformation: TypeAlias = Optional[ClusterInformation]


# TODO: Implement all of the functions needed


class ClusterGameRendererWrapper(
    gym.Wrapper[
        EnvironmentObservation, EnvironmentAction, WrapperObservation, EnvironmentAction
    ]
):
    pass
