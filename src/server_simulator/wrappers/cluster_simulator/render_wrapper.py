from typing import TypeVar, Optional, TypeAlias, SupportsFloat, Any, Literal

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType, RenderFrame

from src.server_simulator.envs import BasicClusterEnv
from src.server_simulator.envs.cluster_simulator.actions import EnvironmentAction
from src.server_simulator.envs.cluster_simulator.base.extractors.information import (
    ClusterBaseInformation,
)
from src.server_simulator.envs.cluster_simulator.base.extractors.observation import BaseClusterObservation
from src.server_simulator.envs.cluster_simulator.base.renderer import (
    AbstractClusterGameRenderer,
)

EnvironmentObservation = TypeVar("EnvironmentObservation", bound=BaseClusterObservation)
WrapperObservation = TypeVar("WrapperObservation", bound=BaseClusterObservation)
ClusterInformation = TypeVar("ClusterInformation", bound=ClusterBaseInformation)
Renderer = TypeVar("Renderer", bound=AbstractClusterGameRenderer)
WrapperInformation: TypeAlias = Optional[ClusterInformation]

T = TypeVar("T", bound=type)

ClusterEnv: TypeAlias = BasicClusterEnv[T, ClusterInformation, EnvironmentObservation]
RenderMode: TypeAlias = Literal["human", "rgb_array", None]


class ClusterGameRendererWrapper(
    gym.Wrapper[
        EnvironmentObservation, EnvironmentAction, WrapperObservation, EnvironmentAction
    ]
):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(self, env: ClusterEnv, renderer: Renderer):
        super().__init__(env)
        self._renderer = renderer
        self._reward = None
        self._observation = None
        self._info = None

    @property
    def render_mode(self):
        return self.env.unwrapped.render_mode


    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._observation, self._reward, terminated, truncated, self._info = self.env.step(
            action
        )
        return self._observation, self._reward, terminated, truncated, self._info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self._observation, self._info = self.env.reset(seed=seed, options=options)
        return self._observation, self._info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._renderer.render(self._info, self._observation)

    def close(self):
        self._renderer.close()
