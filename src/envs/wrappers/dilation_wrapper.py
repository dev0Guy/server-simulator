import logging
from typing import SupportsFloat, TypeVar, TypeAlias

import gymnasium as gym
import typing as tp

import numpy as np

from src.envs.cluster_simulator.core.dilation import (
    AbstractDilation,
    DilationAction,
    DilationState,
    AbstractDilationParams,
)
from src.envs.actions import DilationEnvironmentAction
from src.envs.basic import BasicClusterEnv, EnvironmentAction
from src.envs.cluster_simulator.core.extractors.information import ClusterInformation
from src.envs.cluster_simulator.core.extractors.observation import BaseClusterObservation

EnvironmentObservation = TypeVar("EnvironmentObservation", bound=BaseClusterObservation)
WrapperObservation = TypeVar("WrapperObservation", bound=BaseClusterObservation)
WrapperInformation: TypeAlias = tp.Optional[ClusterInformation]
Dilator = TypeVar("Dilator", bound=AbstractDilation)


class DilatorWrapper(
    gym.Wrapper[
        EnvironmentObservation, EnvironmentAction, WrapperObservation, EnvironmentAction
    ]
):
    def __init__(
        self,
        env: BasicClusterEnv,
        *,
        dilator_cls: tp.Type[Dilator],
        **dilation_params: AbstractDilationParams,
    ):
        super().__init__(env)
        self.dilator_type = dilator_cls
        self._dilation_params = dilation_params
        sampled_obs = self.observation_space.sample()
        reformated_array = self.dilator_type.cast_into_dilation_format(
            sampled_obs["machines"]
        )
        n_jobs = sampled_obs["jobs_usage"].shape[0]
        self._n_machines = sampled_obs["machines"].shape[0]
        self._dilator = self.dilator_type(
            **self._dilation_params, array=reformated_array
        )  # type: ignore
        self.observation_space = self.cast_original_observation_space()
        self.action_space = self.cast_original_action_space(self._dilator, n_jobs)
        self._dilator = None
        self._current_observation = None
        self.logger = logging.getLogger(type(self).__name__)

    def step(
        self, action: DilationEnvironmentAction
    ) -> tuple[WrapperObservation, SupportsFloat, bool, bool, WrapperInformation]:
        if self._dilator is None:
            raise ValueError("Should always call rest before running")

        converted_action = self.run_and_convert(action)
        is_in_dilation = converted_action is None

        if is_in_dilation:
            self._current_observation["machines"] = self._dilator.state.value
            return self._current_observation, 0, False, False, None

        obs, reward, terminated, truncated, info = self.env.step(converted_action)
        return (
            self.update_and_convert_observation(obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self, *, seed: int | None = None, options: WrapperInformation = None
    ) -> tuple[WrapperObservation, WrapperInformation]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.update_and_convert_observation(obs), info

    def cast_original_observation_space(self) -> gym.Space[WrapperObservation]:
        original_obs_space = {k: v for k, v in self.env.observation_space.items()}
        original_machines_space = original_obs_space.pop("machines")

        new_machines_shape = (
            *self._dilator.get_kernel(),
            *original_machines_space.shape[1:],
        )
        machines_space = gym.spaces.Box(  # TODO: understand the problem
            low=0,
            high=np.inf,
            shape=new_machines_shape,
            dtype=original_machines_space.dtype,
        )
        original_obs_space["machines"] = machines_space
        return gym.spaces.Dict(original_obs_space)

    @staticmethod
    def cast_original_action_space(dilator: Dilator, n_jobs: int) -> gym.Space[tuple]:
        return DilationEnvironmentAction.into_action_space(dilator.get_kernel(), n_jobs)

    def run_and_convert(
        self, action: DilationEnvironmentAction
    ) -> tp.Optional[EnvironmentAction]:
        if action.execute_schedule_command:
            self.logger.debug("Executing schedule command (skip time)")
            return EnvironmentAction(should_schedule=True, schedule=(-1, -1))

        if not action.contract and isinstance(
            self._dilator.state, DilationState.FullyExpanded
        ):
            m_index = self._dilator.get_selected_machine(action.selected_machine_cell)

            if m_index >= self._n_machines:
                raise IndexError("Machine index out of bound")

            self.logger.debug("Selecting cell %d on fully expanded.", m_index)
            return EnvironmentAction(
                should_schedule=False, schedule=(m_index, action.selected_job)
            )

        if action.contract:
            self._dilator.execute(DilationAction.Contract())
            return None
        else:
            self._dilator.execute(
                DilationAction.Expand(
                    x=action.selected_machine_cell[0], y=action.selected_machine_cell[1]
                )
            )
            return None

    def dilator_from_machines_obs(self, machines: np.ndarray) -> Dilator:
        array = self.dilator_type.cast_into_dilation_format(machines)
        return self.dilator_type(**self._dilation_params, array=array)

    def update_and_convert_observation(
        self, obs: EnvironmentObservation
    ) -> WrapperObservation:
        self._current_observation = obs.copy()
        self._dilator = self.dilator_from_machines_obs(
            self._current_observation["machines"]
        )
        self._current_observation["machines"] = self._dilator.state.value
        return self._current_observation
