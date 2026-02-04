import logging
from typing import SupportsFloat, Any

import gymnasium as gym
import typing as tp

import numpy as np

from src.cluster.core.cluster import ClusterObservation
from src.cluster.core.dilation import AbstractDilation, DilationAction, DilationState
from src.envs.basic import EnvAction, BasicClusterEnv

WrapperObsType = tp.Type[ClusterObservation]
Dilator = tp.TypeVar('Dilator', bound=AbstractDilation)

class EnvWrapperAction(tp.NamedTuple):
    selected_machine_cell: tp.Tuple[int, int]
    selected_job: int
    execute_schedule_command: bool # a.k.a skip time
    contract: bool


    @classmethod
    def into_action_space(cls, kernel_shape: tp.Tuple[int,int], n_jobs: int) -> gym.Space[tuple]:
        return gym.spaces.Tuple(spaces=(
            gym.spaces.MultiDiscrete(kernel_shape),
            gym.spaces.Discrete(n_jobs),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2)
        ))

# TODO: remove _last_extra_value
# TODO: fix return value observation (on step)

class DilatorWrapper(
    gym.Wrapper[ClusterObservation, EnvAction, ClusterObservation, EnvAction]
):
    def __init__(self, env: BasicClusterEnv[ClusterObservation, EnvAction], *, dilator_type: tp.Type[AbstractDilation], **dilation_params):
        super().__init__(env)
        self.dilator_type = dilator_type
        self._dilation_params = dilation_params
        sampled_obs = self.observation_space.sample()
        reformated_array = self.dilator_type.cast_into_dilation_format(sampled_obs["machines"])
        n_jobs = sampled_obs["jobs"].shape[0]
        self._n_machines = sampled_obs["machines"].shape[0]
        self._dilator = self.dilator_type(**self._dilation_params, array=reformated_array) # type: ignore
        self.observation_space = self.cast_original_observation_space()
        self.action_space = self.cast_original_action_space(self._dilator, n_jobs)
        self._last_extra_value = None
        self._dilator = None
        self.logger = logging.getLogger(type(self).__name__)

    def step(
        self, action: EnvWrapperAction
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._dilator is None:
            raise ValueError("Should always call rest before running")

        converted_action = self.run_and_convert(action)

        if converted_action is None: # Still in dilation
            return self._dilator.state.value,  0, False, False, {} # TODO: replace with previous value

        obs, *self._last_extra_value = self.env.step(converted_action)
        self._dilator = self.dilator_from_machines_obs(obs["machines"])

        return self.convert_observation_space(obs), 0, False, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, *self._last_extra_value = self.env.reset(seed=seed,options=options)
        self._dilator = self.dilator_from_machines_obs(obs["machines"])
        return self.convert_observation_space(obs), *self._last_extra_value

    def cast_original_observation_space(self) -> gym.Space[WrapperObsType]:
        machines_space = self.env.observation_space["machines"] # type: ignore
        new_machines_shape = (*self._dilator.get_kernel() , *machines_space.shape[1:])
        machines_space = gym.spaces.Box(
            low=self.dilator_from_machines_obs(machines_space.low).state.value,
            high=self.dilator_from_machines_obs(machines_space.high).state.value,
            shape=new_machines_shape,
            dtype=machines_space.dtype
        )
        return gym.spaces.Dict( # type: ignore
            ClusterObservation( # type: ignore
                machines=machines_space, # type: ignore
                jobs=self.env.observation_space["jobs"]  # type: ignore
            )
        )

    @staticmethod
    def cast_original_action_space(dilator: Dilator, n_jobs: int) ->  gym.Space[tuple]:
        return EnvWrapperAction.into_action_space(dilator.get_kernel(), n_jobs)

    def run_and_convert(self, action: EnvWrapperAction) -> tp.Optional[EnvAction]:
        if action.execute_schedule_command:
            return EnvAction(should_schedule=True, schedule=(-1,-1))

        if action.contract:
            state = self._dilator.execute(DilationAction.Contract())
        else:
            state = self._dilator.execute(DilationAction.Expand(x=action.selected_machine_cell[0], y=action.selected_machine_cell[1]))

        self.logger.debug("Dilation state: %s", type(state).__name__)
        match state:
            case DilationState.FullyExpanded(_, _, _, _):
                m_index = self._dilator.get_selected_machine(action.selected_machine_cell)
                if m_index >= self._n_machines:
                    raise IndexError("Machine index out of bound")
                return EnvAction(should_schedule=False, schedule=(m_index, action.selected_job))
            case DilationState.Initial(_, _) | DilationState.Expanded(_, _, _, _):
                return None
            case _:
                raise RuntimeError()

    def convert_observation_space(self, obs: ClusterObservation) -> WrapperObsType:
        machines = self._dilator.cast_into_dilation_format(obs["machines"])
        return ClusterObservation(
            machines=machines,
            jobs=obs["jobs"]
        )

    def dilator_from_machines_obs(self, machines: np.ndarray) -> Dilator:
        array = self.dilator_type.cast_into_dilation_format(machines)
        return self.dilator_type(**self._dilation_params, array=array)


