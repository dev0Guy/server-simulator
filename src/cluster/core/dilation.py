import enum
import typing as tp
import numpy as np
import numpy.typing as npt
import gymnasium as gym
import abc

K = tp.TypeVar("K")
State = tp.TypeVar("State", bound=npt.NDArray)
Action = tp.TypeVar("Action", bound=np.int64)

class DilationOperation(enum.IntEnum):
    ZoomOut = enum.auto()
    ZoomIn = enum.auto()
    Execute = enum.auto()

class DilationOperationReturnType(tp.NamedTuple[State]):
    operation: DilationOperation
    state: State

class DilationProtocol(tp.Protocol[K, State]):

    @abc.abstractmethod
    def execute_action(self, action: Action) -> DilationOperationReturnType: ...

    @abc.abstractmethod
    def update(self, original: State) -> State: ...

    @abc.abstractmethod
    def get_selected_machine_idx(self) -> Action: ...

    @abc.abstractmethod
    def get_observation_space_from(self, space: gym.Space[State]): ...

    @abc.abstractmethod
    def get_action_space_from(self, space: gym.Space[Action]) -> gym.Space[Action]: ...
