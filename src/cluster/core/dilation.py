import enum
import typing as tp
import numpy.typing as npt
import abc

K = tp.TypeVar("K")
State = tp.TypeVar("State", bound=npt.NDArray)
Action = tp.TypeVar("Action", bound=int)

class DilationOperation(enum.IntEnum):
    ZoomOut = enum.auto()
    ZoomIn = enum.auto()
    Execute = enum.auto()

class DilationOperationReturnType(tp.NamedTuple):
    operation: DilationOperation
    state: tp.Optional[State]

class DilationProtocol(tp.Protocol[K, State]):

    @abc.abstractmethod
    def execute_action(self, action: Action) -> DilationOperationReturnType: ...

    @abc.abstractmethod
    def update(self, original: State) -> State: ...

    @abc.abstractmethod
    def get_selected_machine_idx(self) -> tp.Optional[Action]: ...

    @abc.abstractmethod
    def kernel_shape(self) -> tp.Tuple[int, int]: ...
