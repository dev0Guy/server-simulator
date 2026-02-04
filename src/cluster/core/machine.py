import abc
import typing as tp

T = tp.TypeVar("T")
MachinesRepresentation = tp.TypeVar("MachinesRepresentation", bound=tp.List)

@tp.runtime_checkable
class Machine(tp.Protocol[T]):
    free_space: T


@tp.runtime_checkable
class MachineCollection(tp.Protocol[MachinesRepresentation]):

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> Machine[T]: ...

    @abc.abstractmethod
    def clean_and_reset(self, seed: tp.Optional[int]) -> None: ...

    @abc.abstractmethod
    def execute_clock_tick(self) -> None: ...

    @abc.abstractmethod
    def get_representation(self) -> MachinesRepresentation: ...
