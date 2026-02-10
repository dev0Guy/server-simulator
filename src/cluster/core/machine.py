import abc
import typing as tp

T = tp.TypeVar("T")
Args = tp.TypeVar("Args", bound=tuple)


@tp.runtime_checkable
class Machine(tp.Protocol[T]):
    free_space: T


@tp.runtime_checkable
class MachineCollection(tp.Protocol[T]):

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> Machine[T]: ...

    @abc.abstractmethod
    def clean_and_reset(self, seed: tp.Optional[int]) -> None: ...

    @abc.abstractmethod
    def execute_clock_tick(self) -> None: ...


@tp.runtime_checkable
class MachineCollectionConvertor(tp.Protocol[T, Args]):

    @abc.abstractmethod
    def to_representation(self, value: MachineCollection[T]) -> Args:
        ...