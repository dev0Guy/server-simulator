import abc
import enum
import typing as tp

import gymnasium as gym

T = tp.TypeVar("T")
JobsRepresentation = tp.TypeVar("JobsRepresentation", bound=tp.Iterable[T])


class Status(enum.IntEnum):
    NotCreated = enum.auto()
    Pending = enum.auto()
    Running = enum.auto()
    Completed = enum.auto()
    Failed = enum.auto()


@tp.runtime_checkable
class Job(tp.Protocol[T]):
    arrival_time: int
    length: int
    run_time: int
    status: Status

    @property
    @abc.abstractmethod
    def usage(self) -> T: ...

    @property
    def tick_left(self) -> tp.Optional[int]:
        if self.status != Status.Running:
            return None
        return self.length - self.run_time


@tp.runtime_checkable
class JobCollection(tp.Protocol[JobsRepresentation]):

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> Job[T]: ...

    @abc.abstractmethod
    def __iter__(self) -> tp.Iterable[Job[T]]: ...

    def execute_clock_tick(self, current_time: int) -> None:
        for job in self:
            match job.status:
                case Status.NotCreated if job.arrival_time == current_time:
                    job.status = Status.Pending
                case Status.Running if job.tick_left == 0:
                    job.status = Status.Completed
                case Status.Running:
                    job.run_time += 1
                case _:
                    ...

    @abc.abstractmethod
    def get_representation(self) -> JobsRepresentation: ...
