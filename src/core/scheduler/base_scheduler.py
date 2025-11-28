import typing as tp
import abc

from src.core.cluster.job import JobCollection, Job
from src.core.cluster.machine import MachineCollection, Machine

T = tp.TypeVar("T")
MachineT = tp.TypeVar("MachineT", bound=Machine)
JobT = tp.TypeVar("JobT", bound=Job)


class ABCScheduler(abc.ABC, tp.Generic[T]):

    def __init__(self, can_run_func: tp.Callable[[MachineT, JobT], bool]):
        self._can_run_func = can_run_func

    @abc.abstractmethod
    def schedule(
            self,
            machines: MachineCollection[T],
            jobs: JobCollection[T]
    ) -> tp.Optional[tp.Tuple[int, int]]: ...
