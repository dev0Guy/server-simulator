import typing as tp

import numpy as np
import numpy.typing as npt

_N_RESOURCE: tp.TypeAlias = int
_N_RESOURCE_CELL: tp.TypeAlias = int
_N_TICKS: tp.TypeAlias = int
_N_JOBS: tp.TypeAlias = int
_N_MACHINES: tp.TypeAlias = int
_DTYPE: tp.TypeAlias = np.bool_

_MACHINE_SHAPE = _JOB_SHAPE = tuple[_N_RESOURCE, _N_RESOURCE_CELL, _N_TICKS]
_JOBS_SHAPE: tp.TypeAlias = tuple[_N_JOBS, _N_RESOURCE, _N_RESOURCE_CELL, _N_TICKS]
_MACHINES_SHAPE = tuple[_N_MACHINES, _N_RESOURCE, _N_RESOURCE_CELL, _N_TICKS]

_JOB_TYPE: tp.TypeAlias = tp.Annotated[npt.NDArray[_DTYPE], _JOB_SHAPE]
_JOBS_TYPE: tp.TypeAlias = tp.Annotated[npt.NDArray[_DTYPE], _JOBS_SHAPE]

_MACHINE_TYPE: tp.TypeAlias = tp.Annotated[npt.NDArray[_DTYPE], _MACHINE_SHAPE]
_MACHINES_TYPE: tp.TypeAlias = tp.Annotated[npt.NDArray[_DTYPE], _MACHINES_SHAPE]
