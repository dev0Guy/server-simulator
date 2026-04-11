from typing import Optional

from gymnasium import Wrapper
from gymnasium.spaces import Tuple, Discrete, Dict, Box
import numpy as np
from src.server_simulator.envs.cluster_simulator.actions import EnvironmentAction
from src.server_simulator.envs.cluster_simulator.base.internal.job import Status  # adjust import

class AutoSelectJobWrapper(Wrapper):
    """
    Automatically selects the first Pending job.
    Agent only needs to decide:
      - should_schedule: bool         (Discrete(2))
      - machine_id:      int          (Discrete(n_machines))
    """

    def __init__(self, env):
        super().__init__(env)

        # Extract n_machines, n_jobs from original space
        # Original: Tuple(Discrete(2), Tuple(Discrete(n_machines), Discrete(n_jobs)))
        original = env.action_space
        self._n_machines = original.spaces[1].spaces[0].n   # Discrete(n_machines)
        self._n_jobs     = original.spaces[1].spaces[1].n   # Discrete(n_jobs)

        self.action_space =  Discrete(self._n_machines + 1)

        # Track currently selected job
        self._selected_job_idx: int = 0

    def _select_job(self, obs) -> Optional[int]:
        """Pick the first job with Pending status. Returns job index."""
        job_statuses = obs["jobs_status"]
        pending_indices = np.where(job_statuses == Status.Pending)[0]

        if len(pending_indices) == 0:
            return None

        return int(pending_indices[0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._selected_job_idx = self._select_job(obs)
        info["selected_job"] = self._selected_job_idx
        return obs, info

    def step(self, action: int):
        should_schedule = self._selected_job_idx is None or action == 0
        machine_id = action - 1

        full_action = EnvironmentAction(
            should_schedule=bool(should_schedule),
            schedule=(int(machine_id), self._selected_job_idx),
        )

        obs, reward, terminated, truncated, info = self.env.step(full_action)

        # Select next pending job for next step
        self._selected_job_idx = self._select_job(obs)
        info["selected_job"] = self._selected_job_idx

        return obs, reward, terminated, truncated, info

    # def action_masks(self) -> np.ndarray:
    #     """
    #     Mask invalid machines for the currently selected job.
    #     Flat mask over Tuple(Discrete(2), Discrete(n_machines)).
    #     """
    #     # should_schedule dim — always both valid
    #     schedule_mask   = np.array([True, True])                      # [2]
    #
    #     # machine dim — mask unavailable machines
    #     machine_mask    = self._get_machine_mask()                    # [n_machines]
    #
    #     # Combine: flat over product(schedule, machine)
    #     flat_mask = np.array([
    #         schedule_mask[s] and machine_mask[m]
    #         for s in range(2)
    #         for m in range(self._n_machines)
    #     ], dtype=bool)
    #
    #     return flat_mask
    #
    # def _get_machine_mask(self) -> np.ndarray:
    #     """
    #     Return bool mask of shape (n_machines,).
    #     True = machine can accept the selected job.
    #     Adjust to use your cluster's internal state.
    #     """
    #     cluster = self.env.unwrapped._cluster
    #     mask = np.zeros(self._n_machines, dtype=bool)
    #
    #     for i in range(self._n_machines):
    #         mask[i] = cluster.is_allocation_possible(
    #             machine_id=i,
    #             job_id=self._selected_job_idx
    #         )
    #
    #     return mask