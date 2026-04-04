from collections import defaultdict
from typing import TypedDict
from gymnasium.wrappers import TimeLimit

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from src.server_simulator.envs.cluster_simulator.base.internal.job import Status

class CustomMetricsCallback(BaseCallback):
    """
    Pulls custom keys from the Monitor's episode_info buffer
    and logs them to W&B at every rollout end.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Keys you want to track — must match what your env puts in `info`
        self._episode_metrics = defaultdict(list)
        self._pending_ticks = defaultdict(int)
        self._first_pending = {}
        self._turnaround_time = defaultdict(int)

    def _reset_episode_state(self):
        self._pending_ticks.clear()
        self._first_pending.clear()
        self._turnaround_time.clear()

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return

        wandb.log({
            "scheduling/avg_pending_time": np.mean(self._episode_metrics["avg_pending_time"]),
            "scheduling/avg_turnaround_time": np.mean(self._episode_metrics["avg_turnaround_time"]),
            "scheduling/avg_utilization": np.mean(self._episode_metrics["avg_utilization"]),
            "scheduling/avg_imbalance": np.mean(self._episode_metrics["avg_imbalance"]),
            "global_step": self.num_timesteps,
        })
        self._episode_metrics.clear()


    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            job_statuses = info["jobs_status"]
            current_tick = info["current_tick"]

            for job_id, status in enumerate(job_statuses):
                if status == Status.Pending:
                    self._pending_ticks[job_id] += 1
                    if job_id not in self._first_pending:
                        self._first_pending[job_id] = current_tick
                if status == Status.Completed:
                    self._turnaround_time[job_id] = current_tick - self._first_pending[job_id]

            if "episode" not in info:
                continue

            if self._pending_ticks:
                avg_pending_time = np.mean(list(self._pending_ticks.values()))
                turnaround_time = np.mean(list(self._turnaround_time.values()))
                self._episode_metrics["avg_pending_time"].append(avg_pending_time)
                self._episode_metrics["avg_turnaround_time"].append(turnaround_time)
                self._episode_metrics["avg_utilization"].append((1 - self.locals["obs_tensor"]["machines"].mean()))
                self._episode_metrics["avg_imbalance"].append(self.locals["obs_tensor"]["machines"].std())
            self._pending_ticks.clear()
        return True