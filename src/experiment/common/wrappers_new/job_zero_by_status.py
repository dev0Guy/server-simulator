from typing import SupportsFloat, Any

from gymnasium import ObservationWrapper

from src.server_simulator.envs.cluster_simulator.base.internal.job import Status
from src.server_simulator.envs.cluster_simulator.base.renderer import ClusterObservation


class ZeroJobsByStatusWrapper(ObservationWrapper):

    def observation(self, observation: ClusterObservation) -> ClusterObservation:
        jobs_not_ready_for_schedule = observation["jobs_status"] != Status.Pending
        observation["jobs_usage"][jobs_not_ready_for_schedule] = 0
        return observation
