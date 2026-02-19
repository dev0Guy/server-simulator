import pygame

from src.envs.cluster_simulator.metric_based import MetricClusterCreator
from src.envs import BasicClusterEnv
from src.envs.cluster_simulator.actions import EnvironmentAction
from src.envs.cluster_simulator.metric_based.renderer import ClusterMetricRenderer
from src.envs.cluster_simulator.base.extractors.information import (
    BaceClusterInformationExtractor,
)
from src.envs.cluster_simulator.metric_based.observation import (
    MetricClusterObservationCreator,
)
from src.envs.cluster_simulator.base.extractors.reward import (
    DifferentInPendingJobsRewardCaculator,
)
from src.experiments.schedulers.random_scheduler import RandomScheduler
from src.wrappers.cluster_simulator.render_wrapper import ClusterGameRendererWrapper


def main(render_mode="human"):
    # -------------------------------
    # Create cluster
    # -------------------------------
    cluster = MetricClusterCreator.generate_default(
        n_machines=20,
        n_jobs=100,
        n_resources=3,
        n_ticks=10,
        is_offline=True,
        seed=42,
    )

    # -------------------------------
    # Create environment
    # -------------------------------
    env = BasicClusterEnv(
        cluster,
        DifferentInPendingJobsRewardCaculator(),
        BaceClusterInformationExtractor(),
        MetricClusterObservationCreator(),
    )

    # -------------------------------
    # Create renderer
    # -------------------------------
    renderer = ClusterMetricRenderer(
        display_size=(1400, 900),
        cell_size=10,
        machine_spacing=10,
        render_mode=render_mode,
    )

    env = ClusterGameRendererWrapper(env, renderer)
    # -------------------------------
    # Reset environment
    # -------------------------------
    current_obs, current_info = env.reset()

    scheduler = RandomScheduler(cluster.is_allocation_possible)

    terminated = False
    truncated = False
    step_count = 0

    clock = pygame.time.Clock()

    print("Starting cluster simulation...")
    print(f"Total jobs: {len(cluster._jobs)}")
    print(f"Total machines: {len(cluster._machines)}")

    # -------------------------------
    # Main simulation loop
    # -------------------------------
    while not (terminated or truncated):
        # -----------------------
        # Scheduling
        # -----------------------
        machines = cluster._machines
        jobs = cluster._jobs

        schedule_result = scheduler.schedule(machines, jobs)

        if schedule_result is None:
            action = EnvironmentAction(True, (-1, -1))
            print(f"Step {step_count}: No allocation possible")
        else:
            machine_idx, job_idx = schedule_result
            action = EnvironmentAction(False, (machine_idx, job_idx))
            print(
                f"Step {step_count}: Scheduling Job {job_idx} → Machine {machine_idx}"
            )
        current_obs, reward, terminated, truncated, current_info = env.step(action)

        print(f"Step {step_count}: Reward = {reward:.2f}")

        step_count += 1

    print("Simulation finished.")

    renderer.close()


if __name__ == "__main__":
    main(render_mode="rgb_array")  # change to "rgb_array" if needed
