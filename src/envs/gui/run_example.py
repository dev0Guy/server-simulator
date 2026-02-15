# Create cluster
from time import sleep

import pygame

from src.cluster.implementation.metric_based import MetricClusterCreator
from src.envs import BasicClusterEnv
from src.envs.actions import EnvironmentAction
from src.envs.gui.cluster.metric_cluster_gui import ClusterMetricRenderer
from src.envs.utils.info_builders.base import BaceClusterInformationExtractor
from src.envs.utils.observation_extractors.metric_observation_extractor import (
    MetricClusterObservationCreator,
)
from src.envs.utils.reward_caculators.base import DifferentInPendingJobsRewardCaculator
from src.scheduler.random_scheduler import RandomScheduler

cluster = MetricClusterCreator.generate_default(
    n_machines=6, n_jobs=20, n_resources=3, n_ticks=50, is_offline=True, seed=42
)

env = BasicClusterEnv(
    cluster,
    DifferentInPendingJobsRewardCaculator(),
    BaceClusterInformationExtractor(),
    MetricClusterObservationCreator(),
)

with ClusterMetricRenderer(
    display_size=(1400, 900),  # Larger window
    cell_size=10,
    machine_spacing=10,
) as renderer:
    current_obs, current_info = env.reset()
    print(current_obs["machines"].shape[-1])
    terminated, truncated = False, False

    scheduler = RandomScheduler(cluster.is_allocation_possible)
    clock = pygame.time.Clock()

    print("Starting cluster simulation...")
    print(f"Total jobs: {len(cluster._jobs)}")
    print(f"Total machines: {len(cluster._machines)}")

    step_count = 0

    while not terminated:
        # Render current state
        renderer.render(current_info, current_obs)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("User closed window")
                break

        machines = cluster._machines
        jobs = cluster._jobs

        # Log current status
        status_counts = {}
        for job in jobs:
            status_name = job.status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        print(f"Step {step_count} - Job statuses: {status_counts}")

        # Schedule next action
        schedule_result = scheduler.schedule(machines, jobs)

        if schedule_result is None:
            # No allocation possible, skip action
            action = EnvironmentAction(
                True, (-1, -1)
            )  # Or however your env expects "no-op" action
            print(f"Step {step_count}: No allocation possible, skipping")
        else:
            machine_idx, job_idx = schedule_result
            action = EnvironmentAction(False, (machine_idx, job_idx))
            print(
                f"Step {step_count}: Scheduling Job {job_idx} to Machine {machine_idx}"
            )

        current_obs, reward, terminated, truncated, current_info = env.step(action)

        print(f"Step {step_count}: Reward = {reward:.2f}")

        # Control frame rate
        clock.tick(2)  # 2 FPS for slower visualization

        step_count += 1
        renderer.render(current_info, current_obs)
