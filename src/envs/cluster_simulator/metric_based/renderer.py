from typing import Tuple, Optional

from pygame import Color

from src.envs.cluster_simulator.base.renderer import AbstractClusterGameRenderer
from src.envs.cluster_simulator.metric_based.observation import (
    MetricClusterObservation,
)
from src.envs.cluster_simulator.base.extractors.information import (
    ClusterBaseInformation,
)
from src.envs.cluster_simulator.base.internal.job import Status
import numpy.typing as npt
import pygame
import numpy as np

from src.wrappers.cluster_simulator.render_wrapper import RenderMode


class ClusterMetricRenderer(
    AbstractClusterGameRenderer[MetricClusterObservation, ClusterBaseInformation]
):
    def __init__(
        self,
        render_mode: RenderMode,
        display_size: Tuple[int, int] = (1400, 900),
        background_color: Color = "#ECF0F1",
        title: str = "Metric Cluster",
        cell_size: int = 5,
        machine_spacing: int = 10,
        label_height: int = 15,
        separator_height: int = 20,
        header_height: int = 40,
    ):
        self._render_mode = render_mode
        self._width, self._height = display_size
        pygame.init()
        if self._render_mode == "Human":
            pygame.display.set_caption(title)
            self.window = pygame.display.set_mode(display_size)
        else:
            self.window = pygame.Surface(display_size)

        self.clock = pygame.time.Clock()
        self.fps = 30
        self.window = pygame.display.set_mode(display_size)
        self.background_color = background_color
        self.grid_color = (0, 0, 0)  # Black grid lines
        self.cell_size = cell_size
        self.margin = 20
        self.machine_spacing = machine_spacing
        self.label_height = label_height
        self.separator_height = separator_height
        self.header_height = header_height

        # Font for labels
        self.font_small = pygame.font.Font(None, 14)
        self.font = pygame.font.Font(None, label_height + 2)
        self.font_large = pygame.font.Font(None, 32)
        self.label_color = (0, 0, 0)  # Black text
        self.separator_color = (50, 50, 50)  # Dark gray separator

        # Status colors for job borders
        self.status_colors = {
            Status.Running: (0, 0, 255),  # Blue
            Status.Completed: (0, 0, 0),  # Black
            Status.Failed: (255, 0, 0),  # Red
            Status.Pending: (0, 255, 0),  # Green
            Status.NotCreated: (128, 128, 128),  # Gray
        }

        # Status order for sorting (priority order)
        self.status_order = {
            Status.Running: 0,
            Status.Pending: 1,
            Status.Failed: 2,
            Status.Completed: 3,
            Status.NotCreated: 4,
        }

    def render(
        self,
        new_info: ClusterBaseInformation,
        new_observation: MetricClusterObservation,
    ) -> Optional[npt.NDArray]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.window.fill(self.background_color)

        # Draw tick counter at the top
        self._draw_tick_counter(new_info)

        # Calculate jobs section height dynamically
        jobs_section_height = self._calculate_jobs_section_height(
            new_observation["jobs_usage"], new_observation["jobs_status"]
        )

        # Draw jobs section at the top (below tick counter)
        self._draw_jobs(
            new_observation["jobs_usage"],
            new_observation["jobs_status"],
            new_observation["arrival_time"],
            jobs_section_height,
        )

        # Draw separator line
        self._draw_separator(jobs_section_height)

        # Draw machines section below jobs
        self._draw_machines(
            new_observation["machines"], jobs_section_height + self.separator_height
        )

        if self._render_mode == "rgb_array":
            return self._get_rgb_array()
        elif self._render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.fps)
        return None

    def close(self) -> None:
        pygame.quit()

    def _get_rgb_array(self) -> np.ndarray:
        """
        Convert the current pygame display to RGB array.
        Shape: (height, width, 3)
        """
        frame = pygame.surfarray.array3d(self.window)
        frame = np.transpose(frame, (1, 0, 2))  # Convert (W,H,3) -> (H,W,3)
        return frame

    @staticmethod
    def _value_to_color_machine(value: float) -> Color:
        """
        Convert value to color for machines: Red (0) -> Green (1)
        """
        value = max(0.0, min(1.0, value))
        red = int(255 * (1 - value))
        green = int(255 * value)
        blue = 0
        return Color(red, green, blue)

    @staticmethod
    def _value_to_color_job(value: float) -> Color:
        """
        Convert value to color for jobs: Gray (0) -> Green (1)
        """
        value = max(0.0, min(1.0, value))

        # Gray (128, 128, 128) when value = 0
        # Green (0, 255, 0) when value = 1

        red = int(128 * (1 - value))
        green = int(128 + 127 * value)  # 128 -> 255
        blue = int(128 * (1 - value))

        return Color(red, green, blue)

    def _draw_tick_counter(self, info: ClusterBaseInformation) -> None:
        """
        Draw the current tick counter at the top of the screen.

        Args:
            info: Information object containing current tick
        """
        # Get current tick from info
        current_tick = info["current_tick"][0]

        # Create tick text
        tick_text = f"Tick: {current_tick}"
        text_surface = self.font_large.render(tick_text, True, self.label_color)

        # Center horizontally, position at top with margin
        text_rect = text_surface.get_rect(
            center=(self._width // 2, self.header_height // 2)
        )

        self.window.blit(text_surface, text_rect)

    def _calculate_jobs_section_height(
        self, jobs: npt.NDArray[float], jobs_status: npt.NDArray
    ) -> int:
        """
        Calculate how much vertical space is needed to display all jobs.

        Args:
            jobs: Array of shape (n_jobs, n_resources, n_time)
            jobs_status: Array of shape (n_jobs,) with Status enum values

        Returns:
            Required height in pixels for jobs section
        """
        if jobs.shape[0] == 0:
            return self.header_height + 50  # Minimum height including header

        n_jobs = jobs.shape[0]
        n_resources = jobs.shape[1]
        n_time = jobs.shape[2]

        # Calculate size of a single job grid
        job_width = n_time * self.cell_size
        job_height = n_resources * self.cell_size

        # Total height including label
        total_job_height = job_height + self.label_height

        job_spacing = self.machine_spacing

        # Calculate how many jobs can fit per row
        available_width = self._width - 2 * self.margin
        jobs_per_row = max(1, available_width // (job_width + job_spacing))

        # Calculate number of rows needed
        n_rows = int(np.ceil(n_jobs / jobs_per_row))

        # Calculate total height needed (including header)
        total_height = (
            self.header_height  # Tick counter space
            + self.margin  # Top margin
            + n_rows * total_job_height  # All job rows
            + (n_rows - 1) * job_spacing  # Spacing between rows
            + self.margin  # Bottom margin
        )

        return total_height

    def _draw_separator(self, jobs_section_height: int) -> None:
        """
        Draw a horizontal line separating jobs and machines sections.

        Args:
            jobs_section_height: Height where separator should be drawn
        """
        separator_y = jobs_section_height + self.separator_height // 2

        # Draw thick horizontal line
        pygame.draw.line(
            self.window,
            self.separator_color,
            (self.margin, separator_y),
            (self._width - self.margin, separator_y),
            3,  # Line thickness
        )

        # Draw text label
        separator_text = "MACHINES"
        text_surface = self.font.render(separator_text, True, self.separator_color)
        text_rect = text_surface.get_rect(center=(self._width // 2, separator_y))

        # Draw white background for text
        bg_rect = text_rect.inflate(20, 4)
        pygame.draw.rect(self.window, self.background_color, bg_rect)

        # Draw text
        self.window.blit(text_surface, text_rect)

    def _draw_jobs(
        self,
        jobs: npt.NDArray[float],
        jobs_status: npt.NDArray,
        jobs_arrival_time: npt.NDArray,
        jobs_section_height: int,
    ) -> None:
        """
        Draw all jobs in multiple rows at the top of the screen, ordered by status.
        Jobs are arranged in a grid layout to fit all on screen.

        Args:
            jobs: Array of shape (n_jobs, n_resources, n_time)
            jobs_status: Array of shape (n_jobs,) with Status enum values
            jobs_arrival_time: Array of shape (n_jobs,) with arrival times
            jobs_section_height: Allocated height for jobs section
        """
        if jobs.shape[0] == 0:
            return

        n_jobs = jobs.shape[0]
        n_resources = jobs.shape[1]
        n_time = jobs.shape[2]

        # Sort jobs by status
        sorted_indices = self._sort_jobs_by_index(jobs_status)

        # Calculate size of a single job grid
        job_width = n_time * self.cell_size
        job_height = n_resources * self.cell_size

        # Total height including label
        total_job_height = job_height + self.label_height

        job_spacing = self.machine_spacing

        # Calculate how many jobs can fit per row
        available_width = self._width - 2 * self.margin
        jobs_per_row = max(1, available_width // (job_width + job_spacing))

        # Starting position (below tick counter)
        start_y = self.header_height + self.margin

        # Draw each job in sorted order
        for display_idx, j_idx in enumerate(sorted_indices):
            row = display_idx // jobs_per_row
            col = display_idx % jobs_per_row

            # Calculate how many jobs are in this specific row
            jobs_in_row = min(jobs_per_row, n_jobs - row * jobs_per_row)
            row_width = jobs_in_row * job_width + (jobs_in_row - 1) * job_spacing
            row_start_x = (self._width - row_width) // 2

            job_x = row_start_x + col * (job_width + job_spacing)
            job_y = start_y + row * (total_job_height + job_spacing)

            # Draw the label with arrival time
            self._draw_job_label(
                job_idx=j_idx,
                arrival_time=jobs_arrival_time[j_idx],
                x=job_x,
                y=job_y,
                width=job_width,
            )

            # Draw the job grid
            self._draw_single_job(
                job=jobs[j_idx],
                status=jobs_status[j_idx],
                x=job_x,
                y=job_y + self.label_height,
            )

    def _sort_jobs_by_index(self, jobs_status: npt.NDArray) -> np.ndarray:
        n_jobs = len(jobs_status)

        sorted_indices = np.arange(n_jobs)

        return sorted_indices

    def _draw_job_label(
        self, job_idx: int, arrival_time: int, x: int, y: int, width: int
    ) -> None:
        """
        Draw label above a job with job index and arrival time.

        Args:
            job_idx: Index of the job
            arrival_time: Arrival time of the job
            x: Left x position of the job
            y: Top y position (where label starts)
            width: Width of the job (for centering)
        """
        label_text = f"Job {job_idx} (t={arrival_time})"
        text_surface = self.font_small.render(label_text, True, self.label_color)

        # Center the text above the job
        text_rect = text_surface.get_rect(
            center=(x + width // 2, y + self.label_height // 2)
        )

        self.window.blit(text_surface, text_rect)

    def _draw_single_job(
        self, job: npt.NDArray[float], status: Status, x: int, y: int
    ) -> None:
        """
        Draw a single job grid at specified position.
        Uses gray-to-green color scheme.

        Args:
            job: Array of shape (n_resources, n_time)
            status: Status enum value for this job
            x: Top-left x position
            y: Top-left y position
        """
        n_resources, n_time = job.shape

        # Get border color based on status
        border_color = self.status_colors.get(status, self.grid_color)
        border_width = 2  # Thicker border to show status clearly

        # Draw outer border first (status color)
        job_width = n_time * self.cell_size
        job_height = n_resources * self.cell_size
        pygame.draw.rect(
            self.window,
            border_color,
            (
                x - border_width,
                y - border_width,
                job_width + 2 * border_width,
                job_height + 2 * border_width,
            ),
            border_width,
        )

        # Draw cells with gray-to-green color scheme
        for r in range(n_resources):
            for t in range(n_time):
                value = job[r, t]

                # Get color for this value (gray -> green)
                color = self._value_to_color_job(value)

                # Calculate cell position
                cell_x = x + t * self.cell_size
                cell_y = y + r * self.cell_size

                # Draw filled rectangle
                pygame.draw.rect(
                    self.window, color, (cell_x, cell_y, self.cell_size, self.cell_size)
                )

                # Draw cell border (thin black lines)
                pygame.draw.rect(
                    self.window,
                    self.grid_color,
                    (cell_x, cell_y, self.cell_size, self.cell_size),
                    1,
                )

    def _draw_machines(
        self, machines: npt.NDArray[float], machines_start_y: int
    ) -> None:
        """
        Draw all machines in a grid layout.

        Args:
            machines: Array of shape (n_machines, n_resources, n_time)
            machines_start_y: Y position where machines section starts
        """
        if machines.ndim == 3:
            # If shape is (n_machines, n_time, n_resources), transpose to (n_machines, n_resources, n_time)
            if (
                machines.shape[2] < machines.shape[1]
            ):  # Heuristic: fewer resources than time steps
                machines = np.transpose(machines, (0, 2, 1))

        n_machines = machines.shape[0]
        n_resources = machines.shape[1]
        n_time = machines.shape[2]

        # Calculate size of a single machine grid
        machine_width = n_time * self.cell_size
        machine_height = n_resources * self.cell_size

        # Total height including label
        total_machine_height = machine_height + self.label_height

        # Calculate how many machines can fit per row
        available_width = self._width - 2 * self.margin
        machines_per_row = max(
            1, available_width // (machine_width + self.machine_spacing)
        )

        # Calculate total grid dimensions
        total_grid_width = (
            machines_per_row * machine_width
            + (machines_per_row - 1) * self.machine_spacing
        )

        # Starting position (centered horizontally, positioned below separator)
        start_x = (self._width - total_grid_width) // 2
        start_y = machines_start_y + self.margin

        # Draw each machine
        for m_idx in range(n_machines):
            row = m_idx // machines_per_row
            col = m_idx % machines_per_row

            # Calculate position for this machine
            machine_x = start_x + col * (machine_width + self.machine_spacing)
            machine_y = start_y + row * (total_machine_height + self.machine_spacing)

            # Draw the label
            self._draw_machine_label(
                machine_idx=m_idx, x=machine_x, y=machine_y, width=machine_width
            )

            # Draw the machine (below the label)
            self._draw_single_machine(
                machine=machines[m_idx], x=machine_x, y=machine_y + self.label_height
            )

    def _draw_machine_label(self, machine_idx: int, x: int, y: int, width: int) -> None:
        """
        Draw label above a machine.

        Args:
            machine_idx: Index of the machine
            x: Left x position of the machine
            y: Top y position (where label starts)
            width: Width of the machine (for centering)
        """
        label_text = f"Machine {machine_idx}"
        text_surface = self.font.render(label_text, True, self.label_color)

        # Center the text above the machine
        text_rect = text_surface.get_rect(
            center=(x + width // 2, y + self.label_height // 2)
        )

        self.window.blit(text_surface, text_rect)

    def _draw_single_machine(self, machine: npt.NDArray[float], x: int, y: int) -> None:
        """
        Draw a single machine grid at specified position.
        Uses red-to-green color scheme.
        Time axis goes horizontally (X-axis), resources vertically (Y-axis).

        Args:
            machine: Array of shape (n_resources, n_time)
            x: Top-left x position
            y: Top-left y position
        """
        n_resources, n_time = machine.shape

        # Draw time on X-axis (horizontal), resources on Y-axis (vertical)
        for r in range(n_resources):
            for t in range(n_time):
                value = machine[r, t]

                # Get color for this value (red -> green)
                color = self._value_to_color_machine(value)

                # Calculate cell position: time (t) on X-axis, resource (r) on Y-axis
                cell_x = x + t * self.cell_size  # Time increases horizontally
                cell_y = y + r * self.cell_size  # Resources stack vertically

                # Draw filled rectangle
                pygame.draw.rect(
                    self.window, color, (cell_x, cell_y, self.cell_size, self.cell_size)
                )

                # Draw cell border
                pygame.draw.rect(
                    self.window,
                    self.grid_color,
                    (cell_x, cell_y, self.cell_size, self.cell_size),
                    1,
                )
