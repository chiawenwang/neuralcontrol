"""Reference trajectories for the middle-node tracking task.

Each generator returns the next T target points (excluding the start), so
the output has shape (T, 2).
"""

import numpy as np


def generate_sin_trajectory(middle_node: np.ndarray, T: int, amplitude: float = 0.05, frequency: float = 3.0):
    x_target = np.linspace(0.0, 1.0, T + 1) + middle_node[0]
    y_target = amplitude * np.sin(frequency * np.pi * (x_target - middle_node[0])) + middle_node[1]
    target = np.vstack((x_target, y_target)).T
    return target[1:, :]


def generate_cos_trajectory(middle_node: np.ndarray, T: int, amplitude: float = 0.05, frequency: float = 3.0):
    x_target = np.linspace(0.0, 1.0, T + 1) + middle_node[0]
    y_target = amplitude * np.cos(frequency * np.pi * (x_target - middle_node[0])) + middle_node[1] - amplitude
    target = np.vstack((x_target, y_target)).T
    return target[1:, :]


def generate_triangle_trajectory(middle_node: np.ndarray, T: int, amplitude: float = 0.05, period: float = 0.5):
    x_target = np.linspace(0.0, 1.0, T + 1) + middle_node[0]
    t_normalized = (x_target - middle_node[0]) / period
    y_raw = -amplitude * (2 * np.abs(2 * (t_normalized - np.floor(t_normalized + 0.5))) - 1)
    # Anchor y to middle_node[1] at t=0 so the trajectory starts on the rod.
    y_target = y_raw - y_raw[0] + middle_node[1]
    target = np.vstack((x_target, y_target)).T
    return target[1:, :]


def generate_semicircle_trajectory(middle_node: np.ndarray, T: int, radius: float = 0.3, direction: str = "up"):
    theta = np.linspace(np.pi, 0, T + 1) if direction == "up" else np.linspace(np.pi, 2 * np.pi, T + 1)
    center_x = middle_node[0] + radius
    center_y = middle_node[1]
    x_target = center_x + radius * np.cos(theta)
    y_target = center_y + radius * np.sin(theta)
    target = np.vstack((x_target, y_target)).T
    return target[1:, :]


def generate_square_trajectory(
    middle_node: np.ndarray,
    T: int,
    amplitude: float = 0.05,
    period: float = 0.25,
    num_segments: int = 10,
):
    """Square-wave path made of `num_segments` alternating horizontal/vertical legs.

    `period` is unused when num_segments is given; kept for call-site compatibility.
    Output is resampled to exactly T+1 points by arc length if the segment count
    does not divide T evenly.
    """
    points_per_segment = (T + 1) // num_segments
    num_horizontal = (num_segments + 1) // 2
    x_per_horizontal = 1.0 / num_horizontal

    trajectory_points = []
    current_x, current_y = middle_node[0], middle_node[1]
    vertical_direction = -1

    for seg in range(num_segments):
        if seg % 2 == 0:
            x_end = current_x + x_per_horizontal
            x_pts = np.linspace(current_x, x_end, points_per_segment, endpoint=False)
            y_pts = np.full_like(x_pts, current_y)
            current_x = x_end
        else:
            y_end = current_y + vertical_direction * amplitude
            x_pts = np.full(points_per_segment, current_x)
            y_pts = np.linspace(current_y, y_end, points_per_segment, endpoint=False)
            current_y = y_end
            vertical_direction *= -1
        for x, y in zip(x_pts, y_pts):
            trajectory_points.append([x, y])

    trajectory_points.append([current_x, current_y])
    trajectory_points = np.array(trajectory_points)

    if len(trajectory_points) != T + 1:
        diffs = np.diff(trajectory_points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        target_lengths = np.linspace(0, total_length, T + 1)
        x_resampled = np.interp(target_lengths, cumulative_length, trajectory_points[:, 0])
        y_resampled = np.interp(target_lengths, cumulative_length, trajectory_points[:, 1])
        trajectory_points = np.vstack((x_resampled, y_resampled)).T

    return trajectory_points[1:, :]


def generate_trajectory(trajectory_type: str, middle_node: np.ndarray, T: int, params: dict) -> np.ndarray:
    """Dispatch to the requested generator."""
    if trajectory_type == "sin":
        return generate_sin_trajectory(
            middle_node, T,
            amplitude=params.get("amplitude", 0.05),
            frequency=params.get("frequency", 3.0),
        )
    if trajectory_type == "cos":
        return generate_cos_trajectory(
            middle_node, T,
            amplitude=params.get("amplitude", 0.05),
            frequency=params.get("frequency", 3.0),
        )
    if trajectory_type == "triangle":
        return generate_triangle_trajectory(
            middle_node, T,
            amplitude=params.get("amplitude", 0.05),
            period=params.get("period", 0.5),
        )
    if trajectory_type == "semicircle":
        return generate_semicircle_trajectory(
            middle_node, T,
            radius=params.get("radius", 0.3),
            direction=params.get("direction", "up"),
        )
    if trajectory_type == "square":
        return generate_square_trajectory(
            middle_node, T,
            amplitude=params.get("square_amplitude", 0.05),
            period=params.get("square_period", 0.25),
            num_segments=params.get("num_segments", 10),
        )
    raise ValueError(
        f"Unknown trajectory type: {trajectory_type}. "
        "Supported types: 'sin', 'cos', 'triangle', 'semicircle', 'square'"
    )


def get_trajectory_description(trajectory_type: str, params: dict) -> str:
    if trajectory_type == "sin":
        return f"Sine wave (amplitude={params.get('amplitude', 0.05)}, frequency={params.get('frequency', 3.0)})"
    if trajectory_type == "cos":
        return f"Cosine wave (amplitude={params.get('amplitude', 0.05)}, frequency={params.get('frequency', 3.0)})"
    if trajectory_type == "triangle":
        return f"Triangle wave (amplitude={params.get('amplitude', 0.05)}, period={params.get('period', 0.5)})"
    if trajectory_type == "semicircle":
        return f"Semicircle (radius={params.get('radius', 0.3)}, direction={params.get('direction', 'up')})"
    if trajectory_type == "square":
        return f"Square wave (amplitude={params.get('square_amplitude', 0.05)}, num_segments={params.get('num_segments', 10)})"
    return trajectory_type
