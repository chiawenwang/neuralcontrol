"""Policy network and small geometric helpers used by the task scripts."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """MLP with optional bounded output via tanh squashing.

    `bounds` may be None (linear output), a scalar / tensor L (symmetric
    [-L, L]), or a (low, high) tuple. `beta` controls the tanh slope so the
    network can saturate gracefully when the unbounded logits grow.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
        output_size: int = 2,
        bounds=None,
        activation=nn.ReLU,
        beta: float = 0.5,
    ):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            lin = nn.Linear(prev, h)
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)
            layers += [lin, activation()]
            prev = h
        self.out = nn.Linear(prev, output_size)
        self.network = nn.Sequential(*layers)
        self.beta = beta
        self._set_bounds(bounds, output_size)

    def _set_bounds(self, bounds, output_size: int) -> None:
        if bounds is None:
            self.register_buffer("low", None, persistent=False)
            self.register_buffer("high", None, persistent=False)
            return
        if isinstance(bounds, tuple):
            low, high = bounds
            low = torch.as_tensor(low).view(1, -1)
            high = torch.as_tensor(high).view(1, -1)
        else:
            L = torch.as_tensor(bounds).view(1, -1)
            if L.numel() == 1:
                L = L.expand(1, output_size)
            low, high = -L, L
        assert low.shape[-1] == output_size and high.shape[-1] == output_size, "Bounds shape mismatch"
        low, high = torch.minimum(low, high), torch.maximum(low, high)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.out(self.network(x))
        if (self.low is None) or (self.high is None):
            return z
        u = torch.tanh(self.beta * z)
        L = (self.high - self.low) * 0.5
        return L * u


def create_policy_model(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    bounds=None,
    actovation=nn.ReLU,
) -> PolicyNetwork:
    """Construct a PolicyNetwork. The misspelled `actovation` kwarg is kept for
    backward compatibility with existing call sites."""
    return PolicyNetwork(
        input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        bounds=bounds,
        activation=actovation,
    )


def sine_curve_between_points(
    A: np.ndarray,
    B: np.ndarray,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    n_points: int = 200,
    mode: str = "sin",
) -> np.ndarray:
    """Sample a sin/cos curve along the line A->B with perpendicular offset."""
    A = np.array(A)
    B = np.array(B)
    d = B - A
    L = np.linalg.norm(d)
    d_hat = d / L
    n_hat = np.array([-d_hat[1], d_hat[0]])
    t = np.linspace(0, 1, n_points)
    line = A + np.outer(t, d)
    if mode == "cos":
        offset = amplitude * np.cos(2 * np.pi * frequency * t)
    else:
        offset = amplitude * np.sin(2 * np.pi * frequency * t)
    return line + np.outer(offset, n_hat)


def translate_and_rotate_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    dx: float,
    dy: float,
    angle: float,
    degrees: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate the segment by (dx, dy), then rotate it about its midpoint."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    if degrees:
        angle = np.deg2rad(angle)
    p1_trans = p1 + np.array([dx, dy])
    p2_trans = p2 + np.array([dx, dy])
    mid = (p1_trans + p2_trans) / 2.0
    v1 = p1_trans - mid
    v2 = p2_trans - mid
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R @ v1 + mid, R @ v2 + mid


def get_vertices(vertices: np.ndarray) -> np.ndarray:
    """Lift (N,2) or (2N,) vertices into the simulator's (N,3) [x,0,z] layout."""
    vertices = vertices.reshape(-1, 2)
    new_vertices = np.vstack([vertices[:, 0], np.zeros_like(vertices[:, 0]), vertices[:, -1]])
    return new_vertices.T


def to_kappa(vertices: np.ndarray) -> np.ndarray:
    """Project (N,3) vertices to a flat [x0,z0,x1,z1,...] vector."""
    vertices_3d = vertices.reshape(-1, 3)
    return vertices_3d[:, [0, 2]].reshape(-1)


def to_3d(vertices: np.ndarray) -> np.ndarray:
    """Same as get_vertices(); kept for the names used at call sites."""
    vertices = vertices.reshape(-1, 2)
    new_vertices = np.vstack([vertices[:, 0], np.zeros_like(vertices[:, 0]), vertices[:, -1]])
    return new_vertices.T


def to_one_hot(vertices: np.ndarray, _3D: bool = True) -> np.ndarray:
    """Flatten vertices: pull (x,z) out of (N,3) when _3D, else flatten as-is."""
    if not _3D:
        return vertices.reshape(-1)
    vertices_3d = vertices.reshape(-1, 3)
    return vertices_3d[:, [0, 2]].reshape(-1)
