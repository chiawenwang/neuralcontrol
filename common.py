"""Shared utilities: thread/seed setup, simulator state IO, SPSA primitives,
and matplotlib animations for the three tasks."""

from __future__ import annotations
import os
import math
import random
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def configure_threads(num_threads: int = 1) -> None:
    """Configure thread count for numerical libraries."""
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_sim_states(sim_manager) -> Dict[str, np.ndarray]:
    """Get current state of the simulator."""
    return {
        "vertices": np.asarray(sim_manager.getAllVertices()).copy(),
        "frames": np.asarray(sim_manager.get_all_frames()).copy(),
    }


def set_sim_states(sim_manager, state: Dict[str, np.ndarray]) -> None:
    """Set simulator state from saved state dictionary."""
    sim_manager.set_all_vertices(
        np.ascontiguousarray(state["vertices"], dtype=np.float64).reshape(-1)
    )
    sim_manager.set_all_frames(
        np.ascontiguousarray(state["frames"], dtype=np.float64)
    )


def reset_sim_with_state(sim_manager, reset_state: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Reset simulator and optionally restore to a saved state."""
    sim_manager.resetSim()
    if reset_state is not None:
        set_sim_states(sim_manager, reset_state)


def reinit_net_(net: nn.Module) -> None:
    """Reinitialize network weights using Kaiming uniform initialization."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
    net.apply(_init)
    with torch.no_grad():
        for name in ["log_mag", "log_mag_xy", "log_mag_a", "rho_xy", "rho_a", "log_metric"]:
            if hasattr(net, name):
                getattr(net, name).zero_()


def rebuild_optimizer(old_opt: torch.optim.Optimizer, net: nn.Module) -> torch.optim.Optimizer:
    """Rebuild optimizer with the same hyperparameters for a (re-initialized) network."""
    return old_opt.__class__(
        [p for p in net.parameters() if p.requires_grad],
        **old_opt.defaults
    )


def clone_state_dict_cpu(net: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a model state dict on CPU for deterministic rollback."""
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def sample_rademacher_like(net: nn.Module) -> List[torch.Tensor]:
    """Sample SPSA perturbations with entries in {-1, +1} for trainable params."""
    deltas = []
    for p in net.parameters():
        if not p.requires_grad:
            continue
        d = torch.empty_like(p)
        d.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        deltas.append(d)
    return deltas


def add_spsa_delta(net: nn.Module, deltas: List[torch.Tensor], scale: float) -> None:
    """Add a scaled SPSA perturbation to trainable model parameters in-place."""
    with torch.no_grad():
        for p, d in zip((p for p in net.parameters() if p.requires_grad), deltas):
            p.add_(d, alpha=scale)


def apply_spsa_update(
    net: nn.Module,
    delta_sets: List[List[torch.Tensor]],
    coeffs: List[float],
    lr: float,
    grad_clip: float,
) -> float:
    """Apply one averaged SPSA SGD step and return the estimated gradient norm."""
    trainable = [p for p in net.parameters() if p.requires_grad]
    if not delta_sets or not coeffs:
        return float("nan")

    grads = [torch.zeros_like(p) for p in trainable]
    inv_m = 1.0 / float(len(coeffs))
    for deltas, coeff in zip(delta_sets, coeffs):
        for g, d in zip(grads, deltas):
            g.add_(d, alpha=float(coeff) * inv_m)

    grad_norm_sq = 0.0
    for g in grads:
        grad_norm_sq += float(torch.sum(g.detach() * g.detach()).cpu())
    grad_norm = float(np.sqrt(max(grad_norm_sq, 0.0)))

    scale = 1.0
    if grad_clip > 0.0 and np.isfinite(grad_norm) and grad_norm > grad_clip:
        scale = float(grad_clip) / max(grad_norm, 1e-12)
        grad_norm = float(grad_clip)

    with torch.no_grad():
        for p, g in zip(trainable, grads):
            p.add_(g, alpha=-float(lr) * scale)
    return grad_norm


def spsa_network_step(
    net: nn.Module,
    evaluate_loss,
    lr: float,
    c: float,
    n_pairs: int,
    grad_clip: float,
    current_loss: Optional[float] = None,
    blocking: bool = False,
    block_tol: float = 0.2,
) -> Dict[str, object]:
    """
    Run one network-parameter SPSA update.

    ``evaluate_loss`` must evaluate the current network and return
    ``(loss, record, eval_time_seconds)``. No adjoint/STO work should happen
    inside that function.
    """
    base_state = clone_state_dict_cpu(net)
    requested_pairs = max(int(n_pairs), 1)
    c = max(float(c), 1e-12)

    delta_sets: List[List[torch.Tensor]] = []
    coeffs: List[float] = []
    loss_plus_values: List[float] = []
    loss_minus_values: List[float] = []
    pair_eval_time = 0.0

    for _ in range(requested_pairs):
        net.load_state_dict(base_state)
        deltas = sample_rademacher_like(net)

        add_spsa_delta(net, deltas, c)
        loss_plus, _record_plus, time_plus = evaluate_loss()

        net.load_state_dict(base_state)
        add_spsa_delta(net, deltas, -c)
        loss_minus, _record_minus, time_minus = evaluate_loss()

        pair_eval_time += float(time_plus + time_minus)
        if np.isfinite(loss_plus) and np.isfinite(loss_minus):
            delta_sets.append(deltas)
            loss_plus_values.append(float(loss_plus))
            loss_minus_values.append(float(loss_minus))
            coeffs.append((float(loss_plus) - float(loss_minus)) / (2.0 * c))

    net.load_state_dict(base_state)
    valid_pairs = len(coeffs)
    grad_norm = apply_spsa_update(net, delta_sets, coeffs, lr, grad_clip)

    loss_current, record_current, time_current = evaluate_loss()
    accepted = int(np.isfinite(loss_current) and valid_pairs > 0)

    if (
        blocking
        and current_loss is not None
        and np.isfinite(loss_current)
        and loss_current > float(current_loss) * (1.0 + float(block_tol))
    ):
        net.load_state_dict(base_state)
        loss_current, record_current, time_recheck = evaluate_loss()
        time_current += time_recheck
        accepted = 0

    if not np.isfinite(loss_current):
        net.load_state_dict(base_state)
        loss_current, record_current, time_recheck = evaluate_loss()
        time_current += time_recheck
        accepted = 0

    return {
        "loss": float(loss_current),
        "record": record_current,
        "forward_eval_time": float(pair_eval_time + time_current),
        "grad_norm": grad_norm,
        "requested_pairs": requested_pairs,
        "valid_pairs": valid_pairs,
        "loss_plus_mean": float(np.mean(loss_plus_values)) if loss_plus_values else float("nan"),
        "loss_minus_mean": float(np.mean(loss_minus_values)) if loss_minus_values else float("nan"),
        "accepted": accepted,
        "forward_rollouts": 2 * requested_pairs + 1 + (1 if accepted == 0 else 0),
    }


def show_animation_any_node(mpc_vertices_list: List[np.ndarray], target: np.ndarray, target_index: int,
                             n_nodes: int = 101, interval: int = 200) -> None:
    """Display animation for any_node tracking (one frame per MPC step)."""
    Nsteps = len(mpc_vertices_list)
    all_xy = np.array([v.reshape(-1, 2) for v in mpc_vertices_list])
    
    xmin, xmax = min(all_xy[:, :, 0].min(), target[0]), max(all_xy[:, :, 0].max(), target[0])
    ymin, ymax = min(all_xy[:, :, 1].min(), target[1]), max(all_xy[:, :, 1].max(), target[1])
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(f"Any Node Tracking (node {target_index})")
    ax.plot(target[0], target[1], 'ro', markersize=10, label="Target")
    config_line, = ax.plot([], [], "b-", lw=2, label="Rod")
    tracked_point, = ax.plot([], [], "go", markersize=8, label=f"Node {target_index}")
    ax.legend(loc="best")
    
    def update(frame_idx):
        config = all_xy[frame_idx]
        config_line.set_data(config[:, 0], config[:, 1])
        tracked_point.set_data([config[target_index, 0]], [config[target_index, 1]])
        ax.set_xlabel(f"Continuation Step {frame_idx+1}/{Nsteps}")
        return config_line, tracked_point
    
    anim = FuncAnimation(fig, update, frames=Nsteps, interval=interval, blit=False, repeat=True)
    plt.show()


def show_animation_letter_curve(mpc_vertices_list: List[np.ndarray], target: np.ndarray,
                                 n_nodes: int = 101, interval: int = 200) -> None:
    """Display animation for letter curve tracking (one frame per MPC step)."""
    Nsteps = len(mpc_vertices_list)
    all_xy = []
    for v in mpc_vertices_list:
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(-1, 2) if v.shape[0] == n_nodes * 2 else v.reshape(-1, 3)[:, [0, 2]]
        elif v.shape[1] == 3:
            v = v[:, [0, 2]]
        all_xy.append(v)
    all_xy = np.array(all_xy)
    
    target_2d = target.reshape(-1, 2) if target.ndim == 1 else target
    if target_2d.shape[1] == 3:
        target_2d = target_2d[:, [0, 2]]
    
    xmin = min(all_xy[:, :, 0].min(), target_2d[:, 0].min())
    xmax = max(all_xy[:, :, 0].max(), target_2d[:, 0].max())
    ymin = min(all_xy[:, :, 1].min(), target_2d[:, 1].min())
    ymax = max(all_xy[:, :, 1].max(), target_2d[:, 1].max())
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title("Letter Curve Tracking")
    ax.plot(target_2d[:, 0], target_2d[:, 1], 'r--', lw=2, label="Target")
    config_line, = ax.plot([], [], "b-", lw=2, label="Rod")
    ax.legend(loc="best")
    
    def update(frame_idx):
        config = all_xy[frame_idx]
        config_line.set_data(config[:, 0], config[:, 1])
        ax.set_xlabel(f"MPC Step {frame_idx+1}/{Nsteps}")
        return (config_line,)
    
    anim = FuncAnimation(fig, update, frames=Nsteps, interval=interval, blit=False, repeat=True)
    plt.show()


def show_animation_middle_tracking(mpc_vertices_list: List[np.ndarray], target_trajectory: np.ndarray,
                                    target_index: int, n_nodes: int = 101, interval: int = 200) -> None:
    """Display animation for middle tracking (one frame per MPC step)."""
    Nsteps = len(mpc_vertices_list)
    all_xy = np.array([v.reshape(-1, 2) for v in mpc_vertices_list])
    
    xmin = min(all_xy[:, :, 0].min(), target_trajectory[:, 0].min())
    xmax = max(all_xy[:, :, 0].max(), target_trajectory[:, 0].max())
    ymin = min(all_xy[:, :, 1].min(), target_trajectory[:, 1].min())
    ymax = max(all_xy[:, :, 1].max(), target_trajectory[:, 1].max())
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(f"Middle Tracking (node {target_index})")
    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], 'r--', lw=1.5, alpha=0.5, label="Target trajectory")
    config_line, = ax.plot([], [], "b-", lw=2, label="Rod")
    tracked_point, = ax.plot([], [], "go", markersize=8, label=f"Node {target_index}")
    ax.legend(loc="best")
    
    def update(frame_idx):
        config = all_xy[frame_idx]
        config_line.set_data(config[:, 0], config[:, 1])
        tracked_point.set_data([config[target_index, 0]], [config[target_index, 1]])
        ax.set_xlabel(f"MPC Step {frame_idx+1}/{Nsteps}")
        return config_line, tracked_point
    
    anim = FuncAnimation(fig, update, frames=Nsteps, interval=interval, blit=False, repeat=True)
    plt.show()
