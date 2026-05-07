"""Adjoint-based receding-horizon control for middle-node trajectory tracking.

Drives a chosen node of a 2D quasi-static rod along a reference path
(sin / cos / triangle / semicircle / square). The inner solver is selected
via --mode (exact adjoint, STO-accelerated adjoint, SPSA, or JFB).
"""

import os
import copy
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch

import nn_der.nn_der as py_der

from utils import create_policy_model
from trajectory import generate_trajectory, get_trajectory_description
from common import (
    configure_threads,
    set_seed,
    get_sim_states,
    reset_sim_with_state,
    reinit_net_,
    rebuild_optimizer,
    spsa_network_step,
    show_animation_middle_tracking,
)
from sto_adapter import (
    MODE_CHOICES,
    apply_run_mode,
    format_reason_counts,
    make_sto_bank,
    mode_label,
    sto_report,
    sto_snapshot,
)


CONFIG = {
    # Run mode. Drives use_rhc / use_sto / use_spsa / use_jfb flags below.
    #   adjoint     - full-horizon exact adjoint
    #   adjoint_sto - full-horizon adjoint with STO reuse
    #   rhc         - receding-horizon exact adjoint
    #   rhc_sto     - receding-horizon adjoint with STO reuse
    #   spsa, jfb   - baselines (see sto_adapter.MODE_CHOICES)
    "mode": "rhc_sto",
    "use_rhc": True,

    # Trajectory types to test: 'cos', 'triangle', 'semicircle'
    "trajectory_types": ["cos", "triangle", "semicircle"],
    
    # Trajectory-specific parameters
    "trajectory_params": {
        # For sin/cos trajectories
        "amplitude": 0.05,          # Wave amplitude
        "frequency": 3.0,           # Wave frequency (number of cycles)
        
        # For triangle wave
        "period": 0.5,              # Period of triangle wave
        
        # For semicircle
        "radius": 0.25,             # Radius of semicircle
        "direction": "down",        # 'up' or 'down'
        
    },
    
    # Target node index (which node to track)
    "target_index": 50,
    
    # MPC parameters
    "T_horizon": 10,                # Steps per MPC epoch
    "total_trajectory_length": 101, # Total length of target trajectory
    "max_inner_iters": 100,         # Max iterations per MPC epoch
    
    # Optimization parameters
    "seed": 42,
    "learning_rate": 0.01,
    "iteration_number": 1000,          # Total iteration limit for the entire case
    "loss_threshold": 1e-6,
    "spsa_lr": 0.01,
    "spsa_c": 0.005,
    "spsa_m": 2,
    "spsa_grad_clip": 1.0,
    "spsa_A": 0.0,
    "spsa_alpha": 0.0,
    "spsa_gamma": 0.0,
    "spsa_blocking": False,
    "spsa_block_tol": 0.2,
    
    # Network parameters
    "hidden_sizes": [64, 64],

    # Per-iteration logs for paper comparisons and STO diagnostics.
    "log_dir": "runs_tuning_task2",
    "run_name": None,

    # STO acceleration. Usually derived from "mode".
    "use_sto": True,
    "sto": {
        "rho_max": 1e-1,
        "kappa_max": 1e10,
        "kappa_warn": 1e8,
        "kappa_check_period": 10,
        "cooldown": 3,
        "n_probes": 5,
        "n_power_iter": 5,
        "max_reuse": 10000,
        "rho_norm": "raw",
        "seed": 42,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Middle-node tracking with modes for exact/STO and adjoint/RHC."
    )
    parser.add_argument("--mode", choices=MODE_CHOICES, default=CONFIG["mode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iteration_number", type=int, default=None)
    parser.add_argument("--max_inner_iters", type=int, default=None)
    parser.add_argument("--loss_threshold", type=float, default=None)
    parser.add_argument("--spsa_lr", type=float, default=None)
    parser.add_argument("--spsa_c", type=float, default=None)
    parser.add_argument("--spsa_m", type=int, default=None)
    parser.add_argument("--spsa_grad_clip", type=float, default=None)
    parser.add_argument("--spsa_A", type=float, default=None)
    parser.add_argument("--spsa_alpha", type=float, default=None)
    parser.add_argument("--spsa_gamma", type=float, default=None)
    parser.add_argument("--spsa_blocking", action="store_true")
    parser.add_argument("--spsa_block_tol", type=float, default=None)
    parser.add_argument("--rho_max", type=float, default=None)
    parser.add_argument("--kappa_warn", type=float, default=None)
    parser.add_argument("--kappa_max", type=float, default=None)
    parser.add_argument("--kappa_check_period", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--n_probes", type=int, default=None)
    parser.add_argument("--n_power_iter", type=int, default=None)
    parser.add_argument("--max_reuse", type=int, default=None)
    parser.add_argument("--rho_norm", choices=("raw", "scaled"), default=None)
    parser.add_argument("--sto_seed", type=int, default=None)
    parser.add_argument(
        "--trajectory_types",
        nargs="+",
        choices=("cos", "triangle", "semicircle"),
        default=None,
        help="subset of Task 2 trajectories to run",
    )
    parser.add_argument("--log_dir", default=CONFIG["log_dir"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_show", action="store_true", help="skip matplotlib animation")
    return parser.parse_args()


def configure_from_args(args):
    config = dict(CONFIG)
    config["sto"] = dict(CONFIG["sto"])
    config["trajectory_params"] = dict(CONFIG["trajectory_params"])
    apply_run_mode(config, args.mode)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.iteration_number is not None:
        config["iteration_number"] = args.iteration_number
    if args.max_inner_iters is not None:
        config["max_inner_iters"] = args.max_inner_iters
    if args.loss_threshold is not None:
        config["loss_threshold"] = args.loss_threshold
    if args.trajectory_types is not None:
        config["trajectory_types"] = list(args.trajectory_types)
    for name in [
        "spsa_lr",
        "spsa_c",
        "spsa_m",
        "spsa_grad_clip",
        "spsa_A",
        "spsa_alpha",
        "spsa_gamma",
        "spsa_block_tol",
    ]:
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.spsa_blocking:
        config["spsa_blocking"] = True
    config["log_dir"] = args.log_dir
    config["run_name"] = args.run_name
    for cli_name, cfg_name in [
        ("rho_max", "rho_max"),
        ("kappa_warn", "kappa_warn"),
        ("kappa_max", "kappa_max"),
        ("kappa_check_period", "kappa_check_period"),
        ("cooldown", "cooldown"),
        ("n_probes", "n_probes"),
        ("n_power_iter", "n_power_iter"),
        ("max_reuse", "max_reuse"),
        ("rho_norm", "rho_norm"),
        ("sto_seed", "seed"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            config["sto"][cfg_name] = value
    config["show_animation"] = not args.no_show
    return config


def safe_run_name(config: dict) -> str:
    raw = config.get("run_name") or config.get("mode", "run")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(raw))


def trace_path_for(config: dict, suffix: str) -> Path:
    log_dir = Path(config.get("log_dir", "runs_tuning_task2"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in suffix)
    return log_dir / f"{safe_run_name(config)}_{safe_suffix}_trace.csv"


def write_trace_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "mode",
        "seed",
        "sto_seed",
        "trajectory_index",
        "trajectory_type",
        "epoch",
        "global_iter",
        "mpc_epoch",
        "inner_iter",
        "loss",
        "best_loss",
        "best_overall_loss",
        "grad_norm",
        "iteration_time",
        "cumulative_time",
        "forward_eval_time",
        "cumulative_forward_eval_time",
        "spsa_forward_rollouts",
        "spsa_m",
        "spsa_valid_pairs",
        "spsa_c",
        "spsa_lr",
        "spsa_loss_plus",
        "spsa_loss_minus",
        "spsa_loss_current",
        "spsa_accepted",
        "gradient_eval_time",
        "cumulative_gradient_eval_time",
        "sim_rollout_time",
        "adjoint_recurrence_time",
        "torch_vjp_time",
        "backward_time",
        "cumulative_backward_time",
        "implicit_tangent_time",
        "cumulative_implicit_tangent_time",
        "exact_tangent_time",
        "sto_query_time",
        "sto_queries",
        "sto_cache_hits",
        "sto_exact_fallbacks",
        "sto_inits",
        "sto_kappa_recomputes",
        "sto_hit_rate",
        "sto_query_time_total",
        "sto_cache_query_time_total",
        "sto_exact_query_time_total",
        "sto_iter_reason_counts",
        "sto_iter_invalid_reason_counts",
        "sto_reason_counts",
        "sto_invalid_reason_counts",
        "sto_last_reason",
        "sto_last_raw_reason",
        "sto_last_rho",
        "sto_rho_min",
        "sto_rho_mean",
        "sto_rho_max",
        "sto_last_eta",
        "sto_eta_mean",
        "sto_eta_max",
        "sto_last_kappa",
        "sto_last_kappa_recomputed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


reset_state = None


def resetSim(sim_manager):
    """Restore the simulator to the saved equilibrium state for this MPC step."""
    reset_sim_with_state(sim_manager, reset_state)


def compute_dL_dtheta(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    target: np.ndarray,
    target_index: int,
    dlam: float,
    jac_reg: float = 1e-6,
    compute_grads: bool = True,
    sto_bank=None,
    use_jfb: bool = False,
):
    """
    Compute gradients for trajectory tracking task.
    
    Parameters
    ----------
    target : np.ndarray
        Shape (T, 2) - target trajectory for the tracked node
    target_index : int
        Index of the node to track
    compute_grads : bool
        If True, compute gradients. If False, only do forward rollout.
    
    Returns
    -------
    grads_list : list[torch.Tensor] or None
        Gradients w.r.t. policy_model parameters. None if compute_grads=False.
    L_total : float
        Scalar loss value.
    vertices_list : list[np.ndarray]
        List of vertices at each time step.
    """
    policy_model.eval()
    grad_eval_start = time.perf_counter()
    timing = {
        "sim_rollout_time": 0.0,
        "exact_tangent_time": 0.0,
        "sto_query_time": 0.0,
        "adjoint_recurrence_time": 0.0,
        "torch_vjp_time": 0.0,
        "gradient_eval_time": 0.0,
        "backward_time": 0.0,
        "implicit_tangent_time": 0.0,
    }

    # Query policy for control sequence
    T = int(lams.numel())
    u_seq_torch = policy_model(lams.view(T, 1))
    u_seq = u_seq_torch.detach().cpu().numpy()

    # Forward rollout in simulator
    resetSim(sim_manager)

    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    verts0_xy = verts0[:, :2]
    N = verts0_xy.shape[0]

    xb_k = verts0_xy[[0, 1, -2, -1], :].reshape(-1).copy()

    # Pre-allocate lists for adjoint
    A_list = np.zeros((T, 8, 8), dtype=np.float64)
    B_list = np.zeros((T, 8, 2), dtype=np.float64)
    dXf_dXb_list = []
    lhs_list = []
    rhs_list = []
    x_free_list = []
    z_boundary_list = []
    vertices_list = []

    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam

        v0 = xb_k[:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()

        v0[0] += dx1
        v1[0] += dx1
        v2[0] += dx2
        v3[0] += dx2

        xb_k = np.hstack((v0, v1, v2, v3))

        sim_t0 = time.perf_counter()
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        sim_manager.step()

        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        vertices_flat = verts_xy.reshape(-1)
        timing["sim_rollout_time"] += time.perf_counter() - sim_t0

        if compute_grads:
            jac = np.asarray(sim_manager.getJacobian()).copy()
            lhs = jac[4:-4, 4:-4]
            rhs = -np.hstack((jac[4:-4, :4], jac[4:-4, -4:]))

            lhs_reg = lhs + jac_reg * np.eye(lhs.shape[0], dtype=np.float64)
            lhs_list.append(lhs_reg.copy())
            rhs_list.append(rhs.copy())
            x_free_list.append(vertices_flat[4:-4].copy())
            z_boundary_list.append(xb_k.copy())

            if sto_bank is None and not use_jfb:
                tangent_t0 = time.perf_counter()
                try:
                    dxf_dxb = np.linalg.solve(lhs_reg, rhs)
                except np.linalg.LinAlgError:
                    dxf_dxb = np.linalg.lstsq(lhs_reg, rhs, rcond=None)[0]
                timing["exact_tangent_time"] += time.perf_counter() - tangent_t0
                dXf_dXb_list.append(dxf_dxb)
            else:
                dXf_dXb_list.append(None)

        A = np.zeros((8, 8), dtype=np.float64)
        B = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B.T

        vertices_list.append(vertices_flat.copy())

    # Compute loss (tracking loss over all steps)
    a_q_array = np.zeros((T, vertices_list[0].shape[0]), dtype=np.float64)
    L_total = 0.0
    for i in range(T):
        v_i = vertices_list[i].reshape(-1, 2)[target_index]
        dv = v_i - target[i]
        L_total += 0.5 * (dv @ dv) * dlam
        a_q = np.zeros((2 * N,), dtype=np.float64)
        a_q[2 * target_index : 2 * target_index + 2] = dv
        a_q_array[i] = a_q.flatten()

    if not compute_grads:
        timing["gradient_eval_time"] = time.perf_counter() - grad_eval_start
        return None, L_total, vertices_list, timing

    # Backward adjoint to compute dL/du_i for the full tracking loss.
    #
    # The previous implementation ran one reverse sweep for each loss time k,
    # creating T(T+1)/2 tangent queries. Because this trajectory loss is a sum
    # of per-step losses and the tangent action is linear in the adjoint vector,
    # we can accumulate the per-step adjoints and run one reverse sweep.
    I8 = np.eye(8, dtype=np.float64)

    v_u = np.zeros((T, 2), dtype=np.float64)
    lam_f = np.zeros((vertices_list[0].shape[0] - 8,), dtype=np.float64)
    lam_b = np.zeros((8,), dtype=np.float64)
    adjoint_t0 = time.perf_counter()
    for i in range(T - 1, -1, -1):
        a_q = a_q_array[i] * dlam
        lam_f = lam_f + a_q[4:-4]
        lam_b = lam_b + np.concatenate([a_q[:4], a_q[-4:]])

        A = A_list[i]
        B = B_list[i]

        if use_jfb:
            tangent_t0 = time.perf_counter()
            free_grad = (rhs_list[i] @ B).T @ lam_f
            free_grad_A = (rhs_list[i] @ A).T @ lam_f
            timing["implicit_tangent_time"] += time.perf_counter() - tangent_t0
        elif sto_bank is None:
            dxf_dxb = dXf_dXb_list[i]
            free_grad = (dxf_dxb @ B).T @ lam_f
        else:
            sto_t0 = time.perf_counter()
            free_grad, _, _, _ = sto_bank[i].query_free_boundary(
                lhs_list[i],
                rhs_list[i],
                B,
                lam_f,
                x_free=x_free_list[i],
                z_boundary=z_boundary_list[i],
            )
            timing["sto_query_time"] += time.perf_counter() - sto_t0

        v_u[i] = dlam * (B.T @ lam_b) + dlam * free_grad
        if use_jfb:
            lam_b = (I8 + dlam * A.T) @ lam_b + dlam * free_grad_A
        elif sto_bank is None:
            lam_b = (I8 + dlam * A.T) @ lam_b + dlam * ((dxf_dxb @ A).T @ lam_f)
        else:
            # A is zero for this endpoint-translation task.
            lam_b = (I8 + dlam * A.T) @ lam_b
    timing["adjoint_recurrence_time"] = time.perf_counter() - adjoint_t0

    vjp_t0 = time.perf_counter()
    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)
    surrogate_total = (u_seq_torch * v_u_torch).sum()

    # Torch VJP
    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(
        surrogate_total, params, retain_graph=False, create_graph=False, allow_unused=False
    )
    timing["torch_vjp_time"] = time.perf_counter() - vjp_t0
    if sto_bank is not None:
        timing["implicit_tangent_time"] = timing["sto_query_time"]
    elif not use_jfb:
        timing["implicit_tangent_time"] = timing["exact_tangent_time"]
    timing["backward_time"] = timing["adjoint_recurrence_time"] + timing["torch_vjp_time"]
    if sto_bank is None and not use_jfb:
        timing["backward_time"] += timing["exact_tangent_time"]
    timing["gradient_eval_time"] = time.perf_counter() - grad_eval_start

    return grads_list, L_total, vertices_list, timing


if __name__ == "__main__":
    args = parse_args()
    config = configure_from_args(args)

    configure_threads(1)
    set_seed(config["seed"])
    device = torch.device("cpu")

    # Simulator setup
    sim_manager = py_der.SimulationManager()
    sim_manager.configure({
        "youngM": 1e5,
        "Poisson": 0.5,
        "density": 1000,
        "deltaTime": 0.01,
        "totalTime": 10.0,
        "gVector": np.array([0, 0, -0.0]),
        "viscosity": 0.000,
        "tol": 1e-4,
        "maxIter": 10000,
        "stol": 1e-4,
        "rodRadius": 1e-3,
        "geometry_file": "vertices.txt",
    })

    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    sim_manager.resetSim()

    verts_init = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts_init.shape[0]

    # Load configuration parameters
    trajectory_types = config["trajectory_types"]
    trajectory_params = config["trajectory_params"]
    target_index = config["target_index"]
    T_horizon = config["T_horizon"]
    total_trajectory_length = config["total_trajectory_length"]
    max_inner_iters = config["max_inner_iters"]
    learning_rate = config["learning_rate"]
    iteration_number = config["iteration_number"]
    loss_threshold = config["loss_threshold"]
    hidden_sizes = config["hidden_sizes"]
    use_rhc = bool(config.get("use_rhc", True))
    use_spsa = bool(config.get("use_spsa", False))
    use_jfb = bool(config.get("use_jfb", False))

    # RHC optimizes short sliding windows. Full-horizon adjoint modes optimize
    # the whole generated trajectory in one segment.
    T = T_horizon if use_rhc else total_trajectory_length
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    lams = torch.tensor(lams_np, dtype=torch.float32, device=device)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([0.1 / dlam, 0.1 / dlam], dtype=torch.float32)

    print(f"\n{'='*60}")
    print(f"Testing {len(trajectory_types)} trajectory types with {mode_label(config)}")
    print(f"Seed: {config['seed']} | STO seed: {config['sto']['seed']}")
    print(f"{'='*60}\n")

    for traj_idx, trajectory_type in enumerate(trajectory_types):
        # Reset to the run seed for each trajectory so modes are comparable.
        set_seed(config["seed"])
        
        # Reset simulator to initial state
        reset_state = None
        sim_manager.resetSim()
        
        # Generate full target trajectory (total_trajectory_length steps)
        middle_node = verts_init[target_index, :].copy()
        target_full = generate_trajectory(trajectory_type, middle_node, total_trajectory_length, trajectory_params)
        
        traj_desc = get_trajectory_description(trajectory_type, trajectory_params)

        # Print configuration
        print(f"\n{'='*60}")
        print(f"[{traj_idx+1}/{len(trajectory_types)}] Trajectory: {traj_desc}")
        print(f"  Target node index: {target_index}")
        print(f"{'='*60}\n")

        # Create fresh network and optimizer for each trajectory
        net = create_policy_model(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=2,
            bounds=bounds,
        ).to(device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

        epoch_dt_hist = []
        total_iters = 0
        mpc_epoch = 0

        best_overall_loss = float("inf")
        
        # Collect u and vertices at end of each MPC step
        mpc_step_u = []  # u[0] from each MPC step
        mpc_step_vertices = []

        # Training loop - RHC uses sliding windows; full-horizon modes run once.
        total_start_time = time.perf_counter()
        sto_bank = make_sto_bank(config, T)
        trace_rows = []
        trace_file = trace_path_for(config, f"traj{traj_idx}_{trajectory_type}")
        cumulative_forward_eval_time = 0.0
        cumulative_gradient_eval_time = 0.0
        cumulative_backward_time = 0.0
        cumulative_implicit_tangent_time = 0.0

        while total_iters < iteration_number and (use_rhc or mpc_epoch == 0):
            t0 = time.perf_counter()
            
            loss_val = float("inf")
            iter_inner_num = 0
            
            # Save current simulator state for this MPC epoch
            reset_state = get_sim_states(sim_manager)
            
            # RHC trains a fresh local controller each segment. Full-horizon
            # adjoint modes keep one controller for the whole run.
            if use_rhc or mpc_epoch == 0:
                reinit_net_(net)
                optimizer = rebuild_optimizer(optimizer, net)

            best_loss = float("inf")
            best_state = None
            current_spsa_loss = None
            
            # Get current target window.
            if use_rhc:
                target_window = target_full[mpc_epoch * T_horizon : mpc_epoch * T_horizon + T, :]
            else:
                target_window = target_full[:T, :]
            
            # Check if we have enough target points left
            if target_window.shape[0] < T:
                print(f"  -> Reached end of target trajectory at epoch {mpc_epoch}")
                break
            
            # Inner optimization loop for this MPC epoch
            inner_limit = max_inner_iters if use_rhc else iteration_number
            while loss_val > loss_threshold and iter_inner_num < inner_limit and total_iters < iteration_number:
                iter_t0 = time.perf_counter()
                if use_spsa:
                    spsa_k = float(iter_inner_num)
                    spsa_A = (
                        float(config["spsa_A"])
                        if float(config["spsa_A"]) >= 0.0
                        else 0.1 * float(max(inner_limit, 1))
                    )
                    spsa_c = float(config["spsa_c"]) / ((spsa_k + 1.0) ** float(config["spsa_gamma"]))
                    spsa_lr = float(config["spsa_lr"]) / (
                        (spsa_k + 1.0 + spsa_A) ** float(config["spsa_alpha"])
                    )

                    def timed_evaluate_spsa_loss():
                        eval_start = time.perf_counter()
                        _grads, loss_eval, vertices_eval, _timing = compute_dL_dtheta(
                            net,
                            lams,
                            sim_manager,
                            target_window,
                            target_index,
                            dlam,
                            compute_grads=False,
                            sto_bank=None,
                            use_jfb=False,
                        )
                        return (
                            float(loss_eval),
                            {"vertices_list": vertices_eval},
                            time.perf_counter() - eval_start,
                        )

                    spsa_result = spsa_network_step(
                        net,
                        timed_evaluate_spsa_loss,
                        lr=spsa_lr,
                        c=spsa_c,
                        n_pairs=config["spsa_m"],
                        grad_clip=config["spsa_grad_clip"],
                        current_loss=current_spsa_loss,
                        blocking=config["spsa_blocking"],
                        block_tol=config["spsa_block_tol"],
                    )
                    loss_val = float(spsa_result["loss"])
                    current_spsa_loss = loss_val if np.isfinite(loss_val) else current_spsa_loss
                    vertices_list = spsa_result["record"].get("vertices_list")

                    if loss_val < best_loss:
                        best_loss = loss_val
                        T_local = int(lams.numel())
                        u_seq_torch = net(lams.view(T_local, 1))
                        u_seq = u_seq_torch.detach().cpu().numpy()
                        best_state = {
                            "mpc_epoch": mpc_epoch,
                            "iter": iter_inner_num,
                            "best_loss": best_loss,
                            "model_state_dict": copy.deepcopy(net.state_dict()),
                            "optimizer_state_dict": {},
                            "best_u": u_seq.copy(),
                        }

                    iteration_time = time.perf_counter() - iter_t0
                    forward_eval_time = float(spsa_result["forward_eval_time"])
                    cumulative_forward_eval_time += forward_eval_time
                    trace_rows.append({
                        "mode": config["mode"],
                        "seed": config["seed"],
                        "sto_seed": config["sto"]["seed"],
                        "trajectory_index": traj_idx,
                        "trajectory_type": trajectory_type,
                        "epoch": total_iters + 1,
                        "global_iter": total_iters,
                        "mpc_epoch": mpc_epoch,
                        "inner_iter": iter_inner_num,
                        "loss": loss_val,
                        "best_loss": best_loss,
                        "best_overall_loss": min(best_overall_loss, best_loss),
                        "grad_norm": spsa_result["grad_norm"],
                        "iteration_time": iteration_time,
                        "cumulative_time": time.perf_counter() - total_start_time,
                        "forward_eval_time": forward_eval_time,
                        "cumulative_forward_eval_time": cumulative_forward_eval_time,
                        "spsa_forward_rollouts": spsa_result["forward_rollouts"],
                        "spsa_m": spsa_result["requested_pairs"],
                        "spsa_valid_pairs": spsa_result["valid_pairs"],
                        "spsa_c": spsa_c,
                        "spsa_lr": spsa_lr,
                        "spsa_loss_plus": spsa_result["loss_plus_mean"],
                        "spsa_loss_minus": spsa_result["loss_minus_mean"],
                        "spsa_loss_current": loss_val,
                        "spsa_accepted": spsa_result["accepted"],
                        "gradient_eval_time": 0.0,
                        "cumulative_gradient_eval_time": cumulative_gradient_eval_time,
                        "sim_rollout_time": 0.0,
                        "adjoint_recurrence_time": 0.0,
                        "torch_vjp_time": 0.0,
                        "backward_time": 0.0,
                        "cumulative_backward_time": cumulative_backward_time,
                        "implicit_tangent_time": 0.0,
                        "cumulative_implicit_tangent_time": cumulative_implicit_tangent_time,
                        "exact_tangent_time": 0.0,
                        "sto_query_time": 0.0,
                        "sto_queries": 0,
                        "sto_cache_hits": 0,
                        "sto_exact_fallbacks": 0,
                        "sto_inits": 0,
                        "sto_kappa_recomputes": 0,
                        "sto_hit_rate": 0.0,
                        "sto_query_time_total": 0.0,
                        "sto_cache_query_time_total": 0.0,
                        "sto_exact_query_time_total": 0.0,
                        "sto_iter_reason_counts": "none",
                        "sto_iter_invalid_reason_counts": "none",
                        "sto_reason_counts": "none",
                        "sto_invalid_reason_counts": "none",
                        "sto_last_reason": "none",
                        "sto_last_raw_reason": "none",
                        "sto_last_rho": np.nan,
                        "sto_rho_min": np.nan,
                        "sto_rho_mean": np.nan,
                        "sto_rho_max": np.nan,
                        "sto_last_eta": np.nan,
                        "sto_eta_mean": np.nan,
                        "sto_eta_max": np.nan,
                        "sto_last_kappa": np.nan,
                        "sto_last_kappa_recomputed": False,
                    })
                    print(
                        f"SPSA {mpc_epoch:03d} | iter {iter_inner_num:03d} "
                        f"| Loss {loss_val:.6e} | grad_norm {float(spsa_result['grad_norm']):.3e} "
                        f"| rollouts {spsa_result['forward_rollouts']} | accepted {spsa_result['accepted']}"
                    )

                    iter_inner_num += 1
                    total_iters += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                sto_before = sto_snapshot(sto_bank)
                
                grads_list, loss, vertices_list, timing = compute_dL_dtheta(
                    net,
                    lams,
                    sim_manager,
                    target_window,
                    target_index,
                    dlam,
                    sto_bank=sto_bank,
                    use_jfb=use_jfb,
                )

                # Assign grads to parameters
                params = [p for p in net.parameters() if p.requires_grad]
                for p, g in zip(params, grads_list):
                    p.grad = g.detach()

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                loss_val = float(loss)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                    # Collect u_seq for this iteration
                    T_local = int(lams.numel())
                    u_seq_torch = net(lams.view(T_local, 1))
                    u_seq = u_seq_torch.detach().cpu().numpy()
                    best_state = {
                        "mpc_epoch": mpc_epoch,
                        "iter": iter_inner_num,
                        "best_loss": best_loss,
                        "model_state_dict": copy.deepcopy(net.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                        "best_u": u_seq.copy(),
                    }

                grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_list)).cpu())
                cumulative_gradient_eval_time += timing["gradient_eval_time"]
                cumulative_backward_time += timing["backward_time"]
                cumulative_implicit_tangent_time += timing["implicit_tangent_time"]
                iteration_time = time.perf_counter() - iter_t0
                sto_stats = sto_snapshot(sto_bank)
                sto_iter_reason_counter = (
                    sto_stats["sto_reason_counter"] - sto_before["sto_reason_counter"]
                )
                sto_iter_invalid_reason_counter = (
                    sto_stats["sto_invalid_reason_counter"]
                    - sto_before["sto_invalid_reason_counter"]
                )
                sto_iter_reasons = format_reason_counts(sto_iter_reason_counter)
                sto_iter_invalid_reasons = format_reason_counts(
                    sto_iter_invalid_reason_counter,
                    include_cache=False,
                )
                trace_rows.append({
                    "mode": config["mode"],
                    "seed": config["seed"],
                    "sto_seed": config["sto"]["seed"],
                    "trajectory_index": traj_idx,
                    "trajectory_type": trajectory_type,
                    "epoch": total_iters + 1,
                    "global_iter": total_iters,
                    "mpc_epoch": mpc_epoch,
                    "inner_iter": iter_inner_num,
                    "loss": loss_val,
                    "best_loss": best_loss,
                    "best_overall_loss": min(best_overall_loss, best_loss),
                    "grad_norm": grad_norm,
                    "iteration_time": iteration_time,
                    "cumulative_time": time.perf_counter() - total_start_time,
                    "forward_eval_time": 0.0,
                    "cumulative_forward_eval_time": cumulative_forward_eval_time,
                    "spsa_forward_rollouts": 0,
                    "spsa_m": 0,
                    "spsa_valid_pairs": 0,
                    "spsa_c": "",
                    "spsa_lr": "",
                    "spsa_loss_plus": "",
                    "spsa_loss_minus": "",
                    "spsa_loss_current": "",
                    "spsa_accepted": "",
                    "gradient_eval_time": timing["gradient_eval_time"],
                    "cumulative_gradient_eval_time": cumulative_gradient_eval_time,
                    "sim_rollout_time": timing["sim_rollout_time"],
                    "adjoint_recurrence_time": timing["adjoint_recurrence_time"],
                    "torch_vjp_time": timing["torch_vjp_time"],
                    "backward_time": timing["backward_time"],
                    "cumulative_backward_time": cumulative_backward_time,
                    "implicit_tangent_time": timing["implicit_tangent_time"],
                    "cumulative_implicit_tangent_time": cumulative_implicit_tangent_time,
                    "exact_tangent_time": timing["exact_tangent_time"],
                    "sto_query_time": timing["sto_query_time"],
                    "sto_queries": sto_stats["sto_queries"],
                    "sto_cache_hits": sto_stats["sto_cache_hits"],
                    "sto_exact_fallbacks": sto_stats["sto_exact_fallbacks"],
                    "sto_inits": sto_stats["sto_inits"],
                    "sto_kappa_recomputes": sto_stats["sto_kappa_recomputes"],
                    "sto_hit_rate": sto_stats["sto_hit_rate"],
                    "sto_query_time_total": sto_stats["sto_query_time_total"],
                    "sto_cache_query_time_total": sto_stats["sto_cache_query_time_total"],
                    "sto_exact_query_time_total": sto_stats["sto_exact_query_time_total"],
                    "sto_iter_reason_counts": sto_iter_reasons,
                    "sto_iter_invalid_reason_counts": sto_iter_invalid_reasons,
                    "sto_reason_counts": sto_stats["sto_reason_counts"],
                    "sto_invalid_reason_counts": sto_stats["sto_invalid_reason_counts"],
                    "sto_last_reason": sto_stats["sto_last_reason"],
                    "sto_last_raw_reason": sto_stats["sto_last_raw_reason"],
                    "sto_last_rho": sto_stats["sto_last_rho"],
                    "sto_rho_min": sto_stats["sto_rho_min"],
                    "sto_rho_mean": sto_stats["sto_rho_mean"],
                    "sto_rho_max": sto_stats["sto_rho_max"],
                    "sto_last_eta": sto_stats["sto_last_eta"],
                    "sto_eta_mean": sto_stats["sto_eta_mean"],
                    "sto_eta_max": sto_stats["sto_eta_max"],
                    "sto_last_kappa": sto_stats["sto_last_kappa"],
                    "sto_last_kappa_recomputed": sto_stats["sto_last_kappa_recomputed"],
                })
                run_kind = "MPC Epoch" if use_rhc else "Full horizon"
                sto_reason_msg = ""
                if sto_bank is not None:
                    sto_reason_msg = f" | STO reasons: {sto_iter_reasons}"
                print(f"{run_kind} {mpc_epoch:03d} | iter {iter_inner_num:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e}{sto_reason_msg}")
                
                iter_inner_num += 1
                total_iters += 1

            epoch_dt = time.perf_counter() - t0
            epoch_dt_hist.append(epoch_dt)

            print(f"  -> Best loss for epoch {mpc_epoch}: {best_loss:.6e}")

            if best_loss < best_overall_loss:
                best_overall_loss = best_loss

            # Load best model for this epoch
            if best_state is not None:
                net.load_state_dict(best_state["model_state_dict"])

            # Run one more forward pass to advance simulator state (no gradient computation)
            _, _, final_vertices_list, _timing = compute_dL_dtheta(
                net, lams, sim_manager, target_window, target_index, dlam,
                compute_grads=False,
                sto_bank=sto_bank,
                use_jfb=use_jfb,
            )

            # Update reset_state for next MPC epoch (continue from current state)
            reset_state = get_sim_states(sim_manager)
            
            # Record current state at end of this MPC step
            # current_verts = np.asarray(sim_manager.getAllVertices()).copy()[:, :2].reshape(-1)
            mpc_step_vertices.append(vertices_list)

            # Get u from best state of this MPC epoch
            if best_state is not None:
                mpc_step_u.append(best_state["best_u"].copy())
            
            mpc_epoch += 1

        total_time = time.perf_counter() - total_start_time
        num_mpc_epochs_done = mpc_epoch
        mpc_step_vertices = np.asarray(mpc_step_vertices).reshape(-1, 202)
        write_trace_csv(trace_file, trace_rows)

        # Print optimal loss and total time
        print(f"\n{'='*60}")
        print(f"[{trajectory_type}] Completed!")
        print(f"  Best Loss: {best_overall_loss:.6e}")
        print(f"  Total Time: {total_time:.4f} s")
        print(f"  Trace CSV: {trace_file}")
        print(f"  {sto_report(sto_bank)}")
        print(f"{'='*60}\n")
        
        # Save MPC control sequence to txt file (one u per MPC step)
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # u_file = os.path.join(script_dir, f"middle_tracking_case{traj_idx}_{trajectory_type}_u.txt")
        # with open(u_file, "w") as f:
        #     f.write(f"# Control sequence for middle_tracking (one u per MPC step)\n")
        #     f.write(f"# Case {traj_idx}: Trajectory type = {trajectory_type}\n")
        #     f.write(f"# Target node index: {target_index}\n")
        #     f.write(f"# Best Loss: {best_overall_loss:.10e}\n")
        #     f.write(f"# Total Time: {total_time:.4f} s\n")
        #     f.write(f"# Format: mpc_step, timestep, u1, u2\n")
        #     for step, u_seq in enumerate(mpc_step_u):
        #         for t, u in enumerate(u_seq):
        #             f.write(f"{step}, {t}, {u[0]:.10e}, {u[1]:.10e}\n")
        # print(f"Control sequence saved to: {u_file}")
        
        # Show animation (one frame per MPC step)
        if config.get("show_animation", True) and len(mpc_step_vertices) > 0:
            show_animation_middle_tracking(
                mpc_step_vertices,
                target_full,
                target_index,
            )
