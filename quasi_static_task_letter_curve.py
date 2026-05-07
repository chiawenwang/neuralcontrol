"""Adjoint-based receding-horizon control for letter-curve deformation.

Drives a 2D quasi-static rod into a target letter shape (C, U, M, ...). The
inner solver uses either the exact implicit-differentiation adjoint or its
STO-accelerated variant, selected via --mode.
"""

import os
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch

import nn_der.nn_der as py_der

from utils import create_policy_model, to_3d, translate_and_rotate_segment, to_one_hot
from common import (
    configure_threads,
    set_seed,
    get_sim_states,
    reset_sim_with_state,
    reinit_net_,
    rebuild_optimizer,
    spsa_network_step,
    show_animation_letter_curve,
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
    #   spsa        - full-horizon SPSA baseline
    #   jfb         - full-horizon Jacobian-free backprop approximation
    "mode": "rhc_sto",
    "use_rhc": True,
    "seed": 1234,

    # List of cases: each case is a dict with "initial" and "target" file paths
    "cases": [
        {"initial": "C_initial.txt", "target": "targets/target_C.txt"},
        {"initial": "C_initial.txt", "target": "targets/target_C_flipped.txt"},
        {"initial": "U_initial.txt", "target": "targets/target_U.txt"},
        {"initial": "M_initial.txt", "target": "targets/target_M.txt"},
    ],
    
    # MPC parameters
    "max_total_iterations": 300,   # Maximum total iterations per case
    "inner_iterations": 50,         # Inner optimization iterations per MPC step
    "learning_rate": 0.01,
    "spsa_lr": 0.01,
    "spsa_c": 0.005,
    "spsa_m": 2,
    "spsa_grad_clip": 1.0,
    "spsa_A": 0.0,
    "spsa_alpha": 0.0,
    "spsa_gamma": 0.0,
    "spsa_blocking": False,
    "spsa_block_tol": 0.2,
    
    # Early stopping
    "patience": 5,
    "min_delta_rel": 1e-4,
    "loss_threshold": 1e-6,
    
    # Time discretization
    "T": 11,                        # Number of time steps per MPC horizon
    
    # Network parameters
    "hidden_sizes": [64, 64, 64],
    
    # Control bounds (will be divided by dlam)
    "bounds_xy": 0.02,
    "bounds_a": 0.2,

    # Per-iteration logs for paper comparisons and STO diagnostics.
    "log_dir": "runs_tuning_task3",
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
        "seed": 1234,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Letter-curve control with modes for exact/STO and adjoint/RHC."
    )
    parser.add_argument("--mode", choices=MODE_CHOICES, default=CONFIG["mode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_total_iterations", type=int, default=None)
    parser.add_argument("--inner_iterations", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--spsa_lr", type=float, default=None)
    parser.add_argument("--spsa_c", type=float, default=None)
    parser.add_argument("--spsa_m", type=int, default=None)
    parser.add_argument("--spsa_grad_clip", type=float, default=None)
    parser.add_argument("--spsa_A", type=float, default=None)
    parser.add_argument("--spsa_alpha", type=float, default=None)
    parser.add_argument("--spsa_gamma", type=float, default=None)
    parser.add_argument("--spsa_blocking", action="store_true")
    parser.add_argument("--spsa_block_tol", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta_rel", type=float, default=None)
    parser.add_argument("--loss_threshold", type=float, default=None)
    parser.add_argument("--rho_max", type=float, default=None)
    parser.add_argument("--rho_norm", choices=("raw", "scaled"), default=None)
    parser.add_argument("--kappa_warn", type=float, default=None)
    parser.add_argument("--kappa_max", type=float, default=None)
    parser.add_argument("--kappa_check_period", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--n_probes", type=int, default=None)
    parser.add_argument("--n_power_iter", type=int, default=None)
    parser.add_argument("--max_reuse", type=int, default=None)
    parser.add_argument("--sto_seed", type=int, default=None)
    parser.add_argument("--log_dir", default=CONFIG["log_dir"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--case_indices",
        nargs="+",
        type=int,
        default=None,
        help="Only run selected case indices, e.g. --case_indices 3.",
    )
    parser.add_argument("--no_show", action="store_true", help="skip matplotlib animation")
    return parser.parse_args()


def configure_from_args(args):
    config = dict(CONFIG)
    config["sto"] = dict(CONFIG["sto"])
    apply_run_mode(config, args.mode)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_total_iterations is not None:
        config["max_total_iterations"] = args.max_total_iterations
    if args.inner_iterations is not None:
        config["inner_iterations"] = args.inner_iterations
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
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
    if args.patience is not None:
        config["patience"] = args.patience
    if args.min_delta_rel is not None:
        config["min_delta_rel"] = args.min_delta_rel
    if args.loss_threshold is not None:
        config["loss_threshold"] = args.loss_threshold
    config["log_dir"] = args.log_dir
    config["run_name"] = args.run_name
    config["case_indices"] = args.case_indices
    for cli_name, cfg_name in [
        ("rho_max", "rho_max"),
        ("rho_norm", "rho_norm"),
        ("kappa_warn", "kappa_warn"),
        ("kappa_max", "kappa_max"),
        ("kappa_check_period", "kappa_check_period"),
        ("cooldown", "cooldown"),
        ("n_probes", "n_probes"),
        ("n_power_iter", "n_power_iter"),
        ("max_reuse", "max_reuse"),
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


def trace_path_for(config: dict, case_idx: int, initial_file: str, target_file: str) -> Path:
    log_dir = Path(config.get("log_dir", "runs_tuning_task3"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    init_name = Path(initial_file).stem
    target_name = Path(target_file).stem
    suffix = f"case{case_idx}_{init_name}_to_{target_name}"
    safe_suffix = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in suffix)
    return log_dir / f"{safe_run_name(config)}_{safe_suffix}_trace.csv"


def write_trace_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "mode",
        "seed",
        "sto_seed",
        "case_idx",
        "initial_file",
        "target_file",
        "epoch",
        "global_iter",
        "rhc_step",
        "inner_iter",
        "loss",
        "best_loss",
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
        "buckled",
        "early_stop",
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
    lams: torch.Tensor,                 # (T,) torch
    sim_manager,
    target: np.ndarray,                 # (2,) numpy
    dlam: float,
    jac_reg: float = 1e-6,
    compute_grads: bool = True,
    sto_bank=None,
    use_jfb: bool = False,
):
    """
    Compute gradients for letter curve tracking task with MPC.
    
    Returns
    -------
    grads_list : list[torch.Tensor]
        Gradients w.r.t. policy_model parameters.
    L_total : float
        Scalar loss value.
    buckled : bool
        Whether the rod buckled during simulation.
    u_seq : np.ndarray
        Control sequence used.
    vertices_list : list
        List of vertex states.
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

    # set the target with the curvature
    kap_target = sim_manager.compute_curvature(to_3d(target))

    # ---- 0) controls with torch graph ----
    T = int(lams.numel())
    u_seq_torch = policy_model(lams.unsqueeze(-1))  # (T, 3)
    u_seq = u_seq_torch.detach().cpu().numpy()      # (T, 3)

    # --------------------------------------
    # 1) Forward rollout in simulator
    # --------------------------------------
    resetSim(sim_manager)

    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts0.shape[0]

    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()

    v0_fixed = xb_k[0:2].copy() 
    v1_fixed = xb_k[2:4].copy()

    # store A_i, B_i, and dxf/dxb for adjoint
    A_list = np.zeros((T, 4, 4), dtype=np.float32)
    B_list = np.zeros((T, 4, 3), dtype=np.float32)
    dXf_dXb_list = []
    lhs_list = []
    rhs_list = []
    x_free_list = []
    z_boundary_list = []
    vertices_list = []
    
    verts = np.asarray(sim_manager.getAllVertices(), dtype=np.float32)
    buckled = False

    for i in range(T):
        uk = u_seq[i]
        dx, dy, da = uk * dlam
        xb0_k = xb_k.copy()
        xf0_k = verts.reshape(-1)[4:-4].copy()  # free vertices

        v2 = xb_k[4:6]
        v3 = xb_k[6:8]
        v2_1, v3_1 = translate_and_rotate_segment(v2, v3, dx, dy, da)
        xb_k = np.hstack([v0_fixed, v1_fixed, v2_1, v3_1])

        sim_t0 = time.perf_counter()
        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        sim_manager.step()

        # get states from simulator
        jac = np.asarray(sim_manager.getJacobian(), dtype=np.float32)
        verts = np.asarray(sim_manager.getAllVertices(), dtype=np.float32)
        timing["sim_rollout_time"] += time.perf_counter() - sim_t0

        lhs = jac[4:-4, 4:-4]
        rhs = -jac[4:-4, -4:]
        lhs_reg = lhs + jac_reg * np.eye(lhs.shape[0], dtype=np.float32)
        lhs_list.append(lhs_reg.copy())
        rhs_list.append(rhs.copy())
        x_free_list.append(verts.reshape(-1)[4:-4].copy())
        z_boundary_list.append(xb_k[-4:].copy())

        if compute_grads and sto_bank is None and not use_jfb:
            tangent_t0 = time.perf_counter()
            dxf_dxb = np.linalg.solve(lhs_reg, rhs)  # (N-4)*2 x 4
            timing["exact_tangent_time"] += time.perf_counter() - tangent_t0
            dXf_dXb_list.append(dxf_dxb)

            # Check for buckling only in exact mode; STO replaces this tangent solve.
            xf_try = xf0_k + dxf_dxb @ (xb_k[-4:] - xb0_k[-4:])
            xf_k = verts.reshape(-1)[4:-4]
            e_metric = np.linalg.norm(xf_try - xf_k)
            if e_metric > 0.1 and i != 0:
                buckled = True
        else:
            dXf_dXb_list.append(None)

        # Build A, B for boundary states
        x2, y2 = v2_1
        x3, y3 = v3_1  
        a = float(uk[2])
        A = np.array([
            [0.0,   -a/2, 0.0,   a/2],
            [a/2,   0.0, -a/2,  0.0],
            [0.0,    a/2, 0.0,  -a/2],
            [-a/2,  0.0,  a/2,  0.0],
        ], dtype=np.float64)

        B = np.array([
            [1.0, 0.0, -0.5 * (y2 - y3)],
            [0.0, 1.0,  0.5 * (x2 - x3)],
            [1.0, 0.0,  0.5 * (y2 - y3)],
            [0.0, 1.0, -0.5 * (x2 - x3)],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B
        vertices_list.append(verts.copy())


    # ----- 2) compute loss and its gradient w.r.t q ------
    coeff_b = np.array([[1e-3, 0.0], [0.0, 1e-3]])  # bending stiffness
    L_kap = sim_manager.compute_curvature_loss(kap_target, coeff_b)
    dkap = sim_manager.compute_dcurvature(kap_target, coeff_b)  # (N,) numpy
    dkap = to_one_hot(dkap)

    L_stretch = sim_manager.compute_stretch_loss(1.0)
    dstretch = sim_manager.compute_stretch_grad(1.0)
    dstretch = to_one_hot(dstretch)

    L_total = L_kap + L_stretch

    if not compute_grads:
        timing["gradient_eval_time"] = time.perf_counter() - grad_eval_start
        return [], L_total, buckled, u_seq, vertices_list, timing
    
    a_q = dkap + dstretch  # (N*2,) numpy
    lam_f = a_q[4:-4]
    lam_b = a_q[-4:]

    # ----- 3) Backward adjoint ------
    v_u = np.zeros((T, 3), dtype = np.float32)
    I4 = np.eye(4, dtype=np.float32)

    adjoint_t0 = time.perf_counter()
    for i in range(T-1, -1, -1):
        A = A_list[i]
        B = B_list[i]

        if use_jfb:
            tangent_t0 = time.perf_counter()
            free_grad_u = (rhs_list[i] @ B).T @ lam_f
            free_grad_A = (rhs_list[i] @ A).T @ lam_f
            timing["implicit_tangent_time"] += time.perf_counter() - tangent_t0
        elif sto_bank is None:
            dxf_dxb = dXf_dXb_list[i]
            free_grad_u = (dxf_dxb @ B).T @ lam_f
            free_grad_A = (dxf_dxb @ A).T @ lam_f
        else:
            sto_t0 = time.perf_counter()
            free_grad_u, _, _, _ = sto_bank[i].query_free_boundary(
                lhs_list[i],
                rhs_list[i],
                B,
                lam_f,
                x_free=x_free_list[i],
                z_boundary=z_boundary_list[i],
            )
            timing["sto_query_time"] += time.perf_counter() - sto_t0
            sto_t0 = time.perf_counter()
            free_grad_A, _, _, _ = sto_bank[i].query_free_boundary(
                lhs_list[i],
                rhs_list[i],
                A,
                lam_f,
                x_free=x_free_list[i],
                z_boundary=z_boundary_list[i],
            )
            timing["sto_query_time"] += time.perf_counter() - sto_t0

        v_u[i] = dlam * B.T @ lam_b + dlam * free_grad_u
        lam_b = (I4 + dlam * A.T) @ lam_b + dlam * free_grad_A
    timing["adjoint_recurrence_time"] = time.perf_counter() - adjoint_t0

    # ---- 4) one torch VJP: grads = (du/dtheta)^T v_u ----
    vjp_t0 = time.perf_counter()
    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)  # (T,3)

    surrogate = (u_seq_torch * v_u_torch).sum()

    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(surrogate, params, retain_graph=False, create_graph=False)
    timing["torch_vjp_time"] = time.perf_counter() - vjp_t0
    if sto_bank is not None:
        timing["implicit_tangent_time"] = timing["sto_query_time"]
    elif not use_jfb:
        timing["implicit_tangent_time"] = timing["exact_tangent_time"]
    timing["backward_time"] = timing["adjoint_recurrence_time"] + timing["torch_vjp_time"]
    if sto_bank is None and not use_jfb:
        timing["backward_time"] += timing["exact_tangent_time"]
    timing["gradient_eval_time"] = time.perf_counter() - grad_eval_start

    return grads_list, L_total, buckled, u_seq, vertices_list, timing


def run_single_case(
    sim_manager,
    initial_file: str,
    target_file: str,
    config: dict,
    device: torch.device,
    case_idx: int,
):
    """
    Run MPC training for a single case.
    
    Returns
    -------
    result : dict
        Contains total_time, best_loss, best_u, target, mpc_step_vertices, etc.
    """
    global reset_state
    
    # Reconfigure simulator with new geometry file
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
        "geometry_file": initial_file,
    })

    # Controller : BC controller
    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    sim_manager.resetSim()
    
    # Initialize reset_state to None for fresh start
    reset_state = None

    # Load target
    target = np.loadtxt(target_file)
    target = target.reshape(-1)

    # Training setup
    T = config["T"]
    max_total_iterations = config["max_total_iterations"]
    inner_iterations = config["inner_iterations"]
    use_rhc = bool(config.get("use_rhc", True))
    use_spsa = bool(config.get("use_spsa", False))
    use_jfb = bool(config.get("use_jfb", False))
    learning_rate = config["learning_rate"]
    hidden_sizes = config["hidden_sizes"]
    bounds_xy = config["bounds_xy"]
    bounds_a = config["bounds_a"]
    patience = config["patience"]
    min_delta_rel = config["min_delta_rel"]
    loss_threshold = config["loss_threshold"]
    
    # Collect u and vertices at end of each MPC step
    mpc_step_u = []  # u[0] from each MPC step
    mpc_step_vertices = []

    lams_np = np.linspace(0.0, 1.0, T).astype(np.float32)
    lams = torch.tensor(lams_np, device=device, requires_grad=True)
    dlam = float(lams_np[1] - lams_np[0])
    sto_bank = make_sto_bank(config, T)

    bounds = torch.tensor([bounds_xy/dlam, bounds_xy/dlam, bounds_a/dlam], dtype=torch.float32)

    net = create_policy_model(
        input_size=1,
        hidden_sizes=hidden_sizes,
        output_size=3,
        bounds=bounds,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

    # MPC Training loop
    best_loss = float('inf')
    epoch_dt_hist = []
    trace_rows = []
    trace_file = trace_path_for(config, case_idx, initial_file, target_file)
    cumulative_forward_eval_time = 0.0
    cumulative_gradient_eval_time = 0.0
    cumulative_backward_time = 0.0
    cumulative_implicit_tangent_time = 0.0

    start_time = time.perf_counter()
    
    mpc_step = 0
    total_iterations = 0

    best_state = get_sim_states(sim_manager)
    run_kind = "RHC" if use_rhc else "full-horizon"
    print(f"\n[Case {case_idx}] Starting {run_kind} optimization ({mode_label(config)})...\n")

    while (
        total_iterations < max_total_iterations
        and best_loss > loss_threshold
        and (use_rhc or mpc_step == 0)
    ):
        t0 = time.perf_counter()
        
        # Save current state for MPC horizon
        reset_state = best_state
        # RHC trains a fresh local controller each segment. Full-horizon
        # adjoint modes keep one controller for the whole run.
        if use_rhc or mpc_step == 0:
            reinit_net_(net)
            optimizer = rebuild_optimizer(optimizer, net)
        
        best_so_far = float('inf')
        stale_steps = 0
        current_spsa_loss = None
        buckled = False
        early_stop = False
        iter_inner = 0
        best_vertices_list = None
        best_inner_iter = 0
        
        inner_limit = inner_iterations if use_rhc else max_total_iterations
        while (iter_inner <= inner_limit or buckled) and total_iterations < max_total_iterations:
            iter_t0 = time.perf_counter()
            if use_spsa:
                spsa_k = float(iter_inner)
                spsa_A = (
                    float(config["spsa_A"])
                    if float(config["spsa_A"]) >= 0.0
                    else 0.1 * float(max(inner_limit, 1))
                )
                spsa_c = float(config["spsa_c"]) / ((spsa_k + 1.0) ** float(config["spsa_gamma"]))
                spsa_lr = float(config["spsa_lr"]) / (
                    (spsa_k + 1.0 + spsa_A) ** float(config["spsa_alpha"])
                )

                def evaluate_spsa_loss():
                    _grads, loss_eval, buckled_eval, u_seq_eval, vertices_eval, timing_eval = compute_dL_dtheta(
                        net,
                        lams,
                        sim_manager,
                        target,
                        dlam,
                        compute_grads=False,
                        sto_bank=None,
                        use_jfb=False,
                    )
                    return (
                        float(loss_eval),
                        {
                            "buckled": buckled_eval,
                            "u_seq": u_seq_eval,
                            "vertices_list": vertices_eval,
                        },
                        float(timing_eval["sim_rollout_time"]),
                    )

                spsa_result = spsa_network_step(
                    net,
                    evaluate_spsa_loss,
                    lr=spsa_lr,
                    c=spsa_c,
                    n_pairs=config["spsa_m"],
                    grad_clip=config["spsa_grad_clip"],
                    current_loss=current_spsa_loss,
                    blocking=config["spsa_blocking"],
                    block_tol=config["spsa_block_tol"],
                )
                record = spsa_result["record"]
                loss_val = float(spsa_result["loss"])
                buckled = bool(record.get("buckled", False))
                u_seq = record.get("u_seq")
                vertices_list = record.get("vertices_list")
                current_spsa_loss = loss_val if np.isfinite(loss_val) else current_spsa_loss

                improve = (best_so_far - loss_val) / max(abs(best_so_far), 1e-12)
                if loss_val < best_so_far:
                    best_so_far = loss_val
                    best_state = get_sim_states(sim_manager)
                    best_vertices_list = vertices_list
                    best_inner_iter = iter_inner

                if improve < min_delta_rel:
                    stale_steps += 1
                else:
                    stale_steps = 0

                if stale_steps >= patience:
                    early_stop = True

                if loss_val < best_loss:
                    best_loss = loss_val

                iteration_time = time.perf_counter() - iter_t0
                forward_eval_time = float(spsa_result["forward_eval_time"])
                cumulative_forward_eval_time += forward_eval_time
                trace_rows.append({
                    "mode": config["mode"],
                    "seed": config["seed"],
                    "sto_seed": config["sto"]["seed"],
                    "case_idx": case_idx,
                    "initial_file": initial_file,
                    "target_file": target_file,
                    "epoch": total_iterations + 1,
                    "global_iter": total_iterations,
                    "rhc_step": mpc_step,
                    "inner_iter": iter_inner,
                    "loss": loss_val,
                    "best_loss": best_loss,
                    "grad_norm": spsa_result["grad_norm"],
                    "iteration_time": iteration_time,
                    "cumulative_time": time.perf_counter() - start_time,
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
                    "sim_rollout_time": forward_eval_time,
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
                    "buckled": buckled,
                    "early_stop": early_stop,
                })
                print(
                    f"Case {case_idx:02d} | SPSA {mpc_step:03d} | iter {iter_inner:03d} "
                    f"| Loss {loss_val:.6e} | grad_norm {float(spsa_result['grad_norm']):.3e} "
                    f"| rollouts {spsa_result['forward_rollouts']} | accepted {spsa_result['accepted']}"
                )

                if early_stop:
                    break
                iter_inner += 1
                total_iterations += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            sto_before = sto_snapshot(sto_bank)
            
            grads_list, loss, buckled, u_seq, vertices_list, timing = compute_dL_dtheta(
                net,
                lams,
                sim_manager,
                target,
                dlam,
                sto_bank=sto_bank,
                use_jfb=use_jfb,
            )
            
            loss_val = float(loss)
            improve = (best_so_far - loss_val) / max(abs(best_so_far), 1e-12)
            if loss_val < best_so_far:
                best_so_far = loss_val
                best_state = get_sim_states(sim_manager)
                best_vertices_list = vertices_list
                best_inner_iter = iter_inner
            
            if improve < min_delta_rel:
                stale_steps += 1
            else:
                stale_steps = 0
            
            if stale_steps >= patience:
                early_stop = True
            
            params = [p for p in net.parameters() if p.requires_grad]
            for p, g in zip(params, grads_list):
                p.grad = g.detach()
            
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            if loss_val < best_loss:
                best_loss = loss_val
            
            grad_norm = float(torch.sqrt(sum((g.detach()**2).sum() for g in grads_list)).cpu())
            iteration_time = time.perf_counter() - iter_t0
            cumulative_gradient_eval_time += timing["gradient_eval_time"]
            cumulative_backward_time += timing["backward_time"]
            cumulative_implicit_tangent_time += timing["implicit_tangent_time"]
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
                "case_idx": case_idx,
                "initial_file": initial_file,
                "target_file": target_file,
                "epoch": total_iterations + 1,
                "global_iter": total_iterations,
                "rhc_step": mpc_step,
                "inner_iter": iter_inner,
                "loss": loss_val,
                "best_loss": best_loss,
                "grad_norm": grad_norm,
                "iteration_time": iteration_time,
                "cumulative_time": time.perf_counter() - start_time,
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
                "buckled": buckled,
                "early_stop": early_stop,
            })
            sto_reason_msg = ""
            if sto_bank is not None:
                sto_reason_msg = f" | STO reasons: {sto_iter_reasons}"
            print(f"Case {case_idx:02d} | {run_kind} {mpc_step:03d} | iter {iter_inner:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | buckled: {buckled}{sto_reason_msg}")
            
            if early_stop:
                break
            iter_inner += 1
            total_iterations += 1

        if best_inner_iter == 0:
            best_state = get_sim_states(sim_manager)
            best_vertices_list = vertices_list

        epoch_dt = time.perf_counter() - t0
        epoch_dt_hist.append(epoch_dt)


        # Record current state at end of this MPC step
        # current_verts = np.asarray(sim_manager.getAllVertices()).copy()
        mpc_step_vertices.append(best_vertices_list)
        mpc_step_u.append(u_seq.copy())  # Only first control input is executed
        
        print(f"\n[Case {case_idx}] {run_kind} step {mpc_step} completed. Best loss so far: {best_loss:.6e}\n")
        mpc_step += 1


    mpc_step_vertices = np.asarray(mpc_step_vertices).reshape(-1, 202)
    mpc_step_u = np.asarray(mpc_step_u).reshape(-1, 3) 

    total_time = time.perf_counter() - start_time
    avg_mpc_time = np.mean(epoch_dt_hist) if epoch_dt_hist else 0.0
    write_trace_csv(trace_file, trace_rows)

    return {
        "initial_file": initial_file,
        "target_file": target_file,
        "mode": config["mode"],
        "total_time": total_time,
        "best_loss": best_loss,
        "total_mpc_steps": mpc_step,
        "avg_mpc_step_time": avg_mpc_time,
        "mpc_step_u": mpc_step_u,
        "target": target,
        "mpc_step_vertices": mpc_step_vertices,
        "sto_report": sto_report(sto_bank),
        "trace_file": str(trace_file),
    }


if __name__ == "__main__":
    args = parse_args()
    config = configure_from_args(args)

    configure_threads(num_threads=1)
    set_seed(config["seed"], deterministic=True)

    device = torch.device("cpu")

    # Load configuration
    cases = config["cases"]
    case_indices = config.get("case_indices")
    if case_indices is None:
        selected_cases = list(enumerate(cases))
    else:
        selected_cases = [(idx, cases[idx]) for idx in case_indices]
    
    # Create simulator
    sim_manager = py_der.SimulationManager()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*70}")
    print(f"Running {len(selected_cases)} cases with {mode_label(config)}")
    print(f"{'='*70}\n")

    for run_i, (case_idx, case) in enumerate(selected_cases):
        # Reset seed for each case to ensure fair comparison
        set_seed(config["seed"])
        
        initial_file = case["initial"]
        target_file = case["target"]

        print(f"\n{'='*70}")
        print(f"[{run_i+1}/{len(selected_cases)}] Case {case_idx}: {os.path.basename(initial_file)} -> {os.path.basename(target_file)}")
        print(f"{'='*70}\n")

        result = run_single_case(
            sim_manager,
            initial_file,
            target_file,
            config,
            device,
            case_idx,
        )

        # Print optimal loss and total time
        print(f"\n{'='*70}")
        print(f"[Case {case_idx}] Completed!")
        print(f"  Best Loss: {result['best_loss']:.6e}")
        print(f"  Total Time: {result['total_time']:.4f} s")
        print(f"  Trace CSV: {result['trace_file']}")
        print(f"  {result['sto_report']}")
        print(f"{'='*70}")
        
        # Save MPC control sequence to txt file (one u per MPC step)
        # init_name = os.path.basename(result['initial_file']).replace('.txt', '')
        # u_file = os.path.join(script_dir, f"letter_curve_case{case_idx}_{init_name}_u.txt")
        # with open(u_file, "w") as f:
        #     f.write(f"# Control sequence for letter_curve tracking (one u per MPC step)\n")
        #     f.write(f"# Case {case_idx}: {result['initial_file']} -> {result['target_file']}\n")
        #     f.write(f"# Best Loss: {result['best_loss']:.10e}\n")
        #     f.write(f"# Total Time: {result['total_time']:.4f} s\n")
        #     f.write(f"# Format: mpc_step, u_x, u_y, u_a\n")
        #     for step, u in enumerate(result['mpc_step_u']):
        #         f.write(f"{step}, {u[0]:.10e}, {u[1]:.10e}, {u[2]:.10e}\n")
        # print(f"Control sequence saved to: {u_file}")
        
        # Show animation (one frame per MPC step)
        if config.get("show_animation", True) and len(result['mpc_step_vertices'])>0:
            show_animation_letter_curve(
                result['mpc_step_vertices'],
                result['target'],
            )
