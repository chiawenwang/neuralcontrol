"""STO adapter for the quasi-static rod simulator.

The simulator exposes the local implicit sensitivity as

    J_ff dx_f + J_fb dx_b = 0
    dx_f / du = -inv(J_ff) J_fb B

where B maps policy controls to boundary-coordinate increments. The generic
STO core answers adjoint tangent queries -G_z.T @ inv(G_x.T) @ v, so we map
G_x = J_ff and G_z = J_fb B and use STO as a drop-in replacement for the
exact (dx_f / du).T @ v primitive in the adjoint/RHC backward pass.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from stateful_tangent_operator import StatefulTangentOperator


DEFAULT_STO_CONFIG: dict[str, Any] = {
    "rho_max": 1e-1,
    "kappa_max": 1e10,
    "kappa_warn": 1e8,
    "kappa_check_period": 10,
    "cooldown": 3,
    "n_probes": 5,
    "n_power_iter": 5,
    "max_reuse": 10000,
    "singular_rcond": 1e-12,
    "rho_norm": "raw",
    "seed": 42,
}

MODE_CHOICES = ("adjoint", "adjoint_sto", "rhc", "rhc_sto", "spsa", "jfb")


def reason_key(reason: str) -> str:
    """Normalize detailed STO query reasons into stable log categories."""
    head = str(reason).split(";", maxsplit=1)[0].strip()
    if head.startswith("rho="):
        return "rho_gt_rho_max"
    if head.startswith("kappa="):
        return "kappa_gt_kappa_max"
    if head.startswith("drift_x="):
        return "drift_x"
    if head.startswith("drift_z="):
        return "drift_z"
    if head.startswith("shape_mismatch"):
        return "shape_mismatch"
    return head or "none"


def format_reason_counts(counts: Counter[str], include_cache: bool = True) -> str:
    """Compact, stable string for console and CSV logs."""
    filtered = {
        key: value
        for key, value in counts.items()
        if value > 0 and (include_cache or key != "cache_hit")
    }
    if not filtered:
        return "none"
    return ";".join(f"{key}={filtered[key]}" for key in sorted(filtered))


def apply_run_mode(config: dict[str, Any], mode: str | None = None) -> str:
    """Set use_rhc / use_sto / use_spsa / use_jfb flags from a single mode string."""
    selected = mode or config.get("mode", "rhc_sto")
    if selected not in MODE_CHOICES:
        raise ValueError(f"Unknown mode {selected!r}; expected one of {MODE_CHOICES}.")

    config["mode"] = selected
    config["use_rhc"] = selected in ("rhc", "rhc_sto")
    config["use_sto"] = selected in ("adjoint_sto", "rhc_sto")
    config["use_spsa"] = selected == "spsa"
    config["use_jfb"] = selected == "jfb"
    return selected


def mode_label(config: dict[str, Any]) -> str:
    labels = {
        "adjoint": "Adjoint exact",
        "adjoint_sto": "Adjoint + STO",
        "rhc": "Adjoint + RHC",
        "rhc_sto": "Adjoint + RHC + STO",
        "spsa": "SPSA",
        "jfb": "JFB",
    }
    return labels.get(config.get("mode", "rhc_sto"), str(config.get("mode")))


class QuasiStaticStripSTO:
    """Thin STO wrapper for the pybind quasi-static strip simulator."""

    def __init__(self, **kwargs: Any) -> None:
        cfg = dict(DEFAULT_STO_CONFIG)
        cfg.update(kwargs)
        seed = int(cfg.pop("seed"))
        self.core = StatefulTangentOperator(
            **cfg,
            rng=np.random.default_rng(seed),
        )
        self._last_n_free = 0
        self.query_time_total = 0.0
        self.query_time_last = 0.0
        self.cache_query_time_total = 0.0
        self.exact_query_time_total = 0.0
        self.reason_counts: Counter[str] = Counter()
        self.last_reason = "none"
        self.last_raw_reason = "none"

    def query_free_boundary(
        self,
        J_ff: np.ndarray,
        rhs_neg_jfb: np.ndarray,
        B: np.ndarray,
        a_free: np.ndarray,
        x_free: np.ndarray | None = None,
        z_boundary: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, float, str]:
        """Return ``(dx_f / du).T @ a_free`` using STO validation/reuse."""
        J_ff = np.asarray(J_ff, dtype=np.float64)
        rhs_neg_jfb = np.asarray(rhs_neg_jfb, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        a_free = np.asarray(a_free, dtype=np.float64)

        # The scripts store rhs = -J_fb, while generic STO expects G_z = J_fb B.
        G_z = (-rhs_neg_jfb) @ B
        self._last_n_free = J_ff.shape[0]

        t0 = time.perf_counter()
        result = self.core.query(
            v=a_free,
            G_x_new=J_ff,
            G_z_new=G_z,
            x_new=None if x_free is None else np.asarray(x_free, dtype=np.float64),
            z_new=None if z_boundary is None else np.asarray(z_boundary, dtype=np.float64),
        )
        dt = time.perf_counter() - t0
        self.query_time_last = dt
        self.query_time_total += dt
        if result.used_cache:
            self.cache_query_time_total += dt
        else:
            self.exact_query_time_total += dt
        self.last_raw_reason = result.reason
        self.last_reason = reason_key(result.reason)
        self.reason_counts[self.last_reason] += 1
        return result.g_z, result.used_cache, result.eta, result.reason

    @property
    def last_rho(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.rho

    @property
    def last_kappa(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.kappa

    @property
    def last_eta(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.eta

    @property
    def last_kappa_recomputed(self) -> bool:
        gate = self.core.last_validation
        return False if gate is None else gate.kappa_recomputed

    @property
    def in_warning(self) -> bool:
        return self.core.in_warning

def make_sto_bank(config: dict[str, Any], n_steps: int) -> list[QuasiStaticStripSTO] | None:
    """Create one STO cache per continuation step, or ``None`` for exact mode."""
    if not config.get("use_sto", True):
        return None
    sto_cfg = dict(DEFAULT_STO_CONFIG)
    sto_cfg.update(config.get("sto", {}))
    seed = int(sto_cfg.pop("seed", 42))
    return [
        QuasiStaticStripSTO(**sto_cfg, seed=seed + 10_000 + i)
        for i in range(n_steps)
    ]


def sto_report(sto_bank: list[QuasiStaticStripSTO] | None) -> str:
    """Compact aggregate stats for logging."""
    stats = sto_snapshot(sto_bank)
    if not stats["enabled"]:
        return "STO disabled"
    return (
        f"STO queries={stats['sto_queries']} cache_hits={stats['sto_cache_hits']} "
        f"exact_fallbacks={stats['sto_exact_fallbacks']} inits={stats['sto_inits']} "
        f"kappa_recomputes={stats['sto_kappa_recomputes']} "
        f"hit_rate={stats['sto_hit_rate']:.1%} "
        f"query_time={stats['sto_query_time_total']:.4f}s "
        f"invalid_reasons={stats['sto_invalid_reason_counts']}"
    )


def sto_snapshot(sto_bank: list[QuasiStaticStripSTO] | None) -> dict[str, Any]:
    """Aggregate STO counters and measured query time for trace logging."""
    if not sto_bank:
        return {
            "enabled": False,
            "sto_queries": 0,
            "sto_cache_hits": 0,
            "sto_exact_fallbacks": 0,
            "sto_inits": 0,
            "sto_kappa_recomputes": 0,
            "sto_hit_rate": 0.0,
            "sto_query_time_total": 0.0,
            "sto_cache_query_time_total": 0.0,
            "sto_exact_query_time_total": 0.0,
            "sto_reason_counter": Counter(),
            "sto_invalid_reason_counter": Counter(),
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
        }

    total = sum(s.core.stats.n_total for s in sto_bank)
    cached = sum(s.core.stats.n_cached for s in sto_bank)
    exact = sum(s.core.stats.n_exact for s in sto_bank)
    inits = sum(s.core.stats.n_init for s in sto_bank)
    kappa = sum(s.core.stats.n_kappa_recompute for s in sto_bank)
    reason_counts: Counter[str] = Counter()
    last_reason = "none"
    last_raw_reason = "none"
    last_rho = np.nan
    last_eta = np.nan
    last_kappa = np.nan
    last_kappa_recomputed = False
    for sto in sto_bank:
        reason_counts.update(sto.reason_counts)
        if sto.last_reason != "none":
            last_reason = sto.last_reason
            last_raw_reason = sto.last_raw_reason
            last_rho = sto.last_rho
            last_eta = sto.last_eta
            last_kappa = sto.last_kappa
            last_kappa_recomputed = sto.last_kappa_recomputed
    rhos = np.asarray([sto.last_rho for sto in sto_bank], dtype=float)
    etas = np.asarray([sto.last_eta for sto in sto_bank], dtype=float)
    finite_rhos = rhos[~np.isnan(rhos)]
    finite_etas = etas[~np.isnan(etas)]
    invalid_reason_counts = Counter(
        {key: value for key, value in reason_counts.items() if key != "cache_hit"}
    )
    return {
        "enabled": True,
        "sto_queries": total,
        "sto_cache_hits": cached,
        "sto_exact_fallbacks": exact,
        "sto_inits": inits,
        "sto_kappa_recomputes": kappa,
        "sto_hit_rate": cached / max(total, 1),
        "sto_query_time_total": sum(s.query_time_total for s in sto_bank),
        "sto_cache_query_time_total": sum(s.cache_query_time_total for s in sto_bank),
        "sto_exact_query_time_total": sum(s.exact_query_time_total for s in sto_bank),
        "sto_reason_counter": reason_counts,
        "sto_invalid_reason_counter": invalid_reason_counts,
        "sto_reason_counts": format_reason_counts(reason_counts),
        "sto_invalid_reason_counts": format_reason_counts(reason_counts, include_cache=False),
        "sto_last_reason": last_reason,
        "sto_last_raw_reason": last_raw_reason,
        "sto_last_rho": last_rho,
        "sto_rho_min": float(np.min(finite_rhos)) if finite_rhos.size else np.nan,
        "sto_rho_mean": float(np.mean(finite_rhos)) if finite_rhos.size else np.nan,
        "sto_rho_max": float(np.max(finite_rhos)) if finite_rhos.size else np.nan,
        "sto_last_eta": last_eta,
        "sto_eta_mean": float(np.mean(finite_etas)) if finite_etas.size else np.nan,
        "sto_eta_max": float(np.max(finite_etas)) if finite_etas.size else np.nan,
        "sto_last_kappa": last_kappa,
        "sto_last_kappa_recomputed": last_kappa_recomputed,
    }
