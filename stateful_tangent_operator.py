"""
Stateful Tangent Operator (STO) — validate-before-reuse implementation.

Purpose
-------
For an implicit equilibrium system

    G(x, z) = 0,

implicit differentiation gives the adjoint tangent action

    T(v) = -G_z^T (G_x^T)^(-1) v.

This class caches the expensive inverse-adjoint action

    M ~= (G_x_anchor^T)^(-1)

at a realized equilibrium branch. Before applying the cached operator to the
actual loss-gradient query v, it performs a query-independent validity check
using fixed sentinel probes stored at initialization time.

Lifecycle
---------
    initialize(G_x_anchor, x_anchor, z_anchor)
        -> store M and fixed validation probes omega_i
        -> seed sigma_G(anchor) and kappa(anchor)

    validate(G_x_new, x_new, z_new)
        -> use only drift checks and sentinel probes, never the actual query v

    apply(v, G_z_new)
        -> allowed only after validation passes

    query(v, G_x_new, G_z_new, x_new, z_new)
        -> convenience wrapper: validate -> apply, or exact fallback -> reinit

Acceptance gates
----------------
Two correctness gates govern reuse. Neither uses the actual query v.

    (a) probe residual gate (cheap, every validate):
            rho := max_i || G_x_new^T M omega_i - omega_i ||
            accept iff rho <= rho_max.
        This directly measures fidelity of M as an inverse of G_x_new^T on
        m random probe directions; with unit-norm probes it lower-bounds
        || I - G_x_new^T M ||_2.

    (b) conditioning gate (expensive, on cadence + kappa warning band):
            kappa := sigma_max(G_x_new) * sigma_max(M)
                  ~  sigma_max(G_x_new) / sigma_min(G_x_anchor)
            accept iff kappa <= kappa_max.
        This rejects regimes where the cached operator amplifies any
        residual into a large gradient error — exactly what fires near
        bifurcations / branch transitions where G_x becomes near-singular.

The bound

        || T_hat(v) - T(v) || <= ||G_z|| * ||G_x_new^{-1}|| * rho * ||v||
                              ~  (some const) * (kappa * rho) * ||v||

(paper Eq. (14)) is reported as

    eta := kappa * rho

for diagnostics, but is NOT used as a gate. The two physical criteria
(operator faithfulness and conditioning) are sufficient and decouple a
cheap, always-on signal from an expensive one.

Adaptive kappa cadence
----------------------
sigma_max(G_x_new) is estimated by power iteration and is the dominant cost
of validate(). We avoid computing it on most calls by exploiting the same
locality that justifies caching M:

  * recompute kappa every kappa_check_period validates (default 10);
  * when a recomputed kappa exceeds kappa_warn, latch into "warning mode";
  * while in warning mode, recompute kappa on every validate to track whether
    conditioning recovers or worsens toward kappa_max;
  * exit warning mode after kappa stays at or below kappa_warn for `cooldown`
    consecutive recomputations.

In the steady regime (kappa <= kappa_warn) kappa is computed once per cadence
period and reused. Once kappa itself indicates possible near-singularity, we
pay for a fresh kappa on every call. Rho remains an independent residual gate.

This is a dense research-grade implementation: M is cached explicitly as a
NumPy array. The validate-before-reuse lifecycle is independent of that
representation, so M can be swapped for a sparse factorization or a
matrix-free approximate inverse without other changes.
"""

from enum import Enum
from typing import NamedTuple, Optional

import numpy as np


class ValidationResult(NamedTuple):
    """Returned by StatefulTangentOperator.validate()."""
    valid: bool
    rho: float                # probe residual: max_i || G_x_new^T M omega_i - omega_i ||
    kappa: float              # amplification proxy: sigma_max(G_x_new) * sigma_max(M)
    eta: float                # reported error proxy: kappa * rho (NOT a gate)
    kappa_recomputed: bool    # True iff sigma_max(G_x_new) was freshly computed
    reason: str


class QueryResult(NamedTuple):
    """Returned by StatefulTangentOperator.query()."""
    g_z: np.ndarray
    used_cache: bool
    eta: float
    reason: str


class InvalidationReason(Enum):
    DRIFT = "drift"
    PROBE_RESIDUAL = "probe_residual"
    CONDITIONING = "conditioning"
    BRANCH_CHANGE = "branch_change"
    MAX_REUSE = "max_reuse"
    SINGULAR = "singular"
    EXPLICIT = "explicit"
    COLD_START = "cold_start"
    SHAPE_MISMATCH = "shape_mismatch"


class STOStats:
    """Amortization bookkeeping."""

    def __init__(self) -> None:
        self.n_init = 0
        self.n_cached = 0
        self.n_exact = 0
        self.n_total = 0
        self.n_validate = 0
        self.n_invalidations = 0
        self.n_kappa_recompute = 0
        self.log: list[tuple[int, str, float]] = []

    @property
    def hit_rate(self) -> float:
        return self.n_cached / max(self.n_total, 1)

    @property
    def tangent_eval_reduction(self) -> float:
        """
        Optimistic, init-free reuse multiplier.

        Equals (n_cached + n_exact) / n_exact, i.e. the number of solves
        that would have been needed under always-exact divided by the
        number actually performed as fallback. Treats both cache hits
        and initializations as free, so it overstates true wall-clock
        savings whenever n_init or C_query is non-negligible.

        Use StatefulTangentOperator.amortized_speedup(cost_ratio) for
        the init-aware speedup that matches the paper's cost model
            baseline = (n_cached + n_exact) * C_solve
            actual   = (n_init + n_exact) * C_solve + n_cached * C_query.
        """
        return self.n_total / max(self.n_exact, 1)

    @property
    def kappa_skip_rate(self) -> float:
        """Fraction of validates that reused a stale kappa."""
        return 1.0 - (self.n_kappa_recompute / max(self.n_validate, 1))

    def summary(self) -> str:
        return (
            "STO Statistics\n"
            f"  Total queries          : {self.n_total}\n"
            f"  Cache hits             : {self.n_cached}\n"
            f"  Exact fallback solves  : {self.n_exact}\n"
            f"  Validate calls         : {self.n_validate}\n"
            f"  (Re-)initializations   : {self.n_init}\n"
            f"  Invalidation events    : {self.n_invalidations}\n"
            f"  Kappa recomputations   : {self.n_kappa_recompute}\n"
            f"  Hit rate               : {self.hit_rate:.1%}\n"
            f"  Kappa skip rate        : {self.kappa_skip_rate:.1%}\n"
            f"  Tangent eval reduction : {self.tangent_eval_reduction:.1f}x"
        )



def _as_1d(v: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _as_2d(A: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def exact_adjoint_query(G_x: np.ndarray, G_z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Exact equilibrium tangent action (paper Eq. (10)):

        T(v) = -G_z^T (G_x^T)^(-1) v.

    Used as fallback when an STO gate fails.
    """
    G_x = _as_2d(G_x, name="G_x")
    G_z = _as_2d(G_z, name="G_z")
    v = _as_1d(v, name="v")

    n_x = G_x.shape[0]
    if G_x.shape != (n_x, n_x):
        raise ValueError(f"G_x must be square, got shape {G_x.shape}.")
    if G_z.shape[0] != n_x:
        raise ValueError(
            f"G_z has incompatible shape {G_z.shape}; expected first dimension {n_x}."
        )
    if v.shape[0] != n_x:
        raise ValueError(f"v has length {v.shape[0]}; expected {n_x}.")

    p = np.linalg.solve(G_x.T, v)
    return -(G_z.T @ p)


def _normalize_columns(U: np.ndarray) -> np.ndarray:
    """Column-normalize U. Robust to all-zero columns via dtype tiny."""
    U = np.asarray(U, dtype=float)
    norms = np.linalg.norm(U, axis=0, keepdims=True)
    eps = np.finfo(U.dtype).tiny
    return U / (norms + eps)


def _power_iter_spectral_norm(
    apply_A,
    apply_AT,
    n: int,
    n_iter: int,
    rng: np.random.Generator,
) -> float:
    """Estimate ||A||_2 = sigma_max(A) by power iteration on A^T A."""
    if n <= 0:
        return 0.0
    if n_iter <= 0:
        u = rng.standard_normal(n)
        u /= np.linalg.norm(u) + 1e-30
        return float(np.linalg.norm(apply_A(u)))

    u = rng.standard_normal(n)
    u /= np.linalg.norm(u) + 1e-30
    for _ in range(n_iter):
        w = apply_AT(apply_A(u))
        nw = np.linalg.norm(w)
        if nw < 1e-30:
            return 0.0
        u = w / nw
    val = float(np.linalg.norm(apply_A(u)))
    return val if np.isfinite(val) else np.inf


def _stable_inverse_transpose(G_x: np.ndarray, rcond: float) -> Optional[np.ndarray]:
    """
    Return M = inv(G_x.T) if G_x is numerically invertible; otherwise None.

    Cheap accept/reject using a 1-norm condition proxy (avoids the O(n^3)
    SVD that np.linalg.cond performs by default). Tight 2-norm conditioning
    is checked by the dedicated kappa gate at validate time.
    """
    try:
        M = np.linalg.inv(G_x.T)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(M)):
        return None
    norm_Gx = float(np.linalg.norm(G_x, ord=1))
    norm_M = float(np.linalg.norm(M, ord=1))
    cond_proxy = norm_Gx * norm_M
    if (not np.isfinite(cond_proxy)) or cond_proxy > 1.0 / max(rcond, 1e-300):
        return None
    return M


def _probe_residual_max(
    G_x_new: np.ndarray,
    M_probes: np.ndarray,
    probes: np.ndarray,
    residual_scale: Optional[np.ndarray] = None,
) -> float:
    """
    rho = max_i || G_x_new^T (M omega_i) - omega_i ||.

    With unit-norm probes this is a probe-direction lower bound on
    || I - G_x_new^T M ||_2, the operator-norm distance from M being
    a true inverse of G_x_new^T.
    """
    W = G_x_new.T @ M_probes - probes
    if residual_scale is None:
        norms = np.linalg.norm(W, axis=0)
    else:
        scale = np.asarray(residual_scale, dtype=float).reshape(-1, 1)
        if scale.shape[0] != W.shape[0]:
            return float("inf")
        denom = np.linalg.norm(scale * probes, axis=0)
        norms = np.linalg.norm(scale * W, axis=0) / np.maximum(denom, 1e-300)
    if not np.all(np.isfinite(norms)):
        return float("inf")
    return float(np.max(norms))


def _residual_scale_from_jacobian(G_x: np.ndarray, mode: str) -> Optional[np.ndarray]:
    """Return diagonal weights for the residual norm, or None for raw norm."""
    selected = str(mode).lower()
    if selected == "raw":
        return None
    if selected != "scaled":
        raise ValueError(f"Unknown rho_norm {mode!r}; expected 'raw' or 'scaled'.")

    # The residual vector lives in the same coordinates as the columns of G_x.
    # Column-norm scaling validates in a diagonally scaled norm, reducing the
    # impact of heterogeneous stiffness/units without changing the STO lifecycle.
    col_norms = np.linalg.norm(G_x, axis=0)
    floor = max(float(np.median(col_norms)) * 1e-12, 1e-300)
    return 1.0 / np.maximum(col_norms, floor)


class StatefulTangentOperator:
    """
    Validate-before-reuse STO with adaptive kappa cadence.

    Acceptance gates (both must pass; neither uses the actual query v):
      (a) rho <= rho_max          — probe residual, cheap, every validate.
      (b) kappa <= kappa_max      — conditioning, expensive, on cadence + warning.

    Reported diagnostics (not gates):
      eta = kappa * rho           — Eq. (14) error-bound proxy.
    """

    def __init__(
        self,
        rho_max: float = 1e-1,
        kappa_max: float = 1e10,
        kappa_warn: float = 1e8,
        kappa_check_period: int = 10,
        cooldown: int = 3,
        drift_radius_x: float = np.inf,
        drift_radius_z: float = np.inf,
        n_probes: int = 5,
        n_power_iter: int = 5,
        max_reuse: int = 10000,
        singular_rcond: float = 1e-12,
        rho_norm: str = "raw",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rho_max <= 0:
            raise ValueError("rho_max must be positive.")
        if kappa_max <= 0:
            raise ValueError("kappa_max must be positive.")
        if kappa_warn <= 0:
            raise ValueError("kappa_warn must be positive.")
        if kappa_warn >= kappa_max:
            raise ValueError(
                f"kappa_warn ({kappa_warn}) must be strictly smaller than "
                f"kappa_max ({kappa_max}); otherwise the warning band has no effect."
            )
        if kappa_check_period < 1:
            raise ValueError("kappa_check_period must be at least 1.")
        if cooldown < 1:
            raise ValueError("cooldown must be at least 1.")
        if n_probes < 1:
            raise ValueError("n_probes must be at least 1.")
        if n_power_iter < 0:
            raise ValueError("n_power_iter must be nonnegative.")
        if max_reuse < 1:
            raise ValueError("max_reuse must be at least 1.")

        self.rho_max = float(rho_max)
        self.kappa_max = float(kappa_max)
        self.kappa_warn = float(kappa_warn)
        self.kappa_check_period = int(kappa_check_period)
        self.cooldown = int(cooldown)
        self.drift_radius_x = float(drift_radius_x)
        self.drift_radius_z = float(drift_radius_z)
        self.n_probes = int(n_probes)
        self.n_power_iter = int(n_power_iter)
        self.max_reuse = int(max_reuse)
        self.singular_rcond = float(singular_rcond)
        if str(rho_norm).lower() not in ("raw", "scaled"):
            raise ValueError("rho_norm must be 'raw' or 'scaled'.")
        self.rho_norm = str(rho_norm).lower()
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Cached operator: M = inv(G_x_anchor.T).
        self._M: Optional[np.ndarray] = None
        self._x_bar: Optional[np.ndarray] = None
        self._z_bar: Optional[np.ndarray] = None
        self._n_x: int = 0
        self._valid: bool = False
        self._reuse_count: int = 0

        # Fixed sentinel probes and their cached inverse actions.
        self._probes: Optional[np.ndarray] = None      # omega_i,    shape (n_x, m)
        self._M_probes: Optional[np.ndarray] = None    # M omega_i,  shape (n_x, m)
        self._residual_scale: Optional[np.ndarray] = None

        # Cadence state for the kappa gate.
        self._sigma_M: float = np.inf                  # fixed at init (M is fixed)
        self._sigma_G_cached: float = np.nan           # most recent sigma_max(G_x_new)
        self._kappa_cached: float = np.nan             # most recent sigma_G_cached * sigma_M
        self._validates_since_kappa: int = 0
        self._in_warning: bool = False
        self._cool_count: int = 0

        # Last-call diagnostics (read by adapters / training loops).
        self.last_validation: Optional[ValidationResult] = None
        self.last_query_metrics: dict[str, float] = {}

        self.stats = STOStats()

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        return self._valid

    @property
    def reuse_count(self) -> int:
        return self._reuse_count

    @property
    def in_warning(self) -> bool:
        return self._in_warning

    @property
    def probes(self) -> Optional[np.ndarray]:
        """Read-only copy of fixed validation probes, for debugging/plots."""
        return None if self._probes is None else self._probes.copy()

    # ------------------------------------------------------------------
    # Cached inverse actions
    # ------------------------------------------------------------------

    def _apply_M(self, V: np.ndarray) -> np.ndarray:
        if self._M is None:
            raise RuntimeError("STO has no cached inverse. Call initialize() first.")
        return self._M @ V

    def _apply_MT(self, V: np.ndarray) -> np.ndarray:
        if self._M is None:
            raise RuntimeError("STO has no cached inverse. Call initialize() first.")
        return self._M.T @ V

    # ------------------------------------------------------------------
    # Lifecycle: initialize
    # ------------------------------------------------------------------

    def initialize(
        self,
        G_x: np.ndarray,
        x_bar: Optional[np.ndarray] = None,
        z_bar: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Initialize STO at a realized equilibrium branch.

        Stores M = inv(G_x^T), draws fixed probes omega_i, and seeds the
        kappa cadence by computing sigma_max(G_x) and sigma_max(M) at the
        anchor. Refuses to initialize if either gate would already fail
        at the anchor (so the cache is born admissible).
        """
        try:
            G_x = _as_2d(G_x, name="G_x")
            n_x = G_x.shape[0]
            if G_x.shape != (n_x, n_x):
                raise ValueError(f"G_x must be square, got shape {G_x.shape}.")
        except ValueError:
            self.invalidate(InvalidationReason.SINGULAR)
            return False

        M = _stable_inverse_transpose(G_x, rcond=self.singular_rcond)
        if M is None or not np.all(np.isfinite(M)):
            self.invalidate(InvalidationReason.SINGULAR)
            return False

        # Provisional cache so the helper apply functions work.
        self._M = M
        self._n_x = n_x
        self._x_bar = np.asarray(x_bar, dtype=float).copy() if x_bar is not None else None
        self._z_bar = np.asarray(z_bar, dtype=float).copy() if z_bar is not None else None

        # Fixed unit-norm probes for this lifecycle, plus their inverse action.
        self._probes = _normalize_columns(self.rng.standard_normal((n_x, self.n_probes)))
        self._M_probes = self._apply_M(self._probes)
        self._residual_scale = _residual_scale_from_jacobian(G_x, self.rho_norm)

        # ||M||_2 is fixed for the lifetime of the cache.
        sigma_M = _power_iter_spectral_norm(
            self._apply_M,
            self._apply_MT,
            n_x,
            self.n_power_iter,
            self.rng,
        )
        if not np.isfinite(sigma_M):
            self.invalidate(InvalidationReason.SINGULAR)
            return False
        self._sigma_M = sigma_M

        # Anchor conditioning: kappa(anchor) = sigma_max(G_x_anchor) * ||M||_2.
        sigma_G_anchor = _power_iter_spectral_norm(
            lambda v: G_x @ v,
            lambda v: G_x.T @ v,
            n_x,
            self.n_power_iter,
            self.rng,
        )
        kappa_anchor = float(sigma_G_anchor * sigma_M)
        if not np.isfinite(kappa_anchor) or kappa_anchor > self.kappa_max:
            self.invalidate(InvalidationReason.CONDITIONING)
            return False

        # Init self-check: numerical floor of the probe residual at the anchor.
        # In double precision rho_init ~ eps * cond(G_x). If it already exceeds
        # rho_max the cache is born too inaccurate to ever satisfy gate (a).
        rho_init = _probe_residual_max(
            G_x, self._M_probes, self._probes, self._residual_scale
        )
        if not np.isfinite(rho_init) or rho_init > self.rho_max:
            self.invalidate(InvalidationReason.PROBE_RESIDUAL)
            return False

        # Seed cadence state. The first validate at the same point will
        # reuse the anchor kappa rather than recompute.
        self._sigma_G_cached = float(sigma_G_anchor)
        self._kappa_cached = kappa_anchor
        self._validates_since_kappa = 0
        self._in_warning = bool(kappa_anchor > self.kappa_warn)
        self._cool_count = 0

        self._valid = True
        self._reuse_count = 0
        self.stats.n_init += 1
        return True

    # ------------------------------------------------------------------
    # Lifecycle: validate
    # ------------------------------------------------------------------

    def _update_warning(self, kappa: float, kappa_warn: float) -> None:
        """Latch into / out of warning mode based on freshly computed kappa."""
        if kappa > kappa_warn:
            self._in_warning = True
            self._cool_count = 0
        elif self._in_warning:
            self._cool_count += 1
            if self._cool_count >= self.cooldown:
                self._in_warning = False
                self._cool_count = 0

    def _record_metrics(
        self,
        rho: float,
        kappa: float,
        eta: float,
        kappa_recomputed: bool,
        sigma_G: float,
        kappa_warn: Optional[float] = None,
    ) -> None:
        kappa_warn_value = self.kappa_warn if kappa_warn is None else float(kappa_warn)
        self.last_query_metrics = {
            "rho": float(rho),
            "kappa": float(kappa),
            "eta": float(eta),
            "kappa_recomputed": float(kappa_recomputed),
            "in_warning": float(self._in_warning),
            "kappa_warn": kappa_warn_value,
            "sigma_G": float(sigma_G),
            "sigma_M": float(self._sigma_M),
            "reuse_count": float(self._reuse_count),
            "validates_since_kappa": float(self._validates_since_kappa),
        }

    def validate(
        self,
        G_x_new: np.ndarray,
        x_new: Optional[np.ndarray] = None,
        z_new: Optional[np.ndarray] = None,
        rho_max: Optional[float] = None,
        kappa_max: Optional[float] = None,
        kappa_warn: Optional[float] = None,
        drift_radius_x: Optional[float] = None,
        drift_radius_z: Optional[float] = None,
    ) -> ValidationResult:
        """
        Query-independent validity gate. See class docstring for the
        gate hierarchy. Per-call overrides accept the same parameter
        names as the constructor; None means "use stored default".
        """
        self.stats.n_validate += 1

        rho_max_eff = self.rho_max if rho_max is None else float(rho_max)
        kappa_max_eff = self.kappa_max if kappa_max is None else float(kappa_max)
        kappa_warn_eff = self.kappa_warn if kappa_warn is None else float(kappa_warn)
        drift_x_eff = self.drift_radius_x if drift_radius_x is None else float(drift_radius_x)
        drift_z_eff = self.drift_radius_z if drift_radius_z is None else float(drift_radius_z)
        if rho_max_eff <= 0:
            raise ValueError("rho_max must be positive.")
        if kappa_max_eff <= 0:
            raise ValueError("kappa_max must be positive.")
        if kappa_warn_eff <= 0:
            raise ValueError("kappa_warn must be positive.")
        if kappa_warn_eff >= kappa_max_eff:
            raise ValueError(
                f"kappa_warn ({kappa_warn_eff}) must be strictly smaller than "
                f"kappa_max ({kappa_max_eff})."
            )

        if (
            not self._valid
            or self._M is None
            or self._probes is None
            or self._M_probes is None
        ):
            result = ValidationResult(False, np.inf, np.inf, np.inf, False, "cold_start")
            self.last_validation = result
            return result
        if self._reuse_count >= self.max_reuse:
            result = ValidationResult(False, 0.0, 0.0, 0.0, False, "max_reuse")
            self.last_validation = result
            return result

        try:
            G_x_new = _as_2d(G_x_new, name="G_x_new")
        except ValueError as exc:
            result = ValidationResult(False, np.inf, np.inf, np.inf, False, str(exc))
            self.last_validation = result
            return result

        if G_x_new.shape != (self._n_x, self._n_x):
            result = ValidationResult(
                False, np.inf, np.inf, np.inf, False,
                f"shape_mismatch: got {G_x_new.shape}, expected {(self._n_x, self._n_x)}",
            )
            self.last_validation = result
            return result

        # ----- (1) Drift gates (optional, geometric early-out). -----
        if np.isfinite(drift_x_eff) and self._x_bar is not None and x_new is not None:
            x_new_arr = np.asarray(x_new, dtype=float)
            if x_new_arr.shape != self._x_bar.shape:
                result = ValidationResult(False, np.inf, np.inf, np.inf, False, "x_shape_mismatch")
                self.last_validation = result
                return result
            drift_x = float(np.linalg.norm(x_new_arr - self._x_bar))
            if drift_x > drift_x_eff:
                result = ValidationResult(
                    False, np.inf, np.inf, np.inf, False,
                    f"drift_x={drift_x:.2e} > {drift_x_eff:.2e}",
                )
                self.last_validation = result
                return result

        if np.isfinite(drift_z_eff) and self._z_bar is not None and z_new is not None:
            z_new_arr = np.asarray(z_new, dtype=float)
            if z_new_arr.shape != self._z_bar.shape:
                result = ValidationResult(False, np.inf, np.inf, np.inf, False, "z_shape_mismatch")
                self.last_validation = result
                return result
            drift_z = float(np.linalg.norm(z_new_arr - self._z_bar))
            if drift_z > drift_z_eff:
                result = ValidationResult(
                    False, np.inf, np.inf, np.inf, False,
                    f"drift_z={drift_z:.2e} > {drift_z_eff:.2e}",
                )
                self.last_validation = result
                return result

        # ----- (2) Probe residual gate (cheap, always). -----
        rho = _probe_residual_max(
            G_x_new, self._M_probes, self._probes, self._residual_scale
        )
        if not np.isfinite(rho):
            self._record_metrics(
                np.inf, self._kappa_cached, np.inf, False, self._sigma_G_cached, kappa_warn_eff
            )
            result = ValidationResult(False, np.inf, self._kappa_cached, np.inf, False, "nonfinite_rho")
            self.last_validation = result
            return result
        if rho > rho_max_eff:
            eta = float(self._kappa_cached * rho) if np.isfinite(self._kappa_cached) else np.inf
            self._record_metrics(
                rho, self._kappa_cached, eta, False, self._sigma_G_cached, kappa_warn_eff
            )
            result = ValidationResult(
                False, rho, self._kappa_cached, eta, False,
                f"rho={rho:.2e} > rho_max={rho_max_eff:.2e}",
            )
            self.last_validation = result
            return result

        # ----- (3) Decide whether to recompute kappa this call. -----
        self._validates_since_kappa += 1
        recompute_kappa = (
            self._in_warning
            or self._validates_since_kappa >= self.kappa_check_period
            or not np.isfinite(self._kappa_cached)
        )

        if recompute_kappa:
            sigma_G = _power_iter_spectral_norm(
                lambda v: G_x_new @ v,
                lambda v: G_x_new.T @ v,
                self._n_x,
                self.n_power_iter,
                self.rng,
            )
            self._sigma_G_cached = float(sigma_G)
            self._kappa_cached = float(sigma_G * self._sigma_M)
            self._validates_since_kappa = 0
            self.stats.n_kappa_recompute += 1
            self._update_warning(self._kappa_cached, kappa_warn_eff)

        kappa = self._kappa_cached
        sigma_G = self._sigma_G_cached

        # ----- (4) Conditioning gate. -----
        if not np.isfinite(kappa):
            self._record_metrics(rho, np.inf, np.inf, recompute_kappa, sigma_G, kappa_warn_eff)
            result = ValidationResult(False, rho, np.inf, np.inf, recompute_kappa, "nonfinite_kappa")
            self.last_validation = result
            return result
        if kappa > kappa_max_eff:
            eta = float(kappa * rho)
            self._record_metrics(rho, kappa, eta, recompute_kappa, sigma_G, kappa_warn_eff)
            result = ValidationResult(
                False, rho, kappa, eta, recompute_kappa,
                f"kappa={kappa:.2e} > kappa_max={kappa_max_eff:.2e}",
            )
            self.last_validation = result
            return result

        # ----- (5) Both gates pass. eta is reported only. -----
        eta = float(kappa * rho)
        self._record_metrics(rho, kappa, eta, recompute_kappa, sigma_G, kappa_warn_eff)
        result = ValidationResult(True, rho, kappa, eta, recompute_kappa, "ok")
        self.last_validation = result
        return result

    # ------------------------------------------------------------------
    # Lifecycle: apply
    # ------------------------------------------------------------------

    def _apply_raw(self, v: np.ndarray, G_z: np.ndarray) -> np.ndarray:
        """T_hat(v) = -G_z^T M v, no bookkeeping. Used internally and by tests."""
        p = self._apply_M(v)
        return -(G_z.T @ p)

    def apply(self, v: np.ndarray, G_z: np.ndarray) -> np.ndarray:
        """
        Apply cached tangent operator (paper Eq. (11)) to the actual
        query v:

            T_hat(v) = -G_z^T M v.

        Assumes validate() has already passed for the current point. Use
        query() for the safe one-shot interface.
        """
        if not self._valid or self._M is None:
            raise RuntimeError("STO.apply() called on invalid cache. Call initialize() first.")

        v = _as_1d(v, name="v")
        G_z = _as_2d(G_z, name="G_z")
        if v.shape[0] != self._n_x:
            raise ValueError(f"v has length {v.shape[0]}; expected {self._n_x}.")
        if G_z.shape[0] != self._n_x:
            raise ValueError(
                f"G_z has incompatible shape {G_z.shape}; expected first dimension {self._n_x}."
            )

        g_z = self._apply_raw(v, G_z)
        self._reuse_count += 1
        self.stats.n_cached += 1
        self.stats.n_total += 1
        return g_z

    # ------------------------------------------------------------------
    # Lifecycle: invalidate
    # ------------------------------------------------------------------

    def invalidate(self, reason: InvalidationReason = InvalidationReason.EXPLICIT) -> None:
        """Discard the current cache and reset cadence state."""
        if self._valid:
            self.stats.n_invalidations += 1
            self.stats.log.append((self.stats.n_total, reason.value, 0.0))

        self._valid = False
        self._M = None
        self._x_bar = None
        self._z_bar = None
        self._n_x = 0
        self._reuse_count = 0
        self._probes = None
        self._M_probes = None
        self._residual_scale = None
        self._sigma_M = np.inf
        self._sigma_G_cached = np.nan
        self._kappa_cached = np.nan
        self._validates_since_kappa = 0
        self._in_warning = False
        self._cool_count = 0

    # ------------------------------------------------------------------
    # Convenience: one-shot query
    # ------------------------------------------------------------------

    def query(
        self,
        v: np.ndarray,
        G_x_new: np.ndarray,
        G_z_new: np.ndarray,
        x_new: Optional[np.ndarray] = None,
        z_new: Optional[np.ndarray] = None,
    ) -> QueryResult:
        """
        Safe one-shot interface:

            validate current cache using the two gates;
            if accepted, apply cache to the actual v;
            otherwise exact adjoint solve, then reinitialize at this point.
        """
        gate = self.validate(G_x_new, x_new=x_new, z_new=z_new)

        if gate.valid:
            g_z = self.apply(v, G_z_new)
            return QueryResult(g_z, True, gate.eta, "cache_hit")

        # Gate failed: never apply cached operator to v.
        self.invalidate(self._reason_from_gate(gate))
        g_z = exact_adjoint_query(G_x_new, G_z_new, v)
        self.stats.n_exact += 1
        self.stats.n_total += 1

        ok = self.initialize(G_x_new, x_bar=x_new, z_bar=z_new)
        if not ok:
            return QueryResult(g_z, False, gate.eta, gate.reason + "; reinit_failed")
        return QueryResult(g_z, False, gate.eta, gate.reason)

    @staticmethod
    def _reason_from_gate(gate: ValidationResult) -> InvalidationReason:
        r = gate.reason
        if r.startswith("drift") or r.endswith("shape_mismatch"):
            return InvalidationReason.DRIFT
        if r.startswith("kappa") or r == "nonfinite_kappa":
            return InvalidationReason.CONDITIONING
        if r.startswith("rho") or r == "nonfinite_rho":
            return InvalidationReason.PROBE_RESIDUAL
        if r == "max_reuse":
            return InvalidationReason.MAX_REUSE
        if r == "cold_start":
            return InvalidationReason.COLD_START
        if r.startswith("shape_mismatch"):
            return InvalidationReason.SHAPE_MISMATCH
        return InvalidationReason.EXPLICIT

    # ------------------------------------------------------------------
    # External branch signal
    # ------------------------------------------------------------------

    def mark_branch_changed(self) -> None:
        """External hook for the forward solver when it detects a branch switch."""
        self.invalidate(InvalidationReason.BRANCH_CHANGE)

    # ------------------------------------------------------------------
    # Amortization
    # ------------------------------------------------------------------

    def amortized_speedup(self, cost_ratio: Optional[float] = None) -> float:
        """
        Init-aware speedup vs always-exact baseline.

        Cost model:
            baseline cost = (n_cached + n_exact) * C_solve
            actual cost   = (n_init + n_exact) * C_solve
                          + n_cached * C_query

        cost_ratio = C_query / C_solve in [0, 1]. For dense Gaussian
        elimination C_solve = O(n_x^3) and C_query = O(n_x^2), so
        cost_ratio ~ 1 / n_x. If None, this default is used (requires
        the STO to currently have an anchor).
        """
        s = self.stats
        work_done = s.n_cached + s.n_exact
        if work_done == 0:
            return 1.0

        if cost_ratio is None:
            if self._n_x > 0:
                cost_ratio = 1.0 / self._n_x
            else:
                raise ValueError(
                    "cost_ratio is None and STO has no current anchor "
                    "(no n_x available); pass cost_ratio explicitly."
                )

        cost_ratio = float(cost_ratio)
        if not (0.0 <= cost_ratio <= 1.0):
            raise ValueError(f"cost_ratio must lie in [0, 1]; got {cost_ratio}.")

        actual = (s.n_init + s.n_exact) + s.n_cached * cost_ratio
        return float(work_done) / max(actual, 1e-12)

    # ------------------------------------------------------------------
    # Reset / repr
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear cache and statistics."""
        self.invalidate(InvalidationReason.EXPLICIT)
        self.stats = STOStats()
        self.last_validation = None
        self.last_query_metrics = {}

    def __repr__(self) -> str:
        status = "VALID" if self._valid else "INVALID"
        anchored = "anchored" if (self._x_bar is not None or self._z_bar is not None) else "unanchored"
        warn = "warn" if self._in_warning else "calm"
        return (
            "StatefulTangentOperator("
            f"status={status}, {anchored}, n_x={self._n_x}, "
            f"rho_max={self.rho_max:.0e}, kappa_max={self.kappa_max:.0e}, "
            f"kappa_warn={self.kappa_warn:.0e}, kappa_period={self.kappa_check_period}, "
            f"cooldown={self.cooldown}, mode={warn}, "
            f"reuse={self._reuse_count}, probes={self.n_probes})"
        )


def verify_sto_gradient(
    G_x: np.ndarray,
    G_z: np.ndarray,
    v: np.ndarray,
    sto: StatefulTangentOperator,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> dict:
    """
    Check cached apply() against exact_adjoint_query() at the same point.

    Uses sto._apply_raw so that stats and the reuse counter are not
    perturbed by the verification call.
    """
    g_exact = exact_adjoint_query(G_x, G_z, v)
    if not sto.is_valid:
        return {
            "pass": False,
            "reason": "STO not valid",
            "g_exact": g_exact,
            "g_cached": None,
            "abs_err": np.inf,
            "rel_err": np.inf,
        }

    v = _as_1d(v, name="v")
    G_z = _as_2d(G_z, name="G_z")
    if v.shape[0] != sto._n_x:
        raise ValueError(f"v has length {v.shape[0]}; expected {sto._n_x}.")
    if G_z.shape[0] != sto._n_x:
        raise ValueError(
            f"G_z has incompatible shape {G_z.shape}; expected first dimension {sto._n_x}."
        )

    g_cached = sto._apply_raw(v, G_z)

    abs_err = float(np.linalg.norm(g_exact - g_cached))
    rel_err = abs_err / max(float(np.linalg.norm(g_exact)), 1e-30)
    return {
        "pass": (abs_err < atol) or (rel_err < rtol),
        "g_exact": g_exact,
        "g_cached": g_cached,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


if __name__ == "__main__":
    # Smoke test 1: fresh cache passes validate at the anchor.
    rng = np.random.default_rng(123)
    n_x, n_z = 12, 4
    A = rng.standard_normal((n_x, n_x))
    G_x = A.T @ A + 2.0 * np.eye(n_x)
    G_z = rng.standard_normal((n_x, n_z))
    v = rng.standard_normal(n_x)

    sto = StatefulTangentOperator(
        rho_max=1e-6, kappa_warn=1e8, kappa_max=1e10,
        kappa_check_period=10, cooldown=3, n_probes=8, rng=rng,
    )
    assert sto.initialize(G_x, x_bar=np.zeros(n_x), z_bar=np.zeros(n_z))
    gate = sto.validate(G_x, x_new=np.zeros(n_x), z_new=np.zeros(n_z))
    assert gate.valid, gate
    check = verify_sto_gradient(G_x, G_z, v, sto)
    assert check["pass"], check

    # Smoke test 2: cadence skips kappa for steady reuse (fresh STO).
    sto2 = StatefulTangentOperator(
        rho_max=1e-6, kappa_warn=1e8, kappa_max=1e10,
        kappa_check_period=5, cooldown=3, n_probes=8, rng=rng,
    )
    assert sto2.initialize(G_x, x_bar=np.zeros(n_x), z_bar=np.zeros(n_z))
    n_kr_before = sto2.stats.n_kappa_recompute
    for i in range(sto2.kappa_check_period - 1):
        gate = sto2.validate(G_x, x_new=np.zeros(n_x), z_new=np.zeros(n_z))
        assert gate.valid
        assert not gate.kappa_recomputed, f"validate #{i+1}: kappa should be skipped"
    gate = sto2.validate(G_x, x_new=np.zeros(n_x), z_new=np.zeros(n_z))
    assert gate.kappa_recomputed, "kappa must be recomputed once period elapses"
    assert sto2.stats.n_kappa_recompute == n_kr_before + 1

    # Smoke test 3: high kappa latches warning mode, then cooldown exits it.
    sto3 = StatefulTangentOperator(
        rho_max=10.0, kappa_warn=1e8, kappa_max=1e12,
        kappa_check_period=10, cooldown=2, n_probes=8, rng=rng,
    )
    assert sto3.initialize(G_x, x_bar=np.zeros(n_x), z_bar=np.zeros(n_z))
    anchor_kappa = sto3._kappa_cached
    sto3.kappa_warn = 1.5 * anchor_kappa
    sto3.kappa_max = 10.0 * anchor_kappa
    sto3._validates_since_kappa = sto3.kappa_check_period - 1
    gate = sto3.validate(2.0 * G_x)
    assert gate.valid, gate
    assert gate.kappa_recomputed, "cadence must recompute kappa"
    assert sto3.in_warning, "warning mode must latch when kappa > kappa_warn"
    gate = sto3.validate(G_x)
    assert gate.valid and gate.kappa_recomputed
    assert sto3.in_warning, "cooldown should require two safe kappa observations"
    gate = sto3.validate(G_x)
    assert gate.valid and gate.kappa_recomputed
    assert not sto3.in_warning, "warning mode must exit after cooldown safe observations"

    # Smoke test 4: kappa above kappa_max rejects reuse.
    sto4 = StatefulTangentOperator(
        rho_max=10.0, kappa_warn=1e8, kappa_max=1e12,
        kappa_check_period=10, cooldown=2, n_probes=8, rng=rng,
    )
    assert sto4.initialize(G_x, x_bar=np.zeros(n_x), z_bar=np.zeros(n_z))
    anchor_kappa = sto4._kappa_cached
    sto4.kappa_warn = 1.2 * anchor_kappa
    sto4.kappa_max = 1.5 * anchor_kappa
    sto4._validates_since_kappa = sto4.kappa_check_period - 1
    gate = sto4.validate(2.0 * G_x)
    assert not gate.valid, "kappa > kappa_max must reject reuse"
    assert gate.reason.startswith("kappa="), gate

    # Smoke test 5: query() still works end-to-end.
    qr = sto.query(v, G_x, G_z, x_new=np.zeros(n_x), z_new=np.zeros(n_z))
    s = sto.amortized_speedup()
    assert np.isfinite(s) and s > 0
    print("Smoke tests passed.")
    print(sto.stats.summary())
    print(f"amortized_speedup = {s:.3f}")
