#!/usr/bin/env python3
"""
3D plasma equilibrium solver (variational formulation).

Implements the workflow described in `veq/thesis` and `veq/readme`:
1) LCFS fitting
2) 3D grid + 1D profiles
3) Geometry / metric update
4) Lambda closure from J^rho = 0 (surface-by-surface 2D Fourier solve)
5) Force balance operators and 12 variational residuals
6) Nonlinear least-squares solve
7) Post-processing and 6 zeta-slice contour plots

Default setup matches the user-provided configuration:
- Grid (rho, theta, zeta) = (100, 160, 80)
- Three helical harmonics (s_0, s_1, s_2 aliases -> s1, s2, s3 in this file)
- N_t = 19
- LCFS:
    R = 10 - cos(theta) - 0.3*cos(theta)*cos(19*zeta) + 0.3*sin(theta)*sin(19*zeta)
    Z = sin(theta) - 0.3*sin(theta)*cos(19*zeta) - 0.3*cos(theta)*sin(19*zeta)
- Pressure:
    P(rho) = 18000*rho^4 - 36000*rho^2 + 18000
- Rotational transform:
    iota(rho) = 1 + 1.5*rho^2

Dynamic input support:
- Use --case-json <path> to pass theory-required inputs externally:
  - 1D profiles (pressure/iota/phi expressions, phi_total, psi_a)
  - LCFS boundary expressions (R_expr/Z_expr)
  - periodicity (N_t/n_helical), Lambda double-spectrum, initial guess vector X0
  - physical constants (c_light) for dimensional MHD operators
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
import numpy as np
from matplotlib.path import Path as MplPath
from scipy import fft as sfft
from scipy.interpolate import CubicSpline, griddata
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TWO_PI = 2.0 * np.pi
N_X_UNKNOWNS = 12

DEFAULT_CASE_CONFIG: Dict[str, Any] = {
    "profiles": {
        "pressure_expr": "18000*rho**4 - 36000*rho**2 + 18000",
        "iota_expr": "1 + 1.5*rho**2",
        "phi_expr": "phi_total * rho**2",
        "phi_total": 1.0,
        # If null, psi_a is inferred from integral of iota*Phi' at rho=1.
        "psi_a": None,
        "constants": {},
    },
    "lcfs": {
        "R_expr": (
            "10 - cos(theta) - 0.3*cos(theta)*cos(19*zeta) "
            "+ 0.3*sin(theta)*sin(19*zeta)"
        ),
        "Z_expr": (
            "sin(theta) - 0.3*sin(theta)*cos(19*zeta) "
            "- 0.3*cos(theta)*sin(19*zeta)"
        ),
        "constants": {},
    },
    "periodicity": {
        "N_t": 19,
        # chi = theta - n_helical*zeta. n_helical=-19 matches LCFS with (+19*zeta) phase.
        "n_helical": -19,
    },
    "solver": {
        # [h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s11, s20, s30]
        "initial_guess": [0.0] * N_X_UNKNOWNS,
        # Jacobian mode: "jax_ad" (automatic differentiation) or "scipy_fd".
        "jacobian_mode": "jax_ad",
        # Legacy note:
        # if users still pass 11-dim initial_guess, optional solver.s11 is accepted
        # and inserted as X[9] automatically.
        # boundary fit seed [R0,Z0,a,kappa_a,c0a,c1a,s1a,s2a,s3a]
        "boundary_seed": [10.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Optional fixed constants; if provided, skip LCFS fitting.
        # "boundary_constants": {
        #   "R0": ..., "Z0": ..., "a": ..., "kappa_a": ...,
        #   "c0a": ..., "c1a": ..., "s1a": ..., "s2a": ..., "s3a": ...
        # }
    },
    # Optional grid override via case file.
    "grid": {"n_rho": 100, "n_theta": 160, "n_zeta": 80, "n_modes": 3},
    # Lambda spectral closure basis:
    # Lambda(rho,theta,zeta)=sum_{m,n} lambda_mn(rho)*sin(m*theta-n*zeta)
    # This is numerically truncated but supports full (m,n) double expansion.
    "lambda_fourier": {
        "m_min": 0,
        "m_max": 3,
        "n_min": -3,
        "n_max": 3,
        "exclude_zero_mode": True,
        # Optional explicit list, e.g. [[1,0],[1,1],[2,-1],...]
        "pairs": [],
    },
    # Physical constants for dimensional consistency in MHD operators.
    "physical_constants": {
        # Set c_light=1.0 for normalized units, or 299792458.0 for SI.
        "c_light": 1.0
    },
}

BOUNDARY_KEYS = ("R0", "Z0", "a", "kappa_a", "c0a", "c1a", "s1a", "s2a", "s3a")

_SAFE_EVAL_GLOBALS = {"__builtins__": {}}
_SAFE_EVAL_LOCALS = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dictionary update, returns base for chaining."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], val)
        else:
            base[key] = val
    return base


def clone_default_case_config() -> Dict[str, Any]:
    """Return a deep copy of default case config without importing copy."""
    return json.loads(json.dumps(DEFAULT_CASE_CONFIG))


def safe_eval_expr(expr: str, variables: Dict[str, Any], constants: Dict[str, Any] | None = None) -> np.ndarray:
    """Evaluate user-configured expression with restricted namespace."""
    local_env = dict(_SAFE_EVAL_LOCALS)
    if constants:
        local_env.update(constants)
    local_env.update(variables)
    try:
        out = eval(expr, _SAFE_EVAL_GLOBALS, local_env)
    except Exception as exc:
        raise ValueError(f"Failed to evaluate expression: {expr!r}. Error: {exc}") from exc
    return np.asarray(out, dtype=float)


def load_case_config(case_json_path: str | None) -> Dict[str, Any]:
    """
    Load and merge external case config.

    If path is None, only defaults are used.
    """
    case_cfg = clone_default_case_config()
    if case_json_path:
        path = Path(case_json_path)
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise ValueError("Case JSON root must be an object.")
        deep_update(case_cfg, user_cfg)

    for section in (
        "profiles",
        "lcfs",
        "periodicity",
        "solver",
        "grid",
        "lambda_fourier",
        "physical_constants",
    ):
        if section in case_cfg and not isinstance(case_cfg[section], dict):
            raise ValueError(f"case.{section} must be an object.")
    for key in ("pressure_expr", "iota_expr", "phi_expr"):
        if key not in case_cfg["profiles"]:
            raise ValueError(f"case.profiles.{key} is required.")
    for key in ("R_expr", "Z_expr"):
        if key not in case_cfg["lcfs"]:
            raise ValueError(f"case.lcfs.{key} is required.")
    return case_cfg


def _spectral_periodic_derivative(arr: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Pseudo-spectral first derivative on periodic axis."""
    a = np.asarray(arr)
    n = a.shape[axis]
    k = TWO_PI * sfft.fftfreq(n, d=spacing)
    shape = [1] * a.ndim
    shape[axis] = n
    ik = (1j * k).reshape(shape)
    a_hat = sfft.fft(a, axis=axis)
    da = sfft.ifft(ik * a_hat, axis=axis)
    return da.real if np.isrealobj(a) else da


def dd_theta(arr: np.ndarray, dtheta: float) -> np.ndarray:
    """Periodic theta-derivative via FFT pseudo-spectral method (axis=1)."""
    return _spectral_periodic_derivative(arr, dtheta, axis=1)


def dd_zeta(arr: np.ndarray, dzeta: float) -> np.ndarray:
    """Periodic zeta-derivative via FFT pseudo-spectral method (axis=2)."""
    return _spectral_periodic_derivative(arr, dzeta, axis=2)


def dd_theta_2d(arr: np.ndarray, dtheta: float) -> np.ndarray:
    """2D periodic theta-derivative via FFT (theta axis=0)."""
    return _spectral_periodic_derivative(arr, dtheta, axis=0)


def dd_zeta_2d(arr: np.ndarray, dzeta: float) -> np.ndarray:
    """2D periodic zeta-derivative via FFT (zeta axis=1)."""
    return _spectral_periodic_derivative(arr, dzeta, axis=1)


def dd_rho(arr: np.ndarray, rho_1d: np.ndarray) -> np.ndarray:
    """
    Rho-derivative (axis=0):
    - uniform grid: 4th-order finite differences (interior + one-sided boundaries)
    - nonuniform grid: cubic-spline derivative
    """
    a = np.asarray(arr)
    rho = np.asarray(rho_1d, dtype=float)
    if a.shape[0] != rho.size:
        raise ValueError(f"dd_rho shape mismatch: axis0={a.shape[0]} vs rho={rho.size}.")

    n = rho.size
    if n < 3:
        raise ValueError("dd_rho requires at least 3 rho points.")

    dr = np.diff(rho)
    uniform = np.allclose(dr, dr[0], rtol=1.0e-10, atol=1.0e-14)
    if (not uniform) or n < 5:
        cs = CubicSpline(rho, a, axis=0, bc_type="natural")
        return cs(rho, 1)

    h = dr[0]
    out = np.empty_like(a, dtype=np.result_type(a.dtype, np.float64))

    out[0] = (-25.0 * a[0] + 48.0 * a[1] - 36.0 * a[2] + 16.0 * a[3] - 3.0 * a[4]) / (12.0 * h)
    out[1] = (-3.0 * a[0] - 10.0 * a[1] + 18.0 * a[2] - 6.0 * a[3] + a[4]) / (12.0 * h)
    out[2:-2] = (-a[4:] + 8.0 * a[3:-1] - 8.0 * a[1:-3] + a[:-4]) / (12.0 * h)
    out[-2] = (3.0 * a[-1] + 10.0 * a[-2] - 18.0 * a[-3] + 6.0 * a[-4] - a[-5]) / (12.0 * h)
    out[-1] = (25.0 * a[-1] - 48.0 * a[-2] + 36.0 * a[-3] - 16.0 * a[-4] + 3.0 * a[-5]) / (12.0 * h)
    return out


def integrate_surface(f2: np.ndarray, theta_1d: np.ndarray, zeta_1d: np.ndarray) -> float:
    """Integrate f(theta,zeta) dtheta dzeta."""
    return float(np.trapezoid(np.trapezoid(f2, zeta_1d, axis=1), theta_1d, axis=0))


def integrate_volume(
    f3: np.ndarray, rho_1d: np.ndarray, theta_1d: np.ndarray, zeta_1d: np.ndarray
) -> float:
    """Integrate f(rho,theta,zeta) drho dtheta dzeta."""
    return float(
        np.trapezoid(
            np.trapezoid(np.trapezoid(f3, zeta_1d, axis=2), theta_1d, axis=1),
            rho_1d,
            axis=0,
        )
    )


@dataclass
class BoundaryParams:
    R0: float
    Z0: float
    a: float
    kappa_a: float
    c0a: float
    c1a: float
    s1a: float
    s2a: float
    s3a: float


@dataclass
class SolverConfig:
    n_rho: int = 100
    n_theta: int = 160
    n_zeta: int = 80
    n_modes: int = 3  # geometry harmonics (Theta uses m=1,2,3 terms here)
    N_t: int = 19
    # chi = theta - n_helical*zeta. To match LCFS with (+19*zeta) phase, default n_helical=-19.
    n_helical: int = -19
    phi_total: float = 1.0
    lambda_reg: float = 1.0e-8
    tol: float = 1.0e-6
    max_nfev: int = 20
    output_dir: str = "veq_output"


class V3DEquilibriumSolver:
    """
    Variational 3D equilibrium solver.

    Unknown vector X (12 vars):
      [h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s11, s20, s30]

    Note on naming:
      user s_0,s_1,s_2  <->  this script s10,s20,s30 (m=1,2,3 sine harmonics).
      s11 is now solved as the 12th variational parameter.
    """

    def __init__(self, cfg: SolverConfig, case_cfg: Dict[str, Any]):
        self.cfg = cfg
        self.case_cfg = case_cfg
        self.profiles_cfg = case_cfg.get("profiles", {})
        self.lcfs_cfg = case_cfg.get("lcfs", {})
        self.solver_case_cfg = case_cfg.get("solver", {})
        self.jacobian_mode = str(self.solver_case_cfg.get("jacobian_mode", "jax_ad")).lower()
        self.physical_cfg = case_cfg.get("physical_constants", {})
        self.profile_constants = dict(self.profiles_cfg.get("constants", {}))
        self.lcfs_constants = dict(self.lcfs_cfg.get("constants", {}))
        self.c_light = float(self.physical_cfg.get("c_light", 1.0))
        if abs(self.c_light) < 1.0e-16:
            raise ValueError("physical_constants.c_light cannot be zero.")
        if self.jacobian_mode not in {"jax_ad", "scipy_fd"}:
            raise ValueError("solver.jacobian_mode must be one of: jax_ad, scipy_fd.")

        # Full-grid (includes axis rho=0 and boundary rho=1) for geometry.
        self.rho = np.linspace(0.0, 1.0, cfg.n_rho)
        # Half-grid (cell centers) for thermodynamic / magnetic quantities.
        self.rho_half = 0.5 * (self.rho[:-1] + self.rho[1:])
        if self.rho_half.size < 3:
            raise ValueError("Need at least 4 full-grid rho points to build stable half-grid.")
        self.theta = np.linspace(0.0, TWO_PI, cfg.n_theta, endpoint=False)
        self.zeta = np.linspace(0.0, TWO_PI, cfg.n_zeta, endpoint=False)
        self.drho = self.rho[1] - self.rho[0]
        self.dtheta = self.theta[1] - self.theta[0]
        self.dzeta = self.zeta[1] - self.zeta[0]

        self.rho3, self.theta3, self.zeta3 = np.meshgrid(
            self.rho, self.theta, self.zeta, indexing="ij"
        )
        self.theta2, self.zeta2 = np.meshgrid(self.theta, self.zeta, indexing="ij")

        # 1D physics profiles on full-grid and half-grid.
        self.P = self._evaluate_profile_expr(str(self.profiles_cfg["pressure_expr"]), self.rho)
        self.P_prime = dd_rho(self.P, self.rho)
        self.iota = self._evaluate_profile_expr(str(self.profiles_cfg["iota_expr"]), self.rho)
        self.Phi = self._evaluate_profile_expr(str(self.profiles_cfg["phi_expr"]), self.rho)
        self.Phi_prime = dd_rho(self.Phi, self.rho)

        self.P_half = self._evaluate_profile_expr(str(self.profiles_cfg["pressure_expr"]), self.rho_half)
        self.P_prime_half = dd_rho(self.P_half, self.rho_half)
        self.iota_half = self._evaluate_profile_expr(str(self.profiles_cfg["iota_expr"]), self.rho_half)
        self.Phi_half = self._evaluate_profile_expr(str(self.profiles_cfg["phi_expr"]), self.rho_half)
        self.Phi_prime_half = dd_rho(self.Phi_half, self.rho_half)
        # Baseline relation from rotational transform profile.
        self.Psi_prime_iota = self.iota * self.Phi_prime
        self.Psi_iota = self._integrate_profile(self.Psi_prime_iota, self.rho)
        self.Psi_prime_iota_half = self.iota_half * self.Phi_prime_half
        self.Psi_iota_half = self._integrate_profile(self.Psi_prime_iota_half, self.rho_half)
        psi_a_raw = self.profiles_cfg.get("psi_a", None)
        self.psi_a = float(self.Psi_iota[-1]) if psi_a_raw is None else float(psi_a_raw)

        # Initial guess for nonlinear unknowns from input config.
        self.initial_guess = self._load_initial_guess()

        # Lambda Fourier double-spectrum pairs (m,n).
        self.lambda_mode_pairs = self._build_lambda_mode_pairs()

        # either use fixed boundary constants or fit from LCFS expressions
        self.boundary = self._load_or_fit_boundary()

    def _evaluate_profile_expr(self, expr: str, rho_1d: np.ndarray) -> np.ndarray:
        arr = safe_eval_expr(
            expr,
            variables={
                "rho": rho_1d,
                "N_t": float(self.cfg.N_t),
                "n_helical": float(self.cfg.n_helical),
                "phi_total": float(self.cfg.phi_total),
            },
            constants=self.profile_constants,
        )
        if arr.ndim == 0:
            return np.full_like(rho_1d, float(arr), dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.shape != rho_1d.shape:
            try:
                arr = np.broadcast_to(arr, rho_1d.shape).astype(float)
            except ValueError as exc:
                raise ValueError(
                    f"Profile expression must evaluate to shape {rho_1d.shape}, got {arr.shape}."
                ) from exc
        return arr

    def _load_initial_guess(self) -> np.ndarray:
        raw = self.solver_case_cfg.get("initial_guess")
        if raw is None:
            return np.zeros(N_X_UNKNOWNS, dtype=float)

        x0 = np.asarray(raw, dtype=float)
        # Backward-compatible upgrade: 11 -> 12 by appending legacy solver.s11
        if x0.shape == (N_X_UNKNOWNS - 1,):
            s11_legacy = float(self.solver_case_cfg.get("s11", 0.0))
            x0 = np.concatenate([x0[:9], np.array([s11_legacy], dtype=float), x0[9:]])
        if x0.shape != (N_X_UNKNOWNS,):
            raise ValueError(
                f"solver.initial_guess must contain {N_X_UNKNOWNS} values (or legacy {N_X_UNKNOWNS - 1}), got shape {x0.shape}."
            )
        return x0

    def _build_lambda_mode_pairs(self) -> np.ndarray:
        """
        Build (m,n) mode pairs for Lambda expansion.
        """
        spec = self.case_cfg.get("lambda_fourier", {})
        pairs_cfg = spec.get("pairs", [])
        exclude_zero_mode = bool(spec.get("exclude_zero_mode", True))

        pairs: list[tuple[int, int]] = []
        if pairs_cfg:
            for item in pairs_cfg:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("case.lambda_fourier.pairs items must be [m, n].")
                m = int(item[0])
                n = int(item[1])
                pairs.append((m, n))
        else:
            m_min = int(spec.get("m_min", 0))
            m_max = int(spec.get("m_max", 3))
            n_min = int(spec.get("n_min", -3))
            n_max = int(spec.get("n_max", 3))
            if m_min > m_max or n_min > n_max:
                raise ValueError(
                    "Invalid lambda_fourier range: require m_min<=m_max and n_min<=n_max."
                )
            for m in range(m_min, m_max + 1):
                for n in range(n_min, n_max + 1):
                    pairs.append((m, n))

        # remove duplicates while preserving order
        uniq: list[tuple[int, int]] = []
        seen = set()
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                uniq.append(pair)

        if exclude_zero_mode:
            uniq = [p for p in uniq if not (p[0] == 0 and p[1] == 0)]
        if not uniq:
            raise ValueError("Lambda mode set is empty after filtering.")

        return np.asarray(uniq, dtype=np.int64)

    def _load_or_fit_boundary(self) -> BoundaryParams:
        fixed = self.solver_case_cfg.get("boundary_constants")
        if fixed is not None:
            missing = [k for k in BOUNDARY_KEYS if k not in fixed]
            if missing:
                raise ValueError(
                    "solver.boundary_constants missing keys: " + ", ".join(missing)
                )
            return BoundaryParams(
                R0=float(fixed["R0"]),
                Z0=float(fixed["Z0"]),
                a=float(fixed["a"]),
                kappa_a=float(fixed["kappa_a"]),
                c0a=float(fixed["c0a"]),
                c1a=float(fixed["c1a"]),
                s1a=float(fixed["s1a"]),
                s2a=float(fixed["s2a"]),
                s3a=float(fixed["s3a"]),
            )
        return self._fit_lcfs_boundary()

    @staticmethod
    def _integrate_profile(yprime: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(yprime)
        seg = 0.5 * (yprime[1:] + yprime[:-1]) * np.diff(x)
        out[1:] = np.cumsum(seg)
        return out

    def _compute_psi_from_nu(self, nu_1d: np.ndarray, rho_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce theoretical constraint:
          psi(rho) = psi_a * rho^2 * [1 + nu(rho)]
        """
        psi = self.psi_a * (rho_1d * rho_1d) * (1.0 + nu_1d)
        psi_prime = dd_rho(psi, rho_1d)
        return psi, psi_prime

    def _target_lcfs(self, theta: np.ndarray, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate configurable LCFS expressions."""
        R = safe_eval_expr(
            str(self.lcfs_cfg["R_expr"]),
            variables={
                "theta": theta,
                "zeta": zeta,
                "N_t": float(self.cfg.N_t),
                "n_helical": float(self.cfg.n_helical),
            },
            constants=self.lcfs_constants,
        )
        Z = safe_eval_expr(
            str(self.lcfs_cfg["Z_expr"]),
            variables={
                "theta": theta,
                "zeta": zeta,
                "N_t": float(self.cfg.N_t),
                "n_helical": float(self.cfg.n_helical),
            },
            constants=self.lcfs_constants,
        )
        if R.shape != theta.shape:
            R = np.broadcast_to(R, theta.shape).astype(float)
        if Z.shape != theta.shape:
            Z = np.broadcast_to(Z, theta.shape).astype(float)
        return R, Z

    def _fit_lcfs_boundary(self) -> BoundaryParams:
        """
        Nonlinear least-squares fit for:
          [R0,Z0,a,kappa_a,c0a,c1a,s1a,s2a,s3a]
        on rho=1.
        """
        Rt, Zt = self._target_lcfs(self.theta2, self.zeta2)

        def model(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            R0, Z0, a, kappa_a, c0a, c1a, s1a, s2a, s3a = p
            chi1 = self.theta2 - self.cfg.n_helical * self.zeta2
            chi2 = 2.0 * self.theta2 - self.cfg.n_helical * self.zeta2
            chi3 = 3.0 * self.theta2 - self.cfg.n_helical * self.zeta2
            Theta_b = (
                chi1
                + c0a
                + c1a * np.cos(chi1)
                + s1a * np.sin(chi1)
                + s2a * np.sin(chi2)
                + s3a * np.sin(chi3)
            )
            Rm = R0 + a * np.cos(Theta_b)
            Zm = Z0 - kappa_a * a * np.sin(chi1)
            return Rm, Zm

        def residual(p: np.ndarray) -> np.ndarray:
            Rm, Zm = model(p)
            return np.concatenate([(Rm - Rt).ravel(), (Zm - Zt).ravel()])

        p0 = np.asarray(
            self.solver_case_cfg.get(
                "boundary_seed",
                [10.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            dtype=float,
        )
        if p0.shape != (9,):
            raise ValueError(f"solver.boundary_seed must contain 9 values, got shape {p0.shape}.")
        fit = least_squares(residual, p0, method="trf", max_nfev=300, xtol=1e-12, ftol=1e-12)
        p = fit.x
        return BoundaryParams(
            R0=float(p[0]),
            Z0=float(p[1]),
            a=float(p[2]),
            kappa_a=float(p[3]),
            c0a=float(p[4]),
            c1a=float(p[5]),
            s1a=float(p[6]),
            s2a=float(p[7]),
            s3a=float(p[8]),
        )

    def _radial_profiles(self, X: np.ndarray, rho_1d: np.ndarray) -> Dict[str, np.ndarray]:
        h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s11, s20, s30 = X
        r = rho_1d
        r2 = r * r

        # h(r)
        h = (h0 + h1 * (2.0 * r2 - 1.0)) * (1.0 - r) ** 2
        # derivative from product rule
        u = (h0 + h1 * (2.0 * r2 - 1.0))
        u_p = 4.0 * h1 * r
        v = (1.0 - r) ** 2
        v_p = -2.0 * (1.0 - r)
        h_p = u_p * v + u * v_p

        # kappa(r)
        kappa = self.boundary.kappa_a + (1.0 - r2) * (kappa0 + kappa1 * (2.0 * r2 - 1.0))
        A = (kappa0 + kappa1 * (2.0 * r2 - 1.0))
        A_p = 4.0 * kappa1 * r
        kappa_p = (-2.0 * r) * A + (1.0 - r2) * A_p

        # nu(r)
        nu = (1.0 - r2) * (nu0 + nu1 * (2.0 * r2 - 1.0))
        B = (nu0 + nu1 * (2.0 * r2 - 1.0))
        B_p = 4.0 * nu1 * r
        nu_p = (-2.0 * r) * B + (1.0 - r2) * B_p

        # c0, c1, s1, s2, s3
        c0 = self.boundary.c0a + c00 * (1.0 - r2)
        c0_p = -2.0 * c00 * r

        c1 = r * (self.boundary.c1a + c10 * (1.0 - r2))
        c1_p = self.boundary.c1a + c10 * (1.0 - 3.0 * r2)

        # Full profile:
        # s1(rho) = rho * [s1a + (1-rho^2) * (s10 + s11*(2*rho^2-1))]
        A = (1.0 - r2)
        B = (s10 + s11 * (2.0 * r2 - 1.0))
        C = self.boundary.s1a + A * B
        C_p = (-2.0 * r) * B + A * (4.0 * s11 * r)
        s1 = r * C
        s1_p = C + r * C_p

        s2 = r2 * (self.boundary.s2a + s20 * (1.0 - r2))
        s2_p = 2.0 * r * (self.boundary.s2a + s20 * (1.0 - 2.0 * r2))

        s3 = r2 * (self.boundary.s3a + s30 * (1.0 - r2))
        s3_p = 2.0 * r * (self.boundary.s3a + s30 * (1.0 - 2.0 * r2))

        return {
            "h": h,
            "h_p": h_p,
            "kappa": kappa,
            "kappa_p": kappa_p,
            "nu": nu,
            "nu_p": nu_p,
            "c0": c0,
            "c0_p": c0_p,
            "c1": c1,
            "c1_p": c1_p,
            "s1": s1,
            "s1_p": s1_p,
            "s2": s2,
            "s2_p": s2_p,
            "s3": s3,
            "s3_p": s3_p,
        }

    def _geometry_metrics(self, X: np.ndarray, rho_1d: np.ndarray) -> Dict[str, np.ndarray]:
        prof = self._radial_profiles(X, rho_1d)
        r = rho_1d[:, None, None]
        th = self.theta2[None, :, :]
        ze = self.zeta2[None, :, :]
        a = self.boundary.a
        n = float(self.cfg.n_helical)

        # broadcast radial profiles to 3D
        h = prof["h"][:, None, None]
        h_p = prof["h_p"][:, None, None]
        kappa = prof["kappa"][:, None, None]
        kappa_p = prof["kappa_p"][:, None, None]
        nu = prof["nu"][:, None, None]
        nu_p = prof["nu_p"][:, None, None]

        c0 = prof["c0"][:, None, None]
        c0_p = prof["c0_p"][:, None, None]
        c1 = prof["c1"][:, None, None]
        c1_p = prof["c1_p"][:, None, None]
        s1 = prof["s1"][:, None, None]
        s1_p = prof["s1_p"][:, None, None]
        s2 = prof["s2"][:, None, None]
        s2_p = prof["s2_p"][:, None, None]
        s3 = prof["s3"][:, None, None]
        s3_p = prof["s3_p"][:, None, None]

        chi1 = th - n * ze
        chi2 = 2.0 * th - n * ze
        chi3 = 3.0 * th - n * ze

        Theta = (
            chi1
            + c0
            + c1 * np.cos(chi1)
            + s1 * np.sin(chi1)
            + s2 * np.sin(chi2)
            + s3 * np.sin(chi3)
        )

        Theta_theta = (
            1.0
            - c1 * np.sin(chi1)
            + s1 * np.cos(chi1)
            + 2.0 * s2 * np.cos(chi2)
            + 3.0 * s3 * np.cos(chi3)
        )
        Theta_zeta = (
            -n
            + n * c1 * np.sin(chi1)
            - n * s1 * np.cos(chi1)
            - n * s2 * np.cos(chi2)
            - n * s3 * np.cos(chi3)
        )
        Theta_rho = (
            c0_p + c1_p * np.cos(chi1) + s1_p * np.sin(chi1) + s2_p * np.sin(chi2) + s3_p * np.sin(chi3)
        )

        R = self.boundary.R0 + h + r * a * np.cos(Theta)
        Z = self.boundary.Z0 + nu - kappa * r * a * np.sin(chi1)

        R_rho = h_p + a * np.cos(Theta) - r * a * np.sin(Theta) * Theta_rho
        R_theta = -r * a * np.sin(Theta) * Theta_theta
        R_zeta = -r * a * np.sin(Theta) * Theta_zeta

        Z_rho = nu_p - (kappa_p * r * a + kappa * a) * np.sin(chi1)
        Z_theta = -kappa * r * a * np.cos(chi1)
        Z_zeta = n * kappa * r * a * np.cos(chi1)

        jac = R_rho * Z_theta - R_theta * Z_rho
        sqrtg = (R / float(self.cfg.N_t)) * jac

        # metric tensor components
        g_rr = R_rho * R_rho + Z_rho * Z_rho
        g_tt = R_theta * R_theta + Z_theta * Z_theta
        g_zz = R_zeta * R_zeta + Z_zeta * Z_zeta + (R * R) / (float(self.cfg.N_t) ** 2)
        g_rt = R_rho * R_theta + Z_rho * Z_theta
        g_rz = R_rho * R_zeta + Z_rho * Z_zeta
        g_tz = R_theta * R_zeta + Z_theta * Z_zeta

        return {
            "rho_1d": rho_1d,
            "rho3": np.broadcast_to(r, R.shape),
            "R": R,
            "Z": Z,
            "Theta": Theta,
            "chi1": chi1,
            "chi2": chi2,
            "chi3": chi3,
            "R_rho": R_rho,
            "R_theta": R_theta,
            "R_zeta": R_zeta,
            "Z_rho": Z_rho,
            "Z_theta": Z_theta,
            "Z_zeta": Z_zeta,
            "jac": jac,
            "sqrtg": sqrtg,
            "g_rr": g_rr,
            "g_tt": g_tt,
            "g_zz": g_zz,
            "g_rt": g_rt,
            "g_rz": g_rz,
            "g_tz": g_tz,
            "nu": nu,
            "nu_1d": prof["nu"],
            "nu_p_1d": prof["nu_p"],
        }

    def _solve_lambda_and_field(
        self,
        geom: Dict[str, np.ndarray],
        psi_prime_1d: np.ndarray,
        phi_prime_1d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Lambda coefficients on each rho-surface from weighted J^rho=0:
          ∫∫ sqrt(g) J^rho sin(m*theta - n*zeta) dtheta dzeta = 0
        and reconstruct (Lambda_theta, Lambda_zeta), (B^theta, B^zeta).
        """
        n_rho = int(geom["rho_1d"].size)

        # 2D Fourier basis over configurable (m,n) pairs.
        pairs = self.lambda_mode_pairs
        m_arr = pairs[:, 0].astype(float)  # (K,)
        n_arr = pairs[:, 1].astype(float)  # (K,)
        K = pairs.shape[0]
        arg = m_arr[:, None, None] * self.theta2[None, :, :] - n_arr[:, None, None] * self.zeta2[None, :, :]
        sin_basis = np.sin(arg)  # (K, Ntheta, Nzeta)
        cos_basis = np.cos(arg)  # (K, Ntheta, Nzeta)
        # Derivative basis d/dtheta and d/dzeta of sin phase.
        dtheta_basis = m_arr[:, None, None] * cos_basis
        dzeta_basis = -n_arr[:, None, None] * cos_basis

        sqrtg = geom["sqrtg"]
        g_tt = geom["g_tt"]
        g_tz = geom["g_tz"]
        g_zz = geom["g_zz"]

        lam_coeff = np.zeros((n_rho, K), dtype=float)
        surf_factor = self.dtheta * self.dzeta

        def jrho_2d(i: int, lam_theta: np.ndarray, lam_zeta: np.ndarray) -> np.ndarray:
            sqrtg2 = sqrtg[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                B_theta = (psi_prime_1d[i] - lam_zeta) / (TWO_PI * sqrtg2)
                B_zeta = (phi_prime_1d[i] + lam_theta) / (TWO_PI * sqrtg2)

            t1 = dd_theta_2d(g_tz[i] * B_theta + g_zz[i] * B_zeta, self.dtheta)
            t2 = dd_zeta_2d(g_tt[i] * B_theta + g_tz[i] * B_zeta, self.dzeta)
            # For J^rho=0 closure we divide by c (equivalent equation), improving conditioning
            # when c_light uses dimensional SI values.
            with np.errstate(divide="ignore", invalid="ignore"):
                jrho = (t1 - t2) / (4.0 * np.pi * sqrtg2)
            if not np.all(np.isfinite(jrho)):
                raise FloatingPointError("Non-finite J^rho encountered on half-grid.")
            return jrho

        # solve each rho surface
        I = np.eye(K)
        for i in range(n_rho):
            sqrtg_i = sqrtg[i]

            # Base J^rho with Lambda=0
            J0 = jrho_2d(
                i,
                lam_theta=np.zeros_like(sqrtg_i),
                lam_zeta=np.zeros_like(sqrtg_i),
            )

            # Right-hand side: -<sqrtg * J0, sin_q>
            ws = sqrtg_i[None, :, :] * sin_basis
            b = -surf_factor * np.einsum("qij,ij->q", ws, J0, optimize=True)

            # Linear response dJ/dlambda_p for all p at once (exact due linearity).
            with np.errstate(divide="ignore", invalid="ignore"):
                dB_theta_basis = -dzeta_basis / (TWO_PI * sqrtg_i[None, :, :])
                dB_zeta_basis = dtheta_basis / (TWO_PI * sqrtg_i[None, :, :])
            t1_basis = dd_theta(
                g_tz[i][None, :, :] * dB_theta_basis + g_zz[i][None, :, :] * dB_zeta_basis,
                self.dtheta,
            )
            t2_basis = dd_zeta(
                g_tt[i][None, :, :] * dB_theta_basis + g_tz[i][None, :, :] * dB_zeta_basis,
                self.dzeta,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                dJ_basis = (t1_basis - t2_basis) / (4.0 * np.pi * sqrtg_i[None, :, :])  # (K, Nt, Nz)
            if not np.all(np.isfinite(dJ_basis)):
                raise FloatingPointError("Non-finite dJ basis encountered on half-grid.")

            # A[q,p] = <sqrtg * dJ_p, sin_q>
            A = surf_factor * np.einsum("qij,pij->qp", ws, dJ_basis, optimize=True)

            A_reg = A + self.cfg.lambda_reg * I
            try:
                lam_coeff[i] = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                lam_coeff[i] = np.linalg.lstsq(A_reg, b, rcond=None)[0]

        # reconstruct Lambda derivatives and B field in 3D
        lam_theta_3 = np.tensordot(lam_coeff, dtheta_basis, axes=(1, 0))
        lam_zeta_3 = np.tensordot(lam_coeff, dzeta_basis, axes=(1, 0))

        with np.errstate(divide="ignore", invalid="ignore"):
            B_theta = (psi_prime_1d[:, None, None] - lam_zeta_3) / (TWO_PI * sqrtg)
            B_zeta = (phi_prime_1d[:, None, None] + lam_theta_3) / (TWO_PI * sqrtg)
        if not (np.all(np.isfinite(B_theta)) and np.all(np.isfinite(B_zeta))):
            raise FloatingPointError("Non-finite B field encountered on half-grid.")

        return lam_coeff, lam_theta_3, B_theta, B_zeta

    def _force_operators(
        self,
        geom: Dict[str, np.ndarray],
        B_theta: np.ndarray,
        B_zeta: np.ndarray,
        psi_prime_1d: np.ndarray,
        phi_prime_1d: np.ndarray,
        p_prime_1d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sqrtg = geom["sqrtg"]
        g_tt = geom["g_tt"]
        g_tz = geom["g_tz"]
        g_zz = geom["g_zz"]
        g_rt = geom["g_rt"]
        g_rz = geom["g_rz"]

        # J^rho
        j1 = dd_theta(g_tz * B_theta + g_zz * B_zeta, self.dtheta)
        j2 = dd_zeta(g_tt * B_theta + g_tz * B_zeta, self.dzeta)
        with np.errstate(divide="ignore", invalid="ignore"):
            J_rho = self.c_light * (j1 - j2) / (4.0 * np.pi * sqrtg)
        if not np.all(np.isfinite(J_rho)):
            raise FloatingPointError("Non-finite J^rho in force operator.")

        # G_rho
        F1 = g_tt * B_theta + g_tz * B_zeta
        F2 = g_rt * B_theta + g_rz * B_zeta
        F3 = g_tz * B_theta + g_zz * B_zeta
        rho_axis = geom["rho_1d"]
        t1 = dd_rho(F1, rho_axis) - dd_theta(F2, self.dtheta)
        t2 = dd_rho(F3, rho_axis) - dd_zeta(F2, self.dzeta)
        G_rho = (B_theta * t1 + B_zeta * t2) / (4.0 * np.pi) + p_prime_1d[:, None, None]

        # inverse-gradient components from local Jacobian inversion
        D = geom["jac"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_R = geom["Z_theta"] / D
            rho_Z = -geom["R_theta"] / D
            theta_R = -geom["Z_rho"] / D
            theta_Z = geom["R_rho"] / D
        zeta_R = np.zeros_like(theta_R)
        zeta_Z = np.zeros_like(theta_Z)

        # General form keeps the 1/c factor explicit.
        G_R = rho_R * G_rho + (J_rho / (TWO_PI * self.c_light)) * (
            theta_R * phi_prime_1d[:, None, None] - zeta_R * psi_prime_1d[:, None, None]
        )
        G_Z = rho_Z * G_rho + (J_rho / (TWO_PI * self.c_light)) * (
            theta_Z * phi_prime_1d[:, None, None] - zeta_Z * psi_prime_1d[:, None, None]
        )
        if not (np.all(np.isfinite(G_R)) and np.all(np.isfinite(G_Z))):
            raise FloatingPointError("Non-finite force projection encountered.")
        return J_rho, G_R, G_Z

    def evaluate_residuals(self, X: np.ndarray) -> np.ndarray:
        large_res = np.full(N_X_UNKNOWNS, 1.0e20, dtype=float)
        try:
            # Staggered-grid evaluation for residual assembly:
            # magnetic closure and force balance are evaluated on half grid (rho>0).
            geom_h = self._geometry_metrics(X, self.rho_half)
            _, psi_prime_h = self._compute_psi_from_nu(geom_h["nu_1d"], self.rho_half)
            _, _, B_theta_h, B_zeta_h = self._solve_lambda_and_field(
                geom_h, psi_prime_h, self.Phi_prime_half
            )
            _, G_R_h, G_Z_h = self._force_operators(
                geom_h, B_theta_h, B_zeta_h, psi_prime_h, self.Phi_prime_half, self.P_prime_half
            )

            # aliases on half-grid
            r = geom_h["rho3"]
            Theta = geom_h["Theta"]
            chi1 = geom_h["chi1"]
            chi2 = geom_h["chi2"]
            chi3 = geom_h["chi3"]
            a = self.boundary.a
            sqrtg_w = np.abs(geom_h["sqrtg"])
            vol = integrate_volume(sqrtg_w, self.rho_half, self.theta, self.zeta)
            if (not np.isfinite(vol)) or vol <= 0.0:
                return large_res

            # 12 residual equations (s11 included)
            R_h0 = (
                integrate_volume(
                    sqrtg_w * (G_R_h * (1.0 - r) ** 2),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_h1 = (
                integrate_volume(
                    sqrtg_w * (G_R_h * (2.0 * r * r - 1.0) * (1.0 - r) ** 2),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_c00 = (
                integrate_volume(
                    sqrtg_w * (-G_R_h * r * a * np.sin(Theta) * (1.0 - r * r)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_c10 = (
                integrate_volume(
                    sqrtg_w * (-G_R_h * r * a * np.sin(Theta) * r * (1.0 - r * r) * np.cos(chi1)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_s10 = (
                integrate_volume(
                    sqrtg_w * (-G_R_h * r * a * np.sin(Theta) * r * (1.0 - r * r) * np.sin(chi1)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_s11 = (
                integrate_volume(
                    sqrtg_w
                    * (
                        -G_R_h
                        * r
                        * a
                        * np.sin(Theta)
                        * r
                        * (1.0 - r * r)
                        * (2.0 * r * r - 1.0)
                        * np.sin(chi1)
                    ),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_s20 = (
                integrate_volume(
                    sqrtg_w * (-G_R_h * r * a * np.sin(Theta) * r * r * np.sin(chi2)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_s30 = (
                integrate_volume(
                    sqrtg_w * (-G_R_h * r * a * np.sin(Theta) * r * r * (1.0 - r * r) * np.sin(chi3)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_nu0 = (
                integrate_volume(
                    sqrtg_w * (G_Z_h * (1.0 - r * r)),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_nu1 = (
                integrate_volume(
                    sqrtg_w * (G_Z_h * (2.0 * r * r - 1.0) * (1.0 - r) ** 2),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            kappa_factor = (1.0 - r * r) * r * a * np.sin(chi1)
            R_kappa0 = (
                integrate_volume(
                    sqrtg_w * (-G_Z_h * kappa_factor),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )
            R_kappa1 = (
                integrate_volume(
                    sqrtg_w * (-G_Z_h * (2.0 * r * r - 1.0) * kappa_factor),
                    self.rho_half,
                    self.theta,
                    self.zeta,
                )
                / vol
            )

            residuals = np.array(
                [
                    R_h0,
                    R_h1,
                    R_kappa0,
                    R_kappa1,
                    R_nu0,
                    R_nu1,
                    R_c00,
                    R_c10,
                    R_s10,
                    R_s11,
                    R_s20,
                    R_s30,
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(residuals)):
                return large_res
            return residuals
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return large_res

    def _build_jax_autodiff_jacobian(self):
        """
        Build automatic-differentiation Jacobian callback using JAX.
        Residual model mirrors the staggered-grid equations and returns a dense dR/dX matrix.
        """
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        rho = jnp.asarray(self.rho_half, dtype=jnp.float64)
        theta = jnp.asarray(self.theta, dtype=jnp.float64)
        zeta = jnp.asarray(self.zeta, dtype=jnp.float64)
        theta2 = jnp.asarray(self.theta2, dtype=jnp.float64)
        zeta2 = jnp.asarray(self.zeta2, dtype=jnp.float64)

        dtheta = float(self.dtheta)
        dzeta = float(self.dzeta)
        surf_factor = float(dtheta * dzeta)
        n_helical = float(self.cfg.n_helical)
        Nt = float(self.cfg.N_t)
        a = float(self.boundary.a)
        c_light = float(self.c_light)
        psi_a = float(self.psi_a)
        lambda_reg = float(self.cfg.lambda_reg)

        phi_prime_h = jnp.asarray(self.Phi_prime_half, dtype=jnp.float64)
        p_prime_h = jnp.asarray(self.P_prime_half, dtype=jnp.float64)

        # Boundary constants
        R0 = float(self.boundary.R0)
        Z0 = float(self.boundary.Z0)
        kappa_a = float(self.boundary.kappa_a)
        c0a = float(self.boundary.c0a)
        c1a = float(self.boundary.c1a)
        s1a = float(self.boundary.s1a)
        s2a = float(self.boundary.s2a)
        s3a = float(self.boundary.s3a)

        # Lambda (m,n) spectrum basis
        pairs = jnp.asarray(self.lambda_mode_pairs, dtype=jnp.float64)
        m_arr = pairs[:, 0]
        n_arr = pairs[:, 1]
        K = int(self.lambda_mode_pairs.shape[0])
        arg = m_arr[:, None, None] * theta2[None, :, :] - n_arr[:, None, None] * zeta2[None, :, :]
        sin_basis = jnp.sin(arg)
        cos_basis = jnp.cos(arg)
        dtheta_basis = m_arr[:, None, None] * cos_basis
        dzeta_basis = -n_arr[:, None, None] * cos_basis
        eyeK = jnp.eye(K, dtype=jnp.float64)

        # Precompute fixed trigonometric phases for geometric harmonics (n fixed by equilibrium family).
        th3 = theta2[None, :, :]
        ze3 = zeta2[None, :, :]
        r3 = rho[:, None, None]
        chi1 = th3 - n_helical * ze3
        chi2 = 2.0 * th3 - n_helical * ze3
        chi3 = 3.0 * th3 - n_helical * ze3

        rho_size = int(self.rho_half.size)
        if rho_size < 2:
            raise ValueError("Need at least two half-grid points for radial derivatives.")
        h_r = float(self.rho_half[1] - self.rho_half[0])

        def dd_per(arr, spacing, axis):
            nloc = arr.shape[axis]
            k = 2.0 * jnp.pi * jnp.fft.fftfreq(nloc, d=spacing)
            shape = [1] * arr.ndim
            shape[axis] = nloc
            ik = (1j * k).reshape(shape)
            a_hat = jnp.fft.fft(arr, axis=axis)
            return jnp.real(jnp.fft.ifft(ik * a_hat, axis=axis))

        def dd_theta_j(arr):
            return dd_per(arr, dtheta, axis=1)

        def dd_zeta_j(arr):
            return dd_per(arr, dzeta, axis=2)

        def dd_theta2_j(arr):
            return dd_per(arr, dtheta, axis=0)

        def dd_zeta2_j(arr):
            return dd_per(arr, dzeta, axis=1)

        def dd_rho_j(arr):
            # 4th-order finite difference on uniform half-grid.
            n = arr.shape[0]
            if n < 5:
                d0 = (-3.0 * arr[0] + 4.0 * arr[1] - arr[2]) / (2.0 * h_r)
                d1 = (arr[2] - arr[0]) / (2.0 * h_r)
                dmid = (arr[2:] - arr[:-2]) / (2.0 * h_r)
                dn2 = (arr[-1] - arr[-3]) / (2.0 * h_r)
                dn1 = (3.0 * arr[-1] - 4.0 * arr[-2] + arr[-3]) / (2.0 * h_r)
                return jnp.concatenate([d0[None, ...], d1[None, ...], dmid[1:-1], dn2[None, ...], dn1[None, ...]], axis=0)

            d0 = (-25.0 * arr[0] + 48.0 * arr[1] - 36.0 * arr[2] + 16.0 * arr[3] - 3.0 * arr[4]) / (12.0 * h_r)
            d1 = (-3.0 * arr[0] - 10.0 * arr[1] + 18.0 * arr[2] - 6.0 * arr[3] + arr[4]) / (12.0 * h_r)
            dmid = (-arr[4:] + 8.0 * arr[3:-1] - 8.0 * arr[1:-3] + arr[:-4]) / (12.0 * h_r)
            dn2 = (3.0 * arr[-1] + 10.0 * arr[-2] - 18.0 * arr[-3] + 6.0 * arr[-4] - arr[-5]) / (12.0 * h_r)
            dn1 = (25.0 * arr[-1] - 48.0 * arr[-2] + 36.0 * arr[-3] - 16.0 * arr[-4] + 3.0 * arr[-5]) / (12.0 * h_r)
            return jnp.concatenate(
                [d0[None, ...], d1[None, ...], dmid, dn2[None, ...], dn1[None, ...]],
                axis=0,
            )

        def integrate_volume_j(f3):
            return jnp.trapezoid(
                jnp.trapezoid(jnp.trapezoid(f3, zeta, axis=2), theta, axis=1),
                rho,
                axis=0,
            )

        def radial_profiles_j(x):
            h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s11, s20, s30 = x
            r = rho
            r2 = r * r

            h = (h0 + h1 * (2.0 * r2 - 1.0)) * (1.0 - r) ** 2
            u = (h0 + h1 * (2.0 * r2 - 1.0))
            u_p = 4.0 * h1 * r
            v = (1.0 - r) ** 2
            v_p = -2.0 * (1.0 - r)
            h_p = u_p * v + u * v_p

            kappa = kappa_a + (1.0 - r2) * (kappa0 + kappa1 * (2.0 * r2 - 1.0))
            A = (kappa0 + kappa1 * (2.0 * r2 - 1.0))
            A_p = 4.0 * kappa1 * r
            kappa_p = (-2.0 * r) * A + (1.0 - r2) * A_p

            nu = (1.0 - r2) * (nu0 + nu1 * (2.0 * r2 - 1.0))
            B = (nu0 + nu1 * (2.0 * r2 - 1.0))
            B_p = 4.0 * nu1 * r
            nu_p = (-2.0 * r) * B + (1.0 - r2) * B_p

            c0 = c0a + c00 * (1.0 - r2)
            c0_p = -2.0 * c00 * r

            c1 = r * (c1a + c10 * (1.0 - r2))
            c1_p = c1a + c10 * (1.0 - 3.0 * r2)

            A = (1.0 - r2)
            B = (s10 + s11 * (2.0 * r2 - 1.0))
            C = s1a + A * B
            C_p = (-2.0 * r) * B + A * (4.0 * s11 * r)
            s1 = r * C
            s1_p = C + r * C_p

            s2 = r2 * (s2a + s20 * (1.0 - r2))
            s2_p = 2.0 * r * (s2a + s20 * (1.0 - 2.0 * r2))

            s3 = r2 * (s3a + s30 * (1.0 - r2))
            s3_p = 2.0 * r * (s3a + s30 * (1.0 - 2.0 * r2))

            return h, h_p, kappa, kappa_p, nu, nu_p, c0, c0_p, c1, c1_p, s1, s1_p, s2, s2_p, s3, s3_p

        def geometry_j(x):
            h, h_p, kappa, kappa_p, nu, nu_p, c0, c0_p, c1, c1_p, s1, s1_p, s2, s2_p, s3, s3_p = radial_profiles_j(x)

            h3 = h[:, None, None]
            h_p3 = h_p[:, None, None]
            kappa3 = kappa[:, None, None]
            kappa_p3 = kappa_p[:, None, None]
            nu3 = nu[:, None, None]
            nu_p3 = nu_p[:, None, None]
            c0_3 = c0[:, None, None]
            c0_p3 = c0_p[:, None, None]
            c1_3 = c1[:, None, None]
            c1_p3 = c1_p[:, None, None]
            s1_3 = s1[:, None, None]
            s1_p3 = s1_p[:, None, None]
            s2_3 = s2[:, None, None]
            s2_p3 = s2_p[:, None, None]
            s3_3 = s3[:, None, None]
            s3_p3 = s3_p[:, None, None]

            Theta = chi1 + c0_3 + c1_3 * jnp.cos(chi1) + s1_3 * jnp.sin(chi1) + s2_3 * jnp.sin(chi2) + s3_3 * jnp.sin(chi3)
            Theta_theta = 1.0 - c1_3 * jnp.sin(chi1) + s1_3 * jnp.cos(chi1) + 2.0 * s2_3 * jnp.cos(chi2) + 3.0 * s3_3 * jnp.cos(chi3)
            Theta_zeta = -n_helical + n_helical * c1_3 * jnp.sin(chi1) - n_helical * s1_3 * jnp.cos(chi1) - n_helical * s2_3 * jnp.cos(chi2) - n_helical * s3_3 * jnp.cos(chi3)
            Theta_rho = c0_p3 + c1_p3 * jnp.cos(chi1) + s1_p3 * jnp.sin(chi1) + s2_p3 * jnp.sin(chi2) + s3_p3 * jnp.sin(chi3)

            R = R0 + h3 + r3 * a * jnp.cos(Theta)
            Z = Z0 + nu3 - kappa3 * r3 * a * jnp.sin(chi1)

            R_rho = h_p3 + a * jnp.cos(Theta) - r3 * a * jnp.sin(Theta) * Theta_rho
            R_theta = -r3 * a * jnp.sin(Theta) * Theta_theta
            R_zeta = -r3 * a * jnp.sin(Theta) * Theta_zeta

            Z_rho = nu_p3 - (kappa_p3 * r3 * a + kappa3 * a) * jnp.sin(chi1)
            Z_theta = -kappa3 * r3 * a * jnp.cos(chi1)
            Z_zeta = n_helical * kappa3 * r3 * a * jnp.cos(chi1)

            jac = R_rho * Z_theta - R_theta * Z_rho
            sqrtg = (R / Nt) * jac

            g_tt = R_theta * R_theta + Z_theta * Z_theta
            g_zz = R_zeta * R_zeta + Z_zeta * Z_zeta + (R * R) / (Nt * Nt)
            g_rt = R_rho * R_theta + Z_rho * Z_theta
            g_rz = R_rho * R_zeta + Z_rho * Z_zeta
            g_tz = R_theta * R_zeta + Z_theta * Z_zeta

            return {
                "rho3": r3 + jnp.zeros_like(R),
                "R": R,
                "Z": Z,
                "Theta": Theta,
                "chi1": chi1,
                "chi2": chi2,
                "chi3": chi3,
                "R_rho": R_rho,
                "R_theta": R_theta,
                "Z_rho": Z_rho,
                "Z_theta": Z_theta,
                "jac": jac,
                "sqrtg": sqrtg,
                "g_tt": g_tt,
                "g_zz": g_zz,
                "g_rt": g_rt,
                "g_rz": g_rz,
                "g_tz": g_tz,
                "nu_1d": nu,
            }

        def compute_psi_j(nu_1d):
            psi = psi_a * (rho * rho) * (1.0 + nu_1d)
            psi_prime = dd_rho_j(psi)
            return psi, psi_prime

        def solve_lambda_and_field_j(geom, psi_prime_1d):
            sqrtg = geom["sqrtg"]
            g_tt = geom["g_tt"]
            g_tz = geom["g_tz"]
            g_zz = geom["g_zz"]
            n_rho = int(sqrtg.shape[0])
            lam_all = []

            for i in range(n_rho):
                sqrtg_i = sqrtg[i]

                def jrho_2d(lam_theta, lam_zeta):
                    B_theta = (psi_prime_1d[i] - lam_zeta) / (TWO_PI * sqrtg_i)
                    B_zeta = (phi_prime_h[i] + lam_theta) / (TWO_PI * sqrtg_i)
                    t1 = dd_theta2_j(g_tz[i] * B_theta + g_zz[i] * B_zeta)
                    t2 = dd_zeta2_j(g_tt[i] * B_theta + g_tz[i] * B_zeta)
                    return (t1 - t2) / (4.0 * jnp.pi * sqrtg_i)

                J0 = jrho_2d(jnp.zeros_like(sqrtg_i), jnp.zeros_like(sqrtg_i))
                ws = sqrtg_i[None, :, :] * sin_basis
                b = -surf_factor * jnp.einsum("qij,ij->q", ws, J0, optimize=True)

                dB_theta_basis = -dzeta_basis / (TWO_PI * sqrtg_i[None, :, :])
                dB_zeta_basis = dtheta_basis / (TWO_PI * sqrtg_i[None, :, :])
                t1_basis = dd_theta_j(g_tz[i][None, :, :] * dB_theta_basis + g_zz[i][None, :, :] * dB_zeta_basis)
                t2_basis = dd_zeta_j(g_tt[i][None, :, :] * dB_theta_basis + g_tz[i][None, :, :] * dB_zeta_basis)
                dJ_basis = (t1_basis - t2_basis) / (4.0 * jnp.pi * sqrtg_i[None, :, :])

                A = surf_factor * jnp.einsum("qij,pij->qp", ws, dJ_basis, optimize=True)
                lam_i = jnp.linalg.solve(A + lambda_reg * eyeK, b)
                lam_all.append(lam_i)

            lam_coeff = jnp.stack(lam_all, axis=0)
            lam_theta_3 = jnp.tensordot(lam_coeff, dtheta_basis, axes=(1, 0))
            lam_zeta_3 = jnp.tensordot(lam_coeff, dzeta_basis, axes=(1, 0))
            B_theta = (psi_prime_1d[:, None, None] - lam_zeta_3) / (TWO_PI * sqrtg)
            B_zeta = (phi_prime_h[:, None, None] + lam_theta_3) / (TWO_PI * sqrtg)
            return lam_coeff, B_theta, B_zeta

        def force_operators_j(geom, B_theta, B_zeta, psi_prime_1d):
            sqrtg = geom["sqrtg"]
            g_tt = geom["g_tt"]
            g_tz = geom["g_tz"]
            g_zz = geom["g_zz"]
            g_rt = geom["g_rt"]
            g_rz = geom["g_rz"]

            j1 = dd_theta_j(g_tz * B_theta + g_zz * B_zeta)
            j2 = dd_zeta_j(g_tt * B_theta + g_tz * B_zeta)
            J_rho = c_light * (j1 - j2) / (4.0 * jnp.pi * sqrtg)

            F1 = g_tt * B_theta + g_tz * B_zeta
            F2 = g_rt * B_theta + g_rz * B_zeta
            F3 = g_tz * B_theta + g_zz * B_zeta
            t1 = dd_rho_j(F1) - dd_theta_j(F2)
            t2 = dd_rho_j(F3) - dd_zeta_j(F2)
            G_rho = (B_theta * t1 + B_zeta * t2) / (4.0 * jnp.pi) + p_prime_h[:, None, None]

            D = geom["jac"]
            rho_R = geom["Z_theta"] / D
            rho_Z = -geom["R_theta"] / D
            theta_R = -geom["Z_rho"] / D
            theta_Z = geom["R_rho"] / D

            G_R = rho_R * G_rho + (J_rho / (TWO_PI * c_light)) * (theta_R * phi_prime_h[:, None, None])
            G_Z = rho_Z * G_rho + (J_rho / (TWO_PI * c_light)) * (theta_Z * phi_prime_h[:, None, None])
            return J_rho, G_R, G_Z

        def residual_jax(x):
            geom = geometry_j(x)
            _, psi_prime_1d = compute_psi_j(geom["nu_1d"])
            _, B_theta, B_zeta = solve_lambda_and_field_j(geom, psi_prime_1d)
            _, G_R, G_Z = force_operators_j(geom, B_theta, B_zeta, psi_prime_1d)

            r = geom["rho3"]
            Theta = geom["Theta"]
            chi1_l = geom["chi1"]
            chi2_l = geom["chi2"]
            chi3_l = geom["chi3"]
            sqrtg_w = jnp.abs(geom["sqrtg"])
            vol = integrate_volume_j(sqrtg_w)

            R_h0 = integrate_volume_j(sqrtg_w * (G_R * (1.0 - r) ** 2)) / vol
            R_h1 = integrate_volume_j(sqrtg_w * (G_R * (2.0 * r * r - 1.0) * (1.0 - r) ** 2)) / vol
            R_c00 = integrate_volume_j(sqrtg_w * (-G_R * r * a * jnp.sin(Theta) * (1.0 - r * r))) / vol
            R_c10 = integrate_volume_j(
                sqrtg_w * (-G_R * r * a * jnp.sin(Theta) * r * (1.0 - r * r) * jnp.cos(chi1_l))
            ) / vol
            R_s10 = integrate_volume_j(
                sqrtg_w * (-G_R * r * a * jnp.sin(Theta) * r * (1.0 - r * r) * jnp.sin(chi1_l))
            ) / vol
            R_s11 = integrate_volume_j(
                sqrtg_w
                * (
                    -G_R
                    * r
                    * a
                    * jnp.sin(Theta)
                    * r
                    * (1.0 - r * r)
                    * (2.0 * r * r - 1.0)
                    * jnp.sin(chi1_l)
                )
            ) / vol
            R_s20 = integrate_volume_j(sqrtg_w * (-G_R * r * a * jnp.sin(Theta) * r * r * jnp.sin(chi2_l))) / vol
            R_s30 = integrate_volume_j(
                sqrtg_w * (-G_R * r * a * jnp.sin(Theta) * r * r * (1.0 - r * r) * jnp.sin(chi3_l))
            ) / vol
            R_nu0 = integrate_volume_j(sqrtg_w * (G_Z * (1.0 - r * r))) / vol
            R_nu1 = integrate_volume_j(sqrtg_w * (G_Z * (2.0 * r * r - 1.0) * (1.0 - r) ** 2)) / vol
            kappa_factor = (1.0 - r * r) * r * a * jnp.sin(chi1_l)
            R_kappa0 = integrate_volume_j(sqrtg_w * (-G_Z * kappa_factor)) / vol
            R_kappa1 = integrate_volume_j(sqrtg_w * (-G_Z * (2.0 * r * r - 1.0) * kappa_factor)) / vol

            res = jnp.array(
                [
                    R_h0,
                    R_h1,
                    R_kappa0,
                    R_kappa1,
                    R_nu0,
                    R_nu1,
                    R_c00,
                    R_c10,
                    R_s10,
                    R_s11,
                    R_s20,
                    R_s30,
                ],
                dtype=jnp.float64,
            )
            return jnp.nan_to_num(res, nan=1.0e20, posinf=1.0e20, neginf=-1.0e20)

        jac_jax = jax.jacfwd(residual_jax)

        def jac_np(x: np.ndarray) -> np.ndarray:
            xj = jnp.asarray(x, dtype=jnp.float64)
            return np.asarray(jac_jax(xj), dtype=float)

        return jac_np

    def solve(self, x0: np.ndarray) -> least_squares:
        jac_for_solver = "2-point"
        if self.jacobian_mode == "jax_ad":
            try:
                jac_for_solver = self._build_jax_autodiff_jacobian()
            except Exception as exc:
                print(f"[warn] JAX Jacobian unavailable, fallback to scipy_fd: {exc}")
                jac_for_solver = "2-point"

        def fun(x: np.ndarray) -> np.ndarray:
            r = self.evaluate_residuals(x)
            print(f"res_inf={np.max(np.abs(r)):.6e}")
            return r

        return least_squares(
            fun,
            x0,
            jac=jac_for_solver,
            method="trf",
            xtol=self.cfg.tol,
            ftol=self.cfg.tol,
            gtol=self.cfg.tol,
            max_nfev=self.cfg.max_nfev,
            verbose=2,
        )

    def postprocess(self, X: np.ndarray, output_dir: Path) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)

        geom = self._geometry_metrics(X, self.rho)
        geom_h = self._geometry_metrics(X, self.rho_half)
        psi_1d, psi_prime_1d = self._compute_psi_from_nu(geom["nu_1d"], self.rho)
        _, psi_prime_h = self._compute_psi_from_nu(geom_h["nu_1d"], self.rho_half)
        lam_coeff, _, _, _ = self._solve_lambda_and_field(geom_h, psi_prime_h, self.Phi_prime_half)
        psi3 = psi_1d[:, None, None] * np.ones_like(geom["R"])
        psi_prime_mismatch = psi_prime_1d - self.Psi_prime_iota

        # 6 zeta slices in [0, 2pi/N_t]
        zeta_slice_targets = np.linspace(0.0, TWO_PI / float(self.cfg.N_t), 6, endpoint=False)
        slice_idx = [int(np.argmin(np.abs(self.zeta - zt))) for zt in zeta_slice_targets]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        axes = axes.ravel()
        interp_nr = max(220, 2 * self.cfg.n_theta)
        interp_nz = max(220, 2 * self.cfg.n_theta)
        for ax, k, zt in zip(axes, slice_idx, zeta_slice_targets):
            R2 = geom["R"][:, :, k]
            Z2 = geom["Z"][:, :, k]
            P2 = psi3[:, :, k]

            # Reconstruct a regular R-Z map for this toroidal slice.
            pts = np.column_stack([R2.ravel(), Z2.ravel()])
            vals = P2.ravel()
            valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(vals)
            pts = pts[valid]
            vals = vals[valid]

            Rb = R2[-1, :]
            Zb = Z2[-1, :]
            bvalid = np.isfinite(Rb) & np.isfinite(Zb)
            Rb = Rb[bvalid]
            Zb = Zb[bvalid]
            if Rb.size >= 3:
                poly = np.column_stack([Rb, Zb])
                if np.hypot(poly[0, 0] - poly[-1, 0], poly[0, 1] - poly[-1, 1]) > 1.0e-14:
                    poly = np.vstack([poly, poly[0]])
                boundary_path = MplPath(poly)
                rmin, rmax = float(np.min(Rb)), float(np.max(Rb))
                zmin, zmax = float(np.min(Zb)), float(np.max(Zb))
            else:
                boundary_path = None
                rmin, rmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                zmin, zmax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

            pad_r = 0.05 * max(rmax - rmin, 1.0e-8)
            pad_z = 0.05 * max(zmax - zmin, 1.0e-8)
            r_lin = np.linspace(rmin - pad_r, rmax + pad_r, interp_nr)
            z_lin = np.linspace(zmin - pad_z, zmax + pad_z, interp_nz)
            RR, ZZ = np.meshgrid(r_lin, z_lin, indexing="xy")

            psi_lin = griddata(pts, vals, (RR, ZZ), method="linear")
            psi_nn = griddata(pts, vals, (RR, ZZ), method="nearest")
            psi_grid = np.where(np.isfinite(psi_lin), psi_lin, psi_nn)

            if boundary_path is not None:
                inside = boundary_path.contains_points(
                    np.column_stack([RR.ravel(), ZZ.ravel()])
                ).reshape(RR.shape)
                psi_plot = np.ma.array(psi_grid, mask=~inside)
            else:
                psi_plot = np.ma.array(psi_grid, mask=~np.isfinite(psi_grid))

            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 1.0e-12:
                levels = np.linspace(vmin, vmax, 24)
                ax.contourf(RR, ZZ, psi_plot, levels=levels, cmap="plasma")
                ax.contour(RR, ZZ, psi_plot, levels=levels[::3], colors="k", linewidths=0.45, alpha=0.55)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=3.0, cmap="plasma")

            if Rb.size >= 3:
                ax.plot(Rb, Zb, color="white", linewidth=1.1, alpha=0.9)

            ax.set_title(f"zeta={zt:.4f} (idx={k})")
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

        fig_path = output_dir / "psi_slices.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

        summary = {
            "grid": [self.cfg.n_rho, self.cfg.n_theta, self.cfg.n_zeta],
            "n_modes": self.cfg.n_modes,
            "N_t": self.cfg.N_t,
            "n_helical": self.cfg.n_helical,
            "phi_total": self.cfg.phi_total,
            "c_light": self.c_light,
            "psi_a": self.psi_a,
            "staggered_grid": {
                "rho_full_size": int(self.rho.size),
                "rho_half_size": int(self.rho_half.size),
                "geometry_on": "full_grid",
                "P_PhiPrime_PsiPrime_B_on": "half_grid",
            },
            "derivative_methods": {
                "theta": "fft_pseudo_spectral",
                "zeta": "fft_pseudo_spectral",
                "rho": "4th_order_fd_uniform_else_cubic_spline",
            },
            "lambda_mode_count": int(self.lambda_mode_pairs.shape[0]),
            "lambda_mode_pairs": self.lambda_mode_pairs.tolist(),
            "inputs": {
                "profiles": {
                    "pressure_expr": str(self.profiles_cfg.get("pressure_expr", "")),
                    "iota_expr": str(self.profiles_cfg.get("iota_expr", "")),
                    "phi_expr": str(self.profiles_cfg.get("phi_expr", "")),
                    "psi_a": self.profiles_cfg.get("psi_a", None),
                    "constants": dict(self.profile_constants),
                },
                "lcfs": {
                    "R_expr": str(self.lcfs_cfg.get("R_expr", "")),
                    "Z_expr": str(self.lcfs_cfg.get("Z_expr", "")),
                    "constants": dict(self.lcfs_constants),
                },
                "lambda_fourier": dict(self.case_cfg.get("lambda_fourier", {})),
                "physical_constants": dict(self.case_cfg.get("physical_constants", {})),
                "solver": {"jacobian_mode": self.jacobian_mode},
                "initial_guess": self.initial_guess.tolist(),
            },
            "boundary": self.boundary.__dict__,
            "X_order": [
                "h0",
                "h1",
                "kappa0",
                "kappa1",
                "nu0",
                "nu1",
                "c00",
                "c10",
                "s10",
                "s11",
                "s20",
                "s30",
            ],
            "X_final": X.tolist(),
            "psi_constraint": "psi = psi_a * rho^2 * (1 + nu(rho))",
            "psi_prime_iota_relation_l2": float(np.sqrt(np.mean(psi_prime_mismatch * psi_prime_mismatch))),
            "psi_prime_iota_relation_linf": float(np.max(np.abs(psi_prime_mismatch))),
            "psi_slice_plot_method": "interpolate_to_regular_RZ_grid_and_mask_by_LCFS",
            "lambda_coeff_shape": list(lam_coeff.shape),
            "outputs": {"psi_slices_png": str(fig_path)},
        }

        with (output_dir / "equilibrium_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            output_dir / "equilibrium_fields.npz",
            rho=self.rho,
            rho_half=self.rho_half,
            theta=self.theta,
            zeta=self.zeta,
            R=geom["R"],
            Z=geom["Z"],
            psi=psi3,
            X_final=X,
            lambda_coeff=lam_coeff,
        )
        return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variational 3D equilibrium solver")
    p.add_argument("--case-json", type=str, default=None, help="External case input JSON.")
    p.add_argument(
        "--write-default-case",
        type=str,
        default=None,
        help="Write default case JSON template and exit.",
    )
    p.add_argument("--n-rho", type=int, default=None)
    p.add_argument("--n-theta", type=int, default=None)
    p.add_argument("--n-zeta", type=int, default=None)
    p.add_argument("--n-modes", type=int, default=None)
    p.add_argument("--N-t", type=int, default=None)
    p.add_argument("--n-helical", type=int, default=None)
    p.add_argument("--phi-total", type=float, default=None)
    p.add_argument("--psi-a", type=float, default=None, help="Override psi_a in psi=psi_a*rho^2*(1+nu).")
    p.add_argument("--c-light", type=float, default=None, help="Physical constant c in MHD operators.")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--max-nfev", type=int, default=None)
    p.add_argument("--lambda-reg", type=float, default=None)
    p.add_argument(
        "--jacobian-mode",
        type=str,
        choices=["jax_ad", "scipy_fd"],
        default=None,
        help="Jacobian mode for least_squares.",
    )
    p.add_argument("--lambda-m-min", type=int, default=None)
    p.add_argument("--lambda-m-max", type=int, default=None)
    p.add_argument("--lambda-n-min", type=int, default=None)
    p.add_argument("--lambda-n-max", type=int, default=None)
    p.add_argument(
        "--s11",
        type=float,
        default=None,
        help="Override initial guess of 12th variational parameter s11.",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-run: overwrite to (36,72,36), max_nfev=4.",
    )
    return p.parse_args()


def main() -> None:
    def pick(cli_value: Any, case_value: Any, default_value: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if case_value is not None:
            return case_value
        return default_value

    args = parse_args()
    if args.write_default_case:
        out = Path(args.write_default_case)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(clone_default_case_config(), f, indent=2, ensure_ascii=False)
        print(f"default case template written: {out}")
        return

    case_cfg = load_case_config(args.case_json)
    default_cfg = SolverConfig()
    case_grid = case_cfg.get("grid", {})
    case_period = case_cfg.get("periodicity", {})
    case_profiles = case_cfg.get("profiles", {})
    case_physical = case_cfg.get("physical_constants", {})

    n_rho = int(pick(args.n_rho, case_grid.get("n_rho"), default_cfg.n_rho))
    n_theta = int(pick(args.n_theta, case_grid.get("n_theta"), default_cfg.n_theta))
    n_zeta = int(pick(args.n_zeta, case_grid.get("n_zeta"), default_cfg.n_zeta))
    n_modes = int(pick(args.n_modes, case_grid.get("n_modes"), default_cfg.n_modes))
    N_t = int(pick(args.N_t, case_period.get("N_t"), default_cfg.N_t))
    n_helical = int(pick(args.n_helical, case_period.get("n_helical"), default_cfg.n_helical))
    phi_total = float(pick(args.phi_total, case_profiles.get("phi_total"), default_cfg.phi_total))
    psi_a = pick(args.psi_a, case_profiles.get("psi_a"), None)
    c_light = float(pick(args.c_light, case_physical.get("c_light"), 1.0))
    tol = float(pick(args.tol, None, default_cfg.tol))
    max_nfev = int(pick(args.max_nfev, None, default_cfg.max_nfev))
    lambda_reg = float(pick(args.lambda_reg, None, default_cfg.lambda_reg))
    output_dir = str(pick(args.output_dir, None, default_cfg.output_dir))

    # keep case expressions and runtime scalar controls aligned
    case_cfg.setdefault("profiles", {})
    case_cfg["profiles"]["phi_total"] = phi_total
    case_cfg["profiles"]["psi_a"] = psi_a
    case_cfg.setdefault("periodicity", {})
    case_cfg["periodicity"]["N_t"] = N_t
    case_cfg["periodicity"]["n_helical"] = n_helical
    case_cfg.setdefault("physical_constants", {})
    case_cfg["physical_constants"]["c_light"] = c_light
    case_cfg.setdefault("lambda_fourier", {})
    if (
        args.lambda_m_min is not None
        or args.lambda_m_max is not None
        or args.lambda_n_min is not None
        or args.lambda_n_max is not None
    ):
        case_cfg["lambda_fourier"]["pairs"] = []
    if args.lambda_m_min is not None:
        case_cfg["lambda_fourier"]["m_min"] = int(args.lambda_m_min)
    if args.lambda_m_max is not None:
        case_cfg["lambda_fourier"]["m_max"] = int(args.lambda_m_max)
    if args.lambda_n_min is not None:
        case_cfg["lambda_fourier"]["n_min"] = int(args.lambda_n_min)
    if args.lambda_n_max is not None:
        case_cfg["lambda_fourier"]["n_max"] = int(args.lambda_n_max)
    case_cfg.setdefault("solver", {})
    if args.jacobian_mode is not None:
        case_cfg["solver"]["jacobian_mode"] = str(args.jacobian_mode)
    if args.s11 is not None:
        x0_case = list(case_cfg["solver"].get("initial_guess", [0.0] * N_X_UNKNOWNS))
        if len(x0_case) == N_X_UNKNOWNS - 1:
            case_cfg["solver"]["s11"] = float(args.s11)
        elif len(x0_case) == N_X_UNKNOWNS:
            x0_case[9] = float(args.s11)
            case_cfg["solver"]["initial_guess"] = x0_case
        else:
            raise ValueError(
                "solver.initial_guess length must be 12 (or legacy 11) when using --s11."
            )

    if args.quick:
        n_rho = 36
        n_theta = 72
        n_zeta = 36
        max_nfev = min(max_nfev, 4)

    cfg = SolverConfig(
        n_rho=n_rho,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_modes=n_modes,
        N_t=N_t,
        n_helical=n_helical,
        phi_total=phi_total,
        tol=tol,
        max_nfev=max_nfev,
        lambda_reg=lambda_reg,
        output_dir=output_dir,
    )
    solver = V3DEquilibriumSolver(cfg, case_cfg)

    print("Boundary constraints:")
    print(json.dumps(solver.boundary.__dict__, indent=2, ensure_ascii=False))

    x0 = solver.initial_guess
    result = solver.solve(x0)

    print("\nSolve done")
    print(f"success={result.success}, status={result.status}")
    print(f"message={result.message}")
    print(f"nfev={result.nfev}, cost={result.cost:.6e}")
    print("X_final=", result.x)

    summary = solver.postprocess(result.x, Path(cfg.output_dir))
    print("\nPost-process outputs:")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
