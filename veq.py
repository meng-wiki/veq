#!/usr/bin/env python3
"""
3D plasma equilibrium solver (variational formulation).

Implements the workflow described in `veq/thesis` and `veq/readme`:
1) LCFS fitting
2) 3D grid + 1D profiles
3) Geometry / metric update
4) Lambda closure from J^rho = 0 (surface-by-surface Fourier solve)
5) Force balance operators and 11 variational residuals
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
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TWO_PI = 2.0 * np.pi


def dd_theta(arr: np.ndarray, dtheta: float) -> np.ndarray:
    """Periodic central difference in theta (axis=1)."""
    return (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1)) / (2.0 * dtheta)


def dd_zeta(arr: np.ndarray, dzeta: float) -> np.ndarray:
    """Periodic central difference in zeta (axis=2)."""
    return (np.roll(arr, -1, axis=2) - np.roll(arr, 1, axis=2)) / (2.0 * dzeta)


def dd_theta_2d(arr: np.ndarray, dtheta: float) -> np.ndarray:
    """Periodic central difference for 2D (theta,zeta), theta axis=0."""
    return (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) / (2.0 * dtheta)


def dd_zeta_2d(arr: np.ndarray, dzeta: float) -> np.ndarray:
    """Periodic central difference for 2D (theta,zeta), zeta axis=1."""
    return (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1)) / (2.0 * dzeta)


def dd_rho(arr: np.ndarray, rho_1d: np.ndarray) -> np.ndarray:
    """Finite difference in rho (axis=0)."""
    return np.gradient(arr, rho_1d, axis=0, edge_order=2)


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
    n_modes: int = 3  # retain m=1,2,3
    N_t: int = 19
    # chi = theta - n_helical*zeta. To match LCFS with (+19*zeta) phase, default n_helical=-19.
    n_helical: int = -19
    phi_total: float = 1.0
    lambda_reg: float = 1.0e-8
    eps: float = 1.0e-10
    tol: float = 1.0e-6
    max_nfev: int = 20
    output_dir: str = "veq_output"


class V3DEquilibriumSolver:
    """
    Variational 3D equilibrium solver.

    Unknown vector X (11 vars):
      [h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s20, s30]

    Note on naming:
      user s_0,s_1,s_2  <->  this script s10,s20,s30 (m=1,2,3 sine harmonics).
    """

    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.rho = np.linspace(0.0, 1.0, cfg.n_rho)
        self.theta = np.linspace(0.0, TWO_PI, cfg.n_theta, endpoint=False)
        self.zeta = np.linspace(0.0, TWO_PI, cfg.n_zeta, endpoint=False)
        self.drho = self.rho[1] - self.rho[0]
        self.dtheta = self.theta[1] - self.theta[0]
        self.dzeta = self.zeta[1] - self.zeta[0]

        self.rho3, self.theta3, self.zeta3 = np.meshgrid(
            self.rho, self.theta, self.zeta, indexing="ij"
        )
        self.theta2, self.zeta2 = np.meshgrid(self.theta, self.zeta, indexing="ij")

        # 1D physics profiles
        self.P = 18000.0 * self.rho**4 - 36000.0 * self.rho**2 + 18000.0
        self.P_prime = 72000.0 * self.rho**3 - 72000.0 * self.rho
        self.iota = 1.0 + 1.5 * self.rho**2
        self.Phi = self.cfg.phi_total * self.rho**2
        self.Phi_prime = 2.0 * self.cfg.phi_total * self.rho
        self.Psi_prime = self.iota * self.Phi_prime
        self.Psi = self._integrate_profile(self.Psi_prime, self.rho)

        # mode list m=1..n_modes
        self.modes = np.arange(1, self.cfg.n_modes + 1, dtype=np.int64)

        # fit LCFS-derived boundary constants once
        self.boundary = self._fit_lcfs_boundary()

    @staticmethod
    def _integrate_profile(yprime: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(yprime)
        seg = 0.5 * (yprime[1:] + yprime[:-1]) * np.diff(x)
        out[1:] = np.cumsum(seg)
        return out

    def _target_lcfs(self, theta: np.ndarray, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        User-provided LCFS geometry.
        """
        n = float(self.cfg.N_t)
        R = (
            10.0
            - np.cos(theta)
            - 0.3 * np.cos(theta) * np.cos(n * zeta)
            + 0.3 * np.sin(theta) * np.sin(n * zeta)
        )
        Z = (
            np.sin(theta)
            - 0.3 * np.sin(theta) * np.cos(n * zeta)
            - 0.3 * np.cos(theta) * np.sin(n * zeta)
        )
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

        p0 = np.array([10.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
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

    def _radial_profiles(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s20, s30 = X
        r = self.rho
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

        s1 = r * (self.boundary.s1a + s10 * (1.0 - r2))
        s1_p = self.boundary.s1a + s10 * (1.0 - 3.0 * r2)

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

    def _geometry_metrics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        prof = self._radial_profiles(X)
        r = self.rho3
        th = self.theta3
        ze = self.zeta3
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
        }

    def _solve_lambda_and_field(
        self, geom: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Lambda coefficients on each rho-surface from weighted J^rho=0:
          ∫∫ sqrt(g) J^rho sin(m*theta - n*zeta) dtheta dzeta = 0
        and reconstruct (Lambda_theta, Lambda_zeta), (B^theta, B^zeta).
        """
        n_rho = self.cfg.n_rho
        n_theta = self.cfg.n_theta
        n_zeta = self.cfg.n_zeta
        n = float(self.cfg.n_helical)
        eps = self.cfg.eps

        # basis on each surface
        sin_basis = []
        cos_basis = []
        for m in self.modes:
            arg = m * self.theta2 - n * self.zeta2
            sin_basis.append(np.sin(arg))
            cos_basis.append(np.cos(arg))
        sin_basis = np.asarray(sin_basis)  # (M, Nt, Nz)
        cos_basis = np.asarray(cos_basis)  # (M, Nt, Nz)

        sqrtg = geom["sqrtg"]
        g_tt = geom["g_tt"]
        g_tz = geom["g_tz"]
        g_zz = geom["g_zz"]

        lam_coeff = np.zeros((n_rho, len(self.modes)), dtype=float)

        def jrho_2d(i: int, coeff: np.ndarray) -> np.ndarray:
            sqrtg2 = sqrtg[i]
            sqrtg_safe = np.where(np.abs(sqrtg2) < eps, np.sign(sqrtg2 + eps) * eps, sqrtg2)

            lam_theta = np.zeros((n_theta, n_zeta), dtype=float)
            lam_zeta = np.zeros((n_theta, n_zeta), dtype=float)
            for j, m in enumerate(self.modes):
                c = coeff[j]
                cm = cos_basis[j]
                lam_theta += m * c * cm
                lam_zeta += -n * c * cm

            B_theta = (self.Psi_prime[i] - lam_zeta) / (TWO_PI * sqrtg_safe)
            B_zeta = (self.Phi_prime[i] + lam_theta) / (TWO_PI * sqrtg_safe)

            t1 = dd_theta_2d(g_tz[i] * B_theta + g_zz[i] * B_zeta, self.dtheta)
            t2 = dd_zeta_2d(g_tt[i] * B_theta + g_tz[i] * B_zeta, self.dzeta)
            # normalized c=1 units
            return (t1 - t2) / (4.0 * np.pi * sqrtg_safe)

        # solve each rho surface
        I = np.eye(len(self.modes))
        for i in range(n_rho):
            J0 = jrho_2d(i, np.zeros(len(self.modes), dtype=float))
            A = np.zeros((len(self.modes), len(self.modes)), dtype=float)
            b = np.zeros(len(self.modes), dtype=float)

            for q in range(len(self.modes)):
                b[q] = -integrate_surface(sqrtg[i] * J0 * sin_basis[q], self.theta, self.zeta)

            for p in range(len(self.modes)):
                e = np.zeros(len(self.modes), dtype=float)
                e[p] = 1.0
                Jp = jrho_2d(i, e)
                dJ = Jp - J0
                for q in range(len(self.modes)):
                    A[q, p] = integrate_surface(sqrtg[i] * dJ * sin_basis[q], self.theta, self.zeta)

            A_reg = A + self.cfg.lambda_reg * I
            try:
                lam_coeff[i] = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                lam_coeff[i] = np.linalg.lstsq(A_reg, b, rcond=None)[0]

        # reconstruct Lambda derivatives and B field in 3D
        lam_theta_3 = np.zeros_like(sqrtg)
        lam_zeta_3 = np.zeros_like(sqrtg)
        for j, m in enumerate(self.modes):
            c = lam_coeff[:, j][:, None, None]
            cm = cos_basis[j][None, :, :]
            lam_theta_3 += m * c * cm
            lam_zeta_3 += -n * c * cm

        sqrtg_safe3 = np.where(np.abs(sqrtg) < eps, np.sign(sqrtg + eps) * eps, sqrtg)
        B_theta = (self.Psi_prime[:, None, None] - lam_zeta_3) / (TWO_PI * sqrtg_safe3)
        B_zeta = (self.Phi_prime[:, None, None] + lam_theta_3) / (TWO_PI * sqrtg_safe3)

        return lam_coeff, lam_theta_3, B_theta, B_zeta

    def _force_operators(
        self, geom: Dict[str, np.ndarray], B_theta: np.ndarray, B_zeta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sqrtg = geom["sqrtg"]
        g_tt = geom["g_tt"]
        g_tz = geom["g_tz"]
        g_zz = geom["g_zz"]
        g_rt = geom["g_rt"]
        g_rz = geom["g_rz"]

        eps = self.cfg.eps
        sqrtg_safe = np.where(np.abs(sqrtg) < eps, np.sign(sqrtg + eps) * eps, sqrtg)

        # J^rho
        j1 = dd_theta(g_tz * B_theta + g_zz * B_zeta, self.dtheta)
        j2 = dd_zeta(g_tt * B_theta + g_tz * B_zeta, self.dzeta)
        J_rho = (j1 - j2) / (4.0 * np.pi * sqrtg_safe)

        # G_rho
        F1 = g_tt * B_theta + g_tz * B_zeta
        F2 = g_rt * B_theta + g_rz * B_zeta
        F3 = g_tz * B_theta + g_zz * B_zeta
        t1 = dd_rho(F1, self.rho) - dd_theta(F2, self.dtheta)
        t2 = dd_rho(F3, self.rho) - dd_zeta(F2, self.dzeta)
        G_rho = (B_theta * t1 + B_zeta * t2) / (4.0 * np.pi) + self.P_prime[:, None, None]

        # inverse-gradient components from local Jacobian inversion
        D = geom["jac"]
        D_safe = np.where(np.abs(D) < eps, np.sign(D + eps) * eps, D)
        rho_R = geom["Z_theta"] / D_safe
        rho_Z = -geom["R_theta"] / D_safe
        theta_R = -geom["Z_rho"] / D_safe
        theta_Z = geom["R_rho"] / D_safe

        # zeta_R=zeta_Z=0 for this coordinate mapping
        G_R = rho_R * G_rho + (J_rho / (TWO_PI)) * (theta_R * self.Phi_prime[:, None, None])
        G_Z = rho_Z * G_rho + (J_rho / (TWO_PI)) * (theta_Z * self.Phi_prime[:, None, None])
        return J_rho, G_R, G_Z

    def evaluate_residuals(self, X: np.ndarray) -> np.ndarray:
        geom = self._geometry_metrics(X)
        _, _, B_theta, B_zeta = self._solve_lambda_and_field(geom)
        J_rho, G_R, G_Z = self._force_operators(geom, B_theta, B_zeta)

        # aliases
        r = self.rho3
        Theta = geom["Theta"]
        chi1 = geom["chi1"]
        chi2 = geom["chi2"]
        chi3 = geom["chi3"]
        a = self.boundary.a
        sqrtg_w = np.abs(geom["sqrtg"])
        vol = integrate_volume(sqrtg_w, self.rho, self.theta, self.zeta) + self.cfg.eps

        # 11 residual equations in thesis order
        R_h0 = integrate_volume(sqrtg_w * (G_R * (1.0 - r) ** 2), self.rho, self.theta, self.zeta) / vol
        R_h1 = (
            integrate_volume(
                sqrtg_w * (G_R * (2.0 * r * r - 1.0) * (1.0 - r) ** 2),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_c00 = (
            integrate_volume(
                sqrtg_w * (-G_R * r * a * np.sin(Theta) * (1.0 - r * r)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_c10 = (
            integrate_volume(
                sqrtg_w * (-G_R * r * a * np.sin(Theta) * r * (1.0 - r * r) * np.cos(chi1)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_s10 = (
            integrate_volume(
                sqrtg_w * (-G_R * r * a * np.sin(Theta) * r * (1.0 - r * r) * np.sin(chi1)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_s20 = (
            integrate_volume(
                sqrtg_w * (-G_R * r * a * np.sin(Theta) * r * r * np.sin(chi2)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_s30 = (
            integrate_volume(
                sqrtg_w * (-G_R * r * a * np.sin(Theta) * r * r * (1.0 - r * r) * np.sin(chi3)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_nu0 = (
            integrate_volume(
                sqrtg_w * (G_Z * (1.0 - r * r)),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_nu1 = (
            integrate_volume(
                sqrtg_w * (G_Z * (2.0 * r * r - 1.0) * (1.0 - r) ** 2),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        kappa_factor = (1.0 - r * r) * r * a * np.sin(chi1)
        R_kappa0 = (
            integrate_volume(
                sqrtg_w * (-G_Z * kappa_factor),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )
        R_kappa1 = (
            integrate_volume(
                sqrtg_w * (-G_Z * (2.0 * r * r - 1.0) * kappa_factor),
                self.rho,
                self.theta,
                self.zeta,
            )
            / vol
        )

        # Keep the same X ordering:
        # [h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s20, s30]
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
                R_s20,
                R_s30,
            ],
            dtype=float,
        )
        return residuals

    def solve(self, x0: np.ndarray) -> least_squares:
        def fun(x: np.ndarray) -> np.ndarray:
            r = self.evaluate_residuals(x)
            print(f"res_inf={np.max(np.abs(r)):.6e}")
            return r

        return least_squares(
            fun,
            x0,
            method="trf",
            xtol=self.cfg.tol,
            ftol=self.cfg.tol,
            gtol=self.cfg.tol,
            max_nfev=self.cfg.max_nfev,
            verbose=2,
        )

    def postprocess(self, X: np.ndarray, output_dir: Path) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)

        geom = self._geometry_metrics(X)
        lam_coeff, _, _, _ = self._solve_lambda_and_field(geom)
        psi3 = self.Psi[:, None, None] * np.ones_like(geom["R"])

        # 6 zeta slices in [0, 2pi/N_t]
        zeta_slice_targets = np.linspace(0.0, TWO_PI / float(self.cfg.N_t), 6, endpoint=False)
        slice_idx = [int(np.argmin(np.abs(self.zeta - zt))) for zt in zeta_slice_targets]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        axes = axes.ravel()
        for ax, k, zt in zip(axes, slice_idx, zeta_slice_targets):
            R2 = geom["R"][:, :, k]
            Z2 = geom["Z"][:, :, k]
            P2 = psi3[:, :, k]
            cs = ax.contour(R2, Z2, P2, levels=24, cmap="plasma")
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
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
            "boundary": self.boundary.__dict__,
            "X_final": X.tolist(),
            "lambda_coeff_shape": list(lam_coeff.shape),
            "outputs": {"psi_slices_png": str(fig_path)},
        }

        with (output_dir / "equilibrium_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            output_dir / "equilibrium_fields.npz",
            rho=self.rho,
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
    p.add_argument("--n-rho", type=int, default=100)
    p.add_argument("--n-theta", type=int, default=160)
    p.add_argument("--n-zeta", type=int, default=80)
    p.add_argument("--n-modes", type=int, default=3)
    p.add_argument("--N-t", type=int, default=19)
    p.add_argument("--n-helical", type=int, default=-19)
    p.add_argument("--phi-total", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1.0e-6)
    p.add_argument("--max-nfev", type=int, default=20)
    p.add_argument("--lambda-reg", type=float, default=1.0e-8)
    p.add_argument("--output-dir", type=str, default="veq_output")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-run: overwrite to (36,72,36), max_nfev=4.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_rho = 36
        args.n_theta = 72
        args.n_zeta = 36
        args.max_nfev = min(args.max_nfev, 4)

    cfg = SolverConfig(
        n_rho=args.n_rho,
        n_theta=args.n_theta,
        n_zeta=args.n_zeta,
        n_modes=args.n_modes,
        N_t=args.N_t,
        n_helical=args.n_helical,
        phi_total=args.phi_total,
        tol=args.tol,
        max_nfev=args.max_nfev,
        lambda_reg=args.lambda_reg,
        output_dir=args.output_dir,
    )
    solver = V3DEquilibriumSolver(cfg)

    print("Boundary fit:")
    print(json.dumps(solver.boundary.__dict__, indent=2, ensure_ascii=False))

    # Initial guess for 11 unknowns:
    # [h0, h1, kappa0, kappa1, nu0, nu1, c00, c10, s10, s20, s30]
    x0 = np.zeros(11, dtype=float)
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
