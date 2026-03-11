import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class VEQ3D_Solver:
    def __init__(self):
        # 1. 物理常数 (SI 单位制)
        self.Nt = 1                  # 周期数
        self.Phi_a = 1.0             # 总环向磁通标量 (Wb)
        self.mu_0 = 4 * np.pi * 1e-7 # 真空磁导率
        
        # 2. 离散化网格
        self.Nr = 20                  
        self.Nt_grid = 24            
        self.Nz_grid = 12             
        
        rho_nodes, self.rho_weights = roots_legendre(self.Nr)
        self.rho = 0.5 * (rho_nodes + 1) * 0.98 + 0.01 
        self.rho_weights *= 0.5 * 0.98
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')

        self.X_edge = {} 
        self.fit_boundary()

    def get_profiles(self, rho):
        """第7节: 磁通与压强剖面映射"""
        P_base = 1.8e4
        P = P_base * (rho**2 - 1)**2
        dP_drho = P_base * 4 * rho * (rho**2 - 1)
        Phi_prime = 2 * rho * self.Phi_a
        iota_rho = 1.0 + 1.5 * rho**2
        psi_prime = iota_rho * Phi_prime
        return P, dP_drho, psi_prime, Phi_prime

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self):
        """预拟合几何边界 (LCFS)"""
        th_fit = np.linspace(0, 2*np.pi, 40); ze_fit = np.linspace(0, 2*np.pi, 20)
        TH_F, ZE_F = np.meshgrid(th_fit, ze_fit)
        R_target = 10 - 0.3 * np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)

        def boundary_residuals(p):
            R0, Z0 = p[0], p[1]
            h = p[2] + p[3]*np.cos(ZE_F) + p[4]*np.sin(ZE_F)
            v = p[5] + p[6]*np.cos(ZE_F) + p[7]*np.sin(ZE_F)
            a = p[8] + p[9]*np.cos(ZE_F) + p[10]*np.sin(ZE_F)
            k = p[11] + p[12]*np.cos(ZE_F) + p[13]*np.sin(ZE_F)
            c0R = p[14] + p[15]*np.cos(ZE_F) + p[16]*np.sin(ZE_F)
            c0Z = p[17] + p[18]*np.cos(ZE_F) + p[19]*np.sin(ZE_F)
            tR, tZ = TH_F + c0R, TH_F + c0Z
            R_mod = R0 + h + a * np.cos(tR); Z_mod = Z0 + v + k * a * np.sin(tZ)
            return np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])

        p0 = np.zeros(20); p0[0], p0[8], p0[11] = 10.0, 0.3, 1.0 
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        keys = ['R0', 'Z0', 'h0', 'h1c', 'h1s', 'v0', 'v1c', 'v1s', 'a0', 'a1c', 'a1s', 'k0', 'k1c', 'k1s', 'c00', 'c01c', 'c01s', 'c00z', 'c01cz', 'c01sz']
        self.X_edge = {k: v for k, v in zip(keys, res.x)}

    def compute_geometry(self, x_core, rho, theta, zeta):
        fac = 1.0 - rho**2
        c0R = (self.X_edge['c00'] + x_core[0]*fac) + (self.X_edge['c01c'] + x_core[1]*fac)*np.cos(zeta) + (self.X_edge['c01s'] + x_core[2]*fac)*np.sin(zeta)
        c0Z = (self.X_edge['c00z'] + x_core[3]*fac) + (self.X_edge['c01cz'] + x_core[4]*fac)*np.cos(zeta) + (self.X_edge['c01sz'] + x_core[5]*fac)*np.sin(zeta)
        h = (self.X_edge['h0'] + x_core[6]*fac) + (self.X_edge['h1c'] + x_core[7]*fac)*np.cos(zeta) + (self.X_edge['h1s'] + x_core[8]*fac)*np.sin(zeta)
        v = (self.X_edge['v0'] + x_core[9]*fac) + (self.X_edge['v1c'] + x_core[10]*fac)*np.cos(zeta) + (self.X_edge['v1s'] + x_core[11]*fac)*np.sin(zeta)
        a = (self.X_edge['a0'] + x_core[15]*fac) + (self.X_edge['a1c'] + x_core[16]*fac)*np.cos(zeta) + (self.X_edge['a1s'] + x_core[17]*fac)*np.sin(zeta)
        k = (self.X_edge['k0'] + x_core[12]*fac) + (self.X_edge['k1c'] + x_core[13]*fac)*np.cos(zeta) + (self.X_edge['k1s'] + x_core[14]*fac)*np.sin(zeta)
        thR, thZ = theta + c0R, theta + c0Z
        R = self.X_edge['R0'] + h + rho * a * np.cos(thR)
        Z = self.X_edge['Z0'] + v + k * rho * a * np.sin(thZ)
        Lambda = fac * (x_core[18] * np.sin(theta + zeta) + x_core[19] * np.sin(theta) + x_core[20] * np.sin(theta - zeta))
        return R, Z, thR, thZ, a, k, Lambda

    def compute_physics(self, x_core):
        """核心物理计算：修复残差向量化的缩放问题"""
        rho, theta, zeta = self.RHO, self.TH, self.ZE
        fac, dfac = 1.0 - rho**2, -2.0 * rho
        eps = 1e-5

        def periodic_grad(f, step, axis):
            return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * step)

        cz, sz = np.cos(zeta), np.sin(zeta)
        def get_vals_derivs(edge, core, cz, sz):
            val = (edge[0]+core[0]*fac) + (edge[1]+core[1]*fac)*cz + (edge[2]+core[2]*fac)*sz
            dr = core[0]*dfac + core[1]*dfac*cz + core[2]*dfac*sz
            dz = -(edge[1]+core[1]*fac)*sz + (edge[2]+core[2]*fac)*cz
            return val, dr, dz

        h, hr, hz = get_vals_derivs([self.X_edge['h0'], self.X_edge['h1c'], self.X_edge['h1s']], x_core[6:9], cz, sz)
        v, vr, vz = get_vals_derivs([self.X_edge['v0'], self.X_edge['v1c'], self.X_edge['v1s']], x_core[9:12], cz, sz)
        a, ar, az = get_vals_derivs([self.X_edge['a0'], self.X_edge['a1c'], self.X_edge['a1s']], x_core[15:18], cz, sz)
        k, kr, kz = get_vals_derivs([self.X_edge['k0'], self.X_edge['k1c'], self.X_edge['k1s']], x_core[12:15], cz, sz)
        c0R, c0Rr, c0Rz = get_vals_derivs([self.X_edge['c00'], self.X_edge['c01c'], self.X_edge['c01s']], x_core[0:3], cz, sz)
        c0Z, c0Zr, c0Zz = get_vals_derivs([self.X_edge['c00z'], self.X_edge['c01cz'], self.X_edge['c01sz']], x_core[3:6], cz, sz)

        thR, thZ = theta + c0R, theta + c0Z
        R = self.X_edge['R0'] + h + rho * a * np.cos(thR)
        Rr = hr + a*np.cos(thR) + rho*ar*np.cos(thR) - rho*a*np.sin(thR)*c0Rr
        Rt = -rho*a*np.sin(thR)
        Zr = vr + kr*rho*a*np.sin(thZ) + k*a*np.sin(thZ) + k*rho*ar*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zr
        Zt = k*rho*a*np.cos(thZ)

        det_J = Rr * Zt - Rt * Zr
        min_det = 1e-4 
        
        # --- 向量化惩罚项 (每一个点的违反情况都是独立的残差分量) ---
        violation = np.maximum(0, min_det - det_J)
        # 惩罚项归一化到与压强梯度相近的量级
        penalty_vec = violation.flatten() * 100.0 
        
        det_safe = np.where(det_J < min_det, min_det, det_J)
        sqrt_g = (R / self.Nt) * det_safe

        rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
        th_R, th_Z = -Zr/det_safe, Rr/det_safe
        
        L1n1, L10, L11 = x_core[18:21]
        L = fac * (L1n1 * np.sin(theta + zeta) + L10 * np.sin(theta) + L11 * np.sin(theta - zeta))
        Lt = fac * (L1n1 * np.cos(theta + zeta) + L10 * np.cos(theta) + L11 * np.cos(theta - zeta))
        Lz = fac * (L1n1 * np.cos(theta + zeta) - L11 * np.cos(theta - zeta))

        P, dP, psip, Phip = self.get_profiles(rho)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)

        g_tt, g_zz, g_rt = Rt**2 + Zt**2, (R/self.Nt)**2, Rr*Rt+Zr*Zt
        Bt_sub, Bz_sub = g_tt * Bt_sup, g_zz * Bz_sup
        Br_sub = g_rt * Bt_sup

        Jz_sup = (np.gradient(Bt_sub, self.rho, axis=0) - periodic_grad(Br_sub, self.dtheta, axis=1)) / sqrt_g
        Jt_sup = (periodic_grad(Br_sub, self.dzeta, axis=2) - np.gradient(Bz_sub, self.rho, axis=0)) / sqrt_g
        Jr_sup = (periodic_grad(Bz_sub, self.dtheta, axis=1) - periodic_grad(Bt_sub, self.dzeta, axis=2)) / sqrt_g

        # J x B 受力平衡 (SI单位)
        J_times_B_rho = sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / self.mu_0
        G_rho = dP - J_times_B_rho
        
        # 归一化到 P_norm 确保残差在 O(1) 量级
        P_norm = 1.8e4
        GR = (rho_R * G_rho + (Jr_sup / self.mu_0) * (th_R * Phip)) / P_norm
        GZ = (rho_Z * G_rho + (Jr_sup / self.mu_0) * (th_Z * Phip)) / P_norm

        residuals = []
        kernels = [
            -fac * rho * a * np.sin(thR), -fac * rho * a * np.sin(thR) * cz, -fac * rho * a * np.sin(thR) * sz, 
            fac * k * rho * a * np.cos(thZ), fac * k * rho * a * np.cos(thZ) * cz, fac * k * rho * a * np.cos(thZ) * sz, 
            fac, fac * cz, fac * sz, fac, fac * cz, fac * sz, 
            rho * a * np.sin(thZ), rho * a * np.sin(thZ) * cz, rho * a * np.sin(thZ) * sz, 
            rho * np.cos(thR), rho * np.cos(thR) * cz, rho * np.cos(thR) * sz
        ]
        
        # 变分投影残差 (均值化处理)
        for i in range(18):
            if i < 3 or (6 <= i < 9):
                res_val = np.mean(sqrt_g * GR * kernels[i])
            elif (3 <= i < 6) or (9 <= i < 15):
                res_val = np.mean(sqrt_g * GZ * kernels[i])
            else:
                res_val = np.mean(sqrt_g * (GR * kernels[i] + GZ * (k * rho * np.sin(thZ) * (1 if i==15 else (cz if i==16 else sz)))))
            residuals.append(res_val)

        # 流函数残差 (均值化)
        for m, n in [(1, -1), (1, 0), (1, 1)]:
            # 这里的 Jr_sup 需要归一化量级
            res_L = np.mean(sqrt_g * (Jr_sup / 1e6) * np.sin(m * self.TH + n * self.ZE))
            residuals.append(res_L)
            
        # 将展平的惩罚向量加入残差 (Vectorized Penalty)
        # 此时 least_squares 最小化的是每个点违反量的平方和
        residuals.extend(penalty_vec.tolist())
            
        return np.array(residuals)

    def solve(self):
        print(">>> 正在启动 VEQ-3D 归一化平衡求解器 (向量化残差)...")
        x0 = np.zeros(21); x0[15] = 0.05
        # 提高迭代次数，减小容差以获得更深度的收敛
        res = least_squares(self.compute_physics, x0, method='trf', 
                            xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=500)
        
        print(f">>> 求解完成！最终残差范数 (L2): {np.linalg.norm(res.fun):.4f}")
        print(f"    迭代次数: {res.nfev}, 终止原因: {res.status}")
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        labels = ['0', r'\pi/3', r'2\pi/3', r'\pi', r'4\pi/3', r'5\pi/3']
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes = axes.flatten()
        rho_plt = np.linspace(0, 1.0, 50); th_plt = np.linspace(0, 2*np.pi, 80)
        RHO_P, TH_P = np.meshgrid(rho_plt, th_plt); PSI_P = self.compute_psi(RHO_P)
        for i, (zv, lb) in enumerate(zip(zetas, labels)):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(RHO_P.flatten(), TH_P.flatten()):
                res_g = self.compute_geometry(x_core, r, t, zv)
                Rm.append(res_g[0]); Zm.append(res_g[1])
            Rm = np.array(Rm).reshape(RHO_P.shape); Zm = np.array(Zm).reshape(RHO_P.shape)
            im = ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='plasma', alpha=0.9)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, 'w-', lw=2 if r_lev==1 else 0.8)
            ax.set_title(rf'$\zeta = {lb}$'); ax.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
