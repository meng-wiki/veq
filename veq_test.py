import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class VEQ3D_Solver:
    def __init__(self):
        # 1. 物理常数 (SI 单位制)
        self.Nt = 1                  # 周期数
        self.Phi_a = 1.0             # 总环向磁通标量 (Wb)
        self.mu_0 = 4 * np.pi * 1e-7 # 真空磁导率 (H/m)
        
        # 2. 离散化网格 (提升分辨率以降低截断误差与混叠误差)
        self.Nr = 35                  
        self.Nt_grid = 64            # 抗混叠：加倍极向网格
        self.Nz_grid = 32            # 抗混叠：加倍环向网格
        
        # 径向 Gauss-Legendre 节点与权重
        rho_nodes, self.rho_weights = roots_legendre(self.Nr)
        self.rho = 0.5 * (rho_nodes + 1) * 0.98 + 0.01 
        self.rho_weights *= 0.5 * 0.98
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        self.X_edge = {} 
        self.fit_boundary()
        
        # 3. 自适应预条件权重初始化
        self.res_scales = np.ones(21)
        self._initialize_scaling()

    def _initialize_scaling(self):
        """执行一次前向计算，记录残差初始量级以实施预条件"""
        print(">>> 正在执行残差预条件归一化...")
        x0 = np.zeros(21); x0[15] = 0.05
        raw_res = self.compute_physics(x0, apply_scaling=False)
        # 避免除以极小值
        self.res_scales = np.maximum(np.abs(raw_res), 1e-3)
        print(f"    预条件完成，Scale范围: [{self.res_scales.min():.2e}, {self.res_scales.max():.2e}]")

    def get_profiles(self, rho):
        P_scale = 1.8e4
        P = P_scale * (rho**2 - 1)**2
        dP_drho = P_scale * 4 * rho * (rho**2 - 1)
        Phi_prime = 2 * rho * self.Phi_a
        iota = 1.0 + 1.5 * rho**2
        psi_prime = iota * Phi_prime
        return P, dP_drho, psi_prime, Phi_prime

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self):
        th_f = np.linspace(0, 2*np.pi, 50); ze_f = np.linspace(0, 2*np.pi, 24)
        TH_F, ZE_F = np.meshgrid(th_f, ze_f)
        R_target = 10 - 0.3 * np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)

        def boundary_residuals(p):
            R0, Z0, h0, h1c, h1s, v0, v1c, v1s, a0, a1c, a1s, k0, k1c, k1s, c00, c01c, c01s, c00z, c01cz, c01sz = p
            h = h0 + h1c*np.cos(ZE_F) + h1s*np.sin(ZE_F)
            v = v0 + v1c*np.cos(ZE_F) + v1s*np.sin(ZE_F)
            a = a0 + a1c*np.cos(ZE_F) + a1s*np.sin(ZE_F)
            k = k0 + k1c*np.cos(ZE_F) + k1s*np.sin(ZE_F)
            c0R = c00 + c01c*np.cos(ZE_F) + c01s*np.sin(ZE_F)
            c0Z = c00z + c01cz*np.cos(ZE_F) + c01sz*np.sin(ZE_F)
            R_mod = R0 + h + a * np.cos(TH_F + c0R)
            Z_mod = Z0 + v + k * a * np.sin(TH_F + c0Z)
            return np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])

        p0 = [10.0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    def compute_physics(self, x_core, apply_scaling=True):
        """核心物理计算：修正混叠、差分与奇异性处理"""
        rho, theta, zeta = self.RHO, self.TH, self.ZE
        fac, dfac = 1.0 - rho**2, -2.0 * rho
        
        def periodic_grad(f, step, axis):
            return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * step)

        # 1. 解析几何偏导数
        cz, sz = np.cos(zeta), np.sin(zeta)
        def eval_derivs(edge, core, cz, sz):
            val = (edge[0]+core[0]*fac) + (edge[1]+core[1]*fac)*cz + (edge[2]+core[2]*fac)*sz
            dr = core[0]*dfac + core[1]*dfac*cz + core[2]*dfac*sz
            dz = -(edge[1]+core[1]*fac)*sz + (edge[2]+core[2]*fac)*cz
            return val, dr, dz

        h, hr, hz = eval_derivs([self.X_edge['h0'], self.X_edge['h1c'], self.X_edge['h1s']], x_core[6:9], cz, sz)
        v, vr, vz = eval_derivs([self.X_edge['v0'], self.X_edge['v1c'], self.X_edge['v1s']], x_core[9:12], cz, sz)
        a, ar, az = eval_derivs([self.X_edge['a0'], self.X_edge['a1c'], self.X_edge['a1s']], x_core[15:18], cz, sz)
        k, kr, kz = eval_derivs([self.X_edge['k0'], self.X_edge['k1c'], self.X_edge['k1s']], x_core[12:15], cz, sz)
        c0R, c0Rr, c0Rz = eval_derivs([self.X_edge['c00'], self.X_edge['c01c'], self.X_edge['c01s']], x_core[0:3], cz, sz)
        c0Z, c0Zr, c0Zz = eval_derivs([self.X_edge['c00z'], self.X_edge['c01cz'], self.X_edge['c01sz']], x_core[3:6], cz, sz)

        thR, thZ = theta + c0R, theta + c0Z
        R = self.X_edge['R0'] + h + rho * a * np.cos(thR)
        Rr = hr + a*np.cos(thR) + rho*ar*np.cos(thR) - rho*a*np.sin(thR)*c0Rr
        Rt = -rho*a*np.sin(thR)
        Rz = hz + rho*az*np.cos(thR) - rho*a*np.sin(thR)*c0Rz
        Zr = vr + kr*rho*a*np.sin(thZ) + k*a*np.sin(thZ) + k*rho*ar*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zr
        Zt = k*rho*a*np.cos(thZ)
        Zz = vz + kz*rho*a*np.sin(thZ) + k*rho*az*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zz

        # 2. 补全度规与平滑雅可比 (避免梯度平原)
        det_J = Rr * Zt - Rt * Zr
        # --- 修复：平滑雅可比逼近 (Softplus-like) ---
        det_safe = 0.5 * (det_J + np.sqrt(det_J**2 + 1e-8))
        sqrt_g = (R / self.Nt) * det_safe

        g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
        g_zz = Rz**2 + (R/self.Nt)**2 + Zz**2 
        g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

        # 3. 逆雅可比
        rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
        th_R, th_Z = -Zr/det_safe, Rr/det_safe
        
        # 4. 磁场与电流 (修正周期梯度与量纲)
        L1n1, L10, L11 = x_core[18:21]
        L = fac * (L1n1 * np.sin(theta + zeta) + L10 * np.sin(theta) + L11 * np.sin(theta - zeta))
        Lt = fac * (L1n1 * np.cos(theta + zeta) + L10 * np.cos(theta) + L11 * np.cos(theta - zeta))
        Lz = fac * (L1n1 * np.cos(theta + zeta) - L11 * np.cos(theta - zeta))

        P, dP, psip, Phip = self.get_profiles(rho)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)

        Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
        Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
        Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

        # 求电流 (修正 np.gradient 在非均匀网格上的使用)
        Jz_sup = (np.gradient(Bt_sub, self.rho, axis=0) - periodic_grad(Br_sub, self.dtheta, axis=1)) / sqrt_g
        Jt_sup = (periodic_grad(Br_sub, self.dzeta, axis=2) - np.gradient(Bz_sub, self.rho, axis=0)) / sqrt_g
        Jr_sup = (periodic_grad(Bz_sub, self.dtheta, axis=1) - periodic_grad(Bt_sub, self.dzeta, axis=2)) / sqrt_g

        # 5. 平衡算子与投影
        J_times_B_rho = sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / self.mu_0
        G_rho = dP - J_times_B_rho
        
        GR = (rho_R * G_rho + (Jr_sup / self.mu_0) * (th_R * Phip))
        GZ = (rho_Z * G_rho + (Jr_sup / self.mu_0) * (th_Z * Phip))

        # 6. 伽辽金变分投影
        dV = self.dtheta * self.dzeta
        residuals = []
        kernels = [
            -fac * rho * a * np.sin(thR), -fac * rho * a * np.sin(thR) * cz, -fac * rho * a * np.sin(thR) * sz, 
            fac * k * rho * a * np.cos(thZ), fac * k * rho * a * np.cos(thZ) * cz, fac * k * rho * a * np.cos(thZ) * sz, 
            fac, fac * cz, fac * sz, fac, fac * cz, fac * sz, 
            rho * a * np.sin(thZ), rho * a * np.sin(thZ) * cz, rho * a * np.sin(thZ) * sz, 
            rho * np.cos(thR), rho * np.cos(thR) * cz, rho * np.cos(thR) * sz
        ]
        
        for i in range(18):
            if i < 3 or (6 <= i < 9):
                integrand = sqrt_g * GR * kernels[i]
            elif (3 <= i < 6) or (9 <= i < 15):
                integrand = sqrt_g * GZ * kernels[i]
            else:
                integrand = sqrt_g * (GR * kernels[i] + GZ * (k * rho * np.sin(thZ) * (1 if i==15 else (cz if i==16 else sz))))
            
            res_val = np.sum(integrand * self.weights_3d) * dV
            residuals.append(res_val)

        for m, n in [(1, -1), (1, 0), (1, 1)]:
            integrand_L = sqrt_g * (Jr_sup / self.mu_0) * np.sin(m * self.TH + n * self.ZE)
            res_L = np.sum(integrand_L * self.weights_3d) * dV
            residuals.append(res_L)
            
        final_res = np.array(residuals)
        
        # --- 修复：实施预条件缩放 ---
        if apply_scaling:
            return final_res / self.res_scales
        return final_res

    def solve(self):
        print(">>> 正在启动 VEQ-3D 混合谱平衡求解器 (预条件+平滑雅可比+抗混叠)...")
        x0 = np.zeros(21); x0[15] = 0.05
        res = least_squares(self.compute_physics, x0, method='trf', xtol=1e-8, ftol=1e-8, max_nfev=500)
        print(f">>> 求解完成！收敛范数: {np.linalg.norm(res.fun):.6e}")
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        labels = ['0', r'\pi/3', r'2\pi/3', r'\pi', r'4\pi/3', r'5\pi/3']
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        rho_p = np.linspace(0, 1.0, 50); th_p = np.linspace(0, 2*np.pi, 80)
        RHO_P, TH_P = np.meshgrid(rho_p, th_p); PSI_P = self.compute_psi(RHO_P)
        for i, (zv, lb) in enumerate(zip(zetas, labels)):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(RHO_P.flatten(), TH_P.flatten()):
                res_g = self.compute_geometry(x_core, r, t, zv)
                Rm.append(res_g[0]); Zm.append(res_g[1])
            Rm = np.array(Rm).reshape(RHO_P.shape); Zm = np.array(Zm).reshape(RHO_P.shape)
            ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, color='#FFD700' if r_lev==1.0 else 'white', lw=2 if r_lev==1.0 else 1.0, alpha=0.8)
            ax.set_title(rf'$\zeta = {lb}$'); ax.set_aspect('equal')
            ax.set_xlim(Rm.min()-0.5, Rm.max()+0.5); ax.set_ylim(Zm.min()-0.5, Zm.max()+0.5)
        plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
