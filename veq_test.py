import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class VEQ3D_Solver:
    def __init__(self):
        # 1. 物理常数与无量纲化参考 (SI -> Normalized)
        self.Nt = 1                  # 周期数
        self.B0 = 1.0                # 特征磁场 (T)
        self.R_ref = 10.0            # 特征长度 (m)
        self.Phi_a = 1.0             # 总环向磁通 (Wb)
        
        # 2. 离散化网格与谱微分矩阵 (Nr=24 以保证高精度)
        self.Nr = 24                  
        self.Nt_grid = 32            
        self.Nz_grid = 16             
        
        rho_nodes, self.rho_weights = roots_legendre(self.Nr)
        # 映射到 [0, 1] 区间
        self.rho = 0.5 * (rho_nodes + 1)
        self.rho_weights *= 0.5
        
        # 预计算勒让德谱微分矩阵 D (Nr x Nr)
        self.D_matrix = self._get_legendre_diff_matrix(self.rho)
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        self.X_edge = {} 
        self.fit_boundary()
        
        # 自适应预条件权重
        self.res_scales = np.ones(22) # 21物理 + 1惩罚
        self._initialize_scaling()

    def _get_legendre_diff_matrix(self, x):
        """计算非均匀节点下的高精度谱微分矩阵"""
        n = len(x)
        D = np.zeros((n, n))
        # 计算重心权重
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] *= (x[i] - x[j])
        w = 1.0 / w
        # 构造矩阵
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, :])
        return D

    def _initialize_scaling(self):
        print(">>> 正在执行物理量纲归一化预条件...")
        x0 = np.zeros(21); x0[15] = 0.05
        raw_res = self.compute_physics(x0, apply_scaling=False)
        self.res_scales = np.maximum(np.abs(raw_res), 1e-2)

    def get_profiles(self, rho):
        """无量纲化剖面: P已相对于 B0^2/mu_0 归一化"""
        beta_0 = 0.02 # 假设中心 beta 约为 2%
        P = beta_0 * (rho**2 - 1)**2
        dP_drho = beta_0 * 4 * rho * (rho**2 - 1)
        Phi_prime = 2 * rho * self.Phi_a
        iota = 1.0 + 1.5 * rho**2
        psi_prime = iota * Phi_prime
        return P, dP_drho, psi_prime, Phi_prime

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self):
        th_f = np.linspace(0, 2*np.pi, 40); ze_f = np.linspace(0, 2*np.pi, 20)
        TH_F, ZE_F = np.meshgrid(th_f, ze_f)
        R_target = 10 - np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
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
        # Lambda 的包络采用 rho^2 以消除磁轴奇异性
        L_fac = rho**2 * fac
        
        c0R = (self.X_edge['c00'] + x_core[0]*fac) + (self.X_edge['c01c'] + x_core[1]*fac)*np.cos(zeta) + (self.X_edge['c01s'] + x_core[2]*fac)*np.sin(zeta)
        c0Z = (self.X_edge['c00z'] + x_core[3]*fac) + (self.X_edge['c01cz'] + x_core[4]*fac)*np.cos(zeta) + (self.X_edge['c01sz'] + x_core[5]*fac)*np.sin(zeta)
        h = (self.X_edge['h0'] + x_core[6]*fac) + (self.X_edge['h1c'] + x_core[7]*fac)*np.cos(zeta) + (self.X_edge['h1s'] + x_core[8]*fac)*np.sin(zeta)
        v = (self.X_edge['v0'] + x_core[9]*fac) + (self.X_edge['v1c'] + x_core[10]*fac)*np.cos(zeta) + (self.X_edge['v1s'] + x_core[11]*fac)*np.sin(zeta)
        a = (self.X_edge['a0'] + x_core[15]*fac) + (self.X_edge['a1c'] + x_core[16]*fac)*np.cos(zeta) + (self.X_edge['a1s'] + x_core[17]*fac)*np.sin(zeta)
        k = (self.X_edge['k0'] + x_core[12]*fac) + (self.X_edge['k1c'] + x_core[13]*fac)*np.cos(zeta) + (self.X_edge['k1s'] + x_core[14]*fac)*np.sin(zeta)
        
        R = self.X_edge['R0'] + h + rho * a * np.cos(theta + c0R)
        Z = self.X_edge['Z0'] + v + k * rho * a * np.sin(theta + c0Z)
        Lambda = L_fac * (x_core[18] * np.sin(theta + zeta) + x_core[19] * np.sin(theta) + x_core[20] * np.sin(theta - zeta))
        return R, Z, theta+c0R, theta+c0Z, a, k, Lambda

    def compute_physics(self, x_core, apply_scaling=True):
        """核心物理计算：谱微分矩阵 + 真实雅可比 + 几何惩罚"""
        rho, theta, zeta = self.RHO, self.TH, self.ZE
        fac, dfac = 1.0 - rho**2, -2.0 * rho
        L_fac = rho**2 * fac
        
        def periodic_grad(f, step, axis):
            return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * step)

        # 1. 解析重构几何导数
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
        Zr = vr + kr*rho*a*np.sin(thZ) + k*a*np.sin(thZ) + k*rho*ar*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zr
        Zt = k*rho*a*np.cos(thZ)

        # 2. 真实度规计算
        det_J = Rr * Zt - Rt * Zr
        det_phys = np.sign(det_J) * np.maximum(np.abs(det_J), 1e-8)
        sqrt_g = (R / self.Nt) * det_phys
        
        # 3. 磁场计算
        L1n1, L10, L11 = x_core[18:21]
        Lambda = L_fac * (L1n1 * np.sin(theta + zeta) + L10 * np.sin(theta) + L11 * np.sin(theta - zeta))
        Lt = L_fac * (L1n1 * np.cos(theta + zeta) + L10 * np.cos(theta) + L11 * np.cos(theta - zeta))
        Lz = L_fac * (L1n1 * np.cos(theta + zeta) - L11 * np.cos(theta - zeta))

        P, dP, psip, Phip = self.get_profiles(rho)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)

        g_tt, g_rt = Rt**2 + Zt**2, Rr*Rt+Zr*Zt
        g_zz = (R/self.Nt)**2
        Bt_sub, Bz_sub, Br_sub = g_tt * Bt_sup, g_zz * Bz_sup, g_rt * Bt_sup

        # --- 谱微分矩阵 D 计算径向导数 ---
        dBt_drho = np.tensordot(self.D_matrix, Bt_sub, axes=(1, 0))
        dBz_drho = np.tensordot(self.D_matrix, Bz_sub, axes=(1, 0))

        Jz_sup = (dBt_drho - periodic_grad(Br_sub, self.dtheta, axis=1)) / det_phys
        Jt_sup = (periodic_grad(Br_sub, self.dzeta, axis=2) - dBz_drho) / det_phys
        Jr_sup = (periodic_grad(Bz_sub, self.dtheta, axis=1) - periodic_grad(Bt_sub, self.dzeta, axis=2)) / det_phys

        # 4. 平衡算子 (无量纲单位)
        G_rho = dP - sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup)
        rho_R, rho_Z = Zt/det_phys, -Rt/det_phys
        GR = (rho_R * G_rho + (Jr_sup / (2 * np.pi)) * (-Zr/det_phys * Phip))
        GZ = (rho_Z * G_rho + (Jr_sup / (2 * np.pi)) * (Rr/det_phys * Phip))

        # 5. 残差构造
        residuals = []
        kernels = [
            -fac * rho * a * np.sin(thR), -fac * rho * a * np.sin(thR) * cz, -fac * rho * a * np.sin(thR) * sz, 
            fac * k * rho * a * np.cos(thZ), fac * k * rho * a * np.cos(thZ) * cz, fac * k * rho * a * np.cos(thZ) * sz, 
            fac, fac * cz, fac * sz, fac, fac * cz, fac * sz, 
            rho * a * np.sin(thZ), rho * a * np.sin(thZ) * cz, rho * a * np.sin(thZ) * sz, 
            rho * np.cos(thR), rho * np.cos(thR) * cz, rho * np.cos(thR) * sz
        ]
        
        dV = self.dtheta * self.dzeta
        for i in range(18):
            term = GR * kernels[i] if (i<3 or (6<=i<9) or i>=15) else GZ * kernels[i]
            residuals.append(np.sum(sqrt_g * term * self.weights_3d) * dV)

        for m, n in [(1, -1), (1, 0), (1, 1)]:
            res_L = np.sum(sqrt_g * Jr_sup * np.sin(m * self.TH + n * self.ZE) * self.weights_3d) * dV
            residuals.append(res_L)
            
        final_res = np.array(residuals)
        
        # 几何约束作为惩罚项
        if np.any(det_J <= 0):
            violation = np.sum(np.where(det_J <= 1e-4, (1e-4 - det_J)**2, 0))
            final_res *= 1.0 + 10.0 * violation 
            
        if apply_scaling:
            return final_res / self.res_scales
        return final_res

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度物理求解器 (谱导数矩阵 + 奇异性抑制)...")
        x0 = np.zeros(21); x0[15] = 0.05
        res = least_squares(self.compute_physics, x0, method='trf', xtol=1e-8, ftol=1e-8, max_nfev=500)
        print(f">>> 最终残差范数: {np.linalg.norm(res.fun):.4e}")
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
        """可视化输出，包含对应的输入 LCFS 目标曲线"""
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘图网格
        rp = np.linspace(0, 1, 50); tp = np.linspace(0, 2*np.pi, 100)
        R_P, T_P = np.meshgrid(rp, tp); PSI_P = self.compute_psi(R_P)
        
        for i, zv in enumerate(zetas):
            ax = axes[i]
            # 计算计算得到的磁面
            Rm, Zm = [], []
            for r, t in zip(R_P.flatten(), T_P.flatten()):
                res_g = self.compute_geometry(x_core, r, t, zv)
                Rm.append(res_g[0]); Zm.append(res_g[1])
            Rm = np.array(Rm).reshape(R_P.shape); Zm = np.array(Zm).reshape(R_P.shape)
            
            # 画出磁通分布图
            im = ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            
            # 画出计算得到的磁面等高线 (rho = 0.2, 0.4, 0.6, 0.8, 1.0)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, color='white', lw=1.2, alpha=0.6)
            
            # --- 核心改进：在每个 zeta 上绘制对应的输入 LCFS 目标曲线 (Red Dashed) ---
            th_target = np.linspace(0, 2*np.pi, 200)
            R_target_line = 10 - np.cos(th_target) - 0.3 * np.cos(th_target + zv)
            Z_target_line = np.sin(th_target) - 0.3 * np.sin(th_target + zv)
            ax.plot(R_target_line, Z_target_line, 'r--', lw=1.5, label='Input LCFS', alpha=0.9)
            
            # 强调计算出来的最外层闭合磁面 (Gold)
            rl_edge, zl_edge = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_edge, zl_edge, color='#FFD700', lw=2.0, label='Solved Boundary')

            ax.set_aspect('equal')
            ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$')
            if i == 0: ax.legend(loc='upper right', fontsize='small')

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
