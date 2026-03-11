import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class VEQ3D_Solver:
    def __init__(self):
        # 1. 物理常数 (严格回归 SI 单位制以对齐 DESC)
        self.Nt = 1                  # 周期数
        self.Phi_a = 1.0             # 总环向磁通 (Wb)
        self.mu_0 = 4 * np.pi * 1e-7 # 真空磁导率 (H/m)
        
        # 2. 离散化网格 (Nr=24, 角向网格采用 2^n 以优化 FFT)
        self.Nr = 24                  
        self.Nt_grid = 32            
        self.Nz_grid = 16             
        
        rho_nodes, self.rho_weights = roots_legendre(self.Nr)
        # 映射到 [0, 1] 区间
        self.rho = 0.5 * (rho_nodes + 1)
        self.rho_weights *= 0.5
        
        # 预计算勒让德谱微分矩阵 D (用于径向谱求导)
        self.D_matrix = self._get_legendre_diff_matrix(self.rho)
        
        # 预计算 FFT 波数向量 (用于角向高精度谱求导)
        self.k_th = np.fft.fftfreq(self.Nt_grid, d=1.0/self.Nt_grid)[None, :, None]
        self.k_ze = np.fft.fftfreq(self.Nz_grid, d=1.0/self.Nz_grid)[None, None, :]
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        self.X_edge = {} 
        # 执行边界拟合
        self.fit_boundary()
        
        # 自适应预条件权重
        self.res_scales = np.ones(21)
        self._initialize_scaling()

    def _get_legendre_diff_matrix(self, x):
        """计算非均匀节点下的高精度谱微分矩阵"""
        n = len(x)
        D = np.zeros((n, n))
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j: w[i] *= (x[i] - x[j])
        w = 1.0 / w
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
        self.res_scales = np.maximum(np.abs(raw_res), 1e-3)

    def get_profiles(self, rho):
        """第7节: 恢复 SI 物理压强剖面"""
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
        """预拟合几何边界 (LCFS) - 修正了致命的初始猜值坍缩问题"""
        print(">>> 正在拟合目标边界位形...")
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

        # 索引: 8 是 a0, 11 是 k0, 14 是 c00 (变分初始相位设为 pi)
        p0 = [10.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, np.pi, 0, 0, 0, 0, 0]
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        keys = ['R0', 'Z0', 'h0', 'h1c', 'h1s', 'v0', 'v1c', 'v1s', 'a0', 'a1c', 'a1s', 'k0', 'k1c', 'k1s', 'c00', 'c01c', 'c01s', 'c00z', 'c01cz', 'c01sz']
        self.X_edge = {k: v for k, v in zip(keys, res.x)}
        print(f"    边界预拟合完成，Cost: {res.cost:.4e}")

    def compute_geometry(self, x_core, rho, theta, zeta):
        fac = 1.0 - rho**2
        L_fac = rho**2 * fac # 消除磁轴奇异性
        
        c0R = (self.X_edge['c00'] + x_core[0]*fac) + (self.X_edge['c01c'] + x_core[1]*fac)*np.cos(zeta) + (self.X_edge['c01s'] + x_core[2]*fac)*np.sin(zeta)
        c0Z = (self.X_edge['c00z'] + x_core[3]*fac) + (self.X_edge['c01cz'] + x_core[4]*fac)*np.cos(zeta) + (self.X_edge['c01sz'] + x_core[5]*fac)*np.sin(zeta)
        h = (self.X_edge['h0'] + x_core[6]*fac) + (self.X_edge['h1c'] + x_core[7]*fac)*np.cos(zeta) + (self.X_edge['h1s'] + x_core[8]*fac)*np.sin(zeta)
        v = (self.X_edge['v0'] + x_core[9]*fac) + (self.X_edge['v1c'] + x_core[10]*fac)*np.cos(zeta) + (self.X_edge['v1s'] + x_core[11]*fac)*np.sin(zeta)
        k = (self.X_edge['k0'] + x_core[12]*fac) + (self.X_edge['k1c'] + x_core[13]*fac)*np.cos(zeta) + (self.X_edge['k1s'] + x_core[14]*fac)*np.sin(zeta)
        a = (self.X_edge['a0'] + x_core[15]*fac) + (self.X_edge['a1c'] + x_core[16]*fac)*np.cos(zeta) + (self.X_edge['a1s'] + x_core[17]*fac)*np.sin(zeta)
        
        R = self.X_edge['R0'] + h + rho * a * np.cos(theta + c0R)
        Z = self.X_edge['Z0'] + v + k * rho * a * np.sin(theta + c0Z)
        Lambda = L_fac * (x_core[18] * np.sin(theta + zeta) + x_core[19] * np.sin(theta) + x_core[20] * np.sin(theta - zeta))
        return R, Z, theta+c0R, theta+c0Z, a, k, Lambda

    def compute_physics(self, x_core, apply_scaling=True):
        """核心物理计算：修复 Lt 解析性、雅可比平滑性与物理惩罚"""
        rho, theta, zeta = self.RHO, self.TH, self.ZE
        fac, dfac = 1.0 - rho**2, -2.0 * rho
        L_fac = rho**2 * fac
        cz, sz = np.cos(zeta), np.sin(zeta)
        
        def spectral_grad_th(f):
            return np.real(np.fft.ifft(1j * self.k_th * np.fft.fft(f, axis=1), axis=1))

        def spectral_grad_ze(f):
            return np.real(np.fft.ifft(1j * self.k_ze * np.fft.fft(f, axis=2), axis=2))

        # 1. 解析几何导数
        def eval_derivs(edge, core, cz, sz):
            val = (edge[0]+core[0]*fac) + (edge[1]+core[1]*fac)*cz + (edge[2]+core[2]*fac)*sz
            dr = core[0]*dfac + core[1]*dfac*cz + core[2]*dfac*sz
            dz = -(edge[1]+core[1]*fac)*sz + (edge[2]+core[2]*fac)*cz
            return val, dr, dz

        h, hr, hz = eval_derivs([self.X_edge['h0'], self.X_edge['h1c'], self.X_edge['h1s']], x_core[6:9], cz, sz)
        v, vr, vz = eval_derivs([self.X_edge['v0'], self.X_edge['v1c'], self.X_edge['v1s']], x_core[9:12], cz, sz)
        k, kr, kz = eval_derivs([self.X_edge['k0'], self.X_edge['k1c'], self.X_edge['k1s']], x_core[12:15], cz, sz)
        a, ar, az = eval_derivs([self.X_edge['a0'], self.X_edge['a1c'], self.X_edge['a1s']], x_core[15:18], cz, sz)
        c0R, c0Rr, c0Rz = eval_derivs([self.X_edge['c00'], self.X_edge['c01c'], self.X_edge['c01s']], x_core[0:3], cz, sz)
        c0Z, c0Zr, c0Zz = eval_derivs([self.X_edge['c00z'], self.X_edge['c01cz'], self.X_edge['c01sz']], x_core[3:6], cz, sz)

        thR, thZ = theta + c0R, theta + c0Z
        R = self.X_edge['R0'] + h + rho * a * np.cos(thR)
        Z = self.X_edge['Z0'] + v + k * rho * a * np.sin(thZ)
        
        Rr = hr + a*np.cos(thR) + rho*ar*np.cos(thR) - rho*a*np.sin(thR)*c0Rr
        Rt = -rho*a*np.sin(thR)
        Rz = hz + rho*az*np.cos(thR) - rho*a*np.sin(thR)*c0Rz
        
        Zr = vr + kr*rho*a*np.sin(thZ) + k*a*np.sin(thZ) + k*rho*ar*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zr
        Zt = k*rho*a*np.cos(thZ)
        Zz = vz + kz*rho*a*np.sin(thZ) + k*rho*az*np.sin(thZ) + k*rho*a*np.cos(thZ)*c0Zz

        # 2. 真实度规计算 (使用平滑非零逼近)
        det_phys = Rr * Zt - Rt * Zr
        det_safe = 0.5 * (det_phys + np.sqrt(det_phys**2 + 1e-8))
        sqrt_g = (R / self.Nt) * det_safe
        
        g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
        g_zz = Rz**2 + (R/self.Nt)**2 + Zz**2 
        g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

        # 3. 磁场计算 (Lt 修正为基于原始角坐标计算精确 Lt)
        L1n1, L10, L11 = x_core[18:21]
        Lt = L_fac * (L1n1 * np.cos(theta + zeta) + L10 * np.cos(theta) + L11 * np.cos(theta - zeta))
        Lz = L_fac * (L1n1 * np.cos(theta + zeta) - L11 * np.cos(theta - zeta))
        
        P, dP, psip, Phip = self.get_profiles(rho)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)

        Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
        Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
        Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

        # 4. 电流与平衡算子 (使用谱微分矩阵与 FFT)
        dBt_drho = np.tensordot(self.D_matrix, Bt_sub, axes=(1, 0))
        dBz_drho = np.tensordot(self.D_matrix, Bz_sub, axes=(1, 0))

        Jz_sup = (dBt_drho - spectral_grad_th(Br_sub)) / sqrt_g
        Jt_sup = (spectral_grad_ze(Br_sub) - dBz_drho) / sqrt_g
        Jr_sup = (spectral_grad_th(Bz_sub) - spectral_grad_ze(Bt_sub)) / sqrt_g

        Jr_phys = Jr_sup / self.mu_0
        G_rho = dP - sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / self.mu_0
        rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
        th_R, th_Z = -Zr/det_safe, Rr/det_safe
        GR = (rho_R * G_rho + (Jr_phys / (2 * np.pi)) * (th_R * Phip))
        GZ = (rho_Z * G_rho + (Jr_phys / (2 * np.pi)) * (th_Z * Phip))

        # 5. 变分残差构造
        residuals = []
        kernels = [
            -fac * rho * a * np.sin(thR), -fac * rho * a * np.sin(thR) * cz, -fac * rho * a * np.sin(thR) * sz, # c0R
            fac * k * rho * a * np.cos(thZ), fac * k * rho * a * np.cos(thZ) * cz, fac * k * rho * a * np.cos(thZ) * sz, # c0Z
            fac, fac * cz, fac * sz, # h
            fac, fac * cz, fac * sz, # v
            fac * rho * a * np.sin(thZ), fac * rho * a * np.sin(thZ) * cz, fac * rho * a * np.sin(thZ) * sz, # k 
            fac * rho * np.cos(thR), fac * rho * np.cos(thR) * cz, fac * rho * np.cos(thR) * sz # a 
        ]
        
        dV = self.dtheta * self.dzeta
        for i in range(18):
            if i < 3 or (6 <= i < 9): term = GR * kernels[i]
            elif (3 <= i < 6) or (9 <= i < 15): term = GZ * kernels[i]
            else:
                kR, kZ = kernels[i], fac * k * rho * np.sin(thZ) * (1 if i==15 else (cz if i==16 else sz))
                term = GR * kR + GZ * kZ
            residuals.append(np.sum(sqrt_g * term * self.weights_3d) * dV)

        for m, n in [(1, -1), (1, 0), (1, 1)]:
            test_function = L_fac * np.sin(m * self.TH - n * self.ZE)
            res_L = np.sum(sqrt_g * Jr_phys * test_function * self.weights_3d) * dV
            residuals.append(res_L)
            
        final_res = np.array(residuals)
        if apply_scaling: 
            final_res = final_res / self.res_scales
            
        # --- 核心修复：保护磁轴物理奇点，仅惩罚真实的网格翻转 (det_phys < -1e-4) ---
        if np.any(det_phys < -1e-4):
            # 采用乘性惩罚，避免广播机制破坏残差独立性
            penalty = np.sum(np.where(det_phys < -1e-4, 100.0 * (-1e-4 - det_phys)**2, 0))
            final_res = final_res * (1.0 + penalty)  
            
        return final_res

    def print_final_parameters(self, x_core):
        """按照 VEQ-3D 命名规则输出所有拟合参数最后的值"""
        print("\n" + "="*60)
        print(f"{'VEQ-3D 拟合参数报告 (Final Parameters)':^60}")
        print("="*60)
        
        # 1. 边界参数 (X_edge)
        print(f"\n[1] 边界固定参数 (X_edge - LCFS Definition):")
        print("-" * 60)
        edge_map = [
            ("R0", "大半径中心偏移"), ("Z0", "垂直中心偏移"),
            ("h0", "水平位移常数 harmonic"), ("h1c", "水平位移 cos(zeta)"), ("h1s", "水平位移 sin(zeta)"),
            ("v0", "垂直位移常数 harmonic"), ("v1c", "垂直位移 cos(zeta)"), ("v1s", "垂直位移 sin(zeta)"),
            ("a0", "截面小半径常数"), ("a1c", "截面小半径 cos(zeta)"), ("a1s", "截面小半径 sin(zeta)"),
            ("k0", "拉长比常数"), ("k1c", "拉长比 cos(zeta)"), ("k1s", "拉长比 sin(zeta)"),
            ("c00", "R变分角相位常数"), ("c01c", "R变分角 cos(zeta)"), ("c01s", "R变分角 sin(zeta)"),
            ("c00z", "Z变分角相位常数"), ("c01cz", "Z变分角 cos(zeta)"), ("c01sz", "Z变分角 sin(zeta)")
        ]
        for key, desc in edge_map:
            val = self.X_edge.get(key, 0.0)
            print(f"{key:<8} | {val:>15.8e} | {desc}")

        # 2. 核心待求参数 (X_core - Radial Variation Coefficients)
        print(f"\n[2] 核心平衡参数 (X_core - Coefficients of (1-rho^2)):")
        print("-" * 60)
        core_names = [
            "c00_1", "c01c_1", "c01s_1",       # c0R radial coefs
            "c00z_1", "c01cz_1", "c01sz_1",    # c0Z radial coefs
            "h0_1", "h1c_1", "h1s_1",          # h radial coefs
            "v0_1", "v1c_1", "v1s_1",          # v radial coefs
            "k0_1", "k1c_1", "k1s_1",          # k radial coefs
            "a0_1", "a1c_1", "a1s_1"           # a radial coefs
        ]
        for i, name in enumerate(core_names):
            print(f"{name:<8} | {x_core[i]:>15.8e} | {name.split('_')[0]} 的径向演化系数")

        # 3. 流函数参数 (Lambda_mn)
        print(f"\n[3] 流函数参数 (Lambda_mn - Coefficients of rho^2(1-rho^2)):")
        print("-" * 60)
        lambda_modes = [("1,-1", "L_1n1_1"), ("1,0", "L_10_1"), ("1,1", "L_11_1")]
        for i, (mode, name) in enumerate(lambda_modes):
            val = x_core[18+i]
            print(f"{name:<8} | {val:>15.8e} | (m,n)=({mode}) 谐波振幅")
            
        print("="*60 + "\n")

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度平衡求解器 (修复磁轴惩罚与几何初值)...")
        x0 = np.zeros(21); x0[15] = 0.05
        res = least_squares(self.compute_physics, x0, method='trf', xtol=1e-11, ftol=1e-11, max_nfev=500)
        
        print(f">>> 最终收敛范数: {np.linalg.norm(res.fun):.4e}")
        
        # 输出最终参数
        self.print_final_parameters(res.x)
        
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
        """可视化输出，包含精确的输入边界参考"""
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        rp = np.linspace(0, 1, 50); tp = np.linspace(0, 2*np.pi, 100)
        R_P, T_P = np.meshgrid(rp, tp); PSI_P = self.compute_psi(R_P)
        for i, zv in enumerate(zetas):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(R_P.flatten(), T_P.flatten()):
                rg = self.compute_geometry(x_core, r, t, zv)
                Rm.append(rg[0]); Zm.append(rg[1])
            Rm = np.array(Rm).reshape(R_P.shape); Zm = np.array(Zm).reshape(R_P.shape)
            ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, color='white', lw=1.0, alpha=0.5)
            # 绘制输入 LCFS (红色虚线)
            th_t = np.linspace(0, 2*np.pi, 200)
            ax.plot(10 - np.cos(th_t) - 0.3*np.cos(th_t+zv), np.sin(th_t) - 0.3*np.sin(th_t+zv), 'r--', lw=1.5, label='Input LCFS')
            rl_e, zl_e = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_e, zl_e, color='#FFD700', lw=2.0, label='Solved Boundary')
            ax.set_aspect('equal'); ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$'); 
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
