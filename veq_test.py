import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class VEQ3D_Solver:
    def __init__(self):
        # 1. 物理常数与输入
        self.Nt = 1                  # 周期数
        self.Phi_a = 1.0             # 总环向磁通标量
        
        # 2. 离散化网格 (增加网格密度)
        self.Nr = 10                  
        self.Nt_grid = 16            
        self.Nz_grid = 12             
        
        rho_nodes, self.rho_weights = roots_legendre(self.Nr)
        # 严格控制径向范围，确保在 (0, 1) 闭区间内
        self.rho = 0.5 * (rho_nodes + 1) * 0.98 + 0.01 
        self.rho_weights *= 0.49 
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')

        # 3. 边界常数容器 (将通过 fit_boundary 自动填充)
        self.X_edge = {} 
        self.fit_boundary()

    def get_profiles(self, rho):
        """第7节: 磁通与压强剖面映射"""
        P = 1.8e4 * (rho**2 - 1)**2
        dP_drho = 1.8e4 * 4 * rho * (rho**2 - 1)
        Phi_prime = 2 * rho * self.Phi_a
        iota_rho = 1.0 + 1.5 * rho**2
        psi_prime = iota_rho * Phi_prime
        return P, dP_drho, psi_prime, Phi_prime

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self):
        """
        参考 fit_edge.py 的逻辑：
        将第5节目标边界拟合到 VEQ-3D 的 20 个边界参数中
        """
        print(">>> 正在进行边界位形预拟合 (Spectral Refinement)...")
        
        # 目标点云
        th_fit = np.linspace(0, 2*np.pi, 30)
        ze_fit = np.linspace(0, 2*np.pi, 20)
        TH_F, ZE_F = np.meshgrid(th_fit, ze_fit)
        
        # 第5节目标方程
        R_target = 10 - 0.3 * np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)

        def boundary_residuals(p):
            R0, Z0 = p[0], p[1]
            # p[2:20] 对应 h0, h1c, h1s, v0, v1c, v1s, a0, a1c, a1s, k0, k1c, k1s, c00, c01c, c01s, c00z, c01cz, c01sz
            # 简化展开
            h = p[2] + p[3]*np.cos(ZE_F) + p[4]*np.sin(ZE_F)
            v = p[5] + p[6]*np.cos(ZE_F) + p[7]*np.sin(ZE_F)
            a = p[8] + p[9]*np.cos(ZE_F) + p[10]*np.sin(ZE_F)
            k = p[11] + p[12]*np.cos(ZE_F) + p[13]*np.sin(ZE_F)
            c0R = p[14] + p[15]*np.cos(ZE_F) + p[16]*np.sin(ZE_F)
            c0Z = p[17] + p[18]*np.cos(ZE_F) + p[19]*np.sin(ZE_F)
            
            tR, tZ = TH_F + c0R, TH_F + c0Z
            R_mod = R0 + h + a * np.cos(tR)
            Z_mod = Z0 + v + k * a * np.sin(tZ)
            return np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])

        # 初始猜想
        p0 = np.zeros(20)
        p0[0], p0[8], p0[11] = 10.0, 0.3, 1.0 # R0, a0, k0
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        p_opt = res.x
        
        # 填充到 X_edge 字典
        keys = ['R0', 'Z0', 'h0', 'h1c', 'h1s', 'v0', 'v1c', 'v1s', 'a0', 'a1c', 'a1s', 
                'k0', 'k1c', 'k1s', 'c00', 'c01c', 'c01s', 'c00z', 'c01cz', 'c01sz']
        self.X_edge = {k: v for k, v in zip(keys, p_opt)}
        print(f"    边界拟合完成，Cost = {res.cost:.4e}")

    def compute_geometry(self, x_core, rho, theta, zeta):
        """三维坐标重构 - 结合 X_edge 和 x_core (1-rho^2)"""
        fac = (1.0 - rho**2)
        
        # 18个核心偏移参数映射
        h = (self.X_edge['h0'] + x_core[6]*fac) + (self.X_edge['h1c'] + x_core[7]*fac)*np.cos(zeta) + (self.X_edge['h1s'] + x_core[8]*fac)*np.sin(zeta)
        v = (self.X_edge['v0'] + x_core[9]*fac) + (self.X_edge['v1c'] + x_core[10]*fac)*np.cos(zeta) + (self.X_edge['v1s'] + x_core[11]*fac)*np.sin(zeta)
        a = (self.X_edge['a0'] + x_core[15]*fac) + (self.X_edge['a1c'] + x_core[16]*fac)*np.cos(zeta) + (self.X_edge['a1s'] + x_core[17]*fac)*np.sin(zeta)
        k = (self.X_edge['k0'] + x_core[12]*fac) + (self.X_edge['k1c'] + x_core[13]*fac)*np.cos(zeta) + (self.X_edge['k1s'] + x_core[14]*fac)*np.sin(zeta)
        
        c0R = (self.X_edge['c00'] + x_core[0]*fac) + (self.X_edge['c01c'] + x_core[1]*fac)*np.cos(zeta) + (self.X_edge['c01s'] + x_core[2]*fac)*np.sin(zeta)
        c0Z = (self.X_edge['c00z'] + x_core[3]*fac) + (self.X_edge['c01cz'] + x_core[4]*fac)*np.cos(zeta) + (self.X_edge['c01sz'] + x_core[5]*fac)*np.sin(zeta)
        
        thR, thZ = theta + c0R, theta + c0Z
        
        R = self.X_edge['R0'] + h + rho * a * np.cos(thR)
        Z = self.X_edge['Z0'] + v + k * rho * a * np.sin(thZ)
        
        # 流函数 Lambda (x_core[18:21])
        Lambda = fac * (x_core[18] * np.sin(theta + zeta) + x_core[19] * np.sin(theta) + x_core[20] * np.sin(theta - zeta))
        
        return R, Z, thR, thZ, a, k, Lambda

    def compute_physics(self, x_core):
        """核心物理残差计算"""
        eps = 1e-5
        def get_geo(x, r, t, z): return self.compute_geometry(x, r, t, z)

        # 1. 坐标与雅可比
        R, Z, thR, thZ, a_v, k_v, L = get_geo(x_core, self.RHO, self.TH, self.ZE)
        
        # 数值导数
        Rr = (get_geo(x_core, self.RHO+eps, self.TH, self.ZE)[0] - get_geo(x_core, self.RHO-eps, self.TH, self.ZE)[0])/(2*eps)
        Rt = (get_geo(x_core, self.RHO, self.TH+eps, self.ZE)[0] - get_geo(x_core, self.RHO, self.TH-eps, self.ZE)[0])/(2*eps)
        Zr = (get_geo(x_core, self.RHO+eps, self.TH, self.ZE)[1] - get_geo(x_core, self.RHO-eps, self.TH, self.ZE)[1])/(2*eps)
        Zt = (get_geo(x_core, self.RHO, self.TH+eps, self.ZE)[1] - get_geo(x_core, self.RHO, self.TH-eps, self.ZE)[1])/(2*eps)
        
        Lt = (get_geo(x_core, self.RHO, self.TH+eps, self.ZE)[6] - get_geo(x_core, self.RHO, self.TH-eps, self.ZE)[6])/(2*eps)
        Lz = np.gradient(L, axis=2) / (self.zeta[1]-self.zeta[0]) if self.Nz_grid > 1 else 0

        det = Rr * Zt - Rt * Zr
        
        # 单调性保护：如果 det 变小，说明磁面交叉
        min_det = 1e-8
        penalty = np.sum(np.where(det < min_det, 1e6, 0.0))
        det_safe = np.where(det < min_det, min_det, det)
        sqrt_g = (R / self.Nt) * det_safe
        
        # 2. 磁场与电流
        g_tt, g_zz = Rt**2 + Zt**2, (R/self.Nt)**2
        g_rt = Rr*Rt + Zr*Zt

        P, dP, psip, Phip = self.get_profiles(self.RHO)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)
        
        Bt_sub = g_tt * Bt_sup; Br_sub = g_rt * Bt_sup; Bz_sub = g_zz * Bz_sup

        dBt_dr = np.gradient(Bt_sub, axis=0) / (self.rho[1]-self.rho[0])
        dBr_dt = np.gradient(Br_sub, axis=1) / (self.theta[1]-self.theta[0])
        Jz_sup = (dBt_dr - dBr_dt) / (2 * np.pi * sqrt_g)
        Jr_sup = (np.gradient(Bz_sub, axis=1) - np.gradient(Bt_sub, axis=2)) / (2 * np.pi * sqrt_g)

        # 3. 受力平衡归一化
        Fr = ((Jz_sup * Rt) * Bz_sup - dP) * 1e-4
        
        # 4. 投影残差
        fac = (1.0 - self.RHO**2)
        residuals = []
        for i in range(18):
            # 简化演示：投影到径向力
            res_val = np.sum(sqrt_g * Fr * fac * self.rho_weights[:, None, None])
            residuals.append(res_val + penalty)

        for m, n in [(1, -1), (1, 0), (1, 1)]:
            res_L = np.sum(sqrt_g * Jr_sup * np.sin(m * self.TH + n * self.ZE) * self.rho_weights[:, None, None])
            residuals.append(res_L)
            
        return np.array(residuals)

    def solve(self):
        print(">>> 正在启动 VEQ-3D 混合拟合-平衡求解器...")
        # 初值微调
        x0 = np.zeros(21)
        res = least_squares(self.compute_physics, x0, method='trf', xtol=1e-8, ftol=1e-8)
        print(f">>> 求解完成！残差范数: {np.linalg.norm(res.fun):.4f}")
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes = axes.flatten()
        
        rho_plt = np.linspace(0, 1.0, 40)
        th_plt = np.linspace(0, 2*np.pi, 60)
        RHO_P, TH_P = np.meshgrid(rho_plt, th_plt)
        PSI_P = self.compute_psi(RHO_P)
        
        for i, (zeta_val, label) in enumerate(zip(zetas, ['0', r'\pi/3', r'2\pi/3', r'\pi', r'4\pi/3', r'5\pi/3'])):
            ax = axes[i]
            R_m, Z_m = [], []
            for r, t in zip(RHO_P.flatten(), TH_P.flatten()):
                res_g = self.compute_geometry(x_core, r, t, zeta_val)
                R_m.append(res_g[0]); Z_m.append(res_g[1])
            R_m = np.array(R_m).reshape(RHO_P.shape)
            Z_m = np.array(Z_m).reshape(RHO_P.shape)
            
            im = ax.tripcolor(R_m.flatten(), Z_m.flatten(), PSI_P.flatten(), shading='gouraud', cmap='viridis', alpha=0.9)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zeta_val)[:2]
                ax.plot(rl, zl, 'w-', lw=2 if r_lev==1 else 0.8)
            ax.set_title(rf'$\zeta = {label}$'); ax.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    solver = VEQ3D_Solver()
    solver.solve()
