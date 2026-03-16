import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time

class VEQ3D_Solver:
    def __init__(self):
        # 初始默认参数 (实际目标求解阶数通过 solve() 函数指定)
        self.M_pol = 0                
        self.N_tor = 1                
        self.L_rad = 1               
        
        # 1. 物理常数 (严格回归 SI 单位制以对齐 DESC)
        self.Nt = 19                  
        self.Phi_a = 1.0             
        self.mu_0 = 4 * np.pi * 1e-7 
        
        # 2. 离散化网格 (Nr=24, 角向网格采用 2^n 以优化 FFT)
        self.Nr = 24                  
        self.Nt_grid = 32            
        self.Nz_grid = 16             
        
        # 初始化动态模式信息与参数长度
        self._setup_modes()
        
        # 径向积分节点: Chebyshev-Fejér 节点
        rho_nodes, self.rho_weights = self._get_chebyshev_nodes_and_weights(self.Nr)
        self.rho = 0.5 * (rho_nodes + 1)
        self.rho_weights *= 0.5
        
        self.D_matrix = self._get_spectral_diff_matrix(self.rho)
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        # 【性能优化 A】: 构造等效于精确 FFT 求导的矩阵算子
        self._build_spectral_matrices()
        
        # 【性能优化 B】: 预计算所有不变的径向和角向基底矩阵
        self._build_basis_matrices()
        self._build_radial_matrices()

        self.p_edge = None
        # 执行初始阶段的边界拟合
        self.fit_boundary()

    def _setup_modes(self):
        """解析并计算当前 M, N, L 设定下的参数维度分布"""
        self.len_1d = 1 + 2 * self.N_tor
        
        self.modes_2d = []
        for m in range(1, self.M_pol + 1):
            for n in range(-self.N_tor, self.N_tor + 1):
                self.modes_2d.append((m, n, 'c'))
                self.modes_2d.append((m, n, 's'))
        self.len_2d = len(self.modes_2d)
        
        self.lambda_modes = []
        for n in range(1, self.N_tor + 1):
            self.lambda_modes.append((0, n))
        for m in range(1, self.M_pol + 2):
            for n in range(-self.N_tor, self.N_tor + 1):
                self.lambda_modes.append((m, n))
        self.len_lam = len(self.lambda_modes)
        
        self.num_geom_params = (6 * self.len_1d + 2 * self.len_2d) * self.L_rad
        self.num_core_params = self.num_geom_params + self.len_lam * self.L_rad
        self.num_edge_params = 2 + 6 * self.len_1d + 2 * self.len_2d

    def _build_spectral_matrices(self):
        """利用单位阵的逆傅里叶变换提取精确的纯实数谱导数矩阵算子"""
        I_th = np.eye(self.Nt_grid)
        self.D_th_matrix = np.real(np.fft.ifft(1j * np.fft.fftfreq(self.Nt_grid, d=1.0/self.Nt_grid)[:, None] * np.fft.fft(I_th, axis=0), axis=0))
        
        I_ze = np.eye(self.Nz_grid)
        self.D_ze_matrix = np.real(np.fft.ifft(1j * np.fft.fftfreq(self.Nz_grid, d=1.0/self.Nz_grid)[:, None] * np.fft.fft(I_ze, axis=0), axis=0))

    def _build_basis_matrices(self):
        """预计算各维度的傅里叶基底及其解析导数"""
        self.basis_1d_val = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_dz  = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_val[0, 0, 0, :] = 1.0
        idx = 1
        for n in range(1, self.N_tor + 1):
            self.basis_1d_val[idx, 0, 0, :] = np.cos(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] = -n * np.sin(n * self.ZE[0,0,:]); idx+=1
            self.basis_1d_val[idx, 0, 0, :] = np.sin(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] =  n * np.cos(n * self.ZE[0,0,:]); idx+=1
            
        self.basis_2d_val = np.zeros((self.len_2d, 1, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dth = np.zeros((self.len_2d, 1, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dze = np.zeros((self.len_2d, 1, self.Nt_grid, self.Nz_grid))
        for i, (m, n, typ) in enumerate(self.modes_2d):
            phase = m * self.TH[0,:,:] - n * self.ZE[0,:,:]
            if typ == 'c':
                self.basis_2d_val[i, 0, :, :] = np.cos(phase); self.basis_2d_dth[i, 0, :, :] = -m * np.sin(phase); self.basis_2d_dze[i, 0, :, :] =  n * np.sin(phase)
            else:
                self.basis_2d_val[i, 0, :, :] = np.sin(phase); self.basis_2d_dth[i, 0, :, :] =  m * np.cos(phase); self.basis_2d_dze[i, 0, :, :] = -n * np.cos(phase)
                
        self.basis_lam_val = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        self.basis_lam_dth = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        self.basis_lam_dze = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        for i, (m, n) in enumerate(self.lambda_modes):
            phase = m * self.TH[0,:,:] - n * self.ZE[0,:,:]
            self.basis_lam_val[i, 0, :, :] = np.sin(phase)
            self.basis_lam_dth[i, 0, :, :] = m * np.cos(phase)
            self.basis_lam_dze[i, 0, :, :] = -n * np.cos(phase)

    def _build_radial_matrices(self):
        """【极致提速】: 将原本计算于目标函数内部的切比雪夫常数张量提取出来预计算，消灭冗余差分算力浪费"""
        rho_1d = self.rho
        x = 2.0 * rho_1d**2 - 1.0
        T = np.zeros((self.L_rad,) + x.shape)
        dTdx = np.zeros((self.L_rad,) + x.shape)
        
        if self.L_rad > 0: 
            T[0] = 1.0
            dTdx[0] = 0.0
        if self.L_rad > 1: 
            T[1] = x
            dTdx[1] = 1.0
        for l in range(2, self.L_rad): 
            T[l] = 2.0 * x * T[l-1] - T[l-2]
            dTdx[l] = 2.0 * T[l-1] + 2.0 * x * dTdx[l-1] - dTdx[l-2]
            
        self.fac_rad = (1.0 - rho_1d**2) * T
        self.dfac_rad = -2.0 * rho_1d * T + 4.0 * rho_1d * (1.0 - rho_1d**2) * dTdx
        self.L_fac_rad = rho_1d[None, :] * self.fac_rad
        
        self.vol_element_base = self.weights_3d * self.dtheta * self.dzeta

    def unpack_edge(self):
        p = self.p_edge; idx = 2
        def get(L): nonlocal idx; c = p[idx:idx+L]; idx+=L; return c
        return p[0], p[1], get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_2d), get(self.len_2d)

    def unpack_core(self, x_core):
        idx = 0
        def get(L): 
            nonlocal idx
            c = x_core[idx:idx+L*self.L_rad].reshape(self.L_rad, L)
            idx += L * self.L_rad
            return c
        return get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_2d), get(self.len_2d), get(self.len_lam)

    def _upgrade_edge_params(self, old_edge, m_old, m_new):
        if m_old == m_new: return old_edge.copy()
        
        base_len = 2 + 6 * self.len_1d
        base = old_edge[:base_len]
        
        len_2d_old = 2 * m_old * (2 * self.N_tor + 1)
        len_2d_new = 2 * m_new * (2 * self.N_tor + 1)
        
        tR_old = old_edge[base_len : base_len + len_2d_old] if len_2d_old > 0 else np.array([])
        tZ_old = old_edge[base_len + len_2d_old : base_len + 2 * len_2d_old] if len_2d_old > 0 else np.array([])
        
        len_add = len_2d_new - len_2d_old
        tR_new = np.concatenate([tR_old, np.zeros(len_add)]) if len_add > 0 else tR_old.copy()
        tZ_new = np.concatenate([tZ_old, np.zeros(len_add)]) if len_add > 0 else tZ_old.copy()
        
        return np.concatenate([base, tR_new, tZ_new])

    def _upgrade_core_params(self, old_params, m_old, m_new):
        if m_old == m_new: return old_params.copy()
        
        len_2d_old = 2 * m_old * (2 * self.N_tor + 1)
        len_lam_old = self.N_tor + (m_old + 1) * (2 * self.N_tor + 1)
        
        idx = 0
        def get_old(L):
            nonlocal idx
            c = old_params[idx:idx+L*self.L_rad].reshape(self.L_rad, L)
            idx += L * self.L_rad
            return c
            
        c0R = get_old(self.len_1d)
        c0Z = get_old(self.len_1d)
        h = get_old(self.len_1d)
        v = get_old(self.len_1d)
        k = get_old(self.len_1d)
        a = get_old(self.len_1d)
        tR_old = get_old(len_2d_old)
        tZ_old = get_old(len_2d_old)
        lam_old = get_old(len_lam_old)
        
        len_2d_new = 2 * m_new * (2 * self.N_tor + 1)
        len_lam_new = self.N_tor + (m_new + 1) * (2 * self.N_tor + 1)
        
        tR_new = np.zeros((self.L_rad, len_2d_new))
        tZ_new = np.zeros((self.L_rad, len_2d_new))
        lam_new = np.zeros((self.L_rad, len_lam_new))
        
        if len_2d_old > 0:
            tR_new[:, :len_2d_old] = tR_old
            tZ_new[:, :len_2d_old] = tZ_old
        lam_new[:, :len_lam_old] = lam_old
        
        return np.concatenate([
            c0R.flatten(), c0Z.flatten(), h.flatten(), v.flatten(), k.flatten(), a.flatten(),
            tR_new.flatten(), tZ_new.flatten(), lam_new.flatten()
        ])

    def _get_chebyshev_nodes_and_weights(self, N):
        k = np.arange(N)
        theta = (2 * (N - 1 - k) + 1) * np.pi / (2 * N)
        x_nodes = np.cos(theta)
        w_nodes = np.zeros(N)
        for i in range(N):
            sum_term = 0.0
            for j in range(1, N // 2 + 1):
                sum_term += 2.0 * np.cos(2 * j * theta[i]) / (4 * j**2 - 1)
            w_nodes[i] = (2.0 / N) * (1.0 - sum_term)
        return x_nodes, w_nodes

    def _get_spectral_diff_matrix(self, x):
        n = len(x); D = np.zeros((n, n)); w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j: w[i] *= (x[i] - x[j])
        w = 1.0 / w
        for i in range(n):
            for j in range(n):
                if i != j: D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, :])
        return D

    def get_profiles(self, rho):
        # 【物理失真修复】: 降低中心压强，解除高比压带来的无解极端 Shafranov 位移
        P_scale = 1.0e3 
        P = P_scale * (rho**2 - 1)**2
        dP_drho = P_scale * 4 * rho * (rho**2 - 1)
        Phi_prime = 2 * rho * self.Phi_a
        iota = 1.0 + 1.5 * rho**2
        psi_prime = iota * Phi_prime
        return P, dP_drho, psi_prime, Phi_prime

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self, use_existing_as_guess=False):
        print(f">>> 正在预拟合目标边界位形(LCFS)...")
        TH_F, ZE_F = self.TH[0], self.ZE[0]
        R_target = 10 - np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)
        
        def eval_1d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        def eval_2d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_2d_val[:, 0, ...], axis=0) if self.len_2d > 0 else 0.0

        def boundary_residuals(p):
            R0, Z0 = p[0], p[1]
            idx = 2
            def get(L): nonlocal idx; c = p[idx:idx+L]; idx+=L; return c
            c0R_v = eval_1d_edge(get(self.len_1d))
            c0Z_v = eval_1d_edge(get(self.len_1d))
            h_c, v_c = get(self.len_1d), get(self.len_1d)
            h_v, v_v = eval_1d_edge(h_c), eval_1d_edge(v_c)
            k_v, a_v = eval_1d_edge(get(self.len_1d)), eval_1d_edge(get(self.len_1d))
            tR_v, tZ_v = eval_2d_edge(get(self.len_2d)), eval_2d_edge(get(self.len_2d))
            
            thR = TH_F + c0R_v + tR_v
            thZ = TH_F + c0Z_v + tZ_v
            R_mod = R0 + h_v + a_v * np.cos(thR)
            Z_mod = Z0 + v_v + k_v * a_v * np.sin(thZ)
            
            res_geom = np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        p0 = np.zeros(self.num_edge_params)
        if use_existing_as_guess and self.p_edge is not None and len(self.p_edge) == self.num_edge_params:
            p0 = self.p_edge.copy()
        else:
            p0[0] = 10.0 
            p0[2] = np.pi 
            p0[2 + 4 * self.len_1d] = 1.0 
            p0[2 + 5 * self.len_1d] = 1.0 
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x
        print(f"    边界 LCFS 投影完毕，极小 Cost: {res.cost:.4e}")

    def compute_geometry(self, x_core, rho, theta, zeta):
        rho, theta, zeta = np.atleast_1d(rho), np.atleast_1d(theta), np.atleast_1d(zeta)
        base_grid = rho + theta + zeta
        
        x = 2.0 * rho**2 - 1.0
        T = np.zeros((self.L_rad,) + x.shape)
        if self.L_rad > 0: T[0] = 1.0
        if self.L_rad > 1: T[1] = x
        for l in range(2, self.L_rad):
            T[l] = 2.0 * x * T[l-1] - T[l-2]
            
        fac_rad = (1.0 - rho**2) * T  
        L_fac_rad = rho * fac_rad     
        
        e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        def ev_1d(c_e, c_c):
            val = c_e[0] + np.tensordot(c_c[:, 0], fac_rad, axes=(0, 0))
            val = val + np.zeros_like(base_grid)  
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_n = c_e[idx]   + np.tensordot(c_c[:, idx],   fac_rad, axes=(0, 0))
                s_n = c_e[idx+1] + np.tensordot(c_c[:, idx+1], fac_rad, axes=(0, 0))
                val = val + c_n * np.cos(n * zeta) + s_n * np.sin(n * zeta)
                idx += 2
            return val
            
        def ev_2d(c_e, c_c):
            val = np.zeros_like(base_grid)
            if self.len_2d == 0: return val
            for i, (m, n, typ) in enumerate(self.modes_2d):
                b = np.cos(m * theta - n * zeta) if typ == 'c' else np.sin(m * theta - n * zeta)
                val = val + (c_e[i] + np.tensordot(c_c[:, i], fac_rad, axes=(0, 0))) * b
            return val

        c0R, c0Z = ev_1d(e_c0R, c_c0R), ev_1d(e_c0Z, c_c0Z)
        h, v = ev_1d(e_h, c_h), ev_1d(e_v, c_v)
        k, a = ev_1d(e_k, c_k), ev_1d(e_a, c_a)
        tR, tZ = ev_2d(e_tR, c_tR), ev_2d(e_tZ, c_tZ)
        
        thR, thZ = theta + c0R + tR, theta + c0Z + tZ
        R = e_R0 + h + rho * a * np.cos(thR)
        Z = e_Z0 + v + k * rho * a * np.sin(thZ)
        
        lam = np.zeros_like(base_grid)
        for i, (m, n) in enumerate(self.lambda_modes):
            lam = lam + np.tensordot(c_lam[:, i], L_fac_rad, axes=(0, 0)) * np.sin(m * theta - n * zeta)
            
        return R, Z, thR, thZ, a, k, lam

    def compute_physics(self, x_core):
        rho, theta, zeta = self.RHO, self.TH, self.ZE
        
        # 直接使用预计算的静态矩阵，极大降低目标函数计算耗时
        fac_rad = self.fac_rad
        dfac_rad = self.dfac_rad
        L_fac_rad = self.L_fac_rad
        
        e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        def spectral_grad_th(f): return np.einsum('tj, rjz -> rtz', self.D_th_matrix, f)
        def spectral_grad_ze(f): return np.einsum('zj, rtj -> rtz', self.D_ze_matrix, f)

        def eval_1d(c_e, c_c):
            core_contrib = np.sum(c_c[:, :, None, None, None] * fac_rad[:, None, :, None, None], axis=0)
            ce_eff = c_e[:, None, None, None] + core_contrib
            val = np.sum(ce_eff * self.basis_1d_val, axis=0)
            dz  = np.sum(ce_eff * self.basis_1d_dz,  axis=0)
            dr_contrib = np.sum(c_c[:, :, None, None, None] * dfac_rad[:, None, :, None, None], axis=0)
            dr  = np.sum(dr_contrib * self.basis_1d_val, axis=0)
            return val, dr, dz

        def eval_2d(c_e, c_c):
            if self.len_2d == 0: return 0.0, 0.0, 0.0, 0.0
            core_contrib = np.sum(c_c[:, :, None, None, None] * fac_rad[:, None, :, None, None], axis=0)
            ce_eff = c_e[:, None, None, None] + core_contrib
            val = np.sum(ce_eff * self.basis_2d_val, axis=0)
            dth = np.sum(ce_eff * self.basis_2d_dth, axis=0)
            dz  = np.sum(ce_eff * self.basis_2d_dze, axis=0)
            dr_contrib = np.sum(c_c[:, :, None, None, None] * dfac_rad[:, None, :, None, None], axis=0)
            dr  = np.sum(dr_contrib * self.basis_2d_val, axis=0)
            return val, dr, dth, dz

        c0R, c0Rr, c0Rz = eval_1d(e_c0R, c_c0R)
        c0Z, c0Zr, c0Zz = eval_1d(e_c0Z, c_c0Z)
        h, hr, hz = eval_1d(e_h, c_h)
        v, vr, vz = eval_1d(e_v, c_v)
        k, kr, kz = eval_1d(e_k, c_k)
        a, ar, az = eval_1d(e_a, c_a)
        
        tR, tRr, tRth, tRz = eval_2d(e_tR, c_tR)
        tZ, tZr, tZth, tZz = eval_2d(e_tZ, c_tZ)
        
        thR, thZ = theta + c0R + tR, theta + c0Z + tZ
        R = e_R0 + h + rho * a * np.cos(thR)
        Z = e_Z0 + v + k * rho * a * np.sin(thZ)
        
        thR_r, thR_th, thR_z = c0Rr + tRr, 1.0 + tRth, c0Rz + tRz
        thZ_r, thZ_th, thZ_z = c0Zr + tZr, 1.0 + tZth, c0Zz + tZz
        
        Rr = hr + a * np.cos(thR) + rho * ar * np.cos(thR) - rho * a * np.sin(thR) * thR_r
        Rt = -rho * a * np.sin(thR) * thR_th
        Rz = hz + rho * az * np.cos(thR) - rho * a * np.sin(thR) * thR_z
        
        Zr = vr + kr * rho * a * np.sin(thZ) + k * a * np.sin(thZ) + k * rho * ar * np.sin(thZ) + k * rho * a * np.cos(thZ) * thZ_r
        Zt = k * rho * a * np.cos(thZ) * thZ_th
        Zz = vz + kz * rho * a * np.sin(thZ) + k * rho * az * np.sin(thZ) + k * rho * a * np.cos(thZ) * thZ_z

        det_phys = Rr * Zt - Rt * Zr
        det_safe = np.where(np.abs(det_phys) < 1e-13, -1e-13, det_phys)
        sqrt_g = (R / self.Nt) * det_safe
        
        g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
        g_zz = Rz**2 + (R/self.Nt)**2 + Zz**2 
        g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

        lam_eff = np.sum(c_lam[:, :, None, None, None] * L_fac_rad[:, None, :, None, None], axis=0) 
        Lt = np.sum(lam_eff * self.basis_lam_dth, axis=0) if self.len_lam > 0 else 0.0
        Lz = np.sum(lam_eff * self.basis_lam_dze, axis=0) if self.len_lam > 0 else 0.0
        
        P, dP, psip, Phip = self.get_profiles(rho)
        Bt_sup = (psip - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)

        Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
        Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
        Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

        dBt_drho = np.tensordot(self.D_matrix, Bt_sub, axes=(1, 0))
        dBz_drho = np.tensordot(self.D_matrix, Bz_sub, axes=(1, 0))

        Jz_sup = (dBt_drho - spectral_grad_th(Br_sub)) / sqrt_g
        Jt_sup = (spectral_grad_ze(Br_sub) - dBz_drho) / sqrt_g
        Jr_sup = (spectral_grad_th(Bz_sub) - spectral_grad_ze(Bt_sub)) / sqrt_g

        Jr_phys = Jr_sup / self.mu_0
        G_rho = dP - sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / self.mu_0
        rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
        th_R, th_Z = -Zr/det_safe, Rr/det_safe
        
        GR = (rho_R * G_rho + (Jr_phys / (2 * np.pi)) * (th_R * (Phip + Lt)))
        GZ = (rho_Z * G_rho + (Jr_phys / (2 * np.pi)) * (th_Z * (Phip + Lt)))

        # ==============================================================================
        # 预计算缓存提取，彻底释放有限差分阶段算力
        # ==============================================================================
        vol_element = sqrt_g * self.vol_element_base
        b1d = self.basis_1d_val[:, 0, 0, :]   
        b2d = self.basis_2d_val[:, 0, :, :]   
        blam = self.basis_lam_val[:, 0, :, :] 
        
        term_R_tR = GR * (-rho * a * np.sin(thR)) * vol_element
        term_Z_tZ = GZ * (k * rho * a * np.cos(thZ)) * vol_element
        term_R_c0R = GR * vol_element
        term_Z_c0Z = GZ * vol_element
        term_Z_k = GZ * (rho * a * np.sin(thZ)) * vol_element
        term_R_Z_a = (GR * rho * np.cos(thR) + GZ * k * rho * np.sin(thZ)) * vol_element
        
        res_1 = np.einsum('rtz, lr, iz -> li', term_R_tR, fac_rad, b1d)
        res_2 = np.einsum('rtz, lr, iz -> li', term_Z_tZ, fac_rad, b1d)
        res_3 = np.einsum('rtz, lr, iz -> li', term_R_c0R, fac_rad, b1d)
        res_4 = np.einsum('rtz, lr, iz -> li', term_Z_c0Z, fac_rad, b1d)
        res_5 = np.einsum('rtz, lr, iz -> li', term_Z_k, fac_rad, b1d)
        res_6 = np.einsum('rtz, lr, iz -> li', term_R_Z_a, fac_rad, b1d)
        
        if self.len_2d > 0:
            res_7 = np.einsum('rtz, lr, itz -> li', term_R_tR, fac_rad, b2d)
            res_8 = np.einsum('rtz, lr, itz -> li', term_Z_tZ, fac_rad, b2d)
            
        residuals_list = []
        for L_idx in range(self.L_rad):
            residuals_list.extend([res_1[L_idx], res_2[L_idx], res_3[L_idx], res_4[L_idx], res_5[L_idx], res_6[L_idx]])
            if self.len_2d > 0:
                residuals_list.extend([res_7[L_idx], res_8[L_idx]])
                
        if self.len_lam > 0:
            res_lam = np.einsum('rtz, lr, itz -> li', Jr_phys * vol_element, L_fac_rad, blam)
            for L_idx in range(self.L_rad):
                residuals_list.append(res_lam[L_idx])
                
        final_res = np.concatenate(residuals_list)
        
        # 【取消假收敛欺骗】: 移除 final_res = final_res / self.res_scales
            
        if self.len_2d > 0:
            idx_start = 6 * self.len_1d * self.L_rad
            idx_end = idx_start + 2 * self.len_2d * self.L_rad
            reg_penalty = x_core[idx_start:idx_end] * 1e-1
            final_res = np.concatenate([final_res, reg_penalty])
            
        if np.any(det_phys > 1e-4):
            # 【修复海森矩阵破坏】: 作为软约束附加到残差末尾
            penalty_res = np.where(det_phys > 1e-4, 100.0 * (det_phys - 1e-4), 0.0).flatten()
            final_res = np.concatenate([final_res, penalty_res])  
            
        return final_res

    def print_final_parameters(self, x_core):
        print("\n" + "="*110)
        print(f"{f'VEQ-3D 动态高维参数报告 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad})':^110}")
        print("="*110)
        
        edge_R0, edge_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        print(f"R0 (大半径中心) = {edge_R0:>15.8e}")
        print(f"Z0 (垂直中心)   = {edge_Z0:>15.8e}")
        print("-" * 110)
        
        header_cols = [f"Chebyshev L={L} 演化系数" for L in range(self.L_rad)]
        header_str = f"{'参数标识':<15} | {'Edge 边界常量 (rho=1)':<25} | " + " | ".join([f"{h:<22}" for h in header_cols])
        print(header_str)
        print("-" * 110)
        
        def print_1d(name, e_arr, c_arr):
            h_str = f"{name+'0':<15} | {e_arr[0]:>25.8e} | " + " | ".join([f"{c_arr[L, 0]:>22.8e}" for L in range(self.L_rad)])
            print(h_str)
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_str = f"{name+str(n)+'c':<15} | {e_arr[idx]:>25.8e} | " + " | ".join([f"{c_arr[L, idx]:>22.8e}" for L in range(self.L_rad)])
                print(c_str)
                s_str = f"{name+str(n)+'s':<15} | {e_arr[idx+1]:>25.8e} | " + " | ".join([f"{c_arr[L, idx+1]:>22.8e}" for L in range(self.L_rad)])
                print(s_str)
                idx+=2

        print_1d("c0R_", e_c0R, c_c0R); print_1d("c0Z_", e_c0Z, c_c0Z)
        print_1d("h_", e_h, c_h); print_1d("v_", e_v, c_v)
        print_1d("k_", e_k, c_k); print_1d("a_", e_a, c_a)
        
        if self.len_2d > 0:
            print("-" * 110); print(">>> 极向高阶摄动分量 (theta_R & theta_Z):")
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tR_str = f"tR_{m}_{n}{typ:<10} | {e_tR[i]:>25.8e} | " + " | ".join([f"{c_tR[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tR_str)
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tZ_str = f"tZ_{m}_{n}{typ:<10} | {e_tZ[i]:>25.8e} | " + " | ".join([f"{c_tZ[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tZ_str)
                
        print("-" * 110); print(f">>> 磁流函数 (Lambda) 谐波分量 ( 比例因子: rho*(1-rho^2)*T_l(x) ):")
        for i, (m, n) in enumerate(self.lambda_modes):
            L_str = f"L_{m}_{n:<12} | {'-- Null --':>25} | " + " | ".join([f"{c_lam[L, i]:>22.8e}" for L in range(self.L_rad)])
            print(L_str)
        print("="*110 + "\n")

    def solve(self, target_M=1, target_N=1, target_L=1):
        print(f">>> 启动 VEQ-3D 谱精度平衡求解器 (优化重构: 物理压强修正 + 残差架构优化 + 零差分开销) ...")
        
        self.M_pol = 0
        self.N_tor = target_N
        self.L_rad = target_L
        self._setup_modes()
        
        self._build_basis_matrices()
        self._build_radial_matrices() # 更新因 L_rad 可能发生变化带来的预计算张量
        
        self.fit_boundary(use_existing_as_guess=False)
        
        print(f"\n[第 1 阶段] 求解零阶骨架包络平衡 (M=0, N={self.N_tor}, L={self.L_rad})")
        x_opt = np.zeros(self.num_core_params)
        start_time = time.time()
        
        res = least_squares(self.compute_physics, x_opt, method='trf', xtol=1e-10, ftol=1e-10, max_nfev=1500)
        print(f"    基础场求解完成! 耗时: {time.time() - start_time:.4f} s | 最终残差范数: {np.linalg.norm(res.fun):.4e}")

        for m in range(1, target_M + 1):
            print(f"\n[第 {m + 1} 阶段] 维度注入，升阶向至 M={m}...")
            old_params = res.x
            
            self.p_edge = self._upgrade_edge_params(self.p_edge, m_old=m-1, m_new=m)
            
            self.M_pol = m
            self._setup_modes()
            
            self._build_basis_matrices()
            self._build_radial_matrices()
            
            self.fit_boundary(use_existing_as_guess=True)
            
            new_x0 = self._upgrade_core_params(old_params, m_old=m-1, m_new=m)
            
            step_start = time.time()
            res = least_squares(self.compute_physics, new_x0, method='trf', xtol=1e-10, ftol=1e-10, max_nfev=1500)
            print(f"    当前阶系解算完成! 耗时: {time.time() - step_start:.4f} s | 最终残差范数: {np.linalg.norm(res.fun):.4e}")

        total_time = time.time() - start_time
        print(f"\n>>> [完成] 全部物理场态构建用时: {total_time:.4f} s")
        
        self.print_final_parameters(res.x)
        self.plot_equilibrium(res.x)
        return res.x

    def plot_equilibrium(self, x_core):
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
            
            th_t = np.linspace(0, 2*np.pi, 200)
            ax.plot(10 - np.cos(th_t) - 0.3*np.cos(th_t+zv), np.sin(th_t) - 0.3*np.sin(th_t+zv), 'r--', lw=1.5, label='Input LCFS')
            
            rl_e, zl_e = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_e, zl_e, color='#FFD700', lw=2.0, label='Solved Boundary')
            ax.set_aspect('equal'); ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$'); 
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve(target_M=1, target_N=2, target_L=2)
