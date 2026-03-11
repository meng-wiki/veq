import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

RRR=1
ZZZ=1

# ==========================================
# 1. 动态阶数的参数解包与模型计算
# ==========================================
ENABLE_POLOIDAL_PERTURBATION = True

def get_num_params(N, M_R, M_Z):
    """计算特定 N, M_R, M_Z 阶数下，总共需要的参数向量长度"""
    base_len = 2 + 6*(1 + 2*N)
    if not ENABLE_POLOIDAL_PERTURBATION:
        return base_len
    return base_len + 2 * M_R * (2*N + 1) + 2 * M_Z * (2*N + 1)

def upgrade_params(p_old, N, M_R_old, M_Z_old, M_R_new, M_Z_new):
    """【核心黑科技】逐步升阶：独立对 M_R 和 M_Z 进行高阶补零升维"""
    if M_R_old == M_R_new and M_Z_old == M_Z_new:
        return p_old.copy()
    
    base_len = 2 + 6*(1 + 2*N)
    base = p_old[:base_len] # 提取基础几何参数
    
    len_pert_R_old = 2 * M_R_old * (2*N + 1) if ENABLE_POLOIDAL_PERTURBATION else 0
    len_pert_Z_old = 2 * M_Z_old * (2*N + 1) if ENABLE_POLOIDAL_PERTURBATION else 0
    
    # 提取旧的摄动参数
    tR_old = p_old[base_len : base_len + len_pert_R_old] if len_pert_R_old > 0 else np.array([])
    tZ_old = p_old[base_len + len_pert_R_old : base_len + len_pert_R_old + len_pert_Z_old] if len_pert_Z_old > 0 else np.array([])
        
    # 对新增的高阶维度补零
    len_add_R = 2 * (M_R_new - M_R_old) * (2*N + 1) if ENABLE_POLOIDAL_PERTURBATION else 0
    len_add_Z = 2 * (M_Z_new - M_Z_old) * (2*N + 1) if ENABLE_POLOIDAL_PERTURBATION else 0
    
    tR_new = np.concatenate([tR_old, np.zeros(len_add_R)]) if len_add_R > 0 else tR_old.copy()
    tZ_new = np.concatenate([tZ_old, np.zeros(len_add_Z)]) if len_add_Z > 0 else tZ_old.copy()
    
    return np.concatenate([base, tR_new, tZ_new])

def unpack_params(p, N, M_R, M_Z):
    idx = 0
    R0 = p[idx]; idx += 1
    Z0 = p[idx]; idx += 1
    
    def get_fourier(p_arr, curr_idx, order):
        length = 1 + 2 * order
        coeffs = p_arr[curr_idx : curr_idx + length]
        return coeffs, curr_idx + length

    h, idx = get_fourier(p, idx, N)
    v, idx = get_fourier(p, idx, N)
    a, idx = get_fourier(p, idx, N)
    kappa, idx = get_fourier(p, idx, N)
    c0R, idx = get_fourier(p, idx, N)
    c0Z, idx = get_fourier(p, idx, N)
    
    len_tR_pert = 2 * M_R * (2 * N + 1)
    len_tZ_pert = 2 * M_Z * (2 * N + 1)
    
    if ENABLE_POLOIDAL_PERTURBATION:
        tR_pert = p[idx : idx + len_tR_pert]; idx += len_tR_pert
        tZ_pert = p[idx : idx + len_tZ_pert]; idx += len_tZ_pert
    else:
        tR_pert = np.zeros(len_tR_pert)
        tZ_pert = np.zeros(len_tZ_pert)
    
    return R0, Z0, h, v, a, kappa, c0R, c0Z, tR_pert, tZ_pert

def eval_fourier(coeffs, zeta, N):
    val = coeffs[0]
    for n in range(1, N + 1):
        val += coeffs[2*n-1] * np.cos(n * zeta) + coeffs[2*n] * np.sin(n * zeta)
    return val

def calc_boundary(p, theta, zeta, N, M_R, M_Z):
    R0, Z0, h, v, a, kappa, c0R, c0Z, tR_pert, tZ_pert = unpack_params(p, N, M_R, M_Z)
    
    h_val = eval_fourier(h, zeta, N)
    v_val = eval_fourier(v, zeta, N)
    a_val = eval_fourier(a, zeta, N)
    k_val = eval_fourier(kappa, zeta, N)
    
    c0R_val = eval_fourier(c0R, zeta, N)
    c0Z_val = eval_fourier(c0Z, zeta, N)
    
    # 分别计算变分角 theta_R 的傅里叶求和 (m 遍历 1 到 M_R)
    tR_sum = 0
    idx_t = 0
    for m in range(1, M_R + 1):
        for n in range(-N, N + 1):
            tR_sum += tR_pert[idx_t] * np.cos(m*theta - n*zeta) + tR_pert[idx_t+1] * np.sin(m*theta - n*zeta)
            idx_t += 2
            
    # 分别计算变分角 theta_Z 的傅里叶求和 (m 遍历 1 到 M_Z)
    tZ_sum = 0
    idx_t = 0 # 对于 Z 摄动参数，索引重新从 0 开始
    for m in range(1, M_Z + 1):
        for n in range(-N, N + 1):
            tZ_sum += tZ_pert[idx_t] * np.cos(m*theta - n*zeta) + tZ_pert[idx_t+1] * np.sin(m*theta - n*zeta)
            idx_t += 2
        
    theta_R = theta + c0R_val + tR_sum
    theta_Z = theta + c0Z_val + tZ_sum
    
    R_mod = R0 + h_val + a_val * np.cos(theta_R)
    Z_mod = Z0 + v_val + k_val * a_val * np.sin(theta_Z)
    
    return R_mod, Z_mod

def residuals(p, theta, zeta, R_target, Z_target, N, M_R, M_Z):
    R_mod, Z_mod = calc_boundary(p, theta, zeta, N, M_R, M_Z)
    return np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])

# ==========================================
# 2. 核心设置与 多级升阶优化策略 (Spectral Refinement)
# ==========================================
N_tor = 1 
M_R_final = 1 
M_Z_final = 1  

N_grid_theta = 33  
N_grid_zeta = 33   

theta_1d = np.linspace(0, 2*np.pi, N_grid_theta)
zeta_1d = np.linspace(0, 2*np.pi, N_grid_zeta)
Theta, Zeta = np.meshgrid(theta_1d, zeta_1d)

# 恢复固定的目标点云计算 (取消 lambda 靶向形变)
R_target = 10 - np.cos(Theta) - 0.3 * np.cos(RRR*Theta + Zeta)
Z_target = np.sin(Theta) - 0.3 * np.sin(ZZZ*Theta + Zeta)

print(f"开始执行多级逐步优化 (Spectral Refinement), 目标 M_R={M_R_final}, M_Z={M_Z_final}...")

# 初始化 M=0 的基础参数 guess
p_current = np.zeros(get_num_params(N_tor, 0, 0))
# p_current[0] = 10.0      # R0
# p_current[1] = 0.0       # Z0
# p_current[2 + 2*(1+2*N_tor)] = 1.0           # a_0
# p_current[2 + 3*(1+2*N_tor)] = 1.0           # kappa_0
# p_current[2 + 4*(1+2*N_tor)] = np.pi         # c_{00}^R

# 恢复基于极向模数的逐步升阶循环
max_steps = max(M_R_final, M_Z_final)
m_R_old, m_Z_old = 0, 0

for step in range(max_steps + 1):
    # 动态计算当前的 M_R 和 M_Z (逐步增加，封顶于用户设定值)
    m_R_current = min(step, M_R_final)
    m_Z_current = min(step, M_Z_final)
    
    print(f"\n>>> 阶段 {step+1}/{max_steps + 1}: 拟合极向模数 (M_R={m_R_current}, M_Z={m_Z_current})")
    
    # 如果不是第一步，则需要从上一步的结果升阶补零
    if step > 0:
        p_current = upgrade_params(p_current, N_tor, m_R_old, m_Z_old, m_R_current, m_Z_current)
    
    # 动态调整容差
    ftol_val = 1e-12 if step == max_steps else 1e-10
    
    # ★ 保留 trf 算法和 huber 鲁棒损失函数
    res = least_squares(residuals, p_current, 
                        args=(Theta, Zeta, R_target, Z_target, N_tor, m_R_current, m_Z_current), 
                        method='trf', loss='huber', ftol=ftol_val)
    
    # 更新当前最优参数，供下一阶段使用
    p_current = res.x
    m_R_old, m_Z_old = m_R_current, m_Z_current
    print(f"    阶段 {step+1} 完成, 当前 Cost = {res.cost:.4e}")

# 将最终的最优解赋值给 p_opt
p_opt = p_current
print(f"\n全部优化流程结束！最终极小 Cost = {res.cost:.4e}")

# ==========================================
# 3. 可视化与验证 (多 zeta 截面)
# ==========================================
N_scatter_points = 20
theta_plot = np.linspace(0, 2*np.pi, 100)
theta_scatter = np.linspace(0, 2*np.pi, N_scatter_points)

zeta_vals = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
zeta_labels = ['0', r'\pi/3', r'2\pi/3', r'\pi', r'4\pi/3', r'5\pi/3']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (zeta_val, label) in enumerate(zip(zeta_vals, zeta_labels)):
    ax = axes[i]
    
    zeta_scatter = np.full_like(theta_scatter, zeta_val)
    # 恢复绘图时的标准目标函数
    R_target_scatter = 10 - np.cos(theta_scatter) - 0.3*np.cos(RRR*theta_scatter + zeta_scatter)
    Z_target_scatter = np.sin(theta_scatter) - 0.3*np.sin(ZZZ*  theta_scatter + zeta_scatter)
    
    zeta_plot_arr = np.full_like(theta_plot, zeta_val)
    R_fit_plot, Z_fit_plot = calc_boundary(p_opt, theta_plot, zeta_plot_arr, N_tor, M_R_final, M_Z_final)
    
    ax.plot(R_target_scatter, Z_target_scatter, 'o', color='crimson', label='Target Points (D-Shape)', markersize=6)
    ax.plot(R_fit_plot, Z_fit_plot, '-', color='navy', label='Fitted Curve', linewidth=2.5)
    
    ax.set_xlabel('R (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title(rf'$\zeta = {label}$', fontsize=14)
    ax.legend(fontsize=10)
    ax.axis('equal') 
    ax.grid(True, linestyle='--', alpha=0.7)

plt.suptitle(rf'Plasma Boundary Spectral Refinement (N={N_tor}, M_R={M_R_final}, M_Z={M_Z_final})', fontsize=18, y=0.95)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

save_name = 'boundary_fit_multi_zeta.pdf'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"多截面拟合图片已保存至当前目录: '{save_name}'")
