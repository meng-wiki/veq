import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 全局配置与阶数设定
# ==========================================
N_thetaR = 1    # thetaR 极向扰动的环向展开阶数
N_thetaZ = 1    # thetaZ 极向扰动的环向展开阶数
N_c0R = 1       # c0R 的环向展开阶数
N_c0Z = 1       # c0Z 的环向展开阶数
N_h = 1         # h 的环向展开阶数
N_nu = 1        # nu (垂直平移 v) 的环向展开阶数
N_a = 1         # a (小半径) 的环向展开阶数
N_kappa = 1     # kappa (拉长比) 的环向展开阶数

M_R_final = 0   # R 的极向扰动阶数 M_R
M_Z_final = 0   # Z 的极向扰动阶数 M_Z
ENABLE_POLOIDAL_PERTURBATION = True

# ==========================================
# 1. 参数解包与基础数学函数
# ==========================================
def get_num_params_sub(M_R, M_Z):
    base_len = (1 + 2 * N_h) + (1 + 2 * N_nu) + (1 + 2 * N_a) + (1 + 2 * N_kappa) + (1 + 2 * N_c0R) + (1 + 2 * N_c0Z)
    if not ENABLE_POLOIDAL_PERTURBATION:
        return base_len
    return base_len + 2 * M_R * (2 * N_thetaR + 1) + 2 * M_Z * (2 * N_thetaZ + 1)

def get_num_params_edge(M_R, M_Z):
    return 2 + get_num_params_sub(M_R, M_Z)

def get_num_params_core(M_R, M_Z):
    return get_num_params_sub(M_R, M_Z)

def eval_fourier(coeffs, zeta, N):
    val = coeffs[0] * np.ones_like(zeta)
    for n in range(1, N + 1):
        val += coeffs[2*n-1] * np.cos(n * zeta) + coeffs[2*n] * np.sin(n * zeta)
    return val

def calc_components(p_sub, theta, zeta, M_R, M_Z):
    idx = 0
    def get_F(order):
        nonlocal idx
        length = 1 + 2 * order
        c = p_sub[idx : idx + length]; idx += length
        return c

    h_c = get_F(N_h); v_c = get_F(N_nu); a_c = get_F(N_a)
    k_c = get_F(N_kappa); c0R_c = get_F(N_c0R); c0Z_c = get_F(N_c0Z)

    h_val = eval_fourier(h_c, zeta, N_h)
    v_val = eval_fourier(v_c, zeta, N_nu)
    a_val = eval_fourier(a_c, zeta, N_a)
    k_val = eval_fourier(k_c, zeta, N_kappa)
    c0R_val = eval_fourier(c0R_c, zeta, N_c0R)
    c0Z_val = eval_fourier(c0Z_c, zeta, N_c0Z)

    tR_sum = np.zeros_like(theta)
    tZ_sum = np.zeros_like(theta)
    
    if ENABLE_POLOIDAL_PERTURBATION:
        len_tR = 2 * M_R * (2 * N_thetaR + 1)
        tR_pert = p_sub[idx : idx + len_tR]; idx += len_tR
        tZ_pert = p_sub[idx : idx + 2 * M_Z * (2 * N_thetaZ + 1)]
        
        idx_t = 0
        for m in range(1, M_R + 1):
            for n in range(-N_thetaR, N_thetaR + 1):
                tR_sum += tR_pert[idx_t] * np.cos(m*theta - n*zeta) + tR_pert[idx_t+1] * np.sin(m*theta - n*zeta)
                idx_t += 2
                
        idx_t = 0
        for m in range(1, M_Z + 1):
            for n in range(-N_thetaZ, N_thetaZ + 1):
                tZ_sum += tZ_pert[idx_t] * np.cos(m*theta - n*zeta) + tZ_pert[idx_t+1] * np.sin(m*theta - n*zeta)
                idx_t += 2

    return h_val, v_val, a_val, k_val, c0R_val, c0Z_val, tR_sum, tZ_sum

# ==========================================
# 2. 全空间物理模型构建
# ==========================================
def calc_full_space(p_edge, p_core, rho, theta, zeta, M_R, M_Z):
    R0, Z0 = p_edge[0], p_edge[1]
    p_edge_sub = p_edge[2:]
    
    E = calc_components(p_edge_sub, theta, zeta, M_R, M_Z)
    C = calc_components(p_core, theta, zeta, M_R, M_Z) if p_core is not None else [0]*8
    
    rho_fac = 1.0 - rho**2
    
    h = E[0] + C[0] * rho_fac
    v = E[1] + C[1] * rho_fac
    a = E[2] + C[2] * rho_fac
    kappa = E[3] + C[3] * rho_fac
    c0R = E[4] + C[4] * rho_fac
    c0Z = E[5] + C[5] * rho_fac
    tR = E[6] + C[6] * rho_fac
    tZ = E[7] + C[7] * rho_fac
    
    theta_R = theta + c0R + tR
    theta_Z = theta + c0Z + tZ
    
    R_mod = R0 + h + rho * a * np.cos(theta_R)
    Z_mod = Z0 + v + kappa * rho * a * np.sin(theta_Z)
    
    return R_mod, Z_mod

# ==========================================
# 3. 残差函数与优化逻辑 (核心修复)
# ==========================================
def residuals_edge(p_edge, theta, zeta, R_target, Z_target, M_R, M_Z):
    R_mod, Z_mod = calc_full_space(p_edge, None, 1.0, theta, zeta, M_R, M_Z)
    res_geom = np.concatenate([(R_mod - R_target), (Z_mod - Z_target)])
    
    idx_h0 = 2
    idx_v0 = 2 + (1 + 2 * N_h)
    idx_a0 = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu)
    
    # 严格确保惩罚项是 1D 数组，防止 numpy 维度广播异常
    res_reg = np.array([p_edge[idx_h0] * 100.0, p_edge[idx_v0] * 100.0])
    
    a0_val = p_edge[idx_a0]
    res_a0 = np.array([(0.1 - a0_val) * 100.0 if a0_val < 0.1 else 0.0])
    
    return np.concatenate([res_geom, res_reg, res_a0])

def residuals_core(p_core, p_edge_fixed, rho, theta, zeta, R_target, Z_target, M_R, M_Z):
    R_mod, Z_mod = calc_full_space(p_edge_fixed, p_core, rho, theta, zeta, M_R, M_Z)
    return np.concatenate([(R_mod - R_target), (Z_mod - Z_target)])

def upgrade_params_edge(p_old, M_R_old, M_Z_old, M_R_new, M_Z_new):
    if M_R_old == M_R_new and M_Z_old == M_Z_new: return p_old.copy()
    base_len = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu) + (1 + 2 * N_a) + (1 + 2 * N_kappa) + (1 + 2 * N_c0R) + (1 + 2 * N_c0Z)
    base = p_old[:base_len]
    len_pR_old = 2 * M_R_old * (2 * N_thetaR + 1)
    len_pZ_old = 2 * M_Z_old * (2 * N_thetaZ + 1)
    tR_old = p_old[base_len : base_len+len_pR_old] if len_pR_old > 0 else np.array([])
    tZ_old = p_old[base_len+len_pR_old : base_len+len_pR_old+len_pZ_old] if len_pZ_old > 0 else np.array([])
    len_add_R = 2 * (M_R_new - M_R_old) * (2 * N_thetaR + 1)
    len_add_Z = 2 * (M_Z_new - M_Z_old) * (2 * N_thetaZ + 1)
    tR_new = np.concatenate([tR_old, np.zeros(len_add_R)]) if len_add_R > 0 else tR_old.copy()
    tZ_new = np.concatenate([tZ_old, np.zeros(len_add_Z)]) if len_add_Z > 0 else tZ_old.copy()
    return np.concatenate([base, tR_new, tZ_new])

# ==========================================
# 4. 数据加载
# ==========================================
def generate_mock_data():
    print(">>> 未检测到 'RZ_data.txt'，正在生成模拟数据...")
    rhos = np.linspace(0.1, 1.0, 10)
    thetas = np.linspace(0, 2*np.pi, 40, endpoint=False)
    zetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    data = []
    for rho in rhos:
        for theta in thetas:
            for zeta in zetas:
                R = 10.0 + 0.05*(1-rho**2) + rho * 1.02 * np.cos(theta + np.pi + 0.1*np.sin(zeta))
                Z = 0.0 + 1.6 * rho * 1.02 * np.sin(theta + np.pi)
                data.append([rho, theta, zeta, R, Z])
    df = pd.DataFrame(data, columns=['rho', 'theta', 'zeta', 'R', 'Z'])
    df.to_csv("RZ_data.txt", index=False)

if not os.path.exists("RZ_data.txt"):
    generate_mock_data()

print(f"\n[第一阶段] 数据加载与分层降采样 (M_R_final={M_R_final}, M_Z_final={M_Z_final})")
df = pd.read_csv("RZ_data.txt")

# 恢复单周期缩放逻辑
if df['zeta'].max() <= 1.0:
    print("  >>> 检测到单周期数据(phi)，执行 zeta = zeta * 19 转换以避免傅里叶病态共线陷阱！")
    df['zeta'] = df['zeta'] * 19

df_edge = df[df['rho'] >= 0.99].copy()
df_core = df[df['rho'] < 0.99].copy()

MAX_CORE_PTS = 5000
if len(df_core) > MAX_CORE_PTS:
    print(f"  >>> 内部点总数为 {len(df_core)}，执行随机降采样至 {MAX_CORE_PTS} 点。")
    df_core = df_core.sample(n=MAX_CORE_PTS, random_state=42)

# ==========================================
# 5. 执行两步走拟合策略
# ==========================================
print("\n[第二阶段] 步骤 1: 纯边界拟合 (智能对齐坐标相位)")
p_edge_opt = np.zeros(get_num_params_edge(0, 0))
p_edge_opt[0] = df_edge['R'].mean()   # R0
p_edge_opt[1] = df_edge['Z'].mean()   # Z0

idx_a0 = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu)
idx_k0 = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu) + (1 + 2 * N_a)
idx_c00R = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu) + (1 + 2 * N_a) + (1 + 2 * N_kappa)

p_edge_opt[idx_a0] = (df_edge['R'].max() - df_edge['R'].min()) / 2.0  
p_edge_opt[idx_k0] = 1.6

# 【核心修复】：智能相位探测。测算 R 和 cos(theta) 的相关性，解除180度死锁
R_cos_corr = np.mean((df_edge['R'] - df_edge['R'].mean()) * np.cos(df_edge['theta']))
if R_cos_corr < 0:
    print("  >>> 智能探测: theta=0 位于内侧 (Inboard)，自动设定变分角基准 c00R = π")
    p_edge_opt[idx_c00R] = np.pi
else:
    print("  >>> 智能探测: theta=0 位于外侧 (Outboard)，自动设定变分角基准 c00R = 0")
    p_edge_opt[idx_c00R] = 0.0

max_steps = max(M_R_final, M_Z_final)
m_R_old, m_Z_old = 0, 0

for step in range(max_steps + 1):
    m_R_curr = min(step, M_R_final)
    m_Z_curr = min(step, M_Z_final)
    print(f"  >>> 边界升阶 ({step}/{max_steps}): M_R={m_R_curr}, M_Z={m_Z_curr}")
    if step > 0:
        p_edge_opt = upgrade_params_edge(p_edge_opt, m_R_old, m_Z_old, m_R_curr, m_Z_curr)
    
    res_edge = least_squares(
        residuals_edge, p_edge_opt, 
        args=(df_edge['theta'].values, df_edge['zeta'].values, 
              df_edge['R'].values, df_edge['Z'].values, m_R_curr, m_Z_curr),
        method='trf', loss='linear', ftol=1e-10
    )
    p_edge_opt = res_edge.x
    m_R_old, m_Z_old = m_R_curr, m_Z_curr
    print(f"      Cost = {res_edge.cost:.4e} | R0={p_edge_opt[0]:.4f}, a0={p_edge_opt[idx_a0]:.4f}")

print("\n[第三阶段] 步骤 2: 冻结边界，拟合内部核心展开系数")
p_core_guess = np.zeros(get_num_params_core(M_R_final, M_Z_final))
res_core = least_squares(
    residuals_core, p_core_guess,
    args=(p_edge_opt, df_core['rho'].values, df_core['theta'].values, df_core['zeta'].values,
          df_core['R'].values, df_core['Z'].values, M_R_final, M_Z_final),
    method='trf', loss='linear', ftol=1e-10
)
p_core_opt = res_core.x
print(f"  >>> 内部拟合完成！最终 Cost = {res_core.cost:.4e}")

# ==========================================
# 6. 最终参数输出报告与可视化
# ==========================================
def print_optimized_parameters(p_edge, p_core, M_R, M_Z):
    print("\n" + "="*85)
    print(f"{'>>> 3D 拟合参数对齐报告 (LCFS vs Core Coefficients) <<<':^85}")
    print("="*85)
    
    print(f"R_0 (Major Radius Center) = {p_edge[0]:.8e}")
    print(f"Z_0 (Vertical Center)     = {p_edge[1]:.8e}")
    
    def print_fourier_group(var_id, N, idx_e_start, idx_c_start):
        idx_e, idx_c = idx_e_start, idx_c_start
        e_val, c_val = p_edge[idx_e], p_core[idx_c]
        name = var_id.replace("0", "_{00}")
        print(f"{name:<15}(edge) = {e_val:>15.8e} | {name:<15}(core) = {c_val:>15.8e}")
        idx_e += 1; idx_c += 1
        
        for n in range(1, N + 1):
            for suffix in ["c", "s"]:
                e_val, c_val = p_edge[idx_e], p_core[idx_c]
                tag = f"{var_id}_{{{n}{suffix}}}"
                print(f"{tag:<15}(edge) = {e_val:>15.8e} | {tag:<15}(core) = {c_val:>15.8e}")
                idx_e += 1; idx_c += 1
        return idx_e, idx_c

    curr_e, curr_c = 2, 0
    print("\n[1] 基础几何参数剖面 (1-rho^2):")
    curr_e, curr_c = print_fourier_group("h", N_h, curr_e, curr_c)
    curr_e, curr_c = print_fourier_group("nu", N_nu, curr_e, curr_c)
    curr_e, curr_c = print_fourier_group("a", N_a, curr_e, curr_c)
    curr_e, curr_c = print_fourier_group("kappa", N_kappa, curr_e, curr_c)
    
    print("\n[2] 基准变分角系数:")
    curr_e, curr_c = print_fourier_group("c0R", N_c0R, curr_e, curr_c)
    curr_e, curr_c = print_fourier_group("c0Z", N_c0Z, curr_e, curr_c)
    
    if ENABLE_POLOIDAL_PERTURBATION:
        print("\n[3] 极向摄动谐波 (theta_R):")
        for m in range(1, M_R + 1):
            for n in range(-N_thetaR, N_thetaR + 1):
                for suffix in ["c", "s"]:
                    e_v, c_v = p_edge[curr_e], p_core[curr_c]
                    label = f"c_{{{m},{n}}}^R" if suffix=="c" else f"s_{{{m},{n}}}^R"
                    print(f"{label:<15}(edge) = {e_v:>15.8e} | {label:<15}(core) = {c_v:>15.8e}")
                    curr_e += 1; curr_c += 1
        
        print("\n[4] 极向摄动谐波 (theta_Z):")
        for m in range(1, M_Z + 1):
            for n in range(-N_thetaZ, N_thetaZ + 1):
                for suffix in ["c", "s"]:
                    e_v, c_v = p_edge[curr_e], p_core[curr_c]
                    label = f"c_{{{m},{n}}}^Z" if suffix=="c" else f"s_{{{m},{n}}}^Z"
                    print(f"{label:<15}(edge) = {e_v:>15.8e} | {label:<15}(core) = {c_v:>15.8e}")
                    curr_e += 1; curr_c += 1
    
    print("="*85 + "\n")

print_optimized_parameters(p_edge_opt, p_core_opt, M_R_final, M_Z_final)

print("\n[第四阶段] 生成截面验证图...")
zeta_plot_vals = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
zeta_labels = ['0', r'\pi/3', r'2\pi/3', r'\pi', r'4\pi/3', r'5\pi/3']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

theta_dense = np.linspace(0, 2*np.pi, 100)
rho_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(rho_levels)))

for i, (zeta_val, label) in enumerate(zip(zeta_plot_vals, zeta_labels)):
    ax = axes[i]
    diff_zeta = np.abs((df['zeta'] - zeta_val + np.pi) % (2 * np.pi) - np.pi)
    df_slice = df[diff_zeta < 0.1]
    zeta_dense = np.full_like(theta_dense, zeta_val)
    for rho_val, c in zip(rho_levels, colors):
        if not df_slice.empty:
            df_rho = df_slice[np.abs(df_slice['rho'] - rho_val) < 0.02]
            if not df_rho.empty:
                ax.scatter(df_rho['R'], df_rho['Z'], s=15, color=c, alpha=0.5)
        R_fit, Z_fit = calc_full_space(p_edge_opt, p_core_opt, rho_val, theta_dense, zeta_dense, M_R_final, M_Z_final)
        lw = 2.5 if rho_val == 1.0 else 1.5
        ax.plot(R_fit, Z_fit, color=c, linewidth=lw, label=rf'$\rho={rho_val:.1f}$')

    ax.set_title(rf'Toroidal Angle $\zeta = {label}$', fontsize=14)
    ax.set_xlabel('R (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.suptitle(f'Full-Volume 3D Plasma Boundary Fit (M_R={M_R_final}, M_Z={M_Z_final})', fontsize=18)
plt.tight_layout()
save_path = 'full_volume_fit.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f">>> 截面绘图已保存至: {save_path}")
