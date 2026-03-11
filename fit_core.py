import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 全局配置与阶数设定
# ==========================================
N_thetaR = 1    # thetaR 极向扰动的环向展开阶数 (对应 PDF 的 N_R)
N_thetaZ = 1    # thetaZ 极向扰动的环向展开阶数 (对应 PDF 的 N_Z)
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
    """计算单个空间（edge或core）的傅里叶及扰动参数总数"""
    base_len = (1 + 2 * N_h) + (1 + 2 * N_nu) + (1 + 2 * N_a) + (1 + 2 * N_kappa) + (1 + 2 * N_c0R) + (1 + 2 * N_c0Z)
    if not ENABLE_POLOIDAL_PERTURBATION:
        return base_len
    return base_len + 2 * M_R * (2 * N_thetaR + 1) + 2 * M_Z * (2 * N_thetaZ + 1)

def get_num_params_edge(M_R, M_Z):
    """边界参数包含 R0, Z0"""
    return 2 + get_num_params_sub(M_R, M_Z)

def get_num_params_core(M_R, M_Z):
    """核心衰减参数不包含 R0, Z0"""
    return get_num_params_sub(M_R, M_Z)

def eval_fourier(coeffs, zeta, N):
    """计算标准的一维环向傅里叶级数"""
    val = coeffs[0] * np.ones_like(zeta)
    for n in range(1, N + 1):
        val += coeffs[2*n-1] * np.cos(n * zeta) + coeffs[2*n] * np.sin(n * zeta)
    return val

def calc_components(p_sub, theta, zeta, M_R, M_Z):
    """
    核心加速黑科技：傅里叶变换是线性的，分别解析edge和core，
    之后通过(1-rho^2)组合避免复杂张量广播。
    """
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
    """
    结合边界系数和核心系数，计算任意 (rho, theta, zeta) 的 R, Z
    """
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
# 3. 残差函数与优化逻辑
# ==========================================
def residuals_edge(p_edge, theta, zeta, R_target, Z_target, M_R, M_Z):
    R_mod, Z_mod = calc_full_space(p_edge, None, 1.0, theta, zeta, M_R, M_Z)
    return np.concatenate([(R_mod - R_target), (Z_mod - Z_target)])

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
# 4. 数据加载与降采样 (Data Pre-processing)
# ==========================================
def generate_mock_data():
    print(">>> 未检测到 'RZ_data.txt'，正在生成数据...")
    rhos = np.linspace(0.1, 1.0, 10)
    thetas = np.linspace(0, 2*np.pi, 30, endpoint=False)
    zetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    data = []
    for rho in rhos:
        for theta in thetas:
            for zeta in zetas:
                h = 0.5 * (1 - rho**2) + 0.1 * np.cos(zeta)
                a = 3.0 - 0.5 * (1 - rho**2)
                kappa = 1.6 + 0.2 * (1 - rho**2)
                tR = 0.1 * rho * np.sin(theta - zeta)
                tZ = 0.1 * rho * np.cos(theta - zeta)
                R = 10.0 + h + rho * a * np.cos(theta + tR)
                Z = 0.0 + kappa * rho * a * np.sin(theta + tZ)
                data.append([rho, theta, zeta/19, R, Z]) # 模拟 phi 数据
    df = pd.DataFrame(data, columns=['rho', 'theta', 'zeta', 'R', 'Z'])
    df.to_csv("RZ_data.txt", index=False)

if not os.path.exists("RZ_data.txt"):
    generate_mock_data()

print("\n[第一阶段] 数据加载与分层降采样")
df = pd.read_csv("RZ_data.txt")

# 【绝对核心修复】：将原数据中的单周期相位 phi 转换回完整的环向角 zeta
df['zeta'] = df['zeta'] * 19

# 回退到干净的按 rho 提取逻辑
df_edge = df[df['rho'] >= 0.99].copy()
df_core = df[df['rho'] < 0.99].copy()

MAX_CORE_PTS = 5000
if len(df_core) > MAX_CORE_PTS:
    print(f"内部点总数为 {len(df_core)}，执行随机降采样至 {MAX_CORE_PTS} 点。")
    df_core = df_core.sample(n=MAX_CORE_PTS, random_state=42)

print(f"  边界拟合点数: {len(df_edge)}")
print(f"  核心拟合点数: {len(df_core)}")

# ==========================================
# 5. 执行两步走拟合策略
# ==========================================
print("\n[第二阶段] 步骤 1: 纯边界拟合 (Spectral Refinement on rho=1)")
p_edge_opt = np.zeros(get_num_params_edge(0, 0))
p_edge_opt[0] = df_edge['R'].mean()   # R0
p_edge_opt[1] = 0.0                   # Z0
idx_a0 = 2 + (1 + 2 * N_h) + (1 + 2 * N_nu) # 动态定位 a0 的位置
p_edge_opt[idx_a0] = (df_edge['R'].max() - df_edge['R'].min())/2.0 # a0

max_steps = max(M_R_final, M_Z_final)
m_R_old, m_Z_old = 0, 0

for step in range(max_steps + 1):
    m_R_curr = min(step, M_R_final)
    m_Z_curr = min(step, M_Z_final)
    print(f"  >>> 边界升阶 ({step}/{max_steps}): M_R={m_R_curr}, M_Z={m_Z_curr}")
    
    if step > 0:
        p_edge_opt = upgrade_params_edge(p_edge_opt, m_R_old, m_Z_old, m_R_curr, m_Z_curr)
    
    ftol_val = 1e-10 if step == max_steps else 1e-6
    res_edge = least_squares(
        residuals_edge, p_edge_opt, 
        args=(df_edge['theta'].values, df_edge['zeta'].values, 
              df_edge['R'].values, df_edge['Z'].values, m_R_curr, m_Z_curr),
        method='trf', loss='huber', ftol=ftol_val
    )
    p_edge_opt = res_edge.x
    m_R_old, m_Z_old = m_R_curr, m_Z_curr
    print(f"      Cost = {res_edge.cost:.4e}")

print("\n[第三阶段] 步骤 2: 冻结边界，拟合内部核心展开系数 (rho < 1)")
p_core_guess = np.zeros(get_num_params_core(M_R_final, M_Z_final))

res_core = least_squares(
    residuals_core, p_core_guess,
    args=(p_edge_opt, df_core['rho'].values, df_core['theta'].values, df_core['zeta'].values,
          df_core['R'].values, df_core['Z'].values, M_R_final, M_Z_final),
    method='trf', loss='huber', ftol=1e-10
)
p_core_opt = res_core.x
print(f"  >>> 内部拟合完成！最终 Cost = {res_core.cost:.4e}")

# ==========================================
# 5.5 输出所有迭代后的参数名称和数值
# ==========================================
def print_optimized_parameters(p_edge, p_core, M_R, M_Z):
    print("\n" + "="*70)
    print(">>> 优化完成！输出最终拟合参数 (依据《拟合边界》命名规范) <<<")
    print("="*70)
    
    print(f"R_0 = {p_edge[0]:.8e}")
    print(f"Z_0 = {p_edge[1]:.8e}")
    
    def print_fourier_group(var_id, N, idx_e_start, idx_c_start):
        idx_e = idx_e_start
        idx_c = idx_c_start
        
        if var_id in ["c0R", "c0Z"]:
            ax = "R" if var_id == "c0R" else "Z"
            print(f"c_{{00}}^{ax}(edge) = {p_edge[idx_e]:.8e}, c_{{00}}^{ax}(core) = {p_core[idx_c]:.8e}")
            idx_e += 1; idx_c += 1
            for n in range(1, N + 1):
                print(f"c_{{0{n}}}^{{{ax},c}}(edge) = {p_edge[idx_e]:.8e}, c_{{0{n}}}^{{{ax},c}}(core) = {p_core[idx_c]:.8e}")
                idx_e += 1; idx_c += 1
                print(f"c_{{0{n}}}^{{{ax},s}}(edge) = {p_edge[idx_e]:.8e}, c_{{0{n}}}^{{{ax},s}}(core) = {p_core[idx_c]:.8e}")
                idx_e += 1; idx_c += 1
        else:
            name = var_id
            print(f"{name}_0(edge) = {p_edge[idx_e]:.8e}, {name}_0(core) = {p_core[idx_c]:.8e}")
            idx_e += 1; idx_c += 1
            for n in range(1, N + 1):
                print(f"{name}_{{{n}c}}(edge) = {p_edge[idx_e]:.8e}, {name}_{{{n}c}}(core) = {p_core[idx_c]:.8e}")
                idx_e += 1; idx_c += 1
                print(f"{name}_{{{n}s}}(edge) = {p_edge[idx_e]:.8e}, {name}_{{{n}s}}(core) = {p_core[idx_c]:.8e}")
                idx_e += 1; idx_c += 1
        return idx_e, idx_c
    
    idx_e = 2
    idx_c = 0
    
    print("\n[基础傅里叶参数 h, \\nu (v), a, \\kappa]")
    idx_e, idx_c = print_fourier_group("h", N_h, idx_e, idx_c)
    idx_e, idx_c = print_fourier_group("\\nu", N_nu, idx_e, idx_c)
    idx_e, idx_c = print_fourier_group("a", N_a, idx_e, idx_c)
    idx_e, idx_c = print_fourier_group("\\kappa", N_kappa, idx_e, idx_c)
    
    print("\n[基准变分角常数 c_0^R, c_0^Z]")
    idx_e, idx_c = print_fourier_group("c0R", N_c0R, idx_e, idx_c)
    idx_e, idx_c = print_fourier_group("c0Z", N_c0Z, idx_e, idx_c)
    
    if ENABLE_POLOIDAL_PERTURBATION:
        print("\n[极向摄动参数 \\theta_R]")
        for m in range(1, M_R + 1):
            for n in range(-N_thetaR, N_thetaR + 1):
                c_edge, s_edge = p_edge[idx_e], p_edge[idx_e+1]
                c_core, s_core = p_core[idx_c], p_core[idx_c+1]
                print(f"c_{{{m},{n}}}^R(edge) = {c_edge:.8e}, c_{{{m},{n}}}^R(core) = {c_core:.8e}")
                print(f"s_{{{m},{n}}}^R(edge) = {s_edge:.8e}, s_{{{m},{n}}}^R(core) = {s_core:.8e}")
                idx_e += 2; idx_c += 2

        print("\n[极向摄动参数 \\theta_Z]")
        for m in range(1, M_Z + 1):
            for n in range(-N_thetaZ, N_thetaZ + 1):
                c_edge, s_edge = p_edge[idx_e], p_edge[idx_e+1]
                c_core, s_core = p_core[idx_c], p_core[idx_c+1]
                print(f"c_{{{m},{n}}}^Z(edge) = {c_edge:.8e}, c_{{{m},{n}}}^Z(core) = {c_core:.8e}")
                print(f"s_{{{m},{n}}}^Z(edge) = {s_edge:.8e}, s_{{{m},{n}}}^Z(core) = {s_core:.8e}")
                idx_e += 2; idx_c += 2
    print("="*70 + "\n")

print_optimized_parameters(p_edge_opt, p_core_opt, M_R_final, M_Z_final)

# ==========================================
# 6. 结果可视化
# ==========================================
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
    
    # 恢复严格的环向截面容差 (0.1)
    diff_zeta = np.abs((df['zeta'] - zeta_val + np.pi) % (2 * np.pi) - np.pi)
    df_slice = df[diff_zeta < 0.1]

    zeta_dense = np.full_like(theta_dense, zeta_val)
    for rho_val, c in zip(rho_levels, colors):
        # 1. 恢复严格的 rho 层容差 (0.02)
        if not df_slice.empty:
            df_rho = df_slice[np.abs(df_slice['rho'] - rho_val) < 0.02]
            if not df_rho.empty:
                ax.scatter(df_rho['R'], df_rho['Z'], s=15, color=c, alpha=0.5)

        # 2. 绘制该 rho 层的拟合闭合曲线
        R_fit, Z_fit = calc_full_space(
            p_edge_opt, p_core_opt, rho_val, theta_dense, zeta_dense, 
            M_R_final, M_Z_final
        )
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
