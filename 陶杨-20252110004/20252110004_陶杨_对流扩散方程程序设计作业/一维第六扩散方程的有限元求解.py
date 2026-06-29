import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# 全局设置：彻底解决中文+负号显示问题
# =========================
# 优先使用系统自带中文字体，Windows下微软雅黑/黑体均可正常显示
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# 强制使用ASCII减号，彻底避免Unicode负号字体缺失警告
rcParams['axes.unicode_minus'] = False
rcParams['text.usetex'] = False


# =========================
# 1. SUPG 最优稳定参数计算
# =========================
def alpha_supg(Pe):
    """
    计算SUPG/Petrov-Galerkin最优稳定系数
    公式：alpha = coth(Pe) - 1/Pe
    Pe→0时自动返回0，避免除以零
    """
    if abs(Pe) < 1e-12:
        return 0.0
    return 1.0 / np.tanh(Pe) - 1.0 / Pe


# =========================
# 2. 单元刚度矩阵
# =========================
def element_matrix(kappa, v, le, alpha):
    """
    生成两节点线性单元的对流扩散单元矩阵
    采用作业指定的等效人工扩散格式：
    kappa_bar = kappa + alpha * v * le / 2
    Ke = 扩散项 + 对流项
    """
    # 等效扩散系数（人工扩散叠加物理扩散）
    kappa_bar = kappa + alpha * v * le / 2.0

    # 扩散项刚度矩阵
    K_diff = (kappa_bar / le) * np.array([[1, -1],
                                          [-1, 1]])

    # 对流项刚度矩阵
    K_conv = (v / 2.0) * np.array([[-1, 1],
                                    [-1, 1]])

    return K_diff + K_conv


# =========================
# 3. 组装未加边界的总体刚度矩阵
# =========================
def assemble_global_matrix(nel, L, v, kappa, alpha):
    """
    组装完整的总体刚度矩阵（未施加边界条件，用于矩阵性质分析）
    """
    nn = nel + 1
    le = L / nel
    K = np.zeros((nn, nn))
    Ke = element_matrix(kappa, v, le, alpha)

    for e in range(nel):
        nodes = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[nodes[i], nodes[j]] += Ke[i, j]
    return K


# =========================
# 4. 精确解（数值稳定写法）
# =========================
def exact_solution(x, v, kappa, L):
    """
    一维稳态对流扩散方程解析解
    大Peclet数下采用渐近形式避免指数溢出
    """
    Pe_global = v * L / kappa
    if Pe_global > 700:
        # 强对流渐近形式：边界层集中在右端
        return np.exp(v * (x - L) / kappa)
    else:
        return (np.exp(v * x / kappa) - 1) / (np.exp(Pe_global) - 1)


# =========================
# 5. 施加Dirichlet边界条件
# =========================
def apply_bc(K, F, left_val, right_val):
    """
    正确施加狄利克雷边界条件：仅修改边界对应行，保留列耦合关系
    """
    K = K.copy()
    F = F.copy()

    # 左边界 x=0
    K[0, :] = 0
    K[0, 0] = 1
    F[0] = left_val

    # 右边界 x=L
    K[-1, :] = 0
    K[-1, -1] = 1
    F[-1] = right_val

    return K, F


# =========================
# 6. 主求解函数
# =========================
def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    完整求解一维对流扩散方程
    输入：单元数、域长、对流速度、扩散系数、稳定参数
    输出：节点坐标、数值解、精确解、原始总体刚度矩阵
    """
    nn = nel + 1
    le = L / nel
    x = np.linspace(0, L, nn)

    # 组装原始总体矩阵
    K_raw = assemble_global_matrix(nel, L, v, kappa, alpha)
    F = np.zeros(nn)

    # 施加边界条件
    K, F = apply_bc(K_raw, F, 0.0, 1.0)

    # 求解线性方程组
    theta = np.linalg.solve(K, F)

    # 计算节点处精确解
    theta_exact = exact_solution(x, v, kappa, L)

    return x, theta, theta_exact, K_raw


# =========================
# 7. 单算例计算+绘图
# =========================
def run_case(Pe_target, nel=20, L=1.0, v=1.0, save_fig=False):
    """
    运行指定单元Peclet数的算例，对比三种方法并绘图
    返回各方法的最大误差字典
    """
    le = L / nel
    kappa = v * le / (2 * Pe_target)  # 由单元Pe反算扩散系数

    # 三种方法的alpha参数
    methods = {
        "标准Galerkin": 0.0,
        "迎风格式": 1.0,
        "SUPG": alpha_supg(Pe_target)
    }

    plt.figure(figsize=(8, 5))

    # 绘制高分辨率精确解曲线
    x_dense = np.linspace(0, L, 500)
    theta_exact_dense = exact_solution(x_dense, v, kappa, L)
    plt.plot(x_dense, theta_exact_dense, "k--", linewidth=2, label="精确解")

    error_dict = {}
    for name, alpha in methods.items():
        x, theta, theta_exact, _ = solve_advection_diffusion(nel, L, v, kappa, alpha)
        max_err = np.max(np.abs(theta - theta_exact))
        error_dict[name] = max_err
        plt.plot(x, theta, marker="o", markersize=4,
                 label=f"{name}，最大误差={max_err:.2e}")

    plt.title(f"一维对流扩散数值解对比（单元Pe={Pe_target}，单元数nel={nel}）")
    plt.xlabel("x")
    plt.ylabel("theta")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"Pe_{Pe_target}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return error_dict


# =========================
# 8. 矩阵性质分析
# =========================
def matrix_analysis(nel=20, L=1.0, v=1.0, Pe_target=3.0):
    """
    分析Pe=3.0下标准Galerkin总体矩阵的性质（未施加边界条件）
    """
    le = L / nel
    kappa = v * le / (2 * Pe_target)
    K = assemble_global_matrix(nel, L, v, kappa, alpha=0.0)

    print("\n" + "=" * 60)
    print(f"矩阵性质分析（标准Galerkin，单元Pe={Pe_target}，未施加边界）")
    print("=" * 60)

    # 1. 对称性检测
    sym_error = np.max(np.abs(K - K.T))
    print(f"1. 对称性检测：最大不对称差值 = {sym_error:.2e}")
    if sym_error < 1e-10:
        print("   结论：矩阵对称")
    else:
        print("   结论：矩阵非对称，由对流项的非对称性导致")

    # 2. 特征值与正定性检测
    eigvals = np.linalg.eigvals(K)
    min_real_eig = np.min(np.real(eigvals))
    print(f"2. 正定性检测：最小实特征值 = {min_real_eig:.6f}")
    if min_real_eig > 1e-10:
        print("   结论：矩阵正定")
    elif min_real_eig > -1e-10:
        print("   结论：矩阵半正定（存在零特征值，对应常数齐次解）")
    else:
        print("   结论：矩阵存在负实特征值，双线性形式丧失椭圆性")

    # 3. 对流项的影响与振荡原因
    print("\n3. 对流项对矩阵性质与数值振荡的影响：")
    print("   (1) 对称性：扩散项刚度矩阵对称，对流项刚度矩阵非对称，叠加后整体非对称；")
    print("   (2) 椭圆性：扩散项保证双线性形式的椭圆性，对流占优（Pe>1）时椭圆性丧失，")
    print("       弱形式不适定，数值解出现空间振荡；")
    print("   (3) 稳定化本质：引入人工扩散增强扩散项占比，恢复椭圆性，消除振荡；")
    print("       迎风格式扩散过量导致精度损失，SUPG取最优alpha兼顾稳定与精度。")
    print("=" * 60)

    return K


# =========================
# 9. 附加题：网格收敛性分析
# =========================
def grid_convergence_study(L=1.0, v=1.0, Pe_global=120, save_fig=False):
    """
    附加题：改变单元数研究网格加密对解的影响
    固定全局Peclet数，单元数取 [10, 20, 40, 80]
    分析Galerkin振荡变化、SUPG误差收敛规律，绘制误差收敛曲线
    """
    nel_list = [10, 20, 40, 80]
    kappa = v * L / Pe_global  # 固定物理参数，全局Pe恒定

    galerkin_errors = []
    supg_errors = []
    le_list = []
    pe_elem_list = []

    print("\n" + "=" * 70)
    print(f"附加题：网格收敛性分析（全局Pe={Pe_global}，固定物理参数）")
    print("=" * 70)
    print(f"{'单元数nel':<10} {'单元长度le':<12} {'单元Pe':<10} {'Galerkin最大误差':<18} {'SUPG最大误差':<18}")
    print("-" * 70)

    # 遍历不同网格密度
    for nel in nel_list:
        le = L / nel
        pe_elem = v * le / (2 * kappa)
        le_list.append(le)
        pe_elem_list.append(pe_elem)

        # Galerkin误差
        _, theta_gal, theta_exact, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=0.0)
        err_gal = np.max(np.abs(theta_gal - theta_exact))
        galerkin_errors.append(err_gal)

        # SUPG误差
        alpha_opt = alpha_supg(pe_elem)
        _, theta_supg, theta_exact, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=alpha_opt)
        err_supg = np.max(np.abs(theta_supg - theta_exact))
        supg_errors.append(err_supg)

        print(f"{nel:<10} {le:<12.4f} {pe_elem:<10.2f} {err_gal:<18.6e} {err_supg:<18.6e}")
    print("=" * 70)

    # 绘制双对数误差收敛曲线
    plt.figure(figsize=(8, 5))
    plt.loglog(le_list, galerkin_errors, 'o-', linewidth=2, markersize=6, label='标准Galerkin')
    plt.loglog(le_list, supg_errors, 's-', linewidth=2, markersize=6, label='SUPG')

    # 标注一阶收敛参考线
    ref_line = 0.5 * np.array(le_list)
    plt.loglog(le_list, ref_line, 'k:', linewidth=1.5, label='一阶收敛参考线')

    plt.xlabel('单元长度 le (对数坐标)')
    plt.ylabel('最大节点误差 (对数坐标)')
    plt.title(f'网格收敛曲线（全局Pe={Pe_global}）')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_fig:
        plt.savefig("网格收敛曲线.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 绘制不同网格下Galerkin解的对比图（观察振荡变化）
    plt.figure(figsize=(8, 5))
    x_dense = np.linspace(0, L, 500)
    theta_exact_dense = exact_solution(x_dense, v, kappa, L)
    plt.plot(x_dense, theta_exact_dense, 'k--', linewidth=2, label='精确解')

    for nel in nel_list:
        x, theta, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=0.0)
        plt.plot(x, theta, marker='o', markersize=3, label=f'Galerkin, nel={nel}')

    plt.xlabel('x')
    plt.ylabel('theta')
    plt.title(f'不同网格密度下Galerkin解的振荡对比（全局Pe={Pe_global}）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        plt.savefig("Galerkin网格加密对比.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 结果分析说明
    print("\n网格收敛分析结论：")
    print("1. 随着网格加密（单元数增加、单元长度减小），单元Peclet数逐步降低；")
    print("2. Galerkin方法：单元Pe>1时存在明显空间振荡，误差较大；当单元Pe<1后振荡消失，误差快速下降；")
    print("3. SUPG方法：在所有网格密度下均无振荡，节点值为精确解，误差仅为浮点舍入级别；")
    print("4. 网格足够密时，Galerkin误差随单元长度一阶收敛，SUPG始终保持最高精度。")
    print("=" * 70)
# =========================
# 10. 主程序
# =========================
if __name__ == "__main__":
    print("一维对流扩散方程有限元求解程序")
    print("课程：3.8 对流扩散方程 程序设计作业")
    print("=" * 60)

    # 任务2：两种Peclet数算例
    print("\n>>> 算例1：单元Pe = 0.1（扩散主导）")
    err_pe01 = run_case(0.1, save_fig=True)

    print("\n>>> 算例2：单元Pe = 3.0（对流占优）")
    err_pe3 = run_case(3.0, save_fig=True)

    # 输出误差对比表
    print("\n" + "=" * 60)
    print("三种格式最大节点误差对比表")
    print("=" * 60)
    print(f"{'方法':<12} {'Pe=0.1':<16} {'Pe=3.0':<16}")
    print("-" * 60)
    for method in err_pe01.keys():
        print(f"{method:<12} {err_pe01[method]:<16.6e} {err_pe3[method]:<16.6e}")
    print("=" * 60)

    # 任务4：矩阵性质分析
    matrix_analysis()

    # 附加题：网格收敛性分析
    grid_convergence_study(save_fig=True)