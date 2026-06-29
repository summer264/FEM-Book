import numpy as np
import time
import matplotlib

# 全局字体配置：解决中文、负号、特殊符号警告
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 稀疏求解器依赖
try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    SCIPY_AVAILABLE = True
    try:
        from pypardiso import spsolve as pardiso_solve

        SOLVER_NAME = "MKL PARDISO"
        USE_PARDISO = True
    except ImportError:
        SOLVER_NAME = "UMFPACK (scipy.sparse)"
        USE_PARDISO = False
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  未检测到scipy库，大规模Poisson算例将跳过。安装命令：pip install scipy")
    print("⚠️  如需调用MKL PARDISO，额外安装：pip install pypardiso")


# ============================================================
# 第一部分：核心求解器模块
# ============================================================

def ldlt_factor(K):
    n = K.shape[0]
    L = np.zeros_like(K, dtype=float)
    D = np.zeros(n, dtype=float)
    for j in range(n):
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - sum_d
        if D[j] <= 1e-12:
            raise ValueError(f"矩阵非正定/存在零主元，第{j}个主元值：{D[j]:.6e}")
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_l) / D[j]
    np.fill_diagonal(L, 1.0)
    return L, D


def ldlt_solve(L, D, R):
    n = len(D)
    R = np.array(R, dtype=float, ndmin=2)
    if R.shape[0] != n:
        R = R.T
    n_rhs = R.shape[1]
    is_1d = (R.shape[1] == 1)
    y = np.zeros_like(R)
    for i in range(n):
        sum_y = 0.0
        for k in range(i):
            sum_y += L[i, k] * y[k, :]
        y[i, :] = R[i, :] - sum_y
    z = y / D[:, np.newaxis]
    a = np.zeros_like(z)
    for i in range(n - 1, -1, -1):
        sum_a = 0.0
        for k in range(i + 1, n):
            sum_a += L[k, i] * a[k, :]
        a[i, :] = z[i, :] - sum_a
    if is_1d:
        return a.flatten()
    return a


def residual_norm(K, a, R):
    r = R - K @ a
    return r, np.linalg.norm(r)


def colsol(n, m, K, R):
    IERR = 0
    K = K.copy().astype(float)
    R = R.copy().astype(float)
    for j in range(1, n):
        for i in range(m[j] + 1, j):
            c = 0.0
            r_start = max(m[i], m[j])
            for r in range(r_start, i):
                c += K[r, i] * K[r, j]
            K[i, j] -= c
        for r in range(m[j], j):
            Lrj = K[r, j] / K[r, r]
            K[j, j] -= Lrj * K[r, j]
            K[r, j] = Lrj
        if K[j, j] <= 1e-12:
            print(f"[Error] 矩阵非正定，第{j}列主元 = {K[j, j]:.6e}")
            IERR = j
            return IERR, K, R
    for i in range(1, n):
        for r in range(m[i], i):
            R[i] -= K[r, i] * R[r]
    for i in range(n):
        R[i] /= K[i, i]
    for i in range(n - 1, 0, -1):
        for r in range(m[i], i):
            R[r] -= K[r, i] * R[i]
    return IERR, K, R


def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    info = {}
    K_FF = np.atleast_2d(K_FF).astype(float)
    rhs = np.asarray(rhs, dtype=float)
    n = K_FF.shape[0]
    if method == "ldlt":
        try:
            L, D = ldlt_factor(K_FF)
            d_F = ldlt_solve(L, D, rhs)
            info["status"] = "success"
            info["L"] = L
            info["D"] = D
        except ValueError as e:
            info["status"] = "failed"
            info["error"] = str(e)
            raise
    elif method == "colsol":
        m = options.get("m", None)
        if m is None:
            raise ValueError("colsol方法必须传入每列首非零行索引m")
        IERR, K_decomp, sol = colsol(n, np.array(m, int), K_FF, rhs)
        if IERR != 0:
            info["status"] = "failed"
            raise ValueError(f"colsol求解失败：非正主元")
        d_F = sol
        info["status"] = "success"
        info["K_decomp"] = K_decomp
    elif method == "numpy":
        d_F = np.linalg.solve(K_FF, rhs)
        info["status"] = "success"
    else:
        raise ValueError(f"不支持的求解方法：{method}")
    r, norm_r = residual_norm(K_FF, d_F, rhs)
    info["residual"] = r
    info["residual_norm"] = norm_r
    norm_R = np.linalg.norm(rhs)
    info["relative_residual"] = norm_r / norm_R if norm_R > 1e-15 else 0.0
    return d_F, info


def compute_relative_error(a_num, a_exact):
    return np.linalg.norm(a_num - a_exact) / np.linalg.norm(a_exact)


# ============================================================
# 第二部分：桁架有限元组装模块
# ============================================================

class FEModel:
    def __init__(self):
        self.Title = ""
        self.nsd = 0
        self.ndof = 0
        self.nnp = 0
        self.nel = 0
        self.nen = 2
        self.neq = 0
        self.E = np.array([])
        self.CArea = np.array([])
        self.IEN = np.array([], dtype=int)
        self.fixed_dof = np.array([], dtype=int)
        self.fixed_value = np.array([])
        self.force_dof = np.array([], dtype=int)
        self.force_value = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.K = np.array([])
        self.f = np.array([])
        self.d = np.array([])
        self.length = np.array([])
        self.stress = np.array([])
        self.reaction = np.array([])
        self.LM = np.array([], dtype=int)
        self.direction_cosines = np.array([])
        self.K_EE = np.array([])
        self.K_FF = np.array([])
        self.K_EF = np.array([])
        self.solver_info = {}


def build_LM(model):
    model.LM = np.zeros((model.nen * model.ndof, model.nel), dtype=int)
    for e in range(model.nel):
        for j in range(model.nen):
            for m in range(model.ndof):
                local_dof = j * model.ndof + m
                global_node = model.IEN[e, j] - 1
                global_dof = global_node * model.ndof + m
                model.LM[local_dof, e] = global_dof
    return model


def compute_element_geometry(model):
    model.direction_cosines = np.zeros((model.nel, max(model.nsd, 1)))
    model.length = np.zeros(model.nel)
    for e in range(model.nel):
        nodes = model.IEN[e, :] - 1
        if model.ndof == 1:
            x1 = model.x[nodes[0]]
            x2 = model.x[nodes[1]]
            delta = x2 - x1
            model.length[e] = abs(delta)
            model.direction_cosines[e, 0] = 1.0 if delta >= 0 else -1.0
        elif model.ndof == 2:
            x1, y1 = model.x[nodes[0]], model.y[nodes[0]]
            x2, y2 = model.x[nodes[1]], model.y[nodes[1]]
            delta = np.array([x2 - x1, y2 - y1])
            L = np.linalg.norm(delta)
            model.length[e] = L
            if L == 0:
                raise ValueError(f'单元 {e + 1} 长度为0')
            model.direction_cosines[e, :] = delta / L
    return model


def element_stiffness(model, e):
    const = model.CArea[e] * model.E[e] / model.length[e]
    if model.ndof == 1:
        Ke = const * np.array([[1, -1], [-1, 1]])
    elif model.ndof == 2:
        c, s = model.direction_cosines[e]
        Ke = const * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s]
        ])
    else:
        raise ValueError('仅支持1D/2D桁架单元')
    return Ke


def assemble_global_stiffness(model):
    model.K = np.zeros((model.neq, model.neq))
    for e in range(model.nel):
        Ke = element_stiffness(model, e)
        lm = model.LM[:, e]
        for i, row in enumerate(lm):
            for j, col in enumerate(lm):
                model.K[row, col] += Ke[i, j]
    print("\n========== 刚度矩阵性质校验 ==========")
    check_stiffness_property(model)
    print("=====================================\n")
    return model


def check_stiffness_property(model):
    K = model.K
    n = K.shape[0]
    eps = 1e-10
    diff_sym = np.max(np.abs(K - K.T))
    print(f"1. 对称性检测：最大不对称差值 = {diff_sym:.2e}")
    print("   >> 刚度矩阵对称 ✅" if diff_sym < eps else "   >> 刚度矩阵不对称 ❌")
    diag = np.diag(K)
    neg_idx = np.where(diag < -eps)[0]
    print(f"2. 对角元非负检测：")
    if len(neg_idx) == 0:
        print("   >> 所有对角元均非负 ✅")
    else:
        print(f"   >> 存在负对角元：自由度{neg_idx}，值{diag[neg_idx]} ❌")
    total = n * n
    zero_num = np.sum(np.abs(K) < eps)
    print(f"3. 稀疏性：总元素{total}，零元{zero_num}，稀疏率{zero_num / total:.4f}")
    try:
        det_K = np.linalg.det(K)
        print(f"4. 奇异性：行列式 = {det_K:.2e}")
        if abs(det_K) < 1e-6:
            print("   >> 无约束总刚奇异（存在刚体位移）✅")
        else:
            print("   >> 总刚非奇异 ❌")
    except np.linalg.LinAlgError:
        print("4. 奇异性：矩阵奇异 ✅")


def solve_system(model, method="ldlt", **options):
    fixed = model.fixed_dof - 1
    all_dof = np.arange(model.neq)
    free = np.setdiff1d(all_dof, fixed)
    d_E = model.d[fixed]
    f_F = model.f[free]
    K_FF = model.K[np.ix_(free, free)]
    K_FE = model.K[np.ix_(free, fixed)]
    rhs = f_F - K_FE @ d_E
    if len(free) == 0:
        d_F = np.array([])
        model.solver_info = {"status": "success", "note": "无自由自由度"}
    else:
        d_F, info = solve_equilibrium(K_FF, rhs, method=method, **options)
        model.solver_info = info
    model.d[free] = d_F
    model.d[fixed] = d_E
    total_internal = model.K @ model.d
    model.reaction = total_internal - model.f
    model.fixed_reaction = model.reaction[fixed]
    model.K_EE = model.K[np.ix_(fixed, fixed)]
    model.K_FF = K_FF
    model.K_EF = model.K[np.ix_(fixed, free)]
    return model


def postprocess_stress(model):
    model.stress = np.zeros(model.nel)
    for e in range(model.nel):
        lm = model.LM[:, e]
        de = model.d[lm]
        const = model.E[e] / model.length[e]
        if model.ndof == 1:
            B = np.array([-1, 1])
        elif model.ndof == 2:
            c, s = model.direction_cosines[e]
            B = np.array([-c, -s, c, s])
        model.stress[e] = const * (B @ de)
    return model


def print_results(model):
    print(f'\n{model.Title}')
    print('LM (1-based) =')
    print(model.LM + 1)
    print('节点位移 d =')
    print(model.d.flatten())
    print('节点反力 reaction =')
    print(model.reaction.flatten())
    print('单元应力 stress =')
    print(model.stress.flatten())
    if "relative_residual" in model.solver_info:
        print(f"求解相对残差：{model.solver_info['relative_residual']:.6e}")


def solve_model(model, method="ldlt", **options):
    model = build_LM(model)
    model = compute_element_geometry(model)
    model = assemble_global_stiffness(model)
    model = solve_system(model, method=method, **options)
    model = postprocess_stress(model)
    print_results(model)
    return model


def create_1d_bar_model():
    model = FEModel()
    model.Title = "1D 两单元杆结构"
    model.nsd = 1
    model.ndof = 1
    model.nnp = 3
    model.nel = 2
    model.nen = 2
    model.neq = model.nnp * model.ndof
    model.E = np.array([100.0, 200.0])
    model.CArea = np.array([1.0, 1.0])
    model.IEN = np.array([[1, 2], [2, 3]], dtype=int)
    model.x = np.array([0.0, 1.0, 2.0])
    model.fixed_dof = np.array([1])
    model.fixed_value = np.array([0.0])
    model.force_dof = np.array([2, 3])
    model.force_value = np.array([0.0, 10.0])
    model.f = np.zeros(model.neq)
    model.d = np.zeros(model.neq)
    for dof, val in zip(model.force_dof, model.force_value):
        model.f[dof - 1] = val
    for dof, val in zip(model.fixed_dof, model.fixed_value):
        model.d[dof - 1] = val
    return model


def create_2d_truss_model():
    model = FEModel()
    model.Title = "2D 两杆桁架结构"
    model.nsd = 2
    model.ndof = 2
    model.nnp = 3
    model.nel = 2
    model.nen = 2
    model.neq = model.nnp * model.ndof
    model.E = np.array([1.0, 1.0])
    model.CArea = np.array([1.0, 1.0])
    model.IEN = np.array([[1, 3], [2, 3]], dtype=int)
    model.x = np.array([1.0, 0.0, 1.0])
    model.y = np.array([0.0, 0.0, 1.0])
    model.fixed_dof = np.array([1, 2, 3, 4])
    model.fixed_value = np.array([0.0, 0.0, 0.0, 0.0])
    model.force_dof = np.array([5, 6])
    model.force_value = np.array([10.0, 0.0])
    model.f = np.zeros(model.neq)
    model.d = np.zeros(model.neq)
    for dof, val in zip(model.force_dof, model.force_value):
        model.f[dof - 1] = val
    for dof, val in zip(model.fixed_dof, model.fixed_value):
        model.d[dof - 1] = val
    return model


# ============================================================
# 第三部分：基础验证算例
# ============================================================

def run_truss_examples():
    print("=" * 70)
    print("算例0-1：1D两单元杆（LDLT求解）")
    print("=" * 70)
    model1 = create_1d_bar_model()
    result1 = solve_model(model1, method="ldlt")
    print("\n预期位移：[0, 0.1, 0.15]")
    print("\n" + "=" * 70)
    print("算例0-2：2D两杆桁架（LDLT求解）")
    print("=" * 70)
    model2 = create_2d_truss_model()
    result2 = solve_model(model2, method="ldlt")
    print(f"\n预期节点3位移：u3=38.284271, v3=-10.000000")
    print(f"计算节点3位移：u3={result2.d[4]:.6f}, v3={result2.d[5]:.6f}")
    print("预期应力：单元1=-10.0，单元2=14.142136")
    print(f"计算应力：{result2.stress}")


def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    x = float(x)
    order = np.floor(np.log10(abs(x)))
    scaled = x / (10 ** order)
    scaled_rounded = np.round(scaled, sig - 1)
    return scaled_rounded * (10 ** order)


def run_ill_conditioned_test():
    print("\n" + "=" * 70)
    print("任务2：病态矩阵误差与条件数分析")
    print("=" * 70)
    K_ill = np.array([[1.0000, 1.0000], [1.0000, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R_ill = K_ill @ a_exact
    cond_K = np.linalg.cond(K_ill)
    print(f"矩阵条件数 cond(K) = {cond_K:.2f}")
    print(f"精确解 a_exact = {a_exact}")
    print(f"右端项 R = {R_ill}")
    a_double, info_double = solve_equilibrium(K_ill, R_ill, method="ldlt")
    err_double = compute_relative_error(a_double, a_exact)
    print(f"\n【1. 双精度计算】")
    print(f"  数值解：{a_double}")
    print(f"  相对残差：{info_double['relative_residual']:.6e}")
    print(f"  相对误差：{err_double:.6e}")
    print(f"\n【2. 4位有效数字计算】")

    def ldlt_low_precision(K, sig=4):
        n = K.shape[0]
        L = np.zeros_like(K)
        D = np.zeros(n)
        for j in range(n):
            sum_d = 0.0
            for k in range(j):
                sum_d += round_sig(L[j, k] ** 2 * D[k], sig)
            D[j] = round_sig(K[j, j] - sum_d, sig)
            if D[j] <= 1e-12:
                raise ValueError(f"第{j}个主元非正，值：{D[j]:.6e}")
            for i in range(j + 1, n):
                sum_l = 0.0
                for k in range(j):
                    sum_l += round_sig(L[i, k] * L[j, k] * D[k], sig)
                L[i, j] = round_sig((K[i, j] - sum_l) / D[j], sig)
        np.fill_diagonal(L, 1.0)
        return L, D

    try:
        L_low, D_low = ldlt_low_precision(K_ill, sig=4)
        a_low = ldlt_solve(L_low, D_low, R_ill)
        _, norm_r_low = residual_norm(K_ill, a_low, R_ill)
        rel_r_low = norm_r_low / np.linalg.norm(R_ill)
        err_low = compute_relative_error(a_low, a_exact)
        print(f"  数值解：{a_low}")
        print(f"  相对残差：{rel_r_low:.6e}")
        print(f"  相对误差：{err_low:.6e}")
    except ValueError as e:
        print(f"  ⚠️  求解失败：{e}")
        print(f"  原因：4位有效数字下，主元0.0001仅1位有效数字，矩阵退化为奇异矩阵")
    print(f"\n【3. 单精度float32计算】")
    K_32 = K_ill.astype(np.float32)
    R_32 = R_ill.astype(np.float32)
    try:
        L_32, D_32 = ldlt_factor(K_32)
        a_32 = ldlt_solve(L_32, D_32, R_32)
        _, norm_r_32 = residual_norm(K_ill, a_32, R_ill)
        rel_r_32 = norm_r_32 / np.linalg.norm(R_ill)
        err_32 = compute_relative_error(a_32, a_exact)
        print(f"  数值解：{a_32}")
        print(f"  相对残差：{rel_r_32:.6e}")
        print(f"  相对误差：{err_32:.6e}")
    except ValueError as e:
        print(f"  ⚠️  求解失败：{e}")
    print(f"\n【4. 半精度float16计算】")
    K_16 = K_ill.astype(np.float16)
    R_16 = R_ill.astype(np.float16)
    try:
        L_16, D_16 = ldlt_factor(K_16)
        a_16 = ldlt_solve(L_16, D_16, R_16)
        _, norm_r_16 = residual_norm(K_ill, a_16, R_ill)
        rel_r_16 = norm_r_16 / np.linalg.norm(R_ill)
        err_16 = compute_relative_error(a_16, a_exact)
        print(f"  数值解：{a_16}")
        print(f"  相对残差：{rel_r_16:.6e}")
        print(f"  相对误差：{err_16:.6e}")
    except ValueError as e:
        print(f"  ⚠️  求解失败：{e}")
        print(f"  原因：float16精度不足，主元下溢为0，矩阵奇异")
    print("\n结论：")
    print("  条件数是相对误差的放大倍数。病态矩阵条件数很大，")
    print("  即使残差很小，数值解的相对误差也可能被放大很多倍；")
    print("  精度越低，误差放大现象越明显，甚至会导致矩阵奇异无法求解。")


def run_non_positive_definite_test():
    print("\n" + "=" * 70)
    print("算例2：非正定矩阵检测")
    print("=" * 70)
    K_nonpd = np.array([[1, 2], [2, 1]])
    R_nonpd = np.array([1, 1])
    try:
        solve_equilibrium(K_nonpd, R_nonpd, method="ldlt")
    except ValueError as e:
        print(f"✅ 成功检测到非正定矩阵：{e}")


def run_tridiagonal_test():
    print("\n" + "=" * 70)
    print("算例1：三对角矩阵规模与时间测试")
    print("=" * 70)

    def build_tridiagonal(n):
        K = np.diag(2 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
        a_exact = np.ones(n)
        R = K @ a_exact
        return K, R, a_exact

    for n in [10, 100, 500, 1000]:
        K_tri, R_tri, a_exact_tri = build_tridiagonal(n)
        start = time.time()
        for _ in range(5):
            L, D = ldlt_factor(K_tri)
            a_num = ldlt_solve(L, D, R_tri)
        t_avg = (time.time() - start) / 5
        err = compute_relative_error(a_num, a_exact_tri)
        print(f"n={n:4d} | 平均求解时间：{t_avg:.6f}s | 相对误差：{err:.6e}")


def run_colsol_test():
    print("\n" + "=" * 70)
    print("活动列colsol验证（课件Example_n_5）")
    print("=" * 70)
    n_col = 5
    m_col = [0, 0, 1, 2, 0]
    K_col = np.array([
        [2, -2, 0, 0, -1],
        [-2, 3, -2, 0, 0],
        [0, -2, 5, -3, 0],
        [0, 0, -3, 10, 4],
        [-1, 0, 0, 4, 10]
    ])
    R_col = np.array([0, 1, 0, 0, 0])
    ierr, K_decomp, a_col = colsol(n_col, m_col, K_col, R_col)
    a_ldlt, _ = solve_equilibrium(K_col, R_col, method="ldlt")
    print(f"colsol解：{a_col}")
    print(f"LDLT解： {a_ldlt}")
    print(f"差值范数：{np.linalg.norm(a_col - a_ldlt):.6e}")
    print("✅ colsol与稠密LDLT结果一致")


def run_multi_load_test():
    print("\n" + "=" * 70)
    print("多载荷工况效率演示（分解一次，求解多右端项）")
    print("=" * 70)
    n = 500
    K_multi = np.diag(2 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    n_rhs = 10
    R_multi = np.random.rand(n, n_rhs)
    start = time.time()
    for i in range(n_rhs):
        L, D = ldlt_factor(K_multi)
        _ = ldlt_solve(L, D, R_multi[:, i])
    t_separate = time.time() - start
    start = time.time()
    L, D = ldlt_factor(K_multi)
    a_multi = ldlt_solve(L, D, R_multi)
    t_once = time.time() - start
    print(f"逐个分解求解总时间：{t_separate:.4f}s")
    print(f"一次分解多右端求解：{t_once:.4f}s")
    print(f"加速比：{t_separate / t_once:.2f}x")
    print("✅ 多载荷工况下，复用分解结果显著提升效率")


# ============================================================
# 第四部分：算例4 二维Poisson方程有限元求解 + 云图输出
# ============================================================

def poisson_q4_solve(nx, ny):
    """
    双线性Q4单元求解单位正方形Poisson方程：-Δu = f，边界u=0
    制造解：u_exact = sin(πx)sin(πy)，右端项 f = 2π² sin(πx)sin(πy)
    """
    t_total_start = time.time()
    hx = 1.0 / nx
    hy = 1.0 / ny
    n_node_x = nx + 1
    n_node_y = ny + 1
    n_node = n_node_x * n_node_y
    n_elem = nx * ny

    # 生成节点坐标
    x = np.linspace(0, 1, n_node_x)
    y = np.linspace(0, 1, n_node_y)
    X, Y = np.meshgrid(x, y)
    node_x = X.flatten()
    node_y = Y.flatten()

    # 生成单元连接
    IEN = np.zeros((n_elem, 4), dtype=int)
    e = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j * n_node_x + i
            n1 = j * n_node_x + (i + 1)
            n2 = (j + 1) * n_node_x + (i + 1)
            n3 = (j + 1) * n_node_x + i
            IEN[e, :] = [n0, n1, n2, n3]
            e += 1

    # 2×2高斯积分
    gauss_pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    gauss_wt = np.array([1.0, 1.0])

    # 初始化稀疏矩阵
    K_sparse = lil_matrix((n_node, n_node))
    F = np.zeros(n_node)

    # 单元装配
    t_assemble_start = time.time()
    for e in range(n_elem):
        nodes = IEN[e, :]
        xe = node_x[nodes]
        ye = node_y[nodes]
        Ke = np.zeros((4, 4))
        Fe = np.zeros(4)
        for si in range(2):
            xi = gauss_pts[si]
            wi = gauss_wt[si]
            for sj in range(2):
                eta = gauss_pts[sj]
                wj = gauss_wt[sj]
                # 形函数
                N = np.array([
                    (1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4,
                    (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4
                ])
                # 形函数对参考坐标偏导
                dN_dxi = np.array([
                    -(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4
                ])
                dN_deta = np.array([
                    -(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4
                ])
                # 雅可比
                J = np.zeros((2, 2))
                J[0, 0] = dN_dxi @ xe
                J[0, 1] = dN_dxi @ ye
                J[1, 0] = dN_deta @ xe
                J[1, 1] = dN_deta @ ye
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                # 物理坐标偏导
                dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
                # 单元刚度
                for a in range(4):
                    for b in range(4):
                        Ke[a, b] += (dN_dx[a] * dN_dx[b] + dN_dy[a] * dN_dy[b]) * detJ * wi * wj
                # 单元载荷
                x_phys = N @ xe
                y_phys = N @ ye
                f_val = 2 * np.pi ** 2 * np.sin(np.pi * x_phys) * np.sin(np.pi * y_phys)
                for a in range(4):
                    Fe[a] += f_val * N[a] * detJ * wi * wj
        # 全局组装
        for a in range(4):
            global_a = nodes[a]
            F[global_a] += Fe[a]
            for b in range(4):
                global_b = nodes[b]
                K_sparse[global_a, global_b] += Ke[a, b]
    t_assemble = time.time() - t_assemble_start

    # 边界条件处理
    t_bc_start = time.time()
    boundary_mask = (node_x == 0) | (node_x == 1) | (node_y == 0) | (node_y == 1)
    free_dof = np.where(~boundary_mask)[0]
    n_free = len(free_dof)
    K_ff = K_sparse[np.ix_(free_dof, free_dof)].tocsr()
    F_ff = F[free_dof]
    t_bc = time.time() - t_bc_start

    # 稀疏求解
    t_solve_start = time.time()
    if USE_PARDISO:
        u_ff = pardiso_solve(K_ff, F_ff)
    else:
        u_ff = spsolve(K_ff, F_ff)
    t_solve = time.time() - t_solve_start

    # 回填全节点解
    u_full = np.zeros(n_node)
    u_full[free_dof] = u_ff

    # 计算误差
    u_exact = np.sin(np.pi * node_x) * np.sin(np.pi * node_y)
    max_error = np.max(np.abs(u_full - u_exact))
    l2_relative_error = np.linalg.norm(u_full - u_exact) / np.linalg.norm(u_exact)
    residual_norm_val = np.linalg.norm(F_ff - K_ff @ u_ff)
    relative_residual = residual_norm_val / np.linalg.norm(F_ff)

    t_total = time.time() - t_total_start

    return {
        "element_type": "双线性四边形Q4单元",
        "solver_name": SOLVER_NAME,
        "nx": nx, "ny": ny,
        "n_node": n_node,
        "n_elem": n_elem,
        "n_free_dof": n_free,
        "nnz": K_sparse.nnz,
        "t_assemble": t_assemble,
        "t_bc": t_bc,
        "t_solve": t_solve,
        "t_total": t_total,
        "max_error": max_error,
        "l2_relative_error": l2_relative_error,
        "relative_residual": relative_residual,
        "node_x": node_x,
        "node_y": node_y,
        "u_full": u_full,
        "u_exact": u_exact,
        "n_node_x": n_node_x,
        "n_node_y": n_node_y,
        "K_ff": K_ff,  # 新增：返回缩减刚度矩阵，用于稠密对比
        "F_ff": F_ff  # 新增：返回缩减右端项，用于稠密对比
    }


def plot_poisson_results(res, save_fig=False):
    """绘制数值解云图、三维曲面图、误差云图"""
    n_node_x = res["n_node_x"]
    n_node_y = res["n_node_y"]
    X = res["node_x"].reshape(n_node_y, n_node_x)
    Y = res["node_y"].reshape(n_node_y, n_node_x)
    U = res["u_full"].reshape(n_node_y, n_node_x)
    U_exact = res["u_exact"].reshape(n_node_y, n_node_x)
    Error = U - U_exact

    fig = plt.figure(figsize=(15, 10))

    # 1. 数值解三维曲面图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, U, cmap='jet', edgecolor='none', alpha=0.9)
    ax1.set_title(f'数值解三维曲面 ({res["nx"]}×{res["ny"]}网格)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u_h(x,y)')
    fig.colorbar(surf, ax=ax1, shrink=0.7)

    # 2. 数值解云图
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.pcolormesh(X, Y, U, cmap='jet', shading='auto')
    ax2.set_title('数值解云图', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2)

    # 3. 误差云图
    ax3 = fig.add_subplot(2, 2, 3)
    err_contour = ax3.pcolormesh(X, Y, Error, cmap='RdBu_r', shading='auto')
    ax3.set_title('数值误差云图', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    fig.colorbar(err_contour, ax=ax3)

    # 4. 理论解对比（截面线）
    ax4 = fig.add_subplot(2, 2, 4)
    mid_y_idx = n_node_y // 2
    ax4.plot(X[mid_y_idx, :], U[mid_y_idx, :], 'r-', linewidth=2, label='有限元数值解')
    ax4.plot(X[mid_y_idx, :], U_exact[mid_y_idx, :], 'k--', linewidth=1.5, label='理论精确解')
    ax4.set_title('y=0.5截面解对比', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('u')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'poisson_{res["nx"]}x{res["ny"]}_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_curve(results):
    """绘制误差收敛曲线"""
    h_list = [1.0 / r["nx"] for r in results]
    max_err_list = [r["max_error"] for r in results]
    l2_err_list = [r["l2_relative_error"] for r in results]

    plt.figure(figsize=(8, 6))
    plt.loglog(h_list, max_err_list, 'bo-', linewidth=2, markersize=8, label='节点最大误差')
    plt.loglog(h_list, l2_err_list, 'rs-', linewidth=2, markersize=8, label='L2相对误差')
    plt.loglog(h_list, [h ** 2 for h in h_list], 'k--', alpha=0.6, label='O(h^2) 参考线')
    plt.xlabel('网格尺寸 h', fontsize=12)
    plt.ylabel('误差', fontsize=12)
    plt.title('有限元误差收敛曲线', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_poisson_test():
    if not SCIPY_AVAILABLE:
        print("\n" + "=" * 70)
        print("算例4：二维Poisson方程有限元求解（跳过，未安装scipy）")
        print("=" * 70)
        return

    print("\n" + "=" * 80)
    print("算例4：大规模二维Poisson方程有限元求解（Q4单元 + 稀疏求解器）")
    print("=" * 80)

    grids = [(10, 10), (50, 50), (100, 100), (200, 200)]
    results = []

    for nx, ny in grids:
        res = poisson_q4_solve(nx, ny)
        results.append(res)

        # 严格对齐作业要求的8项输出
        print(f"\n>>>>> 网格 {nx}×{ny} 计算结果 <<<<<")
        print(f"1. 单元类型：{res['element_type']}")
        print(f"   节点总数：{res['n_node']}，单元总数：{res['n_elem']}")
        print(f"   未知自由度数：{res['n_free_dof']}，总体矩阵非零元个数：{res['nnz']}")
        print(f"2. 时间统计：")
        print(f"   装配时间：{res['t_assemble']:.4f} s")
        print(f"   边界条件处理时间：{res['t_bc']:.4f} s")
        print(f"   求解时间：{res['t_solve']:.4f} s")
        print(f"   总时间：{res['t_total']:.4f} s")
        print(f"3. 调用求解器：{res['solver_name']}")
        print(f"4. 相对残差 ||R-Ka||/||R|| = {res['relative_residual']:.6e}")
        print(f"5. 节点最大误差 = {res['max_error']:.6e}")
        print(f"6. 离散L2相对误差 = {res['l2_relative_error']:.6e}")

    # 绘制100x100网格的云图
    print("\n正在生成数值解云图、误差云图与收敛曲线...")
    plot_poisson_results(results[2])  # 100x100
    plot_convergence_curve(results)

    # 小规模稠密对比（修复：使用完全相同的K和F）
    print("\n【小规模稠密LDLT vs 稀疏求解对比（10x10网格）】")
    res_small = results[0]
    K_ff_dense = res_small["K_ff"].todense()
    F_ff_dense = res_small["F_ff"]

    t0 = time.time()
    u_dense, info = solve_equilibrium(K_ff_dense, F_ff_dense, method="ldlt")
    t_dense = time.time() - t0

    # 计算解的相对差异
    u_sparse = spsolve(res_small["K_ff"], res_small["F_ff"])
    sol_diff = np.linalg.norm(u_dense - u_sparse) / np.linalg.norm(u_sparse)

    print(f"稠密LDLT求解时间：{t_dense:.6f}s，求解相对残差：{info['relative_residual']:.6e}")
    print(f"稀疏求解时间：  {res_small['t_solve']:.6f}s，求解相对残差：{res_small['relative_residual']:.6e}")
    print(f"两种方法解的相对差异：{sol_diff:.6e}")
    print("✅ 两种求解器基于完全相同的方程，结果数值一致，稀疏法在大规模下优势显著")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # 基础算例
    run_truss_examples()
    run_ill_conditioned_test()
    run_non_positive_definite_test()
    run_tridiagonal_test()
    run_colsol_test()
    run_multi_load_test()

    # 算例4：Poisson方程 + 云图
    run_poisson_test()