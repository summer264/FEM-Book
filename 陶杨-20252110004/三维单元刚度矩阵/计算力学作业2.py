import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维桁架单元刚度矩阵。
    x1, x2: 节点坐标 [x, y, z] 作为列表或数组
    E: 弹性模量
    A: 截面面积
    返回: 单元长度 L, 方向余弦 [cx, cy, cz], 刚度矩阵 Ke
    """
    x1 = np.array(x1, dtype=float).flatten()
    x2 = np.array(x2, dtype=float).flatten()
    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("节点坐标必须是长度为3的向量")

    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L == 0:
        raise ValueError("两个节点坐标不能重合")

    n = delta / L
    direction_cosines = n

    projection = np.outer(n, n)
    Ke = (E * A / L) * np.block([[projection, -projection],
                                 [-projection, projection]])
    return L, direction_cosines, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    计算三维桁架单元的应变、应力和轴力。
    x1, x2: 节点坐标 [x, y, z] 作为列表或数组
    E: 弹性模量
    A: 截面面积
    de: 节点位移向量 [u1, v1, w1, u2, v2, w2]
    返回: 应变 epsilon, 应力 sigma, 轴力 N
    """
    x1 = np.array(x1, dtype=float).flatten()
    x2 = np.array(x2, dtype=float).flatten()
    de = np.array(de, dtype=float).flatten()

    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("节点坐标必须是长度为3的向量")
    if de.shape != (6,):
        raise ValueError("位移向量必须是长度为6的向量")

    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L == 0:
        raise ValueError("两个节点坐标不能重合")

    n = delta / L
    B = (1.0 / L) * np.block([-n, n]).reshape(1, 6)
    epsilon = float(B @ de)
    sigma = E * epsilon
    N = A * sigma
    return epsilon, sigma, N


def build_perpendicular_basis(n):
    """选取两个与单元轴线垂直的单位向量，用于构造转动刚体模态"""
    idx = np.argmin(np.abs(n))
    ref = np.zeros(3)
    ref[idx] = 1.0
    perp1 = np.cross(n, ref)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(n, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    return perp1, perp2


def truss3d_element_matrix_properties(Ke, x1, x2):
    """
    检验单元刚度矩阵的基本性质：
    1. 对称性
    2. 奇异性
    3. 半正定性
    4. 零能模态是否只由刚体模态构成
    """
    Ke = np.array(Ke, dtype=float)
    if Ke.shape != (6, 6):
        raise ValueError("Ke 必须是 6x6 矩阵")

    x1 = np.array(x1, dtype=float).flatten()
    x2 = np.array(x2, dtype=float).flatten()
    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("节点坐标必须是长度为3的向量")

    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L == 0:
        raise ValueError("两个节点坐标不能重合")

    n = delta / L
    Ke_sym = 0.5 * (Ke + Ke.T)

    # 对称性检查
    fro_norm_Ke = np.linalg.norm(Ke, 'fro')
    symmetry_tol = 1.0e-10 * max(1.0, fro_norm_Ke)
    symmetry_error = np.linalg.norm(Ke - Ke.T, 'fro')
    is_symmetric = symmetry_error <= symmetry_tol

    # 特征值分析
    eigenvalues = np.sort(np.real(np.linalg.eigvals(Ke_sym)))
    eigen_tol = 1.0e-10 * max(1.0, np.max(np.abs(eigenvalues)))
    matrix_rank = np.linalg.matrix_rank(Ke_sym, tol=eigen_tol)
    nullity = 6 - matrix_rank
    is_singular = matrix_rank < 6
    is_positive_semidefinite = eigenvalues[0] >= -eigen_tol

    # 构造刚体模态
    perp1, perp2 = build_perpendicular_basis(n)
    rigid_modes = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, perp1[0], perp2[0]],
        [0, 1, 0, perp1[1], perp2[1]],
        [0, 0, 1, perp1[2], perp2[2]]
    ])  # 形状 (6, 5)

    rigid_mode_rank = np.linalg.matrix_rank(rigid_modes, tol=eigen_tol)
    rigid_mode_residual = np.linalg.norm(Ke_sym @ rigid_modes, 'fro')
    rigid_tol = 1.0e-10 * max(1.0, np.linalg.norm(Ke_sym, 'fro')) * np.linalg.norm(rigid_modes, 'fro')

    is_rigid_body_only_zero_mode = (
        rigid_mode_rank == 5 and
        nullity == 5 and
        rigid_mode_residual <= rigid_tol
    )

    report = {
        'is_symmetric': is_symmetric,
        'symmetry_error': symmetry_error,
        'symmetry_tolerance': symmetry_tol,
        'is_singular': is_singular,
        'matrix_rank': matrix_rank,
        'nullity': nullity,
        'is_positive_semidefinite': is_positive_semidefinite,
        'min_eigenvalue': eigenvalues[0],
        'eigenvalues': eigenvalues,
        'rigid_mode_rank': rigid_mode_rank,
        'rigid_mode_residual': rigid_mode_residual,
        'rigid_mode_tolerance': rigid_tol,
        'is_rigid_body_only_zero_mode': is_rigid_body_only_zero_mode
    }
    return report


def print_property_report(report):
    """输出任务4的检验结果"""
    def tf_text(flag):
        return "满足" if flag else "不满足"

    print("矩阵性质检查：")
    print(f"1. 对称性：{tf_text(report['is_symmetric'])}（误差 = {report['symmetry_error']:.3e}）")
    print(f"2. 奇异性：{tf_text(report['is_singular'])}（rank = {report['matrix_rank']}, nullity = {report['nullity']}）")
    print(f"3. 半正定性：{tf_text(report['is_positive_semidefinite'])}（最小特征值 = {report['min_eigenvalue']:.3e}）")
    print(f"4. 零能模态仅为刚体模态：{tf_text(report['is_rigid_body_only_zero_mode'])}（刚体模态残差 = {report['rigid_mode_residual']:.3e}）")


def run_case(case_name, x1, x2, E, A, de):
    """统一执行单个算例，避免重复代码"""
    print(f"{case_name}")

    L, direction_cosines, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    Fe = Ke @ de

    report = truss3d_element_matrix_properties(Ke, x1, x2)

    print(f"L = {L:.6g} m")
    print(f"方向余弦 = [{direction_cosines[0]:.6g}, {direction_cosines[1]:.6g}, {direction_cosines[2]:.6g}]")
    print("Ke =")
    print(Ke)
    print(f"epsilon = {epsilon:.6g}")
    print(f"sigma = {sigma:.6g} Pa")
    print(f"N = {N:.6g} N")
    print("Fe =")
    print(Fe)

    print_property_report(report)


def main():
    print("三维桁架单元作业算例\n")

    run_case('算例 1', [0, 0, 0], [2, 0, 0], 200e9, 1.0e-4, [0, 0, 0, 1.0e-3, 0, 0])
    print()
    run_case('算例 2', [0, 0, 0], [1, 2, 2], 210e9, 2.0e-4, [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])


if __name__ == "__main__":
    main()