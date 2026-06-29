import numpy as np

class FEModel:
    def __init__(self):
        self.Title = ""
        self.nsd = 0
        self.ndof = 0
        self.nnp = 0
        self.nel = 0
        self.nen = 0
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
    for e in range(model.nel):
        nodes = model.IEN[e, :] - 1
        if model.ndof == 1:
            x1 = model.x[nodes[0]]
            x2 = model.x[nodes[1]]
            delta = x2 - x1
            model.length[e] = abs(delta)
            model.direction_cosines[e, 0] = 1.0 if delta >= 0 else -1.0
        elif model.ndof == 2:
            x1 = model.x[nodes[0]]
            y1 = model.y[nodes[0]]
            x2 = model.x[nodes[1]]
            y2 = model.y[nodes[1]]
            delta = np.array([x2 - x1, y2 - y1])
            L = np.linalg.norm(delta)
            model.length[e] = L
            if L == 0:
                raise ValueError(f'Element {e+1} has zero length')
            model.direction_cosines[e, :] = delta / L
    return model

def element_stiffness(model, e):
    const = model.CArea[e] * model.E[e] / model.length[e]
    if model.ndof == 1:
        Ke = const * np.array([[1, -1], [-1, 1]])
    elif model.ndof == 2:
        c, s = model.direction_cosines[e, 0], model.direction_cosines[e, 1]
        Ke = const * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
    else:
        raise ValueError('Only 1D and 2D trusses are supported')
    return Ke

def assemble_global_stiffness(model):
    for e in range(model.nel):
        Ke = element_stiffness(model, e)
        lm = model.LM[:, e]
        for i, row in enumerate(lm):
            for j, col in enumerate(lm):
                model.K[row, col] += Ke[i, j]
    # 组装完成后自动校验刚度矩阵四大性质
    print("\n========== 刚度矩阵性质校验 ==========")
    check_stiffness_property(model)
    print("=====================================\n")
    return model

def check_stiffness_property(model):
    K = model.K
    n = K.shape[0]
    eps = 1e-10

    # 1. 对称性检测
    K_T = K.T
    diff_sym = np.max(np.abs(K - K_T))
    print(f"1. 对称性检测：最大不对称差值 = {diff_sym:.2e}")
    if diff_sym < eps:
        print("   >> 刚度矩阵对称 ✅")
    else:
        print("   >> 刚度矩阵不对称 ❌")

    # 2. 对角元非负检测
    diag = np.diag(K)
    neg_diag_idx = np.where(diag < -eps)[0]
    print(f"2. 对角元非负检测：")
    if len(neg_diag_idx) == 0:
        print("   >> 所有对角元均非负 ✅")
    else:
        print(f"   >> 存在负对角元，自由度编号(0基)：{neg_diag_idx} ❌")
        print(f"   >> 对应对角值：{diag[neg_diag_idx]}")

    # 3. 稀疏性计算
    total_elem = n * n
    zero_elem = np.sum(np.abs(K) < eps)
    sparse_rate = zero_elem / total_elem
    print(f"3. 稀疏性检测：总元素数={total_elem}, 零元素数={zero_elem}, 稀疏率={sparse_rate:.4f}")

    # 4. 奇异性检测（无约束完整总刚）
    if n > 0:
        try:
            det_K = np.linalg.det(K)
            print(f"4. 奇异性检测：总刚行列式 = {det_K:.2e}")
            if abs(det_K) < 1e-6:
                print("   >> 无约束整体刚度矩阵奇异（存在刚体位移）✅")
            else:
                print("   >> 整体刚度矩阵非奇异 ❌")
        except np.linalg.LinAlgError:
            print("4. 奇异性检测：矩阵奇异，无法计算行列式 ✅")

def solve_system(model):
    fixed = model.fixed_dof - 1
    all_dof = np.arange(model.neq)
    free = np.setdiff1d(all_dof, fixed)
    d_E = model.d[fixed]
    f_F = model.f[free]
    K_FF = model.K[np.ix_(free, free)]
    K_FE = model.K[np.ix_(free, fixed)]
    if len(free) == 0:
        d_F = np.array([])
    else:
        d_F = np.linalg.solve(K_FF, f_F - K_FE @ d_E)
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
    for e in range(model.nel):
        lm = model.LM[:, e]
        de = model.d[lm]
        const = model.E[e] / model.length[e]
        if model.ndof == 1:
            B = np.array([-1, 1])
        elif model.ndof == 2:
            c, s = model.direction_cosines[e, 0], model.direction_cosines[e, 1]
            B = np.array([-c, -s, c, s])
        model.stress[e] = const * (B @ de)
    return model

def print_results(model):
    print(f'\n{model.Title}')
    print('LM =')
    lm_1based = model.LM + 1
    print(lm_1based)
    print('K =')
    print(model.K)
    print('d =')
    print(model.d.flatten())
    print('reaction =')
    print(model.reaction.flatten())
    print('element stress =')
    print(model.stress.flatten())

def solve_model(model):
    model = build_LM(model)
    model = compute_element_geometry(model)
    model = assemble_global_stiffness(model)
    model = solve_system(model)
    model = postprocess_stress(model)
    print_results(model)
    return model

# ================= 算例1：一维两单元杆 =================
def create_1d_bar_model():
    model = FEModel()
    model.Title = "1D bar example"
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
    model.y = np.array([])
    model.z = np.array([])

    model.fixed_dof = np.array([1])
    model.fixed_value = np.array([0.0])
    model.force_dof = np.array([2, 3])
    model.force_value = np.array([0.0, 10.0])

    model.K = np.zeros((model.neq, model.neq))
    model.f = np.zeros(model.neq)
    model.d = np.zeros(model.neq)
    model.length = np.zeros(model.nel)
    model.stress = np.zeros(model.nel)
    model.reaction = np.zeros(model.neq)

    for dof, val in zip(model.force_dof, model.force_value):
        model.f[dof - 1] = val
    for dof, val in zip(model.fixed_dof, model.fixed_value):
        model.d[dof - 1] = val

    return model

# ================= 算例2：二维两杆桁架 =================
def create_2d_truss_model():
    model = FEModel()
    model.Title = "2D truss example"
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
    model.z = np.array([])

    model.fixed_dof = np.array([1, 2, 3, 4])
    model.fixed_value = np.array([0.0, 0.0, 0.0, 0.0])
    model.force_dof = np.array([5, 6])
    model.force_value = np.array([10.0, 0.0])

    model.K = np.zeros((model.neq, model.neq))
    model.f = np.zeros(model.neq)
    model.d = np.zeros(model.neq)
    model.length = np.zeros(model.nel)
    model.stress = np.zeros(model.nel)
    model.reaction = np.zeros(model.neq)

    for dof, val in zip(model.force_dof, model.force_value):
        model.f[dof - 1] = val
    for dof, val in zip(model.fixed_dof, model.fixed_value):
        model.d[dof - 1] = val

    return model

# ================= 主程序 =================
if __name__ == "__main__":
    print("2.3 Global Stiffness Equations homework examples\n")

    print("\n===== Example 1: 1D bar assembly =====")
    model1 = create_1d_bar_model()
    result1 = solve_model(model1)
    print("Expected d = [0; 0.1; 0.15]")
    print("Computed d =")
    print(result1.d)

    print("\n===== Example 2: 2D truss assembly =====")
    model2 = create_2d_truss_model()
    result2 = solve_model(model2)
    print("Expected u3 = 38.284271, v3 = -10.000000")
    print(f"Computed [u3, v3] = [{result2.d[4]:.6f}, {result2.d[5]:.6f}]")
    print("Expected stress(1) = -10.000000, stress(2) = 14.142136")
    print("Computed stress =")
    print(result2.stress)