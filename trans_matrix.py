import numpy as np
from utilis import *


def Rate2nd2(experiment):
    """
    Second-order geometric factor matrix (no thermal kernel / couplings).
    Returns out[f,i] = sum_{4 electron channels} | <f|(Sx,Sy,Sz,U)|i> · (x,y,z,u) |^2
    """
    # electron-side 2x2 blocks
    x2, y2, z2, u2 = etransport(experiment)
    # print('x2,y2,z2,u2',x2,y2,z2,u2)
    E = np.asarray(experiment.Eigenvec)          # (Nbasis, Nstates)
    N = E.shape[1]

    atoms = experiment.atom
    S=atoms.S

    # build many-body operator acting on the spin system
    if atoms.natoms==1:
        matrxx = Sx(S)
        matrxy = Sy(S)
        matrxz = Sz(S)
        matrxu = atoms.U * S1(S)
    else:
        nr = int(experiment.position)            # 1-based
        nat = atoms.natoms

        base = np.array([[1.0]], dtype=complex)
        for i in range(1, nr):
            base = np.kron(base, S1(S).astype(complex))

        matrxx = base.copy()
        matrxy = base.copy()
        matrxz = base.copy()
        matrxu = base.copy()

        matrxx = np.kron(matrxx, Sx(S).astype(complex))
        matrxy = np.kron(matrxy, Sy(S).astype(complex))
        matrxz = np.kron(matrxz, Sz(S).astype(complex))
        matrxu = atoms.U * np.kron(matrxu, S1(S).astype(complex))

        for i in range(nr+1, nat+1):
            Ii = S1(S).astype(complex)
            matrxx = np.kron(matrxx, Ii)
            matrxy = np.kron(matrxy, Ii)
            matrxz = np.kron(matrxz, Ii)
            matrxu = np.kron(matrxu, Ii)

    # rotate into eigenbasis
    Ec = E.conj().T
    matrxx = Ec @ matrxx @ E
    matrxy = Ec @ matrxy @ E
    matrxz = Ec @ matrxz @ E
    matrxu = Ec @ matrxu @ E
    # print("matrxx:",matrxx)
    # electron channels (use Fortran order to match Scilab linear indexing)
    xf = x2.flatten(order='F')
    yf = y2.flatten(order='F')
    zf = z2.flatten(order='F')
    uf = u2.flatten(order='F')

    out = np.zeros((N, N), dtype=float)
    for k in range(4):
        amp = matrxx * xf[k] + matrxy * yf[k] + matrxz * zf[k] + matrxu * uf[k]
        out += np.abs(amp) ** 2

    return out




# 你需要在工程里提供/导入以下函数与类型：
# - e_epsilon(experiment) -> (e_sig, er_sig, en_sig, enr_sig)  四个 (4,4,4) 张量
# - Sx(atom), Sy(atom), Sz(atom), S1(atom)  返回该原子自旋子空间上的矩阵（numpy.ndarray，复数/实数均可）
# - experiment 可以是 dict 或具名对象；其中
#   experiment.Eigenvec: (N_basis, N_states)
#   experiment.atom:     单个“原子对象”或 list[原子对象]；原子对象需有属性 .J（标量或3分量）与 .U
#   experiment.position, experiment.jposition 为 1-based 索引（与原 Scilab 一致）
def M35(experiment, in_state):
    """
    Python 版 M35：三阶（3 顶点）many-body 幅度矩阵块
    返回四个 (N_states, N_states) 复矩阵：
      e   : tip -> sample,   normal
      er  : tip -> sample,   time-reversed
      en  : sample -> tip,   normal
      enr : sample -> tip,   time-reversed
    """
    # -------- 行/列扩展（模拟 Scilab 的 Kronecker 复制） ----------
    # (mone'.*.row) : 把 1×N 的 row 复制为 N×N，每一行相同
    def row_expand(row):
        row = np.asarray(row)
        N = row.size
        return np.tile(row, (N, 1))

    # (mone.*.col)  : 把 N×1 的 col 复制为 N×N，每一列相同
    def col_expand(col):
        col = np.asarray(col).reshape(-1, 1)
        N = col.size
        return np.tile(col, (1, N))

    # -------- 数值清理（对应 Scilab clean） ----------
    def clean(M, tol=1e-12):
        M = np.asarray(M)
        out = M.copy()
        out[np.abs(out) < tol] = 0.0
        return out

    # -------- 读取基础数据 ----------
    E = np.asarray(experiment.Eigenvec)
    N = E.shape[1]  # 本征态数

    # e_epsilon：纯电子侧三阶“ε 张量”（4×4×4）
    e_sig, er_sig, en_sig, enr_sig = e_epsilon(experiment)
    # print("e_sig:",e_sig)
    # print("er_sig:",er_sig)
    # print("en_sig:",en_sig)
    # print("enr_sig:",enr_sig)

    # 是否为多原子系统：原 Scilab 用 ~isvector(experiment.atom)
    atoms = experiment.atom
    S = atoms.S

    # -------- 构造体系侧算符（针-样通道：matrxx/y/z/u；样-样通道：Smatrxx/y/z/u） ----------
    if atoms.natoms==1:
        # 单自旋系统：直接在该自旋子空间上取算符并旋到能量本征基

        matrxx = E.T @ Sx(S) @ E
        matrxy = E.T @ Sy(S) @ E
        matrxz = E.T @ Sz(S) @ E
        matrxu = atoms.U * (E.T @ S1(S) @ E)

        # J 可以是标量或 3 分量
        J = atoms.J

        Smatrxx = J * matrxx
        Smatrxy = J * matrxy
        Smatrxz = J * matrxz
        Smatrxu = np.zeros_like(Smatrxx, dtype=complex)

    else:
        nr  = int(experiment.position)     # 1-based

        # --- 左边 (1..nr-1) 累乘单位 ---
        base = np.array([[1.0]], dtype=complex)
        for i in range(1, nr):  # 1..nr-1
            base = np.kron(base, S1(S).astype(complex))
        # 初始复制
        matrxx = base.copy()
        matrxy = base.copy()
        matrxz = base.copy()
        matrxu = base.copy()
        # --- 在 nr 位点放算符/单位 ---
        matrxx = np.kron(matrxx, Sx(S).astype(complex))
        matrxy = np.kron(matrxy, Sy(S).astype(complex))
        matrxz = np.kron(matrxz, Sz(S).astype(complex))
        matrxu = atoms.U * np.kron(matrxu, S1(S).astype(complex))

        # --- 右边 (nr+1..natoms) 累乘单位 ---
        for i in range(nr+1, atoms.natoms+1):
            I_i = S1(S).astype(complex)
            matrxx = np.kron(matrxx, I_i)
            matrxy = np.kron(matrxy, I_i)
            matrxz = np.kron(matrxz, I_i)
            matrxu = np.kron(matrxu, I_i)


        # 无纠缠，且样-样顶点与针-样同一位点：直接沿用 matr??，只乘 J 分量
        J = atoms.J

        Smatrxx = J * (E.T @ matrxx @ E)
        Smatrxy = J * (E.T @ matrxy @ E)
        Smatrxz = J * (E.T @ matrxz @ E)
        Smatrxu = np.zeros_like(Smatrxx, dtype=complex)


        # 最后把针-样通道的 matr?? 也转到能量本征基
        matrxx = E.T @ matrxx @ E
        matrxy = E.T @ matrxy @ E
        matrxz = E.T @ matrxz @ E
        matrxu = E.T @ matrxu @ E

    # -------- 组装 so1/so2/so3 列表（索引 0..3 对应 x,y,z,u） ----------
    so1 = [matrxx, matrxy, matrxz, matrxu]           # 针-样 顶点
    so2 = [Smatrxx, Smatrxy, Smatrxz, Smatrxu]       # 样-样 顶点
    so3 = so1                                        # 第三个顶点复用针-样

    # 结果矩阵
    e   = np.zeros((N, N), dtype=complex)
    er  = np.zeros((N, N), dtype=complex)
    en  = np.zeros((N, N), dtype=complex)
    enr = np.zeros((N, N), dtype=complex)

    # Scilab 1-based -> Python 0-based
    in0 = int(in_state) - 1

    # -------- 三重循环累加 (j,k,l = 1..4) ----------
    for j in range(4):
        for k in range(4):
            for l in range(4):
                # 仅在电子侧系数非零时才计算（与原代码一致）
                if e_sig[j, k, l] != 0:
                    A = row_expand(so3[l][in0, :])     # (N×N)
                    B = so2[k].T                       # (N×N)
                    C = col_expand(so1[j][:, in0])     # (N×N)
                    e  = e  - e_sig[j, k, l]  * (A * B * C)

                if er_sig[j, k, l] != 0:
                    A = row_expand(so3[l][in0, :])
                    B = so1[k].T
                    C = col_expand(so2[j][:, in0])
                    er = er - er_sig[j, k, l] * (A * B * C)

                if en_sig[j, k, l] != 0:
                    A = row_expand(so1[l][in0, :])
                    B = so2[k].T
                    C = col_expand(so3[j][:, in0])
                    en = en - en_sig[j, k, l] * (A * B * C)

                if enr_sig[j, k, l] != 0:
                    A = row_expand(so1[l][in0, :])
                    B = so3[k].T
                    C = col_expand(so2[j][:, in0])
                    enr = enr - enr_sig[j, k, l] * (A * B * C)

    # 数值清理
    e   = clean(e)
    er  = clean(er)
    en  = clean(en)
    enr = clean(enr)
    # print("e:",e)
    # print("er:",er)
    # print("en:",en)
    # print("enr:",enr)
    return e, er, en, enr

