import numpy as np
from utilis import *

# 需要你在工程里提供 Hamiltonian 构造器：
# from your_module import hamiltonian  # -> ndarray 或 scipy.sparse 矩阵

def _clean(mat, tol=1e-12):
    """将绝对值小于 tol 的元素置零。"""
    out = np.asarray(mat).copy()
    out[np.abs(out) < tol] = 0.0
    return out

def _kron2(a, b):
    return np.kron(np.asarray(a), np.asarray(b))

def _op_on_site(op, site, nat):
    """
    把单体算符 op（作用在 'site' 的自旋子空间）嵌入整条链：
    I ⊗ ... ⊗ I ⊗ op ⊗ I ⊗ ... ⊗ I
    site: 1-based 索引
    """
    # 左侧
    cur = None
    for i in range(1, site):
        Ii = S1(0.5)
        cur = Ii if cur is None else _kron2(cur, Ii)
    # 目标位点
    cur = op if cur is None else _kron2(cur, op)
    # 右侧
    for i in range(site+1, nat+1):
        Ii = S1(0.5)
        cur = _kron2(cur, Ii)
    return cur

def HHeisenberg(nat, J, n, m):
    """
    Heisenberg 耦合项：在自旋列表 t 中第 n 与第 m 个自旋之间的 J·(SxSx + SySy + SzSz)。

    参数
    ----
    t : list
        原子/自旋对象列表，每个对象需可被 S1/Sx/Sy/Sz 处理。
    J : float 或 (Jx, Jy, Jz)
        耦合常数（各向同性或各向异性）。
    n, m : int
        1-based 位点索引，且 n != m。

    返回
    ----
    e : ndarray 或 scipy.sparse.csr_matrix
        嵌入到 many-body 空间的 Heisenberg 耦合哈密顿量。
    """
    # --- 参数检查与处理 ---

    if n < 1 or m < 1 or n > nat or m > nat or n == m:
        raise ValueError("Invalid pair (n, m) for Heisenberg coupling.")
    if n > m:
        n, m = m, n  # 确保 n < m

    # --- 构造 Sx(n)⊗Sx(m), Sy(n)⊗Sy(m), Sz(n)⊗Sz(m) ---
    # 先把单体算符放到对应位点，再在另一位点放相同分量算符
    # 方式：先在 n 位点放 Sx，然后在 m 位点放 Sx；其余位点放 I
    # 为了高效与清晰，构造时按“在 n 位置放 op_n，在 m 处放 op_m，其余 I”的思路：

    # x 分量
    op_n = _op_on_site(Sx(0.5), n, nat)

    # 把 op_m 嵌入整链（只在 m 位点放 Sx，其余 I）
    op_m = _op_on_site(Sx(0.5), m, nat)

    xterm = (op_n @ op_m)

    # y 分量
    op_n = _op_on_site(Sy(0.5), n, nat)
    op_m = _op_on_site(Sy(0.5), m, nat)
    yterm = (op_n @ op_m)

    # z 分量
    op_n = _op_on_site(Sz(0.5), n, nat)
    op_m = _op_on_site(Sz(0.5), m, nat)
    zterm = (op_n @ op_m)

    # --- 线性组合 ---
    e = J * xterm + J * yterm + J * zterm

    return e


def hamiltonian(experiment):
    """
    组装多自旋体系的总哈密顿量：
      H = H_ani+Zeeman
        + Σ_{i<j} [ (ev(M_ij)*J) * H_Heisenberg_{ij}
                    + (ev(MDM_ij)*J) · H_DM_{ij} ]
    其中 ev() 表示把字符串/标量/向量解析为数值（或向量），并与全局 J 相乘。
    """


    nat = experiment.atom.natoms
    if nat <= 1:
        H=np.eye(2, dtype=complex)
        print("只有一个原子！无需对角化！")
        return H
    else:
        H=np.zeros((2**nat, 2**nat), dtype=complex)
        # 全局耦合常数
        Jglob = float(experiment.heisenberg_coupling)
        # ================= Heisenberg Coupling =================
        M = experiment.matrix

        if M is not None and Jglob != 0.0:
            for i in range(1, nat):          # 1..nat-1
                for j in range(i+1, nat+1):  # i+1..nat
                    Jij=float(M[i-1, j-2])*Jglob
                    if Jij != 0.0:
                        H += HHeisenberg(nat, Jij, i, j)
    print(f"{nat}个原子！对角化以下矩阵：{H}")
    return H

def eigenvalues(experiment):
    """
    计算体系的本征值与本征向量，并写回 experiment.Eigenvec / experiment.Eigenval。
    - 小矩阵：用 numpy.linalg.eigh 全部对角化
    - 大矩阵且给定 max_no_eigenstates：用 scipy.sparse.linalg.eigsh 取最小的 k 个本征对
    """
    H = hamiltonian(experiment)  # 应该是厄米矩阵（密集或稀疏）
    # print('对角化的哈密顿量是：',H)
    # 全部本征对
    vals, vecs = np.linalg.eigh(H)
    # print('vals, vecs:',vals, vecs)

    # 排序（numpy/scipy 通常已升序，但稳妥起见）
    idx = np.argsort(vals.real)
    vals = vals[idx].real
    vecs = vecs[:, idx]

    # 写回 experiment：Eigenval 为 1D 向量（与原代码最终 diag(...) 保持一致）
    experiment.Eigenvec=_clean(vecs)
    experiment.Eigenval=vals
    return vals, vecs




#——————————————————————————#
if __name__ == '__main__':
    # print(HHeisenberg(2, 1.0, 1, 2))
    op_n=_op_on_site(Sz(0.5), 1, 2)
    print(op_n)
    op_m=_op_on_site(Sz(0.5), 2, 2)
    print(op_m)
    zterm = (op_n @ op_m)
    print(zterm)




