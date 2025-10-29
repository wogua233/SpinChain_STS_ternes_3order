import copy
import numpy as np




def Splus(S):
    """
    Ladder operator S+ in the |S, m> basis ordered as [S, S-1, ..., -S].
    Returns a (2S+1)×(2S+1) complex matrix (ħ = 1).
    """
    d = int(round(2 * S + 1))
    # Scilab: e = sqrt((1:2S).*(2S:-1:1)); diag(e, +1)
    k = np.arange(1, int(round(2 * S)) + 1, dtype=float)           # 1 .. 2S
    vals = np.sqrt(k * (2 * S - k + 1.0))                          # length 2S
    M = np.zeros((d, d), dtype=complex)
    M += np.diag(vals, k=+1)
    return M

def Sminus(S):
    """
    Ladder operator S- in the |S, m> basis ordered as [S, S-1, ..., -S].
    Returns a (2S+1)×(2S+1) complex matrix (ħ = 1).
    """
    d = int(round(2 * S + 1))
    k = np.arange(1, int(round(2 * S)) + 1, dtype=float)           # 1 .. 2S
    vals = np.sqrt(k * (2 * S - k + 1.0))                          # length 2S
    M = np.zeros((d, d), dtype=complex)
    M += np.diag(vals, k=-1)
    return M

def Sx(S):
    """
    Sx operator for spin S in the |S, m> basis ordered as [S, S-1, ..., -S].
    """
    return 0.5 * (Splus(S) + Sminus(S))

def Sy(S):
    """
    Sy operator for spin S in the |S, m> basis ordered as [S, S-1, ..., -S].
    """
    return -0.5j * (Splus(S) - Sminus(S))

def Sz(S):
    """
    Sz operator (diagonal with entries S, S-1, ..., -S) in the |S, m> basis.
    """
    m = np.arange(S, -S - 1, -1, dtype=float)  # [S, S-1, ..., -S]
    return np.diag(m.astype(complex))


def S1(S):
    """
    Identity operator on the spin-S subspace.
    """
    # Read spin value (supports dict or object with attribute .S)
    d = int(round(2 * float(S) + 1))
    return np.eye(d, dtype=complex)


def dmatrix(n):
    """
    Spin-1/2 electron density matrix for polarization vector n = (nx, ny, nz),
    with |n| <= 1. Implements: ρ = 1/2 * S1 + nx*Sx + ny*Sy + nz*Sz
    (which equals 1/2 * (I + n·σ) since S = σ/2 for spin-1/2).
    """
    n = np.asarray(n, dtype=float).reshape(3)
    S =  0.5
    return 0.5 * S1(S) + n[0] * Sx(S) + n[1] * Sy(S) + n[2] * Sz(S)

def spec(A):
    # 返回 (W, D)，与 Scilab 的 spec 约定一致
    vals, vecs = np.linalg.eigh(A)   # A 必须是 Hermitian/对称
    D = np.diag(vals)
    return vecs, D

def etransport(experiment):
    """
    Python version of Scilab etransport(experiment).

    Computes 2×2 matrices of electron-side transition elements between the
    tip initial basis and the sample final basis:
      x ≡ <f_s|σx|i_t>, y ≡ <f_s|σy|i_t>, z ≡ <f_s|σz|i_t>, u ≡ -2 <f_s|I|i_t>
    Each is weighted by sqrt(ps ⊗ pt), where ps/pt are eigen-occupations
    of the sample/tip density matrices.

    Required in `experiment`:
      - experiment.ptip    : length-3 array (tip polarization)
      - experiment.psample : length-3 array (sample polarization)
    """
    # density matrices from polarizations
    dt = dmatrix(experiment.ptip)
    ds = dmatrix(experiment.psample)
    # print("dt:",dt)
    # print("ds:",ds)
    # eigen-bases (columns = eigenvectors), diagonal eigenvalue matrices
    wt, pt = spec(dt)  # tip
    ws, ps = spec(ds)  # sample
    # print("wt:",wt)
    # print("ws:",ws)
    # print("ps:",ps)
    # print("pt:",pt)
    # occupation vectors (eigenvalues are real for physical density matrices)
    pt = np.diag(pt).real
    ps = np.diag(ps).real
    # print("pt:",pt)
    # amplitude prefactor sqrt(ps ⊗ pt) -> 2x2
    pp = np.sqrt(np.outer(ps, pt))
    # print("pp:",pp)
    # transform operators to (sample-final × tip-initial) mixed basis and weight
    x = 2*(ws.conj().T @ Sx(0.5) @ wt) * pp
    y = 2*(ws.conj().T @ Sy(0.5) @ wt) * pp
    z = 2*(ws.conj().T @ Sz(0.5) @ wt) * pp
    # print("x:",x)

    # identity channel (potential scattering) with model's factor -2
    u = (-2.0) * (ws.conj().T @ wt) * pp


    return x, y, z, u


def Occupation(ve, T):
    """
    Boltzmann occupations for eigenvalues ve at effective temperature T.
    ve : array-like (1D or square 2D). Energies (e.g., in meV).
    T  : float. Temperature in same units as ve (meV).
    Returns
    -------
    e : (N,) ndarray, normalized occupations.
    """
    ve = np.asarray(ve, dtype=float)

    # If a square matrix is given, take its diagonal
    if ve.ndim == 2 and ve.shape[0] == ve.shape[1]:
        ve = np.diag(ve)

    # Shift so minimum is zero (avoids overflow and matches original code)
    ve = ve - np.min(ve)

    if T != 0:
        e = np.exp(-ve / float(T))
        s = e.sum()
        # Guard against numerical underflow leading to zero sum
        if s == 0:
            # fall back: put all weight on the minimum-energy state(s)
            e = np.zeros_like(ve)
            mask = (ve == 0)
            e[mask] = 1.0 / np.count_nonzero(mask)
        else:
            e /= s
    else:
        # T == 0: all weight equally on the ground-state manifold
        e = np.zeros_like(ve)
        mask = (ve == 0)  # since we shifted by min, ground states are 0
        count = np.count_nonzero(mask)
        if count > 0:
            e[mask] = 1.0 / count
        # else: ve was empty; e stays empty

    return e

def ln_t2(x, e0, T, lt):
    """
    使返回长度与 x 相同 (N) —— 这是 spec3 所需要的。
    """
    x = np.asarray(x, dtype=float)
    N = x.size

    yspan   = x[0] - x[-1]
    ycenter = (x[0] + x[-1]) / 2.0
    i = 2 * N  # 注意：Scilab里用 i=2*length(x)

    # 长度 2N+1 的辅助网格
    y = np.linspace(ycenter + yspan, ycenter - yspan, i + 1)

    # 复对数核（Kondo型），注意加上 i*pi/2
    g = np.log((e0 + np.abs(y - ycenter)) / ( (y - ycenter) + 1j*lt )) + 0.5j*np.pi

    z  = x / T
    ez = np.exp(z)

    f = -((ez + ez*(z - 1.0)) * (ez - 1.0)**-2 - 2.0*ez*(ez*(z - 1.0) + 1.0) * (ez - 1.0)**-3)
    # x≈0 的极限修正
    k = np.where(np.abs(z) < 1e-3)[0]
    f[k] = 1.0/6.0
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # 一维卷积：注意 numpy.convolve(..., 'same') 返回长度 max(N, M) = 2N+1
    conv_same = np.convolve(f, g, mode='same')

    # —— 关键：居中裁剪到长度 N ——
    L = conv_same.size  # 2N+1
    start = (L - N) // 2
    stop  = start + N
    conv_trim = conv_same[start:stop]  # 长度 N，与 x 对齐

    e = 2.0 * conv_trim / (T * i) * abs(yspan)
    return e



def fstep_simple(x):
    """
    Fermi–Dirac step: 1 / (exp(x) + 1)

    Parameters
    ----------
    x : array_like
        Real input (can be scalar or array).

    Returns
    -------
    y : ndarray
        Same shape as x.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore'):
        y = 1.0 / (np.exp(x) + 1.0)
    return y


def fstep(x):
    """
    Doubly temperature-broadened step function used in spec2/spec3.

    Scilab reference:
        y = (1 + (x - 1) * exp(x)) / (exp(x) - 1)^2
        with special handling:
          - near-zero x are snapped to 0,
          - y(x==0) = 0.5,
          - NaNs set to 0.

    Parameters
    ----------
    x : array_like
        Real input (can be scalar or array).

    Returns
    -------
    y : ndarray
        Same shape as x.
    """
    x = np.asarray(x, dtype=float).copy()

    # clean(x, 1e-5): snap near-zero to exactly 0 (improves numerical stability)
    eps = 1e-5
    x[np.abs(x) < eps] = 0.0

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        z = np.exp(x)
        denom = (z - 1.0)
        y = (1.0 + (x - 1.0) * z) / (denom ** 2)

    # enforce y(0) = 0.5 exactly
    zero_mask = (x == 0.0)
    if np.any(zero_mask):
        y[zero_mask] = 0.5

    # replace any residual NaNs/Infs (e.g., extreme x) by 0
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y





def e_epsilon(experiment):
    """
    与 Scilab 版本逐项等价的实现：
      - so3 使用 共轭转置（X^H）
      - so2 使用 仅转置（X^T），不取共轭
      - 用外积构造与 mone=ones(1,2) 完全一致的“复制”矩阵，
        再做逐元素乘法并 sum 全部元素
    返回：e, er, en, enr 皆为 (4,4,4) 复数张量
    """
    # 1) 电子从 tip->sample 的自旋跃迁矩阵（2x2）
    x, y, z, u = etransport(experiment)  # 每个 shape (2,2), 可能为复数

    # 2) 把 tip 极化改成 sample 极化，再算一遍（另一条腿）
    _ptip = experiment.ptip
    experiment.ptip = experiment.psample
    xs, ys, zs, us = etransport(experiment)
    experiment.ptip = _ptip

    # 3) 组装列表；so3 用共轭转置；so2 用仅转置
    so1 = [np.asarray(x),  np.asarray(y),  np.asarray(z),  np.asarray(u)]       # (2,2)
    so2 = [np.asarray(xs), np.asarray(ys), np.asarray(zs), np.asarray(us)]      # (2,2)
    so3 = [A.conj().T for A in so1]                                            # x', y', z', u'  (Hermitian)

    e   = np.zeros((4,4,4), dtype=complex)
    er  = np.zeros_like(e)
    en  = np.zeros_like(e)
    enr = np.zeros_like(e)

    ones_row = np.ones((1, 2), dtype=complex)   # mone  = ones(1,2)
    ones_col = np.ones((2, 1), dtype=complex)   # mone' = ones(2,1)

    # 4) 逐项完全按 Scilab 写法组装
    for j in range(4):
        for k in range(4):
            for l in range(4):
                acc_e  = 0.0 + 0.0j
                acc_er = 0.0 + 0.0j
                acc_en = 0.0 + 0.0j
                acc_enr= 0.0 + 0.0j

                # in = 1..2（Scilab 1-based），Python 0..1
                for in_idx in range(2):
                    # (mone'.*.so3(l)(in,:))  → 外积：ones_col (2x1) ∘ row (1x2)  → (2x2)，每行复制该行向量
                    A_e   = np.outer(ones_col[:,0], so3[l][in_idx, :])                    # (2,2)
                    # (so2(k)).'  → 仅转置（不取共轭）
                    B_e   = so2[k].T                                                      # (2,2)
                    # (mone.*.so1(j)(:,in))  → 外积：col (2x1) ∘ ones_row (1x2) → (2x2)，每列复制该列向量
                    C_e   = np.outer(so1[j][:, in_idx], ones_row[0, :])                   # (2,2)
                    acc_e += np.sum(A_e * B_e * C_e)

                    # er：-(mone'.*.so3(l)(in,:)).*(mone.*.so1(k)(:,in)).*(so2(j).')
                    A_er  = np.outer(ones_col[:,0], so3[l][in_idx, :])
                    B_er  = np.outer(so1[k][:, in_idx], ones_row[0, :])
                    C_er  = so2[j].T
                    acc_er -= np.sum(A_er * B_er * C_er)

                    # en：-(mone.*.so1(l)(:,in)).*(so2(k)).'.*(mone'.*.so3(j)(in,:))
                    A_en  = np.outer(so1[l][:, in_idx], ones_row[0, :])
                    B_en  = so2[k].T
                    C_en  = np.outer(ones_col[:,0], so3[j][in_idx, :])
                    acc_en -= np.sum(A_en * B_en * C_en)

                    # enr：(mone.*.so1(l)(:,in)).*(mone'.*.so3(k)(in,:)).*(so2(j).')
                    A_enr = np.outer(so1[l][:, in_idx], ones_row[0, :])
                    B_enr = np.outer(ones_col[:,0], so3[k][in_idx, :])
                    C_enr = so2[j].T
                    acc_enr += np.sum(A_enr * B_enr * C_enr)

                e[j,  k,  l] = acc_e
                er[j, k,  l] = acc_er
                en[j, k,  l] = acc_en
                enr[j, k,  l] = acc_enr
    # print("e:", e)
    # print("er:", er)
    # print("en:", en)
    # print("enr:", enr)

    return e, er, en, enr

#
# def e_epsilon(experiment):
#     """
#     Python 版 e_epsilon：构造三阶电子侧 ε 张量
#       返回四个 (4,4,4) 张量：
#         e   : tip -> sample,   normal
#         er  : tip -> sample,   time-reversed
#         en  : sample -> tip,   normal
#         enr : sample -> tip,   time-reversed
#     experiment 需包含字段/属性：
#       - ptip, psample  : 三维自旋极化向量（供 etransport 使用）
#     """
#
#     # --- 兼容 dict / 对象属性访问 ---
#     def getf(obj, key):
#         if isinstance(obj, dict):
#             return obj[key]
#         return getattr(obj, key)
#
#     def setf(obj, key, value):
#         if isinstance(obj, dict):
#             obj[key] = value
#         else:
#             setattr(obj, key, value)
#
#     # --- 从 etransport 获取电子 2x2 方块 ---
#     x, y, z, u = etransport(experiment)  # 2x2
#     # 构造一个临时 experiment：把 ptip 换成 psample（样侧基）
#     tempexp = copy.copy(experiment)
#     setf(tempexp, 'ptip', getf(experiment, 'psample'))
#     xs, ys, zs, us = etransport(tempexp)  # 2x2，样-样风格
#
#     # --- 辅助函数：行/列复制，模拟 Scilab 的 Kronecker 复制 ---
#     # (mone'.*.row) : 把 1×2 的 row 复制为 2×2，每一行相同
#     def row_expand(row):
#         row = np.asarray(row)
#         return np.tile(row, (row.size, 1))
#
#     # (mone.*.col) : 把 2×1 的 col 复制为 2×2，每一列相同
#     def col_expand(col):
#         col = np.asarray(col).reshape(-1, 1)
#         return np.tile(col, (1, col.size))
#
#     # --- 打包四种电子算符的 2x2 方块 ---
#     so1 = [x, y, z, u]              # 针->样
#     so2 = [xs, ys, zs, us]          # 样->样
#     so3 = [x.T, y.T, z.T, u.T]      # 另一侧取转置（行/列对调）
#     # print("so1:",so1)
#     # print("so2:",so2)
#     # print("so3:",so3)
#
#     # --- 结果张量 ---
#     e   = np.zeros((4, 4, 4), dtype=complex)
#     er  = np.zeros((4, 4, 4), dtype=complex)
#     en  = np.zeros((4, 4, 4), dtype=complex)
#     enr = np.zeros((4, 4, 4), dtype=complex)
#
#     # --- 三重 (j,k,l) 与内部电子自旋 in=0..1 求和 ---
#     for j in range(4):
#         for k in range(4):
#             for l in range(4):
#                 acc_e = 0.0 + 0.0j
#                 acc_er = 0.0 + 0.0j
#                 acc_en = 0.0 + 0.0j
#                 acc_enr = 0.0 + 0.0j
#
#                 for in_idx in range(2):  # 内部电子自旋求和
#                     # e (tip->sample, normal)
#                     temp = (row_expand(so3[l][in_idx, :]) *
#                             (so2[k].T) *
#                             col_expand(so1[j][:, in_idx]))
#                     acc_e += np.sum(temp)
#
#                     # er (tip->sample, time-reversed)  带负号
#                     temp = -(row_expand(so3[l][in_idx, :]) *
#                              col_expand(so1[k][:, in_idx]) *
#                              (so2[j].T))
#                     acc_er += np.sum(temp)
#
#                     # en (sample->tip, normal)          带负号
#                     temp = -(row_expand(so1[l][in_idx, :]) *
#                              (so2[k].T) *
#                              col_expand(so3[j][:, in_idx]))
#                     acc_en += np.sum(temp)
#
#                     # enr (sample->tip, time-reversed)
#                     temp = (row_expand(so1[l][in_idx, :]) *
#                             (row_expand(so3[k][in_idx, :])) *
#                             (so2[j].T))
#                     # 注意：上式第三项是 (mone'.*.so3(k)(in,:))，等价于 row_expand(...)
#                     acc_enr += np.sum(temp)
#
#                 e[j, k, l]   = acc_e
#                 er[j, k, l]  = acc_er
#                 en[j, k, l]  = acc_en
#                 enr[j, k, l] = acc_enr
#
#     return e, er, en, enr

#_____________________#
if __name__ == '__main__':
    print(Sz(0.5))
    print(S1(0.5))