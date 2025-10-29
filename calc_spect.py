import numpy as np
from utilis import *
from trans_matrix import *
from experiment import *


def spec2(experiment,
          savedata=False,
          precision=1e-3,
          plotspec=False,
          csv_delim=" "):
    """
    Python 版二阶谱计算（对应 Scilab: spec2）。

    Parameters
    ----------
    experiment : dict 或具名对象，需包含：
        - T : float
        - xrange : float 或 1D array
        - ptip, psample, atom, Eigenvec, Eigenval, position: 供 Rate2nd2 使用
    savedata : bool
        是否将每个 (i,j) 通道的曲线另存为 data{i}-{j}SF.dat
    precision : float
        占据阈值（对应 %precission）
    plotspec : bool
        是否绘图
    csv_delim : str
        保存文件的分隔符
    Returns
    -------
    x : (N,) ndarray
    y : (N,) ndarray
    """

    # --- 兼容 dict / 对象 的字段访问 ---

    T = float(experiment.T)
    Eigenval = np.asarray(experiment.Eigenval)
    # 若是方阵，取对角
    if Eigenval.ndim == 2 and Eigenval.shape[0] == Eigenval.shape[1]:
        ve = np.diag(Eigenval).astype(float)
    else:
        ve = Eigenval.astype(float)
    # 平移到最小为 0
    ve = ve - np.min(ve)
    # print(f"ve:",ve)
    # 平衡占据
    occ = Occupation(ve, T)  # (nstates,)
    # print(f"occ:",occ)
    # experiment.position位置的二阶几何因子（tip->sample），以及其转置（sample->tip）
    rate = Rate2nd2(experiment).T   # 与 Scilab 中的 ' 号一致
    rateneg = rate.T                # 再转置一次
    # print('rate:',rate)
    # print('rateneg:',rateneg)
    # 电压网格
    xrange_val = experiment.xrange
    if np.ndim(xrange_val) == 1:
        x = np.asarray(xrange_val, dtype=float).copy()
    else:
        V = abs(float(xrange_val))
        x = np.linspace(V, -V, 1000)  # 保持与原代码相同的降序

    y = np.zeros_like(x, dtype=float)

    # 维度
    Eigenvec = np.asarray(experiment.Eigenvec)
    nstates = Eigenvec.shape[1]

    # 主循环
    absT = abs(T) if T != 0 else 1.0  # 防守式，T=0 时应避免调用本函数的温度核
    for i in range(nstates):
        if occ[i] > precision:
            for j in range(nstates):
                en = (ve[j] - ve[i] - x) / absT  # Δ - V
                ep = (ve[j] - ve[i] + x) / absT  # Δ + V
                ytemp = occ[i] * (rate[i, j] * fstep(ep) + rateneg[i, j] * fstep(en))
                if savedata:
                    ij_name = f"data{i+1}-{j+1}SF.dat"  # 1-based 命名以贴近原实现
                    np.savetxt(ij_name,
                               np.column_stack([x, np.real(ytemp)]),
                               fmt="%.10g", delimiter=csv_delim)
                y += np.real(ytemp)
    # print("y:",y)
    if plotspec:
        try:
            import matplotlib.pyplot as plt
            plt.plot(x, y)
            plt.xlabel("V (same units as energies)")
            plt.ylabel("dI/dV (arb. units)")
            plt.tight_layout()
            plt.title('spec2_in_state') # spec2_in_state本身的数值有问题
            plt.show()
        except Exception as e:
            print("Plot failed:", e)

    return x, y



def spec3(experiment, in_state,
          savedata=False,
          precision=1e-3,
          plotspec=True,
          csv_delim=" "):
    """
    Python 版的第三阶谱计算（与 Scilab spec3 对应）。

    Parameters
    ----------
    experiment : object or dict
        需包含字段/键：
        - T : float
        - xrange : float 或 1D array (电压范围或网格)
        - lt : float (lifetime broadening)
        - ptip, psample : 任意（供 M35/etransport 等使用）
        - atom : list-like（其中 atom[jposition].w 需要存在）
        - Eigenvec : 2D array (本征矢列为态)
        - Eigenval : 1D 或 2D array (本征能)
        - position : int
        - jposition : int（可省略；缺省为 position）
    in_state : int
        初始态编号（保持 Scilab 习惯：**1-based**）。位置
    savedata : bool
        是否保存每个 (in, mid, fin) 的曲线到文件。
    precision : float
        阈值（对应 Scilab 的 %precission）。
    plotspec : bool
        是否实时绘图。
    csv_delim : str
        保存文件的分隔符。
    Returns
    -------
    x : (N,) ndarray
        电压网格
    y : (N,) ndarray
        三阶 normal（含 ts 与 st 的 normal）贡献
    yr : (N,) ndarray
        三阶 time-reversed 贡献
    """


    T  = float(experiment.T)
    lt = float(experiment.lt)
    # print(f"T:{T}, lt:{lt}")

    # 占据（用实部能量）
    Eigenval = np.array(experiment.Eigenval)
    # 若是方阵取对角
    if Eigenval.ndim == 2 and Eigenval.shape[0] == Eigenval.shape[1]:
        ve = np.diag(Eigenval)
    else:
        ve = Eigenval.astype(float).copy()
    ve = ve.real
    # 平移到最小能级为 0
    ve = ve - np.min(ve)
    # print("ve:",ve)
    occ = Occupation(ve, T)  # 期望返回形如 (N,) 向量
    # print("occ:",occ)

    # 三阶幅度（many-body），固定初态
    # 注意：in_state 为 1-based，传给 M35 也保持一致（若你那边要求 0-based，请自行在 M35 内部处理）
    rate, rater, raten, ratenr = M35(experiment, in_state)
    # 电压网格
    xrange_val = experiment.xrange
    if np.ndim(xrange_val) == 1:  # 已给定网格
        x = np.asarray(xrange_val, dtype=float).copy()
    else:
        V = abs(float(xrange_val))
        x = np.linspace(V, -V, 1000)  # 与 Scilab 一致：降序

    # 预分配
    y  = np.zeros_like(x, dtype=float)
    yr = np.zeros_like(x, dtype=float)

    # 维度信息
    Eigenvec = np.array(experiment.Eigenvec)
    N = Eigenvec.shape[1]

    # Scilab 1-based -> Python 0-based
    in0 = int(in_state) - 1

    # 带宽/截断：取 atom[jposition].w # 兼容 atom 为 list/tuple 或对象数组
    w_j = experiment.atom.w

    # 主循环
    for mid in range(N):
        for fin in range(N):
            # 幅度阈值
            if (np.abs(rate[mid, fin]) +
                np.abs(raten[mid, fin]) +
                np.abs(rater[mid, fin]) +
                np.abs(ratenr[mid, fin])) > (precision ** 2):

                # 对数核（Kondo 型）：两种偏置方向
                tlog1 = ln_t2(ve[mid] - ve[in0] + x, w_j, T, lt)  # +V
                tlog2 = ln_t2(ve[mid] - ve[in0] - x, w_j, T, lt)  # -V
                # print('tlog1:',tlog1)
                # print('tlog2:',tlog2)
                # 温度展宽阶跃核
                en = fstep((ve[fin] - ve[in0] - x) / T)  # st：Δ - V
                ep = fstep((ve[fin] - ve[in0] + x) / T)  # ts：Δ + V

                # print(f"Shape of ep: {ep.shape}")
                # print(f"Shape of tlog1: {tlog1.shape}")

                # 四类贡献（取实部、整体系数 -2，与原实现一致）
                pre = -2.0 * float(occ[in0])

                ytemp   = pre * np.real(rate[mid,   fin] * ep * tlog1)   # ts normal
                ytempr  = pre * np.real(rater[mid,  fin] * ep * tlog2)   # ts time-reversed
                ytempn  = pre * np.real(raten[mid,  fin] * en * tlog2)   # st normal
                ytempnr = pre * np.real(ratenr[mid, fin] * en * tlog1)   # st time-reversed
                # 可选保存
                if savedata:
                    # 文件名仍使用 1-based 索引以贴近原代码
                    in1, mid1, fin1 = in0 + 1, mid + 1, fin + 1
                    arr1 = np.column_stack([x, ytemp + ytempn])
                    arr2 = np.column_stack([x, ytempr + ytempnr])
                    np.savetxt(f"data{in1}-{mid1}-{fin1}.dat",  arr1, fmt="%.10g", delimiter=csv_delim)
                    np.savetxt(f"data{in1}-{mid1}-{fin1}r.dat", arr2, fmt="%.10g", delimiter=csv_delim)

                # 累加到 normal / time-reversed 两条曲线
                y  += (ytemp + ytempn)
                yr += (ytempr + ytempnr)

    # 可选绘图
    if plotspec:
        try:
            import matplotlib.pyplot as plt
            plt.plot(x, np.real(y),  label='3rd normal')
            plt.plot(x, np.real(yr), label='3rd time-reversed')
            plt.plot(x, np.real(y + yr), label='3rd total')
            plt.xlabel('V (same unit as energies)')
            plt.ylabel('dI/dV (arb. units)')
            plt.legend()
            plt.tight_layout()
            plt.title('spec3_in_state')
            plt.show()
        except Exception as e:
            print("Plot failed:", e)

    return x, y, yr
