from calc_spect import *
from experiment import *
from hamilton import *


class CalcStore:
    """用来保存计算出的原始谱和偏移缩放后的谱（仿 calc 结构）"""
    def __init__(self):
        self.spec_raw = None   # shape: (N, 1 + n_atoms)
        self.spec = None       # shape: (N, 1 + n_atoms)

def gui_drawspec(experiment,
                 calc: CalcStore,
                 savedata: bool = False,
                 precision: float = 1e-3,
                 normalize: bool = False,
                 show_plot: bool = True,
                 debug: bool = False):
    """
    Python 版的 gui_drawspec（去除了 Scilab GUI 依赖，保留计算逻辑与绘图）。

    Parameters
    ----------
    experiment : object/dict
        需要包含：Eigenval, Eigenvec, T, xrange/xgrid, atom, position/jposition,
                 A, b, x0, y0, third_order_calc, rate_calc 等与原代码一致的字段。
        - atom 可以是单个原子对象或 list[原子对象]
    calc : CalcStore
        存放 spec_raw 与 spec 的容器。
    savedata : bool
        是否把每一步计算曲线保存为文件（与原 %savedata 对应）。
    precision : float
        占据阈值（与 %precission 对应）。
    normalize : bool
        是否做归一尺度（与 %normalize 对应）。
    stopcalc_flag : list|None
        若传入形如 [False] 的单元素列表，你可以在外部把它改为 [True] 来中断长循环。
    show_plot : bool
        是否用 matplotlib 绘图。
    """

    # 若还没有本征系统，就先算
    if experiment.Eigenval is None :
        eigenvalues(experiment)  # 需要你实现：填充 experiment.Eigenvec / Eigenval
    print(experiment.Eigenval)
    # 体系规模
    nbr_of_atoms=experiment.atom.natoms
    print(f'检测到：{nbr_of_atoms}个原子')
    Eigenvec = np.asarray(experiment.Eigenvec)
    nbr_of_states = Eigenvec.shape[1]
    assert 2**nbr_of_atoms == Eigenvec.shape[0], "Eigenvec 与 原子数 维度不匹配"
    maxcnt = nbr_of_states * nbr_of_atoms

    # ------------------- 非 rate 方程 -------------------

    cnt = 0
    print("Calculating Spectrum ...")

    daty_cols = []  # 每个原子一列（y+y2）
    if debug:
        daty_cols_2order=[]
        daty_cols_3order=[]
    occ = Occupation(np.asarray(experiment.Eigenval), float(experiment.T))

    for atnr in range(1, nbr_of_atoms + 1):
        experiment.position=atnr
        experiment.jposition=atnr

        # 二阶
        x, y = spec2(experiment, savedata=False, precision=precision, plotspec=False)
        if savedata:
            np.savetxt(f"data_2nd{atnr}.dat", np.column_stack([x, np.real(y)]), fmt="%.10g")

        # 三阶（把所有占据显著的初态 in 加权累加；spec3 内部已乘 occ(in)）
        y2 = np.zeros_like(y, dtype=float)
        if experiment.third_order_calc:
            for i in range(1, nbr_of_states + 1):
                if occ[i-1] > precision:
                    # 简单的“进度条”输出
                    cnt += 1
                    # print(f'cnt={cnt}')
                    print(f"  in={i}/{nbr_of_states}, tip at atom {atnr} ({cnt}/{maxcnt})")
                    x_, y1, y1r = spec3(experiment, i, savedata=False, precision=precision, plotspec=False)
                    # 与二阶相同电压网格假设；若不同，需要插值到 x 上
                    y2 += np.real(y1 + y1r)
                else:
                    cnt += 1

            if savedata:
                np.savetxt(f"data_inelastic{atnr}.dat",
                           np.column_stack([x, np.real(y2)]),
                           fmt="%.10g")

        # 汇总每个原子的 (y + y2) 二阶+三阶
        col = np.real(y) + np.real(y2)
        daty_cols.append(col)
        if debug:
            col_2order=np.real(y)
            daty_cols_2order.append(col_2order)
            col_3order=np.real(y2)
            daty_cols_3order.append(col_3order)

        if savedata:
            np.savetxt(f"data_inelastic_all{atnr}.dat",
                       np.column_stack([x, np.real(col)]),
                       fmt="%.10g")

    print("\nDone.")
    # 组织 spec_raw: 第一列是电压 x，随后每列是一个原子的曲线
    daty = np.column_stack(daty_cols) if len(daty_cols) > 1 else daty_cols[0].reshape(-1, 1)
    calc.spec_raw = np.column_stack([x, daty])  # shape (N, 1+n_atoms)
    if debug:
        daty_2order = np.column_stack(daty_cols_2order) if len(daty_cols_2order) > 1 else daty_cols_2order[0].reshape(-1, 1)
        spec_raw_2order = np.column_stack([x, daty_2order])  # shape (N, 1+n_atoms)
        daty_3order = np.column_stack(daty_cols_3order) if len(daty_cols_3order) > 1 else daty_cols_3order[0].reshape(-1, 1)
        spec_raw_3order = np.column_stack([x, daty_3order])  # shape (N, 1+n_atoms)
    # ------------------- 生成带偏移与缩放的 spec -------------------
    if calc.spec is None:
        A = float(experiment.A)
        if normalize:
            print('进行normalize')
            # 2 * (S(S+1) + U^2)^-1
            Sval = float(experiment.atom.S)
            Uval = float(experiment.atom.U)
            scale = 2.0 * (Sval * (Sval + 1.0) + Uval**2)**-1
        else:
            print('不进行normalize')
            scale = A

        # 第一列：x 偏移（x + x0）；后续各列：y 偏移 + 线性背景，再乘 scale
        x0 = float(experiment.x0)
        y0 = float(experiment.y0)
        b  = float(experiment.b)
        raw = calc.spec_raw
        # print(raw)
        x_col = raw[:, 0] + x0
        spec_cols = [x_col]
        for k in range(1, raw.shape[1]):
            spec_cols.append((raw[:, k] + y0 + b * raw[:, 0]) * scale)
        calc.spec = np.column_stack(spec_cols)

        if debug:
            spec_cols_2order = [x_col]
            for k in range(1, spec_raw_2order.shape[1]):
                spec_cols_2order.append((spec_raw_2order[:, k] + y0 + b * spec_raw_2order[:, 0]) * scale)
            spec_2order = np.column_stack(spec_cols_2order)
            spec_cols_3order = [x_col]
            for k in range(1, spec_raw_3order.shape[1]):
                spec_cols_3order.append((spec_raw_3order[:, k] + y0 + b * spec_raw_3order[:, 0]) * scale)
            spec_3order = np.column_stack(spec_cols_3order)

    # ------------------- 绘图 -------------------
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            X = calc.spec[:, 0]
            Y = calc.spec[:, 1:]
            plt.plot(X, Y)
            plt.ylabel("dI/dV (a.u.)")
            plt.xlabel("eV (mV)")
            # 图例：Spin #1..N 或单个位置
            # labels = [f"Spin #{i}" for i in range(1, Y.shape[1] + 1)]
            # 如果是多个原子，显示全部图例；否则不显示也行
            # plt.legend(labels, fontsize=9)
            plt.xlim(min(X), max(X))
            plt.tight_layout()
            plt.title('TOTAL')
            plt.show()
        except Exception as e:
            print("Plot failed:", e)
    if debug:
        try:
            import matplotlib.pyplot as plt
            X = spec_2order[:, 0]
            Y = spec_2order[:, 1:]
            plt.plot(X, Y)
            plt.ylabel("dI/dV (a.u.)")
            plt.xlabel("eV (mV)")
            # 图例：Spin #1..N 或单个位置
            labels = [f"Spin #{i}" for i in range(1, Y.shape[1] + 1)]
            # 如果是多个原子，显示全部图例；否则不显示也行
            plt.legend(labels, fontsize=9)
            plt.xlim(min(X), max(X))
            plt.tight_layout()
            plt.title('TOTAL_2order')
            plt.show()
        except Exception as e:
            print("Plot failed:", e)
        try:
            import matplotlib.pyplot as plt
            X = spec_3order[:, 0]
            Y = spec_3order[:, 1:]
            plt.plot(X, Y)
            plt.ylabel("dI/dV (a.u.)")
            plt.xlabel("eV (mV)")
            # 图例：Spin #1..N 或单个位置
            labels = [f"Spin #{i}" for i in range(1, Y.shape[1] + 1)]
            # 如果是多个原子，显示全部图例；否则不显示也行
            plt.legend(labels, fontsize=9)
            plt.xlim(min(X), max(X))
            plt.tight_layout()
            plt.title('TOTAL_3order')
            plt.show()
        except Exception as e:
            print("Plot failed:", e)

if __name__ == "__main__":

    experiment=Experiment()
    calcStore = CalcStore()
    gui_drawspec(experiment, calcStore, debug=True)
