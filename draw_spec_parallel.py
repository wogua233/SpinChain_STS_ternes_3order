# draw_spec_parallel.py
from calc_spect import *
from experiment import *
from hamilton import *
import os, copy, numpy as np
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

# —— 可选：限制 BLAS/OMP 线程，避免过度并行 ——
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ===== 把 worker 放在模块顶层（不是函数内部！）=====
def _spec3_worker(exp_pack, in_state, precision):
    """
    exp_pack: 一个“可 picklable”的轻量对象（例如 SimpleNamespace/dict），
              里面包含 spec3 所必需的字段。
    """
    # 如果你那边的 spec3 接受的是“富对象 experiment”，
    # 就在这里把它还原；否则直接传 dict/namespace 进去。
    # 这里示范用 SimpleNamespace：
    experiment = SimpleNamespace(**exp_pack.__dict__)
    # 深拷贝防止子进程彼此写冲突（保守做法）
    experiment = copy.deepcopy(experiment)

    x_, y1, y1r = spec3(experiment, in_state,
                        savedata=False,
                        precision=precision,
                        plotspec=False)
    return (in_state, np.asarray(x_), np.real(np.asarray(y1) + np.asarray(y1r)))

# ===== 你原来的 gui_draw 里，把“并行三阶”的那段替换为下方版本 =====
def accumulate_third_order_parallel(experiment, occ, nbr_of_states, x, atnr, precision):
    """
    返回 y2（与原串行版本含义相同）
    """
    # 仅挑选占据显著的初态
    todo = [i for i in range(1, nbr_of_states + 1) if occ[i-1] > precision]
    if not todo:
        return np.zeros_like(x, dtype=float)

    # —— 把 experiment 打包成“可 picklable”的轻量对象（重要）——
    # 最稳妥做法：只放 spec3 需要的字段；不要塞进复杂/不可序列化成员
    exp_pack = SimpleNamespace(
        T=float(experiment.T),
        lt=float(experiment.lt),
        xrange=np.array(experiment.xrange, dtype=float) if np.ndim(experiment.xrange)==1 else float(experiment.xrange),
        ptip=np.array(experiment.ptip, dtype=float),
        psample=np.array(experiment.psample, dtype=float),
        position=int(experiment.position),
        jposition=int(getattr(experiment, "jposition", experiment.position)),
        A=float(getattr(experiment, "A", 1.0)),
        b=float(getattr(experiment, "b", 0.0)),
        x0=float(getattr(experiment, "x0", 0.0)),
        y0=float(getattr(experiment, "y0", 0.0)),
        third_order_calc=bool(getattr(experiment, "third_order_calc", True)),
        # 关键的量：
        Eigenvec=np.array(experiment.Eigenvec),
        Eigenval=np.array(experiment.Eigenval),
        # atom：尽量保证其为简单结构。若是自定义类，建议在此打平成简单 namespace
        atom=SimpleNamespace(
            S=float(experiment.atom.S),
            g=float(experiment.atom.g),
            D=float(experiment.atom.D),
            E=float(experiment.atom.E),
            J=np.array(experiment.atom.J) if np.ndim(experiment.atom.J) else float(experiment.atom.J),
            U=float(experiment.atom.U),
            w=float(experiment.atom.w),
            natoms=int(experiment.atom.natoms),
        ),
        rate_calc=bool(getattr(experiment, "rate_calc", False)),
        entanglement=bool(getattr(experiment, "entanglement", False)),
    )

    y2 = np.zeros_like(x, dtype=float)
    n_workers = min(os.cpu_count() or 1, len(todo))
    # Windows 下建议使用 spawn 上下文（ProcessPoolExecutor 默认就是 spawn）
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_spec3_worker, exp_pack, i, precision) for i in todo]
        for fut in as_completed(futures):
            i_done, x_ret, yret = fut.result()
            if x_ret.shape != x.shape or np.max(np.abs(x_ret - x)) > 1e-12:
                # 如果网格不一致，插值到主网格
                yret = np.interp(x, x_ret, yret)
            y2 += yret
    return y2

def gui_drawspec_parallel(experiment,
                 calc=None,
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
        y2 = accumulate_third_order_parallel(experiment, occ, nbr_of_states, x, atnr, precision)


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