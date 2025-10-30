import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

# 设置支持中文字体，避免中文显示乱码或缺失
rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
def get_data(filepath,xlabel="Bias (meV)", ylabel="dI/dV (a.u.)", title=None,
             savepath=None, show=False, markersize=12, alpha=0.9):
    """读取路径，返回数据"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 判断首行是否为表头
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    tokens = first_line.split()
    has_header = False
    try:
        _ = [float(t) for t in tokens]
    except ValueError:
        has_header = True

    data = np.loadtxt(filepath, delimiter=None, ndmin=2, skiprows=1 if has_header else 0)
    if data.shape[1] < 2:
        raise ValueError("数据列数不足：需要至少两列（mev 和 didv）。")
    mev, didv = data[:, 0], data[:, 1]
    if show:
        plt.figure(figsize=(6.5, 4.2))
        plt.scatter(mev, didv, s=markersize, alpha=alpha)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()
    return mev, didv


def plot_dat(filepath, xlabel="Bias (meV)", ylabel="dI/dV (a.u.)", title=None,
             savepath=None, show=True, markersize=12, alpha=0.9):
    """读取路径，直接作图"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 判断首行是否为表头
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    tokens = first_line.split()
    has_header = False
    try:
        _ = [float(t) for t in tokens]
    except ValueError:
        has_header = True

    data = np.loadtxt(filepath, delimiter=None, ndmin=2, skiprows=1 if has_header else 0)
    if data.shape[1] < 2:
        raise ValueError("数据列数不足：需要至少两列（mev 和 didv）。")

    mev, didv = data[:, 0], data[:, 1]


    # 纯散点图（不连线）
    if show:
        plt.figure(figsize=(6.5, 4.2))
        plt.scatter(mev, didv, s=markersize, alpha=alpha)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()
    return mev, didv


# —— 文件对话框版本（仍然画散点图） ——
def plot_dat_with_dialog(initialdir=None, **kwargs):
    """
    弹出文件选择框，选择 .dat/.txt 文件后调用 plot_dat 画【散点图】。
    其他参数通过 **kwargs 透传给 plot_dat（如 title、savepath、show、markersize、alpha 等）。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError("此功能需要 tkinter 支持。请在有桌面环境的 Python 中运行。") from e

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="选择数据文件（两列：mev  didv）",
        initialdir=initialdir or os.getcwd(),
        filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*.*")]
    )

    if not path:
        messagebox.showinfo("提示", "未选择文件。")
        return None, None

    try:
        return plot_dat(path, **kwargs)
    except Exception as e:
        messagebox.showerror("错误", f"处理文件失败：\n{e}")
        return None, None

import os
import numpy as np
import matplotlib.pyplot as plt

def _load_xy(filepath):
    """读取两列 X,Y 数据；自动跳过表头。"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    toks = first.split()
    skip = 0
    try:
        _ = [float(t) for t in toks]
    except ValueError:
        skip = 1
    data = np.loadtxt(filepath, ndmin=2, skiprows=skip)
    if data.shape[1] < 2:
        raise ValueError(f"{filepath} 数据列不足两列")
    return data[:, 0], data[:, 1]

def plot_pairs_2n(
    natoms=1,
    calc=None,
    filepaths=None,
    use_dialog=False,
    xlabel="Bias (meV)",
    ylabel="dI/dV (a.u.)",
    titles=None,           # 可传入长度为 n 的标题列表
    save_dir=None,         # 若提供，则保存到该目录
    markersize=12,
    alpha=0.9,
    linewidth=1.2,
    show=True,
    matrix=None
):
    """
    从 2*nimage 份 (X,Y) 数据文件生成 nimage 个图像画布；每个画布上画2组数据。
    实验曲线来自filepath
    拟合曲线来自Calc
    参数
    ----
    filepaths : list[str] or None
        数据文件路径列表（长度必须为偶数）。若为 None 且 use_dialog=True，则弹框选择。
    use_dialog : bool
        是否使用图形对话框一次性多选文件（Windows/macOS/Linux）。
    titles : list[str] or None
        每个画布的标题（长度应为 n）。不传则用自动标题。
    save_dir : str or None
        保存目录；若提供则每个画布保存为 pair_01.png, pair_02.png, ...
    style : "scatter" | "line"
        绘图样式；默认散点。你也可改为 "line" 画折线。
    其余参数传递给画图的外观。
    """
    # 选文件（可多选）
    if use_dialog and filepaths is None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except Exception as e:
            raise RuntimeError("需要 tkinter 支持才能弹出文件选择框。") from e
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(
            title="选择 2n 个数据文件（每个含两列 X Y）",
            filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*.*")]
        )
        if not paths:
            messagebox.showinfo("提示", "未选择文件。")
            return
        filepaths = list(paths)

    if not filepaths or len(filepaths) == 0:
        raise ValueError("请提供文件路径列表，或设置 use_dialog=True 进行选择。")

    if len(filepaths)  != natoms:
        raise ValueError(f"文件数为 {len(filepaths)}，必须为{natoms}。")

    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if titles is not None and len(titles) != natoms:
        raise ValueError(f"titles 长度应为 {natoms}，但收到 {len(titles)}。")

    # 计算子图网格
    ncols = int(np.ceil(np.sqrt(natoms+3)))
    nrows = int(np.ceil((natoms+3) / ncols))

    # 增加 figsize，并为 tight_layout 增加填充
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.5, nrows * 5.0))  # 稍微增大尺寸
    # 如果只有一个子图，axes 可能不是数组，需要处理
    if natoms == 1:
        axes = np.array([axes])
    axes = axes.flatten()  # 将 axes 展平，方便迭代
    # 实验数据集合
    for j in range(natoms):
        f1 = filepaths[j]
        x1, y1 = _load_xy(f1)
        ax = axes[0]  # 获取当前子图
        label1 = os.path.basename(f1)
        ax.scatter(x1, y1, s=markersize, alpha=alpha, label=label1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_title(f"exp di/dv")
    # 理论数据集合
    for k in range(natoms):
        x2 = calc.spec[:, 0]
        y2 = calc.spec[:, k + 1]
        ax = axes[1]  # 获取当前子图
        ax.plot(x2, y2, linewidth=linewidth, label=f"calc spin{k+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_title(f"calc di/dv")
    # 单个位置实验vs理论
    for i in range(natoms):
        f1 = filepaths[i]

        x1, y1 = _load_xy(f1)
        x2 = calc.spec[:, 0]
        y2 = calc.spec[:, i + 1]

        ax = axes[i+2]  # 获取当前子图

        label1 = os.path.basename(filepaths[i])
        label2 = f"calc spin{i+1}"

        ax.scatter(x1, y1, s=markersize, alpha=alpha, label=label1)
        ax.plot(x2, y2, linewidth=linewidth, label=label2,color='red')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        if titles is not None:
            ax.set_title(titles[i])
        else:
            ax.set_title(f"spin {i + 1} di/dv")
    # 所用的海森堡矩阵
    if matrix is not None:
        ax = axes[natoms+2]
        x = np.arange(len(matrix))
        ax.bar(x, matrix, color='skyblue')
        ax.set_title("J Matrix")
        ax.set_xlabel('COUPLE')
        ax.set_ylabel('J/meV')


    # 隐藏未使用的子图
    for j in range(natoms+3, nrows * ncols):
        fig.delaxes(axes[j])
    plt.tight_layout(pad=2.0)  # 增加填充，避免重叠
    if save_dir:
        # 保存整个画布，而不是单个子图
        out = os.path.join(save_dir, "combined_plots.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return [fig]  # 返回包含单一 figure 对象的列表


# --------- 使用示例 ----------
# 1) 直接给路径列表（顺序配对：0-1, 2-3, ...）
# figs = plot_pairs_2n(
#     filepaths=["a1.dat", "a2.dat", "b1.dat", "b2.dat"],
#     titles=["样品A", "样品B"],
#     save_dir="pairs_out",
#     style="scatter",   # 或 "line"
#     show=True
# )

# 2) 弹窗多选文件（一次选 2n 个文件；顺序配对）
# figs = plot_pairs_2n(use_dialog=True, save_dir="pairs_out", style="scatter")




# if __name__ == "__main__":
    # mev, didv = plot_dat_with_dialog(title="dI/dV vs Bias (scatter)", savepath="didv.png", show=True)
