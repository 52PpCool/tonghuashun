# -*- coding: utf-8 -*-
"""
================================================================================
 32 | Purged K-Fold CV 与 Embargo：金融机器学习交叉验证的正规做法
================================================================================
承接 07_机器学习选股。课07 讲了"标签对齐"与"统计量样本"两类泄漏，
本课补上第三块拼图：**交叉验证怎么切，才不会把未来渗进训练集。**

先剧透结论（正文用数字证明）：
  1) 普通 KFold(shuffle) 用在金融面板上是错的：随机打乱后，验证样本的
     "邻居日"（标签窗口与它大量重叠、特征高度自相关、同一截面强相关）
     几乎全在训练集里 —— 模型等于提前见过验证答案，OOF IC 大幅虚高；
  2) TimeSeriesSplit 方向对了，但每折边界处训练集的标签窗口仍会伸进
     验证期（5 日标签重叠），且验证块之后没有留任何缓冲；
  3) Purged K-Fold + Embargo（López de Prado《金融机器学习》第 7 章）：
       purge   = 剔除训练集中"标签窗口与验证期重叠"的样本（挡住验证块
                 之前、标签朝前伸 H 日的那条通道）；
       embargo = 验证块之后再挖掉 h 个交易日（h >= 特征最长回看窗口），
                 挡住另一条通道：验证块之后的训练样本用 60 日滚动窗口算
                 特征，会把验证期行情"背"进训练集。
     为什么 embargo 在验证集"之后"？因为特征回看窗口朝过去看，只有
     验证集之后的训练样本，其特征才可能装进验证期的信息；验证集之前的
     越界者是标签窗口（朝未来伸 H 日），归 purge 管。两条真空带缺一不可。
  4) 但 CV 是"评估器"不是"回测器"：purged CV 的 OOF IC 应接近
     walk-forward 样本外 IC（真值参照），KFold 则差得很远；
     (c) CV 分数再好看也要过交易成本关、再上模拟盘（López de Prado
     三重屏障：CV / 回测 / 模拟盘）。

数据：data/cache/ 30 只真实 A 股日K（与课 07 同源同口径，utf-8-sig）；
      特征 = 动量 5/20/60 日 + 60 日波动率；标签 = 未来 5 日收益
      close[t+5]/close[t] - 1。
运行：../../.venv/bin/python main.py   （约 1 分钟，种子 42，可复现）
================================================================================
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面环境出图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import lightgbm as lgb
from sklearn.model_selection import KFold, TimeSeriesSplit

warnings.filterwarnings("ignore")

# ============================================================ 全局配置（S1）
SEED = 42
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)  # 泄漏注入实验用的独立随机源

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "cache")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2018-01-01"    # 数据起点（与课 07 同口径）
FIRST_OOS = "2023-01-01"     # walk-forward 样本外起点（与课 07 同口径：训练 2018-2022）
H = 5                        # 标签持有期：未来 5 日收益（交易日）
N_SPLITS = 5                 # 三种 CV 统一 5 折
EMBARGO = 60                 # embargo 天数：>= 特征最长回看窗口（mom60/vol60 用 60 日）
REFIT_EVERY = 126            # walk-forward 约每半年（126 个交易日）重训一次
MIN_CS_N = 8                 # 计算日度截面 IC 所需最少股票数（与课 07 一致）

# 特征口径：本课聚焦"切分方式"，特征精简为课 07 的前 4 个
FEATURES = ["mom5", "mom20", "mom60", "vol60"]
FEATURES_LEAK = FEATURES + ["leak"]

LGB_PARAMS = dict(           # 与课 07 相同的极小 LightGBM
    n_estimators=100,
    num_leaves=15,
    learning_rate=0.05,
    verbose=-1,
    random_state=SEED,
)

IS_SYNTHETIC = False         # 数据缺失被迫用合成数据时置 True 并显著标注


# ============================================================ 数据加载（S2）
def load_close_panel():
    """读取 data/cache/stock_*.csv -> 收盘价宽表（index=交易日, columns=代码）。

    本课只用到 close（动量与波动率都由它算出）。数据缺失时退回合成数据，
    保证课程永远能跑通（绝不带错误退出）。
    """
    global IS_SYNTHETIC
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("stock_") and f.endswith(".csv")]
    if not files:
        IS_SYNTHETIC = True
        print("[警告] 未找到 data/cache/stock_*.csv，改用合成数据演示（所有图表将标注）")
        dates = pd.bdate_range(START_DATE, "2026-09-17")
        rng = np.random.default_rng(SEED)
        cols = {}
        for i in range(30):
            ret = rng.normal(0.0003, 0.02, len(dates))
            cols[f"S{i:03d}"] = 20 * np.exp(np.cumsum(ret))
        return pd.DataFrame(cols, index=dates)

    frames = []
    for fname in sorted(files):
        code = fname[:-4].split("_")[1]
        df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding="utf-8-sig",
                         usecols=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= START_DATE]
        frames.append(df.set_index("date")["close"].rename(code))
    close_w = pd.concat(frames, axis=1).sort_index()
    return close_w


def build_dataset(close_w):
    """宽表 -> 特征/标签长表。与课 07 同口径：

    特征（全部只用 t 日及以前的数据，t 日收盘后才能确定）：
      mom5  = close[t]/close[t-5] - 1      mom20 / mom60 同理
      vol60 = 过去 60 日日收益率标准差
    标签：fwd5 = close[t+5]/close[t] - 1 （收益发生在 t+1..t+5 共 5 个交易日）
    """
    ret1 = close_w / close_w.shift(1) - 1.0
    panels = {
        "mom5": close_w / close_w.shift(5) - 1.0,
        "mom20": close_w / close_w.shift(20) - 1.0,
        "mom60": close_w / close_w.shift(60) - 1.0,
        "vol60": ret1.rolling(60).std(),
        "fwd5": close_w.shift(-H) / close_w - 1.0,   # 标签：严格错位到未来
    }
    dates = close_w.index
    codes = close_w.columns
    n_d, n_c = close_w.shape
    long = pd.DataFrame({k: v.values.ravel() for k, v in panels.items()})  # 宽表按日期优先展开
    long["date"] = np.repeat(dates.values, n_c)
    long["code"] = np.tile(codes.values, n_d)
    long = long.dropna().reset_index(drop=True)      # 前 60 日热身 + 末尾 5 日无标签的行剔除
    return long


# ============================================================ 手写 Purged K-Fold + Embargo（S3）
class PurgedKFold:
    """López de Prado 式 Purged K-Fold + Embargo（截面面板版，本课灵魂）。

    与普通 KFold 的三点不同：
    1. 以"交易日"为整组切分：同一交易日的 30 只股票是一个截面，绝不能
       拆到不同折（同截面样本强相关，拆开本身就是泄漏）；
    2. purge：剔除训练集中"标签窗口 [t+1, t+H] 与验证期重叠"的样本 ——
       即验证块开始前 H 个交易日之内的训练样本（它们的标签与验证集的
       标签共享同一批未来收益，等于把验证答案泄露给模型）；
    3. embargo：验证块结束后 h 个交易日内的训练样本也剔除 —— 这些样本
       的特征用 60 日回看窗口算出，窗口会伸进验证期，把验证期行情带进
       训练集；h 取 >= 特征最长回看窗口（本课 60 日）。

    split() 以"交易日序号"做索引运算：uniq 为排序后的交易日数组，
    pos 为每条样本所属交易日的序号，purge/embargo 就是序号区间上的剔除。
    """

    def __init__(self, n_splits=N_SPLITS, label_horizon=H, embargo=EMBARGO):
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.fold_spans_ = []   # 每折的时间区间（画甘特图用）

    def split(self, row_dates):
        """row_dates: 每条样本的交易日数组。返回 [(train_idx, val_idx), ...]。"""
        uniq, pos = np.unique(row_dates, return_inverse=True)   # uniq=交易日序列, pos=每行所属序号
        n = len(uniq)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        folds = []
        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]                   # 验证块 = 交易日序号 [lo, hi)
            val_idx = np.where((pos >= lo) & (pos < hi))[0]
            tr_idx = np.where((pos < lo) | (pos >= hi))[0]
            tr_pos = pos[tr_idx]

            # ---- purge：验证块之前的训练样本，标签窗口 [t+1, t+H] 伸进验证期者剔除 ----
            #     交易序号 t 的标签覆盖 [t+1, t+H]，与验证期 [lo, hi) 重叠 <=> t >= lo - H
            pre = tr_pos < lo
            drop_purge = pre & (tr_pos >= lo - self.label_horizon)
            # ---- embargo：验证块结束后 h 个交易日内的训练样本剔除 ----
            post = tr_pos >= hi
            drop_embargo = post & (tr_pos <= hi - 1 + self.embargo)

            keep = ~(drop_purge | drop_embargo)
            tr_idx = tr_idx[keep]
            folds.append((tr_idx, val_idx))

            # ---- 记录甘特图所需的时间区间（验证块两侧各留一条真空带）----
            has_before = lo - self.label_horizon > 0
            has_after = hi + self.embargo < n
            self.fold_spans_.append(dict(
                k=k, lo=lo, hi=hi,
                train_before=(uniq[0], uniq[lo - self.label_horizon - 1]) if has_before else None,
                purge=(uniq[max(lo - self.label_horizon, 0)], uniq[lo - 1]) if lo > 0 else None,
                val=(uniq[lo], uniq[hi - 1]),
                embargo=(uniq[hi], uniq[min(hi - 1 + self.embargo, n - 1)]) if hi < n else None,
                train_after=(uniq[min(hi + self.embargo, n - 1)], uniq[n - 1]) if has_after else None,
                n_train=len(tr_idx), n_val=len(val_idx),
                n_purged=int(drop_purge.sum()), n_embargoed=int(drop_embargo.sum()),
            ))
        return folds


# ============================================================ CV 执行与 IC 评估（S4）
def fit_predict_folds(X, y, folds):
    """在给定折切分上训练/预测：返回 OOF 预测、平均增益重要性与每折样本量。

    每条样本恰好处在一个折的验证集中、且预测它的模型训练时没见过它 ——
    这就是"袋外(OOF)预测"。三种 CV 都走这一个函数，保证唯一差别是切分方式。
    """
    oof = np.full(len(y), np.nan)
    imp = np.zeros(X.shape[1])
    sizes = []
    for tr, va in folds:
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
        imp += model.booster_.feature_importance(importance_type="gain")
        sizes.append((len(tr), len(va)))
    return oof, imp / len(folds), sizes


def walk_forward(X, y, row_dates, first_oos=FIRST_OOS, refit_every=REFIT_EVERY):
    """expanding walk-forward（真值参照）：训练集永远在验证集之前。

    每隔 refit_every 个交易日重训一次；同样做边界 purge（训练样本的标签
    窗口不得伸进验证期），否则"真值"自己也有 5 日的边界泄漏。
    注意：walk-forward 不需要右侧 embargo —— 训练集永远在验证期之前，
    不存在"验证期之后的训练样本"。
    """
    uniq, pos = np.unique(row_dates, return_inverse=True)
    n = len(uniq)
    oos0 = int(np.searchsorted(uniq, np.datetime64(FIRST_OOS)))
    oof = np.full(len(y), np.nan)
    imp = np.zeros(X.shape[1])
    n_models = 0
    for c0 in range(oos0, n, refit_every):
        c1 = min(c0 + refit_every, n)
        va = np.where((pos >= c0) & (pos < c1))[0]
        tr = np.where(pos < c0 - H)[0]           # 边界 purge：标签窗口止于验证期开始之前
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
        imp += model.booster_.feature_importance(importance_type="gain")
        n_models += 1
    return oof, imp / n_models


def daily_ic_series(dates, pred, label, min_n=MIN_CS_N):
    """日度截面 Spearman IC：每个交易日内 (OOF预测, 标签) 的秩相关。

    一个日期的有效 (股票) 样本数 < min_n 时记 NaN（与课 07 同一纪律）。
    """
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "pred": pred, "label": label}).dropna()
    grp = df.groupby("date")
    df["rp"] = grp["pred"].rank()
    df["rl"] = grp["label"].rank()
    # 秩的 Pearson 相关 = Spearman IC；用组内矩公式全向量化计算（几千个交易日秒级完成）
    df["rp2"], df["rl2"], df["rpl"] = df["rp"] ** 2, df["rl"] ** 2, df["rp"] * df["rl"]
    agg = df.groupby("date")[["rp", "rl", "rp2", "rl2", "rpl"]].mean()
    cnt = grp.size()
    var_p = agg["rp2"] - agg["rp"] ** 2
    var_l = agg["rl2"] - agg["rl"] ** 2
    cov = agg["rpl"] - agg["rp"] * agg["rl"]
    ic = cov / np.sqrt(var_p * var_l)
    ic[cnt[cnt < min_n].index] = np.nan          # 截面股票数不足的日期不计
    return ic


def ic_stats(ic_s, start=None):
    """汇总一段日度 IC：均值 / 标准差 / ICIR（日频与其年化值）。"""
    s = ic_s.dropna()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    sd = s.std()
    icir = s.mean() / sd if sd and sd > 0 else np.nan
    return dict(n_days=int(len(s)), ic=s.mean(), ic_std=sd,
                icir=icir, icir_ann=icir * np.sqrt(252) if np.isfinite(icir) else np.nan)


# ============================================================ 绘图（S5）
C_TRAIN, C_VAL, C_PURGE, C_EMB = "#4878A8", "#E8853D", "#C44E52", "#55A868"


def plot_gantt(pk, uniq, tag=""):
    """【全课核心图】5 折的 训练集-验证集-embargo 甘特图 + 两处边界局部放大。

    5 日的 purge 空档在 8 年全景里只有 1 个像素宽，所以下方配两个"放大镜"：
    分别放大第 3 折验证块的起点（purge）与终点（embargo），看清三段如何衔接。
    uniq 为排序后的交易日数组（放大窗口按交易日序号取边界）。
    """
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.95, 0.95], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    n_folds = len(pk.fold_spans_)
    d2n = lambda t: mdates.date2num(pd.Timestamp(t))

    def draw_fold(a, f, y=0.0, h=0.7):
        """把一折的 训练|purge|验证|embargo|训练 五段画成横向色带。"""
        spans = [("train_before", C_TRAIN, None), ("purge", C_PURGE, "///"), ("val", C_VAL, None),
                 ("embargo", C_EMB, "\\\\\\"), ("train_after", C_TRAIN, None)]
        for key, color, hatch in spans:
            if f.get(key):
                x0, x1 = d2n(f[key][0]), d2n(f[key][1])
                a.broken_barh([(x0, x1 - x0)], (y - h / 2, h),
                              facecolors=color, hatch=hatch, edgecolor="white" if hatch else "none")

    for f in pk.fold_spans_:
        draw_fold(ax, f, y=n_folds - 1 - f["k"])       # 第 1 折画在最上面

    handles = [Patch(fc=C_TRAIN, label="训练集"),
               Patch(fc=C_VAL, label="验证集"),
               Patch(fc=C_PURGE, hatch="///", edgecolor="white", label="purge（标签重叠剔除）"),
               Patch(fc=C_EMB, hatch="\\\\\\", edgecolor="white", label="embargo（特征回看剔除）")]
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=9.5, framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-0.7, n_folds - 0.3)
    ax.set_yticks(range(n_folds), [f"第 {n_folds - i} 折" for i in range(n_folds)])
    suffix = "（合成数据）" if IS_SYNTHETIC else ""
    ax.set_title(f"Purged K-Fold(5) + Embargo 切分示意{suffix}：验证块两侧各留一条真空带\n"
                 f"标签窗口 H={H} 日，embargo={EMBARGO} 日（>= 特征最长回看窗口 60 日）"
                 + (f"  {tag}" if tag else ""))
    ax.grid(axis="x", alpha=0.25)

    # ---- 两个"放大镜"：第 3 折（中间折）验证块的起点与终点 ----
    mid = pk.fold_spans_[n_folds // 2]
    k_mid = mid["k"] + 1
    zoom_specs = [
        (gs[1], mid["lo"] - 22, mid["lo"] + 38, "purge",
         f"局部放大 1 | 第 {k_mid} 折 验证块起点：训练 | purge(5日) | 验证 三段衔接\n"
         "purge 掉的训练日，其标签窗口 [t+1, t+5] 与验证期共享同一批未来收益 —— 不剔就是给模型递答案"),
        (gs[2], mid["hi"] - 18, min(mid["hi"] - 1 + EMBARGO + 18, len(uniq) - 1), "embargo",
         f"局部放大 2 | 第 {k_mid} 折 验证块终点：验证 | embargo(60日) | 训练 三段衔接\n"
         "embargo 挡住的训练日，其 60 日回看特征把验证期行情背进训练集（滚动窗口泄漏）"),
    ]
    for spec, i0, i1, key, ztitle in zoom_specs:
        axz = fig.add_subplot(spec)
        draw_fold(axz, mid)
        axz.set_xlim(d2n(uniq[max(i0, 0)]), d2n(uniq[min(i1, len(uniq) - 1)]))
        axz.set_yticks([])
        axz.set_title(ztitle, fontsize=9.5)
        axz.grid(axis="x", alpha=0.3)
        axz.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        axz.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axz.xaxis.get_major_locator()))
        # 关键边界竖线（虚线 = 验证块边界；三段含义见各放大图标题）
        axz.axvline(d2n(mid["val"][0] if key == "purge" else mid["val"][1]),
                    color="k", ls="--", lw=1)
    fig.savefig(os.path.join(OUT_DIR, "02_PurgedCV切分甘特图.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_comparison(stats, ic_dict, wf_ic_mean, win_name):
    """图01：三种 CV 的验证集 IC 对比（柱状 + 日度 IC 分布）。"""
    names = ["KFold(shuffle)", "TimeSeriesSplit", "PurgedKFold\n+Embargo"]
    keys = ["KFold", "TSC", "Purged"]
    means = [stats[k]["ic"] for k in keys]
    ses = [stats[k]["ic_std"] / np.sqrt(stats[k]["n_days"]) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bars = ax1.bar(names, means, yerr=ses, capsize=4,
                   color=[C_PURGE, "#8172B2", C_EMB], alpha=0.88, width=0.55)
    ax1.axhline(wf_ic_mean, ls="--", color="black", lw=1.4,
                label=f"walk-forward 样本外 IC（真值参照）= {wf_ic_mean:.4f}")
    for b, m, se, k in zip(bars, means, ses, keys):
        ratio = m / wf_ic_mean if wf_ic_mean else np.nan
        ax1.text(b.get_x() + b.get_width() / 2, m + se + 0.003,
                 f"{m:.4f}\n(真值的 {ratio:.2f} 倍)", ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"验证集日均 IC 对比（{win_name}）\n误差棒 = 1 标准误")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    data = [ic_dict[k].dropna().values for k in keys]
    bp = ax2.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [C_PURGE, "#8172B2", C_EMB]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_title("日度截面 IC 的分布（箱体不含离群点）")
    ax2.grid(axis="y", alpha=0.3)
    suffix = "（合成数据）" if IS_SYNTHETIC else ""
    fig.suptitle(f"普通 KFold 在金融时序上虚高 IC —— 三种交叉验证对比{suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "01_三种CV的IC对比.png"), dpi=150)
    plt.close(fig)


def plot_purged_vs_wf(ic_dict, win_name):
    """图04：purged CV 的 OOF IC 是否贴近 walk-forward 真值。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    styles = {"KFold": (C_PURGE, "KFold(shuffle) IC（虚高参照）"),
              "Purged": (C_EMB, "PurgedKFold+Embargo OOF IC"),
              "WF": ("black", "walk-forward 样本外 IC（真值）")}
    for k, (c, lab) in styles.items():
        s = ic_dict[k].dropna()
        ax1.plot(s.index, s.cumsum(), color=c, lw=1.6, label=lab + f"（终值 {s.sum():.2f}）")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.legend(fontsize=9)
    ax1.set_title(f"累计日度 IC（{win_name}）\npurged CV 紧贴真值，KFold 一路飞走")
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    m_p = ic_dict["Purged"].resample("ME").mean()
    m_w = ic_dict["WF"].resample("ME").mean()
    pair = pd.concat([m_p, m_w], axis=1, keys=["purged", "wf"]).dropna()
    r = pair["purged"].corr(pair["wf"])
    ax2.scatter(pair["wf"], pair["purged"], s=26, color=C_EMB, alpha=0.75)
    lims = [min(pair.min()), max(pair.max())]
    ax2.plot(lims, lims, "k--", lw=1, label="45°线（完美一致）")
    ax2.set_xlabel("walk-forward 月度 IC（真值）")
    ax2.set_ylabel("Purged CV 月度 OOF IC")
    ax2.set_title(f"月度 IC 散点：purged CV vs 真值\nPearson r = {r:.3f}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    suffix = "（合成数据）" if IS_SYNTHETIC else ""
    fig.suptitle(f"Purged CV 评估 vs walk-forward 真值对照{suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, "04_purgedCV与walkforward对照.png"), dpi=150)
    plt.close(fig)


def plot_leak_experiment(imp_kf_leak, imp_pk_leak, leak_stats, leak_col, win_name):
    """图03：泄漏注入实验 —— 特征重要性 + 注入前后各 CV 的 IC 变化。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 9))

    # ---- 上图：注入泄漏后的特征重要性（增益占比）----
    feats = list(imp_kf_leak.index)
    x = np.arange(len(feats))
    w = 0.38
    b1 = ax1.bar(x - w / 2, imp_kf_leak.values * 100, w, color=C_PURGE, alpha=0.85, label="KFold(shuffle)")
    b2 = ax1.bar(x + w / 2, imp_pk_leak.values * 100, w, color=C_EMB, alpha=0.85, label="PurgedKFold+Embargo")
    for b in (b1, b2):
        for rect in b:
            ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6,
                     f"{rect.get_height():.1f}%", ha="center", fontsize=8)
    idx = feats.index(leak_col)
    for b in (b1[idx], b2[idx]):
        b.set_edgecolor("black")
        b.set_linewidth(1.6)
    ax1.set_xticks(x, feats)
    ax1.set_ylabel("增益重要性占比 %")
    ax1.legend(fontsize=9)
    ax1.set_title(f"特征重要性（含泄漏列 '{leak_col}' = 未来5日收益 + 噪声，仅含标签约1%信息）\n"
                  "黑框列即泄漏列：模型大方地承认'我用了它'——重要性自己不会报警，要靠正确的 CV 对照真值才能识破")
    ax1.grid(axis="y", alpha=0.3)

    # ---- 下图：注入泄漏前后，各方法的验证集 IC ----
    order = ["KFold", "TSC", "Purged", "WF"]
    labels = ["KFold(shuffle)", "TimeSeriesSplit", "PurgedKFold+Embargo", "walk-forward(真值)"]
    before = [leak_stats[k]["ic_before"] for k in order]
    after = [leak_stats[k]["ic_after"] for k in order]
    x = np.arange(len(order))
    w = 0.38
    ax2.bar(x - w / 2, before, w, color="#9BB7D4", label="无泄漏（4 特征）")
    ax2.bar(x + w / 2, after, w, color=C_PURGE, alpha=0.9, label="注入 1% 泄漏列")
    ax2.set_ylim(0, max(max(before), max(after)) * 1.22)   # 顶部留白，避免数值与标题重叠
    for i, (b0, a0) in enumerate(zip(before, after)):
        rel = (a0 - b0) / abs(b0) * 100 if b0 else np.nan
        ax2.text(i - w / 2, b0 + 0.0015, f"{b0:.4f}", ha="center", fontsize=9)
        ax2.text(i + w / 2, a0 + 0.0015, f"{a0:.4f}\n({rel:+.0f}%)", ha="center", fontsize=9)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("日均 IC")
    ax2.set_title(f"注入泄漏列后各方法的验证集 IC（{win_name}）\n"
                  "purged CV 的跳变与真值同量级（泄漏被如实检出）；KFold 的虚高基线把泄漏增量吞掉，IC 却始终偏离真值")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    suffix = "（合成数据）" if IS_SYNTHETIC else ""
    fig.suptitle(f"泄漏注入实验：你的 CV 能不能'看见'泄漏？{suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "03_泄漏注入实验.png"), dpi=150)
    plt.close(fig)


# ============================================================ 主流程（S6）
def main():
    t0 = time.time()
    print("=" * 72)
    print("32 | Purged K-Fold CV 与 Embargo：金融ML交叉验证的正规做法")
    print("=" * 72)

    # ---------- 数据 ----------
    close_w = load_close_panel()
    df = build_dataset(close_w)
    if IS_SYNTHETIC:
        df.attrs["note"] = "合成数据"
    print(f"[数据] {df['code'].nunique()} 只股票 | "
          f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d} | "
          f"样本 {len(df):,} 行（特征={FEATURES}，标签=fwd5 未来{H}日收益）")

    X = df[FEATURES].to_numpy()
    y = df["fwd5"].to_numpy()
    row_dates = df["date"].to_numpy()

    # ---------- 三种切分器 ----------
    folds_kf = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X))
    folds_tsc = list(TimeSeriesSplit(n_splits=N_SPLITS).split(X))
    pk = PurgedKFold(n_splits=N_SPLITS, label_horizon=H, embargo=EMBARGO)
    folds_pk = pk.split(row_dates)

    n_purged = sum(f["n_purged"] for f in pk.fold_spans_)
    n_emb = sum(f["n_embargoed"] for f in pk.fold_spans_)
    print(f"[purge 代价] 全部 5 折共剔除：purge {n_purged:,} 行 + embargo {n_emb:,} 行 "
          f"（占样本 {(n_purged + n_emb) / len(df) * 100:.1f}% —— 这是防泄漏的样本成本）")

    # 切分核对：三种 CV 各折"验证集"覆盖的日期范围（一眼看出 KFold 的荒谬之处）
    print("[切分核对] 各折验证集的日期范围：")
    for name, folds in [("KFold", folds_kf), ("TSC  ", folds_tsc), ("Purged", folds_pk)]:
        for i, (_, va) in enumerate(folds):
            d = df["date"].to_numpy()[va]
            print(f"    {name} 折{i + 1}: {pd.Timestamp(d.min()):%Y-%m-%d} ~ {pd.Timestamp(d.max()):%Y-%m-%d}"
                  + ("   <- 随机打乱，每折都横跨全程" if name == "KFold" else ""))

    # ---------- 实验1：无泄漏数据上的三种 CV ----------
    print("\n" + "-" * 72)
    print("实验1：同一份数据（无泄漏特征）上的三种交叉验证")
    print("-" * 72)
    oof = {}
    imps = {}
    for name, folds in [("KFold", folds_kf), ("TSC", folds_tsc), ("Purged", folds_pk)]:
        oof[name], imps[name], _ = fit_predict_folds(X, y, folds)
        print(f"  {name:<7s} 5 折训练/验证完成")
    oof["WF"], imps["WF"] = walk_forward(X, y, row_dates)
    print(f"  WF      walk-forward 滚动训练完成（每 {REFIT_EVERY} 日重训，样本外 {FIRST_OOS} 起）")

    ic_all = {k: daily_ic_series(df["date"].to_numpy(), oof[k], y) for k in oof}
    win = "2023年以来，与walk-forward同窗口"
    stats = {k: ic_stats(s, start=FIRST_OOS) for k, s in ic_all.items()}
    print(f"\n  {'方法':<22s}{'覆盖天数':>8s}{'日均IC':>10s}{'ICIR(日)':>10s}{'ICIR(年化)':>11s}")
    for k, lab in [("KFold", "KFold(shuffle)"), ("TSC", "TimeSeriesSplit"),
                   ("Purged", "PurgedKFold+Embargo"), ("WF", "walk-forward(真值)")]:
        s = stats[k]
        print(f"  {lab:<22s}{s['n_days']:>8d}{s['ic']:>10.4f}{s['icir']:>10.3f}{s['icir_ann']:>11.3f}")

    ic_kf, ic_wf = stats["KFold"]["ic"], stats["WF"]["ic"]
    print(f"\n  >>> KFold IC {ic_kf:.4f} 是 walk-forward 真值 {ic_wf:.4f} 的 "
          f"{ic_kf / ic_wf:.2f} 倍（虚高 {(ic_kf - ic_wf) / abs(ic_wf) * 100:+.0f}%）—— 这就是标签重叠+时间交错的泄漏")
    print(f"  >>> Purged CV IC {stats['Purged']['ic']:.4f}，与真值仅差 "
          f"{(stats['Purged']['ic'] - ic_wf) / abs(ic_wf) * 100:+.0f}% —— 评估基本回到了地面")

    plot_gantt(pk, np.unique(row_dates))
    print("\n[图] 02_PurgedCV切分甘特图.png 已保存（全课核心图，含验证块两侧边界局部放大）")

    # ---------- 实验3：purged CV vs walk-forward 对照 ----------
    ic_2023 = {k: s[s.index >= pd.Timestamp(FIRST_OOS)] for k, s in ic_all.items()}
    plot_ic_comparison(stats, ic_2023, ic_wf, win)
    plot_purged_vs_wf(ic_2023, win)
    print("[图] 01_三种CV的IC对比.png / 04_purgedCV与walkforward对照.png 已保存")

    # ---------- 实验4：泄漏注入 ----------
    print("\n" + "-" * 72)
    print("实验4：注入'未来信息'——特征里加一列 fwd5 的噪声副本（约 1% 标签信息）")
    print("-" * 72)
    sigma = df["fwd5"].std()
    df["leak"] = df["fwd5"] + RNG.normal(0, np.sqrt(99) * sigma, len(df))
    corr_leak = df["leak"].corr(df["fwd5"])
    # 泄漏列的"截面强度"：日度截面内 corr(leak, fwd5) 的均值 —— 截面 IC 的世界里这才是有效强度
    _d = pd.DataFrame({"date": df["date"].to_numpy(), "a": df["leak"].to_numpy(), "b": y}).dropna()
    _g = _d.groupby("date")
    _d["ra"], _d["rb"] = _g["a"].rank(), _g["b"].rank()
    _agg = _d.groupby("date")[["ra", "rb"]].corr().unstack()["ra"]["rb"]
    cs_corr = _agg.mean()
    print(f"  构造：leak = fwd5 + N(0, (sqrt(99)*sigma)^2)，sigma={sigma:.4f}")
    print(f"  实测 corr(leak, fwd5) = {corr_leak:.4f}（全样本 R^2≈{corr_leak**2*100:.1f}%，"
          f"即该列只包含标签约 1% 的信息）")
    print(f"  但日度截面内 corr(leak, fwd5) 平均 = {cs_corr:.3f} —— 截面选股看的是横截面排序，"
          f"这点'1%泄漏'在截面 IC 的世界里已是可观的信号")

    X2 = df[FEATURES_LEAK].to_numpy()
    oof2, imp2_all = {}, {}
    for name, folds in [("KFold", folds_kf), ("TSC", folds_tsc), ("Purged", folds_pk)]:
        oof2[name], imp2_all[name], _ = fit_predict_folds(X2, y, folds)
    oof2["WF"], imp2_all["WF"] = walk_forward(X2, y, row_dates)
    ic2 = {k: daily_ic_series(df["date"].to_numpy(), oof2[k], y) for k in oof2}
    stats2 = {k: ic_stats(s, start=FIRST_OOS) for k, s in ic2.items()}

    leak_stats = {}
    print(f"\n  {'方法':<22s}{'无泄漏IC':>10s}{'注入后IC':>10s}{'绝对增量':>10s}{'相对跳变':>10s}")
    for k, lab in [("KFold", "KFold(shuffle)"), ("TSC", "TimeSeriesSplit"),
                   ("Purged", "PurgedKFold+Embargo"), ("WF", "walk-forward(真值)")]:
        b, a = stats[k]["ic"], stats2[k]["ic"]
        rel = (a - b) / abs(b) * 100
        leak_stats[k] = dict(ic_before=b, ic_after=a)
        print(f"  {lab:<22s}{b:>10.4f}{a:>10.4f}{a - b:>+10.4f}{rel:>+9.0f}%")

    gap_b = stats["Purged"]["ic"] / ic_wf - 1
    gap_a = stats2["Purged"]["ic"] / stats2["WF"]["ic"] - 1
    inc_p = stats2["Purged"]["ic"] - stats["Purged"]["ic"]
    inc_w = stats2["WF"]["ic"] - ic_wf
    inc_k = stats2["KFold"]["ic"] - ic_kf
    print("\n  >>> 解读（数字说话）：")
    print("      1. 泄漏列在预测时就能拿到，所以所有方法的 IC 都被直接抬升 ——")
    print("         任何 CV 都防不了'特征本身是未来函数'，那要靠特征审计（见坑5）。")
    print(f"      2. purged CV 两次评估都忠实：无泄漏时与真值差 {gap_b * 100:+.0f}%，"
          f"注入后差 {gap_a * 100:+.0f}%；它相对自己基线的跳变 (+{inc_p:.4f}) 与真值的跳变 "
          f"(+{inc_w:.4f}) 同量级 —— 泄漏被如实'检出'。")
    print(f"      3. KFold 被自己的虚高淹没：无泄漏时已虚报 {ic_kf / ic_wf:.2f} 倍，"
          f"注入泄漏后增量只剩 +{inc_k:.4f}（真值的增量是它的 {inc_w / inc_k:.1f} 倍）——")
    print("         虚高的基线把泄漏的边际效应吞掉了：你既看不见泄漏发生，也估不准它值多少。")

    imp_kf_leak = pd.Series(imp2_all["KFold"], index=FEATURES_LEAK)
    imp_pk_leak = pd.Series(imp2_all["Purged"], index=FEATURES_LEAK)
    print("\n  注入泄漏后的特征重要性（增益占比，节选自图03上图）：")
    print(f"    {'特征':<8s}{'KFold':>9s}{'Purged':>9s}")
    for f in FEATURES_LEAK:
        print(f"    {f:<8s}{imp_kf_leak[f] / imp_kf_leak.sum() * 100:>8.1f}%"
              f"{imp_pk_leak[f] / imp_pk_leak.sum() * 100:>8.1f}%")
    plot_leak_experiment(imp_kf_leak / imp_kf_leak.sum(), imp_pk_leak / imp_pk_leak.sum(),
                         leak_stats, "leak", win)
    print("[图] 03_泄漏注入实验.png 已保存")

    # ---------- cv_results.csv ----------
    rows = []
    conds = [("无泄漏", stats, ic_all, ""), ("注入1%泄漏", stats2, ic2, "特征含 leak=fwd5+噪声(R2≈1%)")]
    win_full = "全样本(各自覆盖)"
    win_2023 = f"{FIRST_OOS}起样本外窗口"
    for cond, st, icmap, note in conds:
        for k, lab in [("KFold", "KFold(shuffle)"), ("TSC", "TimeSeriesSplit"),
                       ("Purged", "PurgedKFold+Embargo"), ("WF", "walk-forward(真值)")]:
            for wname, start in [(win_full, None), (win_2023, FIRST_OOS)]:
                s = ic_stats(icmap[k], start=start)
                if s["n_days"] == 0:
                    continue
                extra = []
                if wname == win_full:
                    extra.append("各方法覆盖期不同，仅供纵向参考" if k != "WF" else "walk-forward 仅有2023起的样本外预测")
                if note:
                    extra.append(note)
                rows.append(dict(方法=lab, 数据条件=cond, 评估窗口=wname, 覆盖交易日数=s["n_days"],
                                 日均IC=round(s["ic"], 5), IC标准差=round(s["ic_std"], 5),
                                 ICIR日频=round(s["icir"], 4), ICIR年化=round(s["icir_ann"], 3),
                                 备注="；".join(extra)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "cv_results.csv"),
                              index=False, encoding="utf-8-sig")
    print("[表] cv_results.csv 已保存")

    # ---------- 实践清单 ----------
    print("\n" + "=" * 72)
    print("实践清单（本课要点，建议背下来）")
    print("=" * 72)
    print("""
1. 什么时候用 walk-forward？
   回答"这个模型历史上真实部署能赚多少"时 —— 它是唯一完全按时间顺序、
   可直接映射到实盘的评估。代价是样本效率低：早期数据只能当训练集，
   且每个时点的模型只用到过去的样本。
2. 什么时候用 Purged K-Fold + Embargo？
   回答"这个因子/模型在样本期内平均有没有稳定预测力"时（调参、选因子、
   做研究迭代）。样本效率高（每条数据都当过验证集），但必须 purge+embargo；
   它允许训练集包含验证期之后的数据，因此对"市场状态漂移"不免疫。
3. embargo 取多长？
   >= 特征最长回看窗口（本课 mom60/vol60 为 60 日），并与标签持有期比较取大者；
   标签窗口越长、特征窗口越长，purge/embargo 的空档就越宽。
4. 无论哪种 CV，都要先做特征审计：任何"预测时拿不到"的列（未来收益、
   全样本统计量、事后修正的财务数据）进了特征，再正确的 CV 也只会
   诚实地把虚高呈现出来 —— CV 防的是"切分泄漏"，不防"特征泄漏"。
5. López de Prado 三重屏障：CV 通过 -> 历史回测（含交易成本、滑点、容量）
   通过 -> 模拟盘/小资金实盘通过，才谈得上上实盘。CV 分数只是第一关。
""")

    print(f"总耗时 {time.time() - t0:.1f}s" + ("（合成数据）" if IS_SYNTHETIC else "（真实数据）"))
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
