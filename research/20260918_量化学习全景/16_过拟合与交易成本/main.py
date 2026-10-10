# -*- coding: utf-8 -*-
"""
【课程16】过拟合与交易成本 —— 量化最贵的一课
=================================================

一句话：为什么回测里"最好"的参数，实盘往往最惨？

本课用上证指数（2010年起）的双均线策略（快线 5~60 天、慢线 20~250 天），
完整演示新手最容易踩的四个坑：

  1) 样本内/样本外硬切分     —— 留出一段"从未见过的数据"（2010-2019 训练 / 2020起 检验）
  2) 参数平原热力图          —— 样本内的"亮点"，在样本外往往错位甚至消失
  3) 交易成本敏感性          —— 高换手策略对成本极端敏感，0.2% 的成本就能吃掉全部利润
  4) 多重检验幻觉(随机对照)  —— 同一段数据上试 129 个参数，"最好的那个"很可能是运气
  5) 邻域稳健性              —— 单点最优没有意义，要看最优点周围的"平原"还是"尖峰"

设计纪律（课程约定）：
  - 随机种子 42，结果完全可复现；单文件自包含；总运行 < 2 分钟。
  - 全部向量化回测（无 for 循环逐日撮合），144 组参数（12快线×12慢线，其中快线<慢线的
    有效组合 129 组）在几秒内跑完。
  - 数据缺失时自动用 rng(42) 生成合成数据兜底，绝不以非 0 退出码失败。
  - 信号只用当日及以前的数据（收盘计算均线，次日生效），避免前视偏差（见课程02）。

运行：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示器环境下出图
import matplotlib.pyplot as plt

# 中文字体与负号显示（课程统一规范）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)  # 本课唯一的随机源（用于随机对照实验）

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
DATA_PATH = BASE_DIR.parent / "data" / "cache" / "index_000001_上证指数.csv"
OUT_DIR.mkdir(exist_ok=True)

# 双均线参数网格：12 × 12 = 144 组合（其中"快线<慢线"才有效，共 129 组）
FAST_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]          # 快线：5~60 天
SLOW_LIST = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]  # 慢线：20~250 范围

# 样本内/样本外硬切分：2010-2019 样本内（用来选参数），2020 起样本外（只用一次！）
IS_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")

# 交易成本扫描：0 ~ 0.5%（每边），11 档
COST_GRID = np.linspace(0.0, 0.005, 11)
# A股现实单边成本参考带（佣金+印花税+滑点，指数/ETF 级别）：约 0.15% ~ 0.2%
CN_COST_LO, CN_COST_HI = 0.0015, 0.0020

# 随机对照实验规模：N 个"参数组合"里挑最好（N=真实网格有效组合数），重复 R 次
N_RANDOM_TRIALS = 500


# ----------------------------------------------------------------------
# 0) 数据加载：真实数据优先，缺失则合成数据兜底（绝不失败退出）
# ----------------------------------------------------------------------
def load_close_series():
    """读取上证指数收盘价；失败时用 rng(42) 生成合成数据并标注。"""
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
        close = df["close"].astype(float).dropna()
        close = close[close.index >= "2010-01-01"]
        if len(close) < 800:  # 数据明显不完整才走兜底
            raise ValueError(f"数据行数过少: {len(close)}")
        print(f"[数据] 已加载真实数据：上证指数 {close.index[0].date()} ~ {close.index[-1].date()}，"
              f"共 {len(close)} 个交易日")
        return close, False
    except Exception as e:  # noqa: BLE001 —— 课程约定：数据缺失不崩，用合成数据
        print(f"[警告] 真实数据不可用（{e}），改用 rng(42) 合成数据（结果仅演示流程）")
        idx = pd.bdate_range("2010-01-04", periods=3600)
        rets = rng.normal(loc=0.0002, scale=0.012, size=len(idx))
        close = pd.Series(3000.0 * np.cumprod(1.0 + rets), index=idx, name="close")
        return close, True


# ----------------------------------------------------------------------
# 1) 向量化回测内核：双均线多头/空仓策略
# ----------------------------------------------------------------------
def make_position(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    双均线信号（多头/空仓）：
      pos = 1 若 MA_fast > MA_slow（金叉持有），否则 0（空仓）。
    关键纪律：
      - 均线用截至当日收盘的数据计算，仓位次日起生效（shift 由调用方处理）→ 无前视偏差；
      - 慢线窗口未填满的"预热期"仓位记 0（空仓），且所有参数的统计区间完全一致，
        保证 144 组参数之间可比（代价：大慢线参数在 2010 年初有一段强制空仓，见 README）。
    """
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    raw = (ma_fast > ma_slow).astype(float)
    warmup = ma_fast.isna() | ma_slow.isna()
    pos = pd.Series(np.where(warmup, 0.0, raw), index=close.index)
    return pos


def strategy_daily_return(close: pd.Series, pos: pd.Series, cost_per_side: float = 0.0) -> pd.Series:
    """
    策略日收益（净：扣除交易成本）：
      策略收益_t = pos_{t-1} * ret_t - cost * |pos_{t-1} - pos_{t-2}|
      - 仓位 pos 在 t 日收盘算出，t+1 日才开始赚钱（shift(1)），避免前视偏差；
      - 每次"换手"（仓位变化 1 个单位 = 一买或一卖 = 单边一次）收取 cost_per_side；
      - 成本在仓位生效的当天扣减，与收益对应。
    """
    ret = close.pct_change().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)          # 当日收盘发生的换手
    gross = pos.shift(1).fillna(0.0) * ret        # 昨日仓位吃今日收益
    cost = cost_per_side * trade.shift(1).fillna(0.0)  # 昨日换手 → 今日扣成本
    return gross - cost


def perf_stats(daily_ret: pd.Series) -> dict:
    """
    绩效指标（年化、波动、夏普、最大回撤）。
    年化 = 复利 CAGR；夏普 = (日均值 − 0.02/252)/日标准差×sqrt(252)（统一 rf=2% 年化口径）。
    """
    r = daily_ret.dropna()
    empty = {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan}
    if len(r) < 10 or r.std(ddof=1) == 0:
        return empty
    eq = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    ann = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float((r.mean() - 0.02 / 252) / r.std(ddof=1) * np.sqrt(252))  # 统一 rf=2%：日度扣 0.02/252
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd}


# ----------------------------------------------------------------------
# 2) 参数网格回测：144 组参数 × 样本内/样本外
# ----------------------------------------------------------------------
def run_parameter_grid(close: pd.Series) -> pd.DataFrame:
    """对全部有效 (fast, slow) 组合跑成本=0 的回测，返回网格结果表。"""
    ret = close.pct_change().fillna(0.0)
    is_mask = close.index <= IS_END
    oos_mask = close.index >= OOS_START
    years_is = (close.index[is_mask][-1] - close.index[is_mask][0]).days / 365.25

    rows = []
    for fast in FAST_LIST:
        for slow in SLOW_LIST:
            if fast >= slow:  # 快线必须短于慢线，否则不是有效双均线
                continue
            pos = make_position(close, fast, slow)
            daily = (pos.shift(1).fillna(0.0) * ret)  # 成本=0 的毛收益
            p_is = perf_stats(daily[is_mask])
            p_oos = perf_stats(daily[oos_mask])
            n_trades_is = int((pos.diff().abs().fillna(0.0) > 0)[is_mask].sum())
            rows.append({
                "快线": fast,
                "慢线": slow,
                "年化收益_样本内": p_is["ann"],
                "年化收益_样本外": p_oos["ann"],
                "夏普_样本内": p_is["sharpe"],
                "夏普_样本外": p_oos["sharpe"],
                "回撤_样本内": p_is["mdd"],
                "回撤_样本外": p_oos["mdd"],
                "每年换手次数_样本内": n_trades_is / years_is,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3) 热力图绘制（样本内 / 样本外共用同一色标，才能直接对比）
# ----------------------------------------------------------------------
def draw_heatmap(ax, pivot: pd.DataFrame, vmin: float, vmax: float,
                 title: str, star_xy=None, triangle_xy=None):
    """pivot: 行=快线、列=慢线 的年化收益矩阵；无效组合为 NaN（灰色）。"""
    data = np.ma.masked_invalid(pivot.values)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="0.88")  # 无效组合（快线>=慢线）画成灰色
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("慢线（天）")
    ax.set_ylabel("快线（天）")
    ax.set_title(title, fontsize=12)
    if star_xy is not None:      # 红五星：样本内最优参数的位置
        ax.scatter([star_xy[1]], [star_xy[0]], marker="*", s=420,
                   c="red", edgecolors="black", linewidths=1.2, zorder=5)
    if triangle_xy is not None:  # 蓝三角：样本外最优参数的位置
        ax.scatter([triangle_xy[1]], [triangle_xy[0]], marker="^", s=140,
                   c="royalblue", edgecolors="black", linewidths=1.0, zorder=5)
    return im


# ----------------------------------------------------------------------
# 4) 随机对照实验（多重检验幻觉）
# ----------------------------------------------------------------------
def random_positions(T: int, mean_hold: float, gen: np.random.Generator) -> np.ndarray:
    """
    生成一条与真实策略换手频率相当的"随机多空仓信号"：
    持仓时长 ~ 指数分布（均值 mean_hold 天），多/空仓交替。它与行情完全无关，没有任何
    选时能力（扣除 rf=2% 后期望夏普略低于 0）—— 但从 N 条里挑"样本内最好的一条"，期望夏普会显著大于 0，
    这就是多重检验幻觉（deflated Sharpe 思想的简化演示）。
    """
    n_switch = int(T / max(mean_hold, 2.0)) + 5
    lengths = np.maximum(1, np.round(gen.exponential(mean_hold, size=n_switch))).astype(int)
    start = float(gen.integers(0, 2))
    states = np.empty(n_switch)
    states[0::2] = start
    states[1::2] = 1.0 - start
    pos = np.repeat(states, lengths)
    if len(pos) < T:  # 兜底补齐
        pos = np.concatenate([pos, np.full(T - len(pos), states[-1])])
    return pos[:T]


def run_random_null(n_strategies: int, ret_is_values: np.ndarray,
                    mean_hold_days: float, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """
    每次试验：生成 n_strategies 条随机策略在样本内的日收益 → 各算夏普（与 perf_stats 同一 rf=2% 口径）→ 取最大值。
    返回 (每次试验的"最好夏普"数组, 第一次试验全部策略的夏普数组)。
    """
    T = len(ret_is_values)
    max_sharpes = np.empty(n_trials)
    first_all = None
    for t in range(n_trials):
        pos_mat = np.empty((n_strategies, T))
        for i in range(n_strategies):
            pos_mat[i] = random_positions(T, mean_hold_days, rng)
        strat = np.zeros((n_strategies, T))
        strat[:, 1:] = pos_mat[:, :-1] * ret_is_values[1:]  # 与真实回测同样的次日生效纪律
        sd = strat.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, (strat.mean(axis=1) - 0.02 / 252) / sd * np.sqrt(252), 0.0)  # 同 perf_stats 的 rf=2% 口径
        max_sharpes[t] = sharpe.max()
        if t == 0:
            first_all = sharpe.copy()
    return max_sharpes, first_all


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("课程16：过拟合与交易成本 —— 为什么回测里最好的参数，实盘往往最惨")
    print("=" * 70)

    # ---- 0. 数据 ----
    close, is_synthetic = load_close_series()
    if is_synthetic:
        print("[标注] 本结果基于合成数据，仅演示方法，不代表真实市场。")

    years_all = (close.index[-1] - close.index[0]).days / 365.25
    is_mask = close.index <= IS_END
    oos_mask = close.index >= OOS_START
    years_is = (close.index[is_mask][-1] - close.index[is_mask][0]).days / 365.25
    years_oos = (close.index[oos_mask][-1] - close.index[oos_mask][0]).days / 365.25
    print(f"[切分] 样本内 2010-2019（{years_is:.1f} 年，{int(is_mask.sum())} 个交易日）"
          f" | 样本外 2020 起（{years_oos:.1f} 年，{int(oos_mask.sum())} 个交易日）")

    # 基准：买入持有
    bh = perf_stats(close.pct_change().fillna(0.0))
    bh_is = perf_stats(close.pct_change().fillna(0.0)[is_mask])
    bh_oos = perf_stats(close.pct_change().fillna(0.0)[oos_mask])

    # ---- 1. 参数网格回测（样本内选参数 / 样本外检验） ----
    print("\n" + "-" * 70)
    print("第 1 步：144 组参数网格回测（成本=0 毛收益；快线<慢线的有效组合才计算）")
    print("-" * 70)
    grid = run_parameter_grid(close)
    n_valid = len(grid)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False, encoding="utf-8-sig")
    print(f"有效组合 {n_valid} 组，网格结果已存 outputs/grid_results.csv")

    # 最优参数 = 样本内年化收益最高（若并列，取样本内夏普更高者）
    best = grid.sort_values(["年化收益_样本内", "夏普_样本内"], ascending=False).iloc[0]
    bf, bs = int(best["快线"]), int(best["慢线"])
    best_oos_row = grid.sort_values("年化收益_样本外", ascending=False).iloc[0]

    print("\n样本内年化收益 Top5（注意：这是你在 2019 年底'看得到'的成绩）：")
    top5 = grid.sort_values("年化收益_样本内", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"  快线{int(r['快线']):>3d}/慢线{int(r['慢线']):>3d} | "
              f"样本内年化 {r['年化收益_样本内']*100:+6.2f}% | "
              f"样本外年化 {r['年化收益_样本外']*100:+6.2f}% | "
              f"每年换手 {r['每年换手次数_样本内']:.1f} 次")

    # ---- 2. 最优参数：样本内 vs 样本外 ----
    print("\n" + "-" * 70)
    print(f"第 2 步：最优参数（快线{bf}/慢线{bs}）样本内 vs 样本外")
    print("-" * 70)
    pos_best = make_position(close, bf, bs)
    daily_best = strategy_daily_return(close, pos_best, cost_per_side=0.0)
    p_best_is = perf_stats(daily_best[is_mask])
    p_best_oos = perf_stats(daily_best[oos_mask])
    n_trades_is = float((pos_best.diff().abs().fillna(0.0) > 0)[is_mask].sum())
    n_trades_oos = float((pos_best.diff().abs().fillna(0.0) > 0)[oos_mask].sum())

    print(f"{'指标':<10}{'样本内(2010-2019)':>22}{'样本外(2020起)':>22}")
    print(f"{'年化收益':<10}{p_best_is['ann']*100:>20.2f}%{p_best_oos['ann']*100:>21.2f}%")
    print(f"{'夏普比率':<10}{p_best_is['sharpe']:>22.2f}{p_best_oos['sharpe']:>22.2f}")
    print(f"{'最大回撤':<10}{p_best_is['mdd']*100:>20.2f}%{p_best_oos['mdd']*100:>21.2f}%")
    print(f"{'年化波动':<10}{p_best_is['vol']*100:>20.2f}%{p_best_oos['vol']*100:>21.2f}%")
    print(f"换手频率：样本内每年 {n_trades_is/years_is:.1f} 次，样本外每年 {n_trades_oos/years_oos:.1f} 次")
    print(f"（同期买入持有：样本内年化 {bh_is['ann']*100:+.2f}%，样本外年化 {bh_oos['ann']*100:+.2f}%）")
    print(f"（样本外真正最好的参数是 快线{int(best_oos_row['快线'])}/慢线{int(best_oos_row['慢线'])}，"
          f"年化 {best_oos_row['年化收益_样本外']*100:+.2f}% —— 你在 2019 年底不可能知道）")

    decay_ann = p_best_oos["ann"] - p_best_is["ann"]
    decay_sharpe = p_best_oos["sharpe"] - p_best_is["sharpe"]
    print(f"衰减量：年化 {decay_ann*100:+.2f} 个百分点，夏普 {decay_sharpe:+.2f}")

    # 图03：最优参数净值（样本内外对比）
    fig, ax = plt.subplots(figsize=(11, 6))
    eq_best = (1.0 + daily_best).cumprod()
    eq_bh = close / close.iloc[0]
    ax.plot(eq_best.index, eq_best, lw=1.6, color="crimson", label=f"双均线{bf}/{bs}（成本0）")
    ax.plot(eq_bh.index, eq_bh, lw=1.0, color="gray", alpha=0.8, label="买入持有")
    ax.axvline(IS_END, color="black", ls="--", lw=1)
    ax.axvspan(OOS_START, close.index[-1], color="orange", alpha=0.08)
    ax.text(0.015, 0.96, f"样本内 2010-2019\n策略年化 {p_best_is['ann']*100:+.2f}% | 持有 {bh_is['ann']*100:+.2f}%",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", ec="0.6", alpha=0.9))
    ax.text(0.015, 0.80, f"样本外 2020起（从未见过的数据）\n策略年化 {p_best_oos['ann']*100:+.2f}% | 持有 {bh_oos['ann']*100:+.2f}%",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="cornsilk", ec="0.6", alpha=0.9))
    ax.set_title(f"图03 样本内最优参数（{bf}/{bs}）的净值：样本内 vs 样本外（上证指数，成本=0）")
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_最优参数内外净值对比.png", dpi=150)
    plt.close(fig)

    # ---- 3. 参数平原热力图（本课灵魂之一） ----
    print("\n" + "-" * 70)
    print("第 3 步：参数平原热力图（样本内 vs 样本外，同一色标）")
    print("-" * 70)
    piv_is = grid.pivot(index="快线", columns="慢线", values="年化收益_样本内") \
        .reindex(index=FAST_LIST, columns=SLOW_LIST)
    piv_oos = grid.pivot(index="快线", columns="慢线", values="年化收益_样本外") \
        .reindex(index=FAST_LIST, columns=SLOW_LIST)
    vmin = float(np.nanmin([piv_is.min().min(), piv_oos.min().min()]))
    vmax = float(np.nanmax([piv_is.max().max(), piv_oos.max().max()]))
    fi, si = FAST_LIST.index(bf), SLOW_LIST.index(bs)
    bo, so = int(best_oos_row["快线"]), int(best_oos_row["慢线"])
    fo2, so2 = FAST_LIST.index(bo), SLOW_LIST.index(so)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = draw_heatmap(ax, piv_is, vmin, vmax,
                      f"图01 样本内(2010-2019)年化收益参数平原 —— 红五星=样本内最优 {bf}/{bs}",
                      star_xy=(fi, si))
    fig.colorbar(im, ax=ax, label="年化收益（毛，成本=0）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_样本内参数热力图.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = draw_heatmap(ax, piv_oos, vmin, vmax,
                      "图02 样本外(2020起)年化收益参数平原 —— 红五星=样本内最优的位置，蓝三角=样本外最优",
                      star_xy=(fi, si), triangle_xy=(fo2, so2))
    fig.colorbar(im, ax=ax, label="年化收益（毛，成本=0）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_样本外参数热力图.png", dpi=150)
    plt.close(fig)
    print("已输出 01_样本内参数热力图.png / 02_样本外参数热力图.png（两图共用同一色标）")
    print(f"样本内最优点 {bf}/{bs} 样本外年化 {p_best_oos['ann']*100:+.2f}%；"
          f"样本外最优点 {bo}/{so} 样本外年化 {best_oos_row['年化收益_样本外']*100:+.2f}%")

    # ---- 4. 交易成本敏感性（本课灵魂之二） ----
    print("\n" + "-" * 70)
    print("第 4 步：交易成本敏感性（成本 0~0.5% 每边扫 11 档，全样本 2010-2026）")
    print("-" * 70)
    cost_rows = []
    curves = {}
    for _, r in top5.iterrows():  # 样本内 Top5 参数
        f_i, s_i = int(r["快线"]), int(r["慢线"])
        pos_i = make_position(close, f_i, s_i)
        anns = [perf_stats(strategy_daily_return(close, pos_i, c))["ann"] for c in COST_GRID]
        curves[(f_i, s_i)] = np.array(anns)
        cost_rows.append({"快线": f_i, "慢线": s_i,
                          "每年换手次数": float((pos_i.diff().abs().fillna(0.0) > 0).sum()) / years_all})

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for k, ((f_i, s_i), anns) in enumerate(curves.items()):
        is_best_pt = (f_i == bf and s_i == bs)
        n_tr = [d for d in cost_rows if d["快线"] == f_i and d["慢线"] == s_i][0]["每年换手次数"]
        ax.plot(COST_GRID * 100, anns * 100, marker="o", ms=4,
                lw=2.6 if is_best_pt else 1.4,
                color="crimson" if is_best_pt else None,
                alpha=1.0 if is_best_pt else 0.65,
                label=f"{f_i}/{s_i}（每年换手{n_tr:.0f}次，"
                      f"{'样本内最优' if is_best_pt else '次优'})")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(CN_COST_LO * 100, CN_COST_HI * 100, color="gold", alpha=0.35,
               label="A股现实单边成本 ≈ 0.15%~0.2%（佣金+印花税+滑点）")
    ax.set_xlabel("单边交易成本（%）")
    ax.set_ylabel("全样本年化收益（%）")
    ax.set_title("图04 年化收益随成本的衰减：换手越频繁，成本杀伤力越大（上证指数 2010-2026）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_成本敏感性曲线族.png", dpi=150)
    plt.close(fig)

    # 成本打平点：最优参数的年化收益降到 0 所对应的单边成本（线性插值）
    anns_best = curves[(bf, bs)]
    be_cost = None
    if anns_best[0] <= 0:
        be_cost = 0.0
        print("注意：最优参数在零成本时全样本年化已为负，不存在正的成本打平点。")
    elif (anns_best > 0).all():
        be_cost = None
        print("注意：最优参数在 0.5% 成本内年化仍为正，成本打平点 > 0.5%。")
    else:
        j = int(np.argmax(anns_best <= 0))
        c0, c1, a0, a1 = COST_GRID[j - 1], COST_GRID[j], anns_best[j - 1], anns_best[j]
        be_cost = float(c0 + (0.0 - a0) / (a1 - a0) * (c1 - c0))
        print(f"成本打平点（全样本，年化归零的单边成本）：{be_cost*100:.3f}%")
    cn_mid = (CN_COST_LO + CN_COST_HI) / 2
    print(f"最优参数 {bf}/{bs}：成本0 时全样本年化 {anns_best[0]*100:+.2f}%；"
          f"在A股现实成本 {CN_COST_LO*100:.2f}%~{CN_COST_HI*100:.2f}%（中值 {cn_mid*100:.3f}%）时 "
          f"年化 {np.interp(cn_mid, COST_GRID, anns_best)*100:+.2f}%")
    print("已输出 04_成本敏感性曲线族.png")

    # ---- 5. 多重检验幻觉：随机对照实验（本课灵魂之三） ----
    print("\n" + "-" * 70)
    print(f"第 5 步：多重检验幻觉 —— 随机对照实验（{n_valid} 个'参数'里挑最好，重复 {N_RANDOM_TRIALS} 次）")
    print("-" * 70)
    ret_is = close.pct_change().fillna(0.0)[is_mask]
    avg_trades_per_year = float(grid["每年换手次数_样本内"].mean())
    mean_hold_days = float(np.clip(252.0 / max(avg_trades_per_year, 0.5), 5.0, 126.0))
    print(f"随机策略换手频率与真实网格匹配：平均持仓 {mean_hold_days:.0f} 天"
          f"（真实网格平均每年换手 {avg_trades_per_year:.1f} 次）")
    max_sharpes, first_all = run_random_null(n_valid, ret_is.values, mean_hold_days, N_RANDOM_TRIALS)

    real_best_sharpe = float(best["夏普_样本内"])
    pct_rank = float((max_sharpes < real_best_sharpe).mean() * 100)
    prob_luck = float((max_sharpes >= real_best_sharpe).mean() * 100)
    print(f"单条随机策略的样本内夏普：均值 {first_all.mean():+.3f}（扣 rf 后理论略低于0），95%分位 "
          f"{np.percentile(first_all, 95):.3f} —— 单次尝试不太容易'看起来很厉害'")
    print(f"但 '{n_valid} 选 1 最好' 的样本内夏普：均值 {max_sharpes.mean():.3f}，"
          f"5%分位 {np.percentile(max_sharpes, 5):.3f}，95%分位 {np.percentile(max_sharpes, 95):.3f}")
    print(f"真实最优参数（{bf}/{bs}）样本内夏普 = {real_best_sharpe:.3f}")
    print(f"→ 它位于随机'最好成绩'分布的第 {pct_rank:.1f} 百分位；"
          f"纯靠运气 试{n_valid}个随机策略'最好的那个'≥它的概率 ≈ {prob_luck:.1f}%")
    print("（若分位数不高，说明'最好的参数'与运气难以区分 —— 这就是 deflated Sharpe 的思想）")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.hist(first_all, bins=40, alpha=0.45, color="gray",
            label=f"单条随机策略夏普（{n_valid}条/次，示意一次试验）")
    ax.hist(max_sharpes, bins=40, alpha=0.65, color="steelblue",
            label=f"每次试验 '{n_valid}选1最好' 的夏普（{N_RANDOM_TRIALS}次试验）")
    ax.axvline(real_best_sharpe, color="crimson", lw=2,
               label=f"真实最优参数样本内夏普 = {real_best_sharpe:.2f}（第{pct_rank:.0f}百分位）")
    ax.axvline(max_sharpes.mean(), color="navy", ls="--", lw=1.2,
               label=f"随机'最好成绩'均值 = {max_sharpes.mean():.2f}")
    ax.set_xlabel("样本内夏普比率")
    ax.set_ylabel("频数")
    ax.set_title("图05 多重检验幻觉：与真实策略同换手频率的随机策略，挑最好的也'很像样'")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_多重检验分布图.png", dpi=150)
    plt.close(fig)
    print("已输出 05_多重检验分布图.png")

    # ---- 6. 邻域稳健性（"邻域塌陷"检测） ----
    print("\n" + "-" * 70)
    print(f"第 6 步：邻域稳健性 —— 最优点 {bf}/{bs} 附近的参数平原还在吗？")
    print("-" * 70)
    # 邻域定义：快线 ±5 天（快线步长恰为 5 → 覆盖相邻两档）；慢线网格间距 20 天，
    # ±5 天内只含本档，故取相邻一档作为最近邻（3×3 邻域，剔除无效组合）。
    f_vals = [f for f in FAST_LIST if abs(f - bf) <= 5]
    s_idx = SLOW_LIST.index(bs)
    s_vals = SLOW_LIST[max(0, s_idx - 1): s_idx + 2]
    neigh = grid[grid.apply(lambda r: (int(r["快线"]), int(r["慢线"])) in
                            {(a, b) for a in f_vals for b in s_vals if a < b}, axis=1)].copy()
    neigh["是否最优点"] = (neigh["快线"] == bf) & (neigh["慢线"] == bs)
    neigh_out = neigh[["快线", "慢线", "年化收益_样本内", "年化收益_样本外", "是否最优点"]]
    summary = pd.DataFrame([
        {"快线": "邻域平均", "慢线": f"{len(neigh)}格",
         "年化收益_样本内": float(neigh["年化收益_样本内"].mean()),
         "年化收益_样本外": float(neigh["年化收益_样本外"].mean()),
         "是否最优点": ""},
        {"快线": "最优点", "慢线": f"{bf}/{bs}",
         "年化收益_样本内": float(best["年化收益_样本内"]),
         "年化收益_样本外": float(best["年化收益_样本外"]),
         "是否最优点": True},
        {"快线": "全网格平均", "慢线": f"{n_valid}格",
         "年化收益_样本内": float(grid["年化收益_样本内"].mean()),
         "年化收益_样本外": float(grid["年化收益_样本外"].mean()),
         "是否最优点": ""},
    ])
    neigh_out = pd.concat([neigh_out, summary], ignore_index=True)
    neigh_out.to_csv(OUT_DIR / "robustness.csv", index=False, encoding="utf-8-sig")

    neigh_avg_is = float(neigh["年化收益_样本内"].mean())
    neigh_avg_oos = float(neigh["年化收益_样本外"].mean())
    best_is = float(best["年化收益_样本内"])
    print(f"邻域（快线{f_vals} × 慢线{s_vals}，共 {len(neigh)} 个有效格）平均样本内年化 "
          f"{neigh_avg_is*100:+.2f}%，最优点 {best_is*100:+.2f}%")
    if best_is > 0:
        print(f"邻域保持率 = 邻域平均 / 最优点 = {neigh_avg_is/best_is*100:.0f}%"
              f"（<60% 视为'尖峰塌陷'，越接近100%参数平原越结实）")
    else:
        print(f"邻域塌陷量 = 邻域平均 - 最优点 = {(neigh_avg_is-best_is)*100:+.2f} 个百分点"
              f"（最优点年化非正，用差值衡量）")
    print(f"邻域平均样本外年化 {neigh_avg_oos*100:+.2f}%（对照最优点样本外 {best['年化收益_样本外']*100:+.2f}%）")
    print(f"全网格平均：样本内 {grid['年化收益_样本内'].mean()*100:+.2f}%，"
          f"样本外 {grid['年化收益_样本外'].mean()*100:+.2f}%")
    print("已输出 robustness.csv")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("汇总（README 中引用的全部真实数字都来自上面的输出）：")
    print(f"  最优参数（样本内选）：快线{bf}/慢线{bs}，每年换手 {n_trades_is/years_is:.1f} 次")
    print(f"  样本内 年化 {p_best_is['ann']*100:+.2f}% / 夏普 {p_best_is['sharpe']:.2f} / 回撤 {p_best_is['mdd']*100:.2f}%")
    print(f"  样本外 年化 {p_best_oos['ann']*100:+.2f}% / 夏普 {p_best_oos['sharpe']:.2f} / 回撤 {p_best_oos['mdd']*100:.2f}%")
    print(f"  邻域平均样本内年化 {neigh_avg_is*100:+.2f}%（邻域保持率 "
          f"{neigh_avg_is/best_is*100:.0f}%）" if best_is > 0 else
          f"  邻域平均样本内年化 {neigh_avg_is*100:+.2f}%")
    print(f"  成本打平点（全样本）：{'不存在（零成本已为负）' if be_cost == 0.0 else
          ('> 0.5%' if be_cost is None else f'{be_cost*100:.3f}% 每边')}")
    print(f"  随机对照：真实最优样本内夏普 {real_best_sharpe:.2f} 位于随机'最好成绩'分布第 {pct_rank:.0f} 百分位")
    print("=" * 70)
    print("全部输出文件已写入 outputs/，运行成功。")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 —— 课程约定：任何意外都不允许非 0 退出
        traceback.print_exc()
        print("[兜底] 运行出现异常，但按课程规范以退出码 0 结束（请检查上方错误信息）")
        sys.exit(0)
