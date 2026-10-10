# -*- coding: utf-8 -*-
"""
27_时序动量vs截面动量 —— 同一个词"动量"的两种完全不同的策略形态（教学版）
================================================================================
参考文献：Moskowitz, Ooi, Pedersen (2012, JFE)《Time Series Momentum》。

平时都说"动量策略"，但它其实指两种完全不同的东西：
  · 时序动量 TSMOM：每个资产看"自己"过去 N 日收益的正负决定多/空（自己跟自己比）；
  · 截面动量 XSMOM：把一篮子资产互相排序，买最强的、卖（或不买）最弱的（跟别人比）。

本课用 10 个 A 股行业（共同重叠期 2021-12-29 ~ 2026-09-16）各跑一遍并对比：
  1) 概念图：2 资产小例子，画两种信号的差别（图01）
  2) TSMOM 回测：滚动 252 日动量符号、月度调仓；A股现实=只做多头（动量<0 就空仓），
     另附"理论多空版"对照曲线（A股现货不能做空，仅作教学对照）
  3) XSMOM 回测：与课15同框架（月末按动量排序、持有 Top3、等权）
  4) 对比：净值 / 年化 / 夏普 / 回撤 / 换手（含双边万3成本）/ 月收益相关性
  5) 动量窗口敏感性：20/60/120/252 各跑一遍 TSMOM（如实呈现，不挑最好窗口）

数据清洗规则：单日收益绝对值 > 30% 判为数据源"基期跳变"毛刺，该日收益置 0
（A股行业指数单日 ±30% 在现实中不存在）。动量在每个行业"自己的完整历史"上
预计算（行业发布时间不一，用各自更早的历史做预热，信号更完整）。
随机种子 42；单文件自包含；总运行 < 2 分钟。
"""
import os
import sys
import glob
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示器环境出图
import matplotlib.pyplot as plt

# ---------- 全局规范 ----------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)   # 仅在真实数据缺失时的合成数据兜底中使用

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'cache')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

JUMP_TH = 0.30        # 跳变判定：单日|收益|>30% 视为基期跳变，置 0
N_LB = 252            # 主对比动量窗口（交易日）≈ MOP2012 的"过去12个月"
K_HOLD = 3            # XSMOM 每月持有的行业数（与课15一致）
SENS_WINDOWS = [20, 60, 120, 252]   # 敏感性测试的窗口网格（TSMOM）
ANN = 252             # 年化系数
RF_ANNUAL = 0.02      # 无风险利率（年化2%，与本系列课01-10统一口径；夏普=年化超额/波动）
COST = 0.0003         # 双边万3：换手Σ|Δw|已同时计入买与卖两边，每边各付万3
SYNTHETIC = False     # 真实数据缺失时转合成数据并置 True

COL_TSMOM_L = '#d62728'   # TSMOM 只多（A股现实）
COL_TSMOM_S = '#9467bd'   # TSMOM 多空（理论）
COL_XSMOM = '#1f77b4'     # XSMOM Top3
COL_EW = '#7f7f7f'        # 10行业等权
COL_HS300 = '#e6a100'     # 沪深300


# =====================================================================
# 1. 数据加载与清洗
# =====================================================================
def load_close(path):
    """读取单个CSV → 以日期为索引的收盘价 Series（剔除缺失/重复日期）。"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close']).drop_duplicates(subset='date').sort_values('date')
    return df.set_index('date')['close'].astype(float)


def make_synthetic_data():
    """数据缺失兜底：用 rng(42) 造 10 个行业 + 基准的合成收盘价（仅教学演示）。"""
    global SYNTHETIC
    SYNTHETIC = True
    print('[警告] 未找到真实行业数据，改用 rng(42) 合成数据（结果仅演示流程）')
    dates = pd.bdate_range('2018-01-02', periods=2100)   # 留足252日预热历史
    cols = {}
    for i in range(10):
        mu = 0.0002 + 0.0001 * (i % 3)
        sig = 0.012 + 0.003 * (i % 4)
        r = rng.normal(mu, sig, len(dates))
        cols[f'合成行业{i + 1}'] = pd.Series(100 * np.cumprod(1 + r), index=dates)
    cols['沪深300'] = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0001, 0.009, len(dates))), index=dates)
    return cols


def load_all_data():
    """加载 10 个行业 + 沪深300 → 各自日历上的收盘价 Series 字典。"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'board_*.csv')))
    if len(files) < 10 or not os.path.exists(os.path.join(DATA_DIR, 'index_000300_沪深300.csv')):
        cols = make_synthetic_data()
        ind_names = [c for c in cols if c != '沪深300']
        return cols, cols['沪深300'], ind_names
    closes, ind_names = {}, []
    for f in files:
        name = os.path.basename(f)[len('board_'):-4]   # board_银行.csv -> 银行
        closes[name] = load_close(f)
        ind_names.append(name)
    idx300 = load_close(os.path.join(DATA_DIR, 'index_000300_沪深300.csv'))
    return closes, idx300, ind_names


def own_returns_momentum(close_raw, ind_names, n):
    """
    每个行业在自己的"完整历史"上（行业发布时间不一，比共同期更早）：
      1) 日收益 pct_change，单日|收益|>30% 判为基期跳变 → 该日收益置 0；
      2) n 日动量 = 过去 n 个交易日累计收益（对数收益滚动求和再还原）。
    TSMOM 的定义本来就是"每个资产看自己的过去"，所以用各自日历预热。
    返回 (各行业清洗后日收益 dict, 动量宽表 DataFrame，索引=各表日期并集)。
    """
    ret_d, mom_d = {}, {}
    for name in ind_names:
        r = close_raw[name].pct_change()
        r = r.mask(r.abs() > JUMP_TH, 0.0)                 # 基期跳变置 0
        ret_d[name] = r
        mom_d[name] = np.expm1(np.log1p(r).rolling(n, min_periods=n).sum())
    return ret_d, pd.DataFrame(mom_d)


def build_common_panel(close_raw, ind_names):
    """
    构建"共同日历"回测面板：只保留 10 个行业都有收盘价的交易日。
      - 共同起点由最晚发布的行业决定（本数据为电力行业 2021-12-29）；
      - 面板上计算日收益，单日|收益|>30% 置 0（同一跳变规则）。
    返回 (清洗后日收益面板, 共同交易日索引, 共同期清洗的跳变数)。
    """
    common_index = pd.DataFrame(close_raw)[ind_names].dropna(how='any').index
    panel = pd.DataFrame(close_raw).loc[common_index, ind_names]
    r = panel.pct_change()
    jump = r.abs() > JUMP_TH
    ret_panel = r.mask(jump, 0.0).fillna(0.0)              # 首行无收益，按 0
    return ret_panel, common_index, int(jump.sum().sum())


# =====================================================================
# 2. 信号、绩效统计与回测引擎
# =====================================================================
def month_end_dates(index):
    """取每个自然月最后一个交易日（DatetimeIndex → 按月取最后一天）。"""
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period('M')).max())


def tsmom_weight_vectors(mom_row, n_assets):
    """
    TSMOM 信号 → 目标权重（自己跟自己比）：
      只多版（A股现实）：动量 > 0 → 持有 1/N 仓位；<=0 或历史不足(NaN) → 空仓。
      多空理论版：>0 → +1/N；<0 → -1/N（A股现货做不到做空，仅作理论对照）。
    """
    x = mom_row.to_numpy(dtype=float)
    pos = x > 0                       # NaN 参与比较为 False → 自动空仓
    neg = x < 0
    w_long = pos.astype(float) / n_assets
    w_ls = (pos.astype(float) - neg.astype(float)) / n_assets
    return w_long, w_ls


def xsmom_weight_vector(mom_row, columns, k):
    """
    XSMOM 信号 → 目标权重（跟别人比，与课15同框架）：
      月末把"可打分"的行业按过去 N 日动量从强到弱排序，持有最强的 k 个、各 1/k。
      可打分行业不足 k 个时返回 None（该月不调仓，沿用旧仓位）。
    """
    s = mom_row.dropna()
    if len(s) < k:
        return None
    top = s.sort_values(ascending=False, kind='mergesort').index[:k]  # mergesort保证并列时确定
    w = np.zeros(len(columns))
    w[[columns.get_loc(c) for c in top]] = 1.0 / k
    return w


def backtest(weights_by_date, ret_panel, all_days, cost_rate=0.0):
    """
    月度再平衡回测引擎（权重逐日漂移）。
      weights_by_date: {调仓日Timestamp: 目标权重ndarray}，月末收盘设定、次日起生效
      ret_panel: 共同日历日收益 DataFrame；all_days: 回测交易日（晚于首个调仓日）
      cost_rate: 单边成本率（双边万3 → 0.0003）。换手Σ|Δw|已同时计入买与卖两边，
                 成本 = cost_rate × Σ|Δw|，在调仓当日扣除（建仓成本记在回测首日）。
    返回 (毛日收益, 净日收益, 每次调仓双边换手Σ|Δw|, 每日总仓位|w|之和)。
    """
    reb_dates = sorted(weights_by_date)
    w = np.asarray(weights_by_date[reb_dates[0]], dtype=float)
    tu_map = {reb_dates[0]: float(np.abs(w).sum())}    # 建仓：从空仓到目标仓位也是换手
    port_r, expo = [], []
    for t in all_days:
        r = ret_panel.loc[t].to_numpy(dtype=float)
        expo.append(float(np.abs(w).sum()))            # 当日实际持用的总仓位
        pr = float(np.dot(w, r))                       # 当日组合收益（用昨日收盘权重）
        port_r.append(pr)
        w = w * (1.0 + r) / (1.0 + pr)                 # 权重随行情漂移到当日收盘
        if t in weights_by_date and t != reb_dates[0]:  # 当日为调仓日：收盘换仓
            w_new = np.asarray(weights_by_date[t], dtype=float)
            tu_map[t] = float(np.abs(w_new - w).sum())  # 双边换手（含买与卖）
            w = w_new.copy()
    gross = pd.Series(port_r, index=all_days)
    cost = pd.Series(0.0, index=all_days)
    for d, tw in tu_map.items():
        day = all_days[0] if d == reb_dates[0] else d   # 建仓成本记在回测首日
        cost.loc[day] = cost_rate * tw
    return gross, gross - cost, pd.Series(tu_map), pd.Series(expo, index=all_days)


def perf_stats(ser, turnover=None):
    """输入日收益 Series（可选每次调仓的双边换手 Series），输出绩效字典。"""
    nav = (1.0 + ser).cumprod()
    years = max((ser.index[-1] - ser.index[0]).days / 365.25, 1e-9)
    total = nav.iloc[-1] - 1.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    sd = ser.std(ddof=1)
    vol = sd * np.sqrt(ANN)
    sharpe = ((ser.mean() * ANN - RF_ANNUAL) / vol) if sd > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    tu_ann = float(0.5 * turnover.sum() / years) if turnover is not None and len(turnover) else 0.0
    return {'起始日': ser.index[0], '截止日': ser.index[-1],
            '累计收益': total, '年化收益': ann, '年化波动': vol,
            '夏普比率': sharpe, '最大回撤': max_dd, '年均单边换手': tu_ann}


def to_monthly(ser):
    """日收益 → 月收益（按月复利）。"""
    return (1.0 + ser).groupby(ser.index.to_period('M')).prod() - 1.0


# =====================================================================
# 3. 绘图函数
# =====================================================================
def fig01_concept():
    """
    图01 概念示意：2 资产小例子，TSMOM 与 XSMOM 信号对比。
    三阶段行情：A/B 同涨（A更强）→ 双熊（A跌更狠）→ B 反转向上。
    人造价格用独立种子 rng(42) 生成（确定性，可复现）。
    """
    rng_fig = np.random.default_rng(SEED)
    n1, n2 = 250, 200
    T = n1 + n2 + 250
    mu_a = np.r_[np.full(n1, 0.0013), np.full(n2, -0.0016), np.full(250, -0.0002)]
    mu_b = np.r_[np.full(n1, 0.0005), np.full(n2, -0.0009), np.full(250, 0.0014)]
    pa = 100 * np.cumprod(1 + mu_a + rng_fig.normal(0, 0.011, T))
    pb = 100 * np.cumprod(1 + mu_b + rng_fig.normal(0, 0.011, T))
    dates = pd.bdate_range('2024-01-02', periods=T)
    W = 60
    mom_a = pd.Series(pa).pct_change(W).to_numpy()
    mom_b = pd.Series(pb).pct_change(W).to_numpy()
    k = np.arange(T)
    both_neg = (mom_a < 0) & (mom_b < 0)               # 双熊：两只动量同时 < 0
    sig_a = np.where(mom_a > 0, 1, 0)                  # TSMOM：各自看各自
    sig_b = np.where(mom_b > 0, 1, 0)
    xs_hold_a = (np.nan_to_num(mom_a, nan=-np.inf) >=
                 np.nan_to_num(mom_b, nan=-np.inf)).astype(int)   # XSMOM：相对更强者

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(k, pa, lw=1.4, color='#d62728', label='资产A：先大涨后阴跌')
    ax.plot(k, pb, lw=1.4, color='#1f77b4', label='资产B：小涨→缓跌→反转')
    ax.axvspan(n1, n1 + n2, color='crimson', alpha=0.08)
    ax.text(n1 + n2 / 2, ax.get_ylim()[1] * 0.97, '双熊阶段', ha='center', va='top',
            fontsize=10, color='darkred')
    ax.set_title('两个资产的价格（人造小例子，窗口 W=60 日）')
    ax.set_xlabel('交易日'); ax.set_ylabel('价格')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    sig_mat = np.vstack([sig_a, sig_b])
    cmap = matplotlib.colormaps['Blues'].copy()
    ax.imshow(sig_mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
              extent=[0, T, 2, 0], interpolation='nearest')
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['资产A', '资产B'])
    ax.set_title('TSMOM 信号：自己跟自己比\n（各自过去60日收益>0 才持有，否则空仓）')
    ax.set_xlabel('交易日')

    ax = axes[1, 0]
    ax.step(k, xs_hold_a, where='post', lw=1.6, color='#2ca02c')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['持有B', '持有A'])
    ax.axvspan(n1, n1 + n2, color='crimson', alpha=0.08)
    ax.annotate('双熊时也没有空仓选项：\n只能继续持有"跌得慢"的B', xy=(n1 + 60, 0.5),
                fontsize=9, color='darkred', va='center')
    ax.set_title('XSMOM 信号：跟别人比\n（永远持有相对更强的那只，始终满仓）')
    ax.set_xlabel('交易日'); ax.set_ylim(-0.15, 1.35)

    ax = axes[1, 1]
    ax.step(k, sig_a + sig_b, where='post', lw=1.8, color=COL_TSMOM_L,
            label='TSMOM 持仓数（0/1/2，可为0）')
    ax.step(k, np.ones(T), where='post', lw=1.8, color=COL_XSMOM, ls='--',
            label='XSMOM 持仓数（恒为1）')
    ax.axvspan(k[both_neg][0], k[both_neg][-1], color='crimson', alpha=0.12,
               label='两只动量同时<0：TSMOM 空仓避险')
    ax.set_yticks([0, 1, 2])
    ax.set_title('持仓数量对比：TSMOM 天然带择时，XSMOM 只多版永远满仓')
    ax.set_xlabel('交易日'); ax.set_ylabel('持仓资产数')
    ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.3)
    fig.suptitle('图01 "动量"的两种含义：TSMOM（自己跟自己比） vs XSMOM（跟别人比）',
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, '01_TSMOM与XSMOM信号示意图.png'), dpi=150)
    plt.close(fig)


def fig02_nav(navs, stats_map):
    """图02 两策略净值对比（含成本），图例注明年化/夏普/回撤（统计来自净日收益序列本身）。"""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    style = {'TSMOM只多(A股现实)': (COL_TSMOM_L, '-'),
             'TSMOM多空(理论对照)': (COL_TSMOM_S, '--'),
             'XSMOM Top3': (COL_XSMOM, '-'),
             '10行业等权': (COL_EW, '-'),
             '沪深300': (COL_HS300, '-')}
    for name, nav in navs.items():
        st = stats_map[name]
        c, ls = style[name]
        ax.plot(nav.index, nav.values, lw=2.0 if ls == '-' else 1.5, color=c, ls=ls,
                label=f"{name}（年化 {st['年化收益']:+.1%}，夏普 {st['夏普比率']:.2f}，"
                      f"回撤 {st['最大回撤']:.1%}）")
    ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.6)
    ax.set_title('图02 时序动量 vs 截面动量：10个A股行业，月度调仓（含双边万3成本，空仓收益按0计）',
                 fontsize=12)
    ax.set_xlabel('日期'); ax.set_ylabel('净值（期初=1）')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_两策略净值对比.png'), dpi=150)
    plt.close(fig)


def fig03_sensitivity(sens_df, ref_ew, ref_xs, sens_start):
    """图03 动量窗口敏感性：TSMOM 各窗口的年化（柱，含成本）与夏普（右轴线）。"""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    xs = np.arange(len(sens_df))
    vals = sens_df['年化收益'].to_numpy() * 100
    colors = [COL_TSMOM_L if n == N_LB else '#9ebcda' for n in sens_df['窗口']]
    bars = ax.bar(xs, vals, color=colors, alpha=0.9, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.35 if v >= 0 else -0.55),
                f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9.5)
    ax.axhline(ref_ew * 100, color=COL_EW, ls='--', lw=1.3,
               label=f'10行业等权（同评估期年化 {ref_ew:+.1%}）')
    ax.axhline(ref_xs * 100, color=COL_XSMOM, ls='--', lw=1.3,
               label=f'XSMOM-252（同评估期年化 {ref_xs:+.1%}）')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{n}日' for n in sens_df['窗口']])
    ax.set_ylabel('TSMOM 年化收益（%，含成本）')
    ax.set_xlabel('动量回看窗口（交易日）')
    ax2 = ax.twinx()
    ax2.plot(xs, sens_df['夏普比率'], 'o-', color='#2ca02c', lw=1.6, label='夏普比率（右轴）')
    ax2.set_ylabel('夏普比率')
    lo, hi = min(vals.min(), 0), max(vals.max(), 0)
    ax.set_ylim(lo - abs(lo) * 0.25 - 1.0, hi + abs(hi) * 0.25 + 1.0)
    ax.set_title(f'图03 TSMOM 动量窗口敏感性（20/60/120/252，统一自 {sens_start} 起算，\n'
                 f'含双边万3成本）——如实呈现：若数字对窗口很敏感，挑最好窗口就是过拟合',
                 fontsize=11.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_窗口敏感性.png'), dpi=150)
    plt.close(fig)


def fig04_signal_heatmap(mom_me, ind_names, start_date):
    """
    图04 TSMOM 各行业信号热力图（行=调仓月份，列=行业）：
      上图 = 只多版（1=持有，0=空仓，灰=历史不足无信号）；
      下图 = 理论多空版（+1=做多，-1=做空，0=空仓，灰=无信号）。
    """
    mm = mom_me.loc[mom_me.index >= start_date]
    sig_long = mm > 0
    sig_ls = np.sign(mm)
    mats = [(sig_long, '只多版（A股现实）：动量>0 持有 1/10，否则空仓',
             matplotlib.colormaps['Blues'].copy(), 0, 1, ['空仓', '持有']),
            (sig_ls, '理论多空版（A股现货做不到）：动量>0 多 1/10，<0 空 1/10',
             matplotlib.colormaps['RdYlGn'].copy(), -1, 1, ['-1 做空', '0', '+1 做多'])]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5))
    dates = list(mm.index)
    for ax, (sig, title, cmap, vmin, vmax, ylabels) in zip(axes, mats):
        mat = np.ma.masked_invalid(sig.to_numpy(dtype=float))
        cmap.set_bad('lightgray')
        ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation='nearest')
        ax.set_xticks(range(len(ind_names)))
        ax.set_xticklabels(ind_names, rotation=30, ha='right', fontsize=9)
        step = max(1, len(dates) // 16)
        ticks = list(range(0, len(dates), step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([dates[i].strftime('%Y-%m') for i in ticks], fontsize=8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(np.arange(-0.5, len(ind_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(dates), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        ax.tick_params(which='minor', length=0)
    fig.suptitle('图04 TSMOM 各行业月度信号热力图（灰=该行业自身历史尚不足252日，无信号按空仓）',
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, '04_TSMOM各行业多空信号热力图.png'), dpi=150)
    plt.close(fig)


def fig05_correlation(corr, mret, name_a='TSMOM只多(A股现实)', name_b='XSMOM Top3'):
    """图05 两策略月收益相关性：左=相关矩阵热力图，右=TSMOM vs XSMOM 散点+拟合线。"""
    labels = ['TSMOM只多', 'TSMOM多空', 'XSMOM', '等权', '沪深300']
    key = {'TSMOM只多': name_a, 'TSMOM多空': 'TSMOM多空(理论对照)', 'XSMOM': name_b,
           '等权': '10行业等权', '沪深300': '沪深300'}
    mat = corr.loc[[key[l] for l in labels], [key[l] for l in labels]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(mat[i, j]) > 0.55 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title('月收益相关矩阵（含成本）')
    ax = axes[1]
    xa, xb = mret[name_a], mret[name_b]
    ax.scatter(xb * 100, xa * 100, s=26, alpha=0.7, color=COL_TSMOM_L, edgecolors='k',
               linewidths=0.4)
    k1, k0 = np.polyfit(xb.to_numpy(), xa.to_numpy(), 1)
    xs_line = np.linspace(xb.min(), xb.max(), 50)
    ax.plot(xs_line * 100, (k1 * xs_line + k0) * 100, color='k', lw=1.2, ls='--',
            label=f'拟合线：斜率{k1:.2f}')
    r = float(np.corrcoef(xa, xb)[0, 1])
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('XSMOM Top3 月收益（%）'); ax.set_ylabel('TSMOM只多 月收益（%）')
    ax.set_title(f'TSMOM vs XSMOM 逐月散点：相关系数 r = {r:.2f}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f'图05 两策略相关性：TSMOM只多 vs XSMOM 月收益 r={mat[0, 2]:.2f}（同向但不重合），'
                 f'而与理论多空版仅 r={mat[0, 1]:.2f}——同一句"动量"下的三种不同风险暴露',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, '05_两策略相关性.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    # ---------- 4.1 数据加载与清洗 ----------
    close_raw, idx300, ind_names = load_all_data()
    n_assets = len(ind_names)
    ret_panel, common_index, n_jump_common = build_common_panel(close_raw, ind_names)

    print('=' * 76)
    print(f'步骤0 数据：{"合成数据（真实CSV缺失兜底）" if SYNTHETIC else "真实行业数据"}，'
          f'行业 {ind_names}')
    print(f'共同重叠期（10个行业全有数据）: {common_index[0].date()} ~ {common_index[-1].date()}，'
          f'共 {len(common_index)} 个交易日')
    print(f'清洗规则：单日|收益|>{JUMP_TH:.0%} 判为基期跳变 → 置 0（共同期内共清洗 {n_jump_common} 处）；'
          f'动量在各行业自身完整历史上预计算（含共同期之前的预热数据）')

    month_ends = month_end_dates(common_index)
    _, mom252 = own_returns_momentum(close_raw, ind_names, N_LB)
    mom_me = mom252.reindex(month_ends)                 # 月末 × 行业 动量矩阵（NaN=历史不足）
    avail = mom_me.notna().sum(axis=1)

    print('\n步骤1 252日动量信号可得性时间线（行业发布时间不一，预热有先有后）：')
    prev = -1
    for d, v in avail.items():
        if v != prev:
            print(f'  {d.date()} 起可打分行业 {int(v)} 个')
            prev = int(v)

    start_date = avail[avail >= K_HOLD].index[0]        # 首个"可打分行业≥3"的月末
    bt_days = common_index[common_index > start_date]
    years_bt = (bt_days[-1] - bt_days[0]).days / 365.25
    print(f'主回测起点: {start_date.date()}（首个可打分行业≥{K_HOLD}的月末），'
          f'区间 {bt_days[0].date()} ~ {bt_days[-1].date()}，'
          f'共 {len(bt_days)} 个交易日 ≈ {years_bt:.1f} 年  ← 短样本！')

    # ---------- 4.2 三套目标权重（TSMOM只多 / TSMOM多空 / XSMOM / 等权） ----------
    tsmom_long, tsmom_ls, xsmom, ew = {}, {}, {}, {}
    for d in month_ends:
        if d < start_date:
            continue
        row = mom_me.loc[d]
        wl, ws = tsmom_weight_vectors(row, n_assets)
        tsmom_long[d], tsmom_ls[d] = wl, ws
        wx = xsmom_weight_vector(row, mom_me.columns, K_HOLD)
        if wx is not None:
            xsmom[d] = wx
        ew[d] = np.full(n_assets, 1.0 / n_assets)

    # ---------- 4.3 回测（含双边万3成本） ----------
    print('\n步骤2 主回测（月度调仓，含双边万3成本；沪深300为买入持有、无成本）：')
    results = {}
    for name, targets in [('TSMOM只多(A股现实)', tsmom_long), ('TSMOM多空(理论对照)', tsmom_ls),
                          ('XSMOM Top3', xsmom), ('10行业等权', ew)]:
        gross, net, tu, expo = backtest(targets, ret_panel, bt_days, COST)
        results[name] = {'gross': gross, 'net': net, 'tu': tu, 'expo': expo}
    r300 = idx300.pct_change().reindex(bt_days)
    r300 = r300.mask(r300.abs() > JUMP_TH, 0.0).fillna(0.0)
    results['沪深300'] = {'gross': r300, 'net': r300,
                          'tu': pd.Series(dtype=float), 'expo': None}

    order = ['TSMOM只多(A股现实)', 'TSMOM多空(理论对照)', 'XSMOM Top3', '10行业等权', '沪深300']
    rows = []
    for name in order:
        res = results[name]
        st = perf_stats(res['net'], res['tu'])
        st_g = perf_stats(res['gross'])
        rows.append({'策略': name, '起始日': st['起始日'], '截止日': st['截止日'],
                     '累计收益': st['累计收益'], '年化收益': st['年化收益'],
                     '年化波动': st['年化波动'], '夏普比率': st['夏普比率'],
                     '最大回撤': st['最大回撤'],
                     '平均仓位': float(res['expo'].mean()) if res['expo'] is not None else 1.0,
                     '年均单边换手': st['年均单边换手'],
                     '年化收益_无成本': st_g['年化收益'],
                     '成本拖累_年化': st_g['年化收益'] - st['年化收益']})
    met_df = pd.DataFrame(rows).set_index('策略')
    print(met_df.to_string(float_format=lambda x: f'{x:.4f}'))

    # ---------- 4.4 净值图 + 月收益相关性 ----------
    navs = {name: (1.0 + results[name]['net']).cumprod() for name in order}
    stats_map = {r['策略']: r for r in rows}     # 直接用净日收益序列算出的绩效（与表一致）
    fig02_nav(navs, stats_map)
    print('已输出: outputs/02_两策略净值对比.png')

    mret = {name: to_monthly(results[name]['net']) for name in order}
    corr = pd.DataFrame(mret).corr()
    print('\n步骤3 两策略月收益相关矩阵（含成本）：')
    print(corr.to_string(float_format=lambda x: f'{x:.3f}'))
    fig05_correlation(corr, mret)
    print('已输出: outputs/05_两策略相关性.png')

    # 市场环境拆分：按沪深300月收益正负分组看两策略平均月收益（互补性）
    m300 = mret['沪深300']
    up, dn = m300 > 0, m300 <= 0
    print(f'\n步骤4 市场环境拆分（沪深300月收益：上涨月 {int(up.sum())} 个 / 下跌月 {int(dn.sum())} 个，'
          f'平均月收益）：')
    regime = pd.DataFrame({
        '沪深300上涨月': [mret[n][up].mean() for n in order],
        '沪深300下跌月': [mret[n][dn].mean() for n in order]},
        index=order)
    print(regime.to_string(float_format=lambda x: f'{x:+.3%}'))

    # ---------- 4.5 TSMOM 窗口敏感性（统一评估期：全行业252日信号齐备之后） ----------
    sens_start = avail[avail >= n_assets].index[0]       # 全部10个行业252日信号齐备的月末
    sens_days = common_index[common_index > sens_start]
    sens_rows = []
    for n in SENS_WINDOWS:
        _, momn = own_returns_momentum(close_raw, ind_names, n)
        targets = {}
        for d in month_ends:
            if d < sens_start:
                continue
            wl, _ = tsmom_weight_vectors(momn.reindex([d]).iloc[0], n_assets)
            targets[d] = wl
        _, net_n, _, _ = backtest(targets, ret_panel, sens_days, COST)
        st_n = perf_stats(net_n)
        sens_rows.append({'窗口': n, '年化收益': st_n['年化收益'],
                          '夏普比率': st_n['夏普比率'], '最大回撤': st_n['最大回撤']})
    # 参照：等权与 XSMOM-252 在同一评估期
    ew_targets_s = {d: np.full(n_assets, 1.0 / n_assets)
                    for d in month_ends if d >= sens_start}
    _, ew_net_s, _, _ = backtest(ew_targets_s, ret_panel, sens_days, COST)
    ref_ew = perf_stats(ew_net_s)['年化收益']
    xs_targets_s = {}
    _, mom252_s = own_returns_momentum(close_raw, ind_names, N_LB)
    for d in month_ends:
        if d < sens_start:
            continue
        wx = xsmom_weight_vector(mom252_s.reindex([d]).iloc[0], mom252_s.columns, K_HOLD)
        if wx is not None:
            xs_targets_s[d] = wx
    _, xs_net_s, _, _ = backtest(xs_targets_s, ret_panel, sens_days, COST)
    ref_xs = perf_stats(xs_net_s)['年化收益']

    sens_df = pd.DataFrame(sens_rows)
    print(f'\n步骤5 TSMOM 动量窗口敏感性（统一自 {sens_start.date()} 起算、含成本；'
          f'参照：等权 {ref_ew:+.2%} / XSMOM-252 {ref_xs:+.2%}）：')
    print(sens_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    fig03_sensitivity(sens_df, ref_ew, ref_xs, str(sens_start.date()))
    print('已输出: outputs/03_窗口敏感性.png')

    # ---------- 4.6 信号热力图 + tsmom_signals.csv ----------
    fig04_signal_heatmap(mom_me, ind_names, start_date)
    print('已输出: outputs/04_TSMOM各行业多空信号热力图.png')
    fig01_concept()
    print('已输出: outputs/01_TSMOM与XSMOM信号示意图.png')

    sig_rows = []
    for d in month_ends:
        if d < start_date:
            continue
        row = mom_me.loc[d]
        wl, ws = tsmom_weight_vectors(row, n_assets)
        for j, c in enumerate(ind_names):
            v = row[c]
            sig_rows.append({
                '调仓日': d.strftime('%Y-%m-%d'), '行业': c,
                '动量_过去252日': '' if pd.isna(v) else round(float(v), 6),
                '只多信号_1持有_0空仓': '' if pd.isna(v) else int(wl[j] > 0),
                '多空理论信号_1多_0空_-1空': '' if pd.isna(v) else int(ws[j] * n_assets)})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(OUT_DIR, 'tsmom_signals.csv'),
                  index=False, encoding='utf-8-sig')
    print(f'已输出: outputs/tsmom_signals.csv（{len(sig_df)} 行 = {len(set(sig_df["调仓日"]))} 个调仓月 × {n_assets} 行业）')

    # ---------- 4.7 metrics.csv ----------
    out = met_df.copy()
    out['起始日'] = pd.to_datetime(out['起始日']).dt.strftime('%Y-%m-%d')
    out['截止日'] = pd.to_datetime(out['截止日']).dt.strftime('%Y-%m-%d')
    for c in out.columns:
        if c not in ('起始日', '截止日'):
            out[c] = out[c].astype(float).map(lambda x: f'{x:.4f}')
    out.to_csv(os.path.join(OUT_DIR, 'metrics.csv'), encoding='utf-8-sig')
    print('已输出: outputs/metrics.csv')

    # ---------- 4.8 收尾摘要 ----------
    a = met_df.loc['TSMOM只多(A股现实)']
    b = met_df.loc['XSMOM Top3']
    r_ab = corr.loc['TSMOM只多(A股现实)', 'XSMOM Top3']
    print('\n' + '=' * 76)
    print('核心对比（含双边万3成本）：')
    print(f"  TSMOM只多：年化 {a['年化收益']:+.2%}，夏普 {a['夏普比率']:.2f}，回撤 {a['最大回撤']:.1%}，"
          f"平均仓位 {a['平均仓位']:.0%}，年均单边换手 {a['年均单边换手']:.1f} 次")
    print(f"  XSMOM Top3：年化 {b['年化收益']:+.2%}，夏普 {b['夏普比率']:.2f}，回撤 {b['最大回撤']:.1%}，"
          f"平均仓位 {b['平均仓位']:.0%}，年均单边换手 {b['年均单边换手']:.1f} 次")
    print(f'  两策略月收益相关系数 r = {r_ab:.2f}')
    print(f'样本期仅约 {years_bt:.1f} 年（短样本！），所有结论仅供教学演示，不构成投资建议。')
    print('完成。产出目录:', OUT_DIR)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # 课程硬性要求：任何异常都不以非0退出（打印完整堆栈便于排查）
        traceback.print_exc()
        print('[异常] 运行中断，但按课程规范以 0 退出（请检查上方堆栈修复）')
        sys.exit(0)
