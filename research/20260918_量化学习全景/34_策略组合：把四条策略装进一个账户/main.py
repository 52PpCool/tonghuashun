# -*- coding: utf-8 -*-
"""
34_策略组合：把四条策略装进一个账户（毕业设计）
====================================================

单策略都平平无奇，合起来会怎样？——策略层面的分散化实战。

把前面上过的四条策略复刻成四个"独立子账户"，各自算出日净收益后放进同一个面板，
再比较四种配钱方式（等权 / 逆波动率 / 最小方差 / 风险平价代理）：

  S1 双均线      ：上证指数 SMA20/60 金叉死叉，多/空仓切换（课02规则，信号次日生效）
  S2 海龟简化    ：贵州茅台 20日唐奇安突破入场 / 10日反向离场 / 2N止损（课03简化版，不加仓）
  S3 行业轮动    ：10个行业板块 60日动量 Top3，月度调仓（课15/27框架，共同重叠期起）
  S4 波动率目标  ：上证+茅台各半仓，各按 10% 年化目标波动缩放仓位（日频调仓）

统一口径（README 有详细说明）：
  * 成本：单边万3（0.0003/次），买卖两边各收一次；"一买一卖"完整回合 = 双边万6。
    （与课02/03 "佣金双边各万2.5" 的口径习惯一致，本课统一简化为单边万3。）
  * 现金：子账户中未投资的仓位按无风险利率 2% 年化计息（日利率 2%/252）。
  * 净值：日频，(1+日净收益) 连乘。
  * 数据毛刺：单日 |收益| > 30% 视为基期跳变毛刺，该日【收益率】置 0、价位不动（改价位会制造次日"假分母"链式伪影，见 clean_glitch）。

产出（outputs/）：
  01_四策略净值与回撤.png / 02_策略相关性热力图.png / 03_组合方式净值对比.png
  04_组合权重演变.png / 05_最大回撤期贡献分解.png
  strategy_metrics.csv / portfolio_metrics.csv（utf-8-sig）

运行：cd 到本目录后用工作区 .venv 的 python 执行 main.py，总耗时 < 2 分钟。
"""

import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 全局参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'cache'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)          # 随机种子 42（合成数据兜底时使用）
TRADING_DAYS = 252
RF_ANNUAL = 0.02                          # 无风险利率 2% 年化（现金部分计息）
RF_DAILY = RF_ANNUAL / TRADING_DAYS
COST = 3e-4                               # 单边万3：每发生 1 单位单边换手收 0.0003
TARGET_VOL = 0.10                         # S4 目标年化波动 10%
EST_WINDOW = 252                          # 组合层估计窗口（252 个交易日）
STRESS_WIN = ('2024-09-10', '2024-10-18')  # 2024年"924行情"普涨普跌事件窗

SYNTH_FLAG = False                        # 数据缺失时置 True（合成数据兜底）

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

COL_S = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']   # 四策略配色


# ================================================================ 第0步 数据加载与清洗
def _synth_ohlc(start, end, s0, mu, sigma, name):
    """数据缺失时的合成数据兜底（几何布朗运动，种子42），并显著标注。"""
    global SYNTH_FLAG
    SYNTH_FLAG = True
    print(f'!! 警告：缓存缺失，"{name}" 使用【合成数据】兜底（rng(42) 几何布朗运动）')
    days = pd.bdate_range(start, end)
    ret = RNG.normal(mu / TRADING_DAYS, sigma / np.sqrt(TRADING_DAYS), len(days))
    close = s0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(RNG.normal(0, 0.004, len(days))))
    low = close * (1 - np.abs(RNG.normal(0, 0.004, len(days))))
    open_ = low + (high - low) * RNG.uniform(0.2, 0.8, len(days))
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=days)
    return df


def load_ohlc(fname, synth_kwargs, name):
    """读取缓存 CSV（utf-8-sig），缺失则合成兜底；返回 OHLC DataFrame（按日期索引）。"""
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date'], index_col='date')
        df = df.sort_index()
        df = df.dropna(subset=['close'])          # 板块末行偶有空值行
        df = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    return _synth_ohlc(**synth_kwargs, name=name)


def clean_glitch(close):
    """基期跳变清洗（与课15口径一致）：单日|收益|>30% 视为数据源基期切换毛刺，该日【收益率】置0。
    注意：只动收益率、不改价位——若改价位（置为前收盘），跳变次日的收益会被"假分母"放大，
    制造出新的未标记毛刺（链式伪影）。A股指数/板块单日|收益|>30% 只可能是数据问题。"""
    r = close.pct_change()
    glitch = r.abs() > 0.30
    glitch.iloc[0] = False                        # 首行无前收盘
    r_clean = r.mask(glitch, 0.0)
    return r_clean, int(glitch.sum())


# ================================================================ 第1步 四条子策略
def perf_stats(ret):
    """由日净收益序列计算绩效四件套：年化收益(CAGR)、年化波动、夏普(超额=年化-2%)、最大回撤。"""
    ret = ret.dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(年化收益=np.nan, 年化波动=np.nan, 夏普=np.nan, 最大回撤=np.nan)
    nav = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = nav.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS - RF_ANNUAL) / vol
    dd = (nav / nav.cummax() - 1).min()
    return dict(年化收益=cagr, 年化波动=vol, 夏普=sharpe, 最大回撤=dd)


def strat_s1_dual_ma(ohlc_idx, ret_idx):
    """S1 双均线（课02规则）：上证指数 SMA20/60，金叉持有、死叉空仓；信号收盘生成、次日生效。"""
    close = ohlc_idx['close']
    ret = ret_idx                                  # 清洗后收益（毛刺置0）
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    signal = (sma20 > sma60).astype(float)        # 收盘时的目标仓位
    pos = signal.shift(1).fillna(0.0)             # 信号次日执行：今日持有的仓位=昨日信号
    gross = pos * ret + (1 - pos) * RF_DAILY      # 空仓部分按 2% 年化计息
    cost = COST * pos.diff().abs().fillna(0.0)    # 仓位切换按换手收单边万3
    return gross - cost


def strat_s2_turtle(ohlc_mt, ret_mt):
    """S2 海龟简化（课03简化版）：茅台 20日突破入场 / 10日反向离场 / 2N止损；不加仓、全仓进出。
    （茅台经检查无单日|收益|>30% 毛刺，唐奇安/ATR 直接用原始OHLC是安全的。）"""
    close, high, low = ohlc_mt['close'], ohlc_mt['high'], ohlc_mt['low']
    ret = ret_mt                                   # 清洗后收益（毛刺置0）
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    n_val = tr.rolling(20).mean()                          # N = ATR(20)（简单均值，课03同款）
    don_up = high.rolling(20).max().shift(1)               # 前20日最高价（不含今日）
    don_low = low.rolling(10).min().shift(1)               # 前10日最低价（不含今日）

    n_days = len(close)
    pos_held = np.zeros(n_days)                            # 今日实际持有的仓位（0/1）
    state, entry, stop = 0, 0.0, np.nan
    close_v = close.values
    n_v, up_v, low_v = n_val.values, don_up.values, don_low.values
    for t in range(n_days):
        pos_held[t] = state                                # 今日仓位=昨日收盘决策的结果
        c = close_v[t]
        if state == 1:                                     # 持仓中：今日收盘检查离场
            exit_rev = (not np.isnan(low_v[t])) and (c < low_v[t])       # 10日反向离场
            exit_stop = (not np.isnan(stop)) and (c < stop)              # 2N止损
            if exit_rev or exit_stop:
                state = 0                                       # 今日收盘卖出 → 明日空仓
        else:                                              # 空仓中：今日收盘检查突破
            if (not np.isnan(up_v[t])) and (not np.isnan(n_v[t])) and (c > up_v[t]):
                state = 1                                   # 今日收盘买入 → 明日持仓
                entry = c                                   # 入场价≈突破日收盘（向量化近似）
                stop = entry - 2.0 * n_v[t]                 # 止损线 = 入场价 - 2N（不加仓不上移）
    pos = pd.Series(pos_held, index=close.index)
    gross = pos * ret + (1 - pos) * RF_DAILY
    cost = COST * pos.diff().abs().fillna(0.0)
    return gross - cost


# ---------------- 月度再平衡引擎（S3 子策略与第4步组合层共用） ----------------
def month_end_mask(idx):
    """每月最后一个交易日为再平衡日（首日作为初始建仓日另行处理）。"""
    months = idx.to_period('M')
    me = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        me[:-1] = months.values[:-1] != months.values[1:]
    me[-1] = True
    return me


def monthly_engine(panel, target_fn, cost_rate=COST, init_buyin=True):
    """
    月度再平衡回测引擎：权重逐日漂移 + 再平衡日扣换手成本。

    panel     : DataFrame 日收益面板（各列已含各自的现金收益与内部成本）
    target_fn : target_fn(t, panel) -> np.ndarray，在第 t 日收盘后决定目标权重
                （只允许使用 ≤t 的数据；返回 None 表示本月无可行目标、维持现状）
    规则：
      第 t 日收益 = w·r_t + (1-Σw)·rf_daily   （w 为昨日收盘后的实际权重；现金部分计息）
      收盘后权重按当日收益漂移；
      若 t 是再平衡日：cost = cost_rate × Σ|w_target − w_drift|（每个单边各收一次），
                       净值再乘 (1−cost)，权重设为 w_target。
    返回 nav, w_used(每日"用前"权重，供贡献分解), w_after(每日收盘后权重), cost, turnover
    """
    dates = panel.index
    R = np.nan_to_num(panel.values, nan=0.0)   # 首行 pct_change 的 NaN 置0，防止毒化整条净值
    T, N = R.shape
    rebal = month_end_mask(dates)
    w = np.zeros(N)
    nav = np.ones(T + 1)
    w_used = np.zeros((T, N))
    w_after = np.zeros((T, N))
    cost = np.zeros(T)
    turn = np.zeros(T)
    for t in range(T):
        w_used[t] = w
        inv = w.sum()
        gross = w @ R[t] + (1 - inv) * RF_DAILY          # 未投资部分按无风险计息
        nav[t + 1] = nav[t] * (1 + gross)
        if (1 + gross) > 1e-12:                          # 权重逐日漂移
            w = w * (1 + R[t]) / (1 + gross)
        else:
            w = np.zeros(N)
        if (t == 0 and init_buyin) or rebal[t]:          # 初始建仓 / 月末再平衡
            tgt = target_fn(t, panel)
            if tgt is not None:
                tgt = np.asarray(tgt, dtype=float)
                turn[t] = np.abs(tgt - w).sum()
                cost[t] = cost_rate * turn[t]
                nav[t + 1] *= (1 - cost[t])
                w = tgt.copy()
        w_after[t] = w
    nav_s = pd.Series(nav[1:], index=dates)
    wu = pd.DataFrame(w_used, index=dates, columns=panel.columns)
    wa = pd.DataFrame(w_after, index=dates, columns=panel.columns)
    return nav_s, wu, wa, pd.Series(cost, index=dates), pd.Series(turn, index=dates)


def strat_s3_rotation(board_ret):
    """S3 行业轮动（课15/27框架）：10行业 60日动量 Top3 等权，月度调仓（用共用引擎）。
    动量 = 过去60个交易日复利收益 = expm1(Σlog(1+r))——必须用【清洗后收益】累乘：
    板块指数有基期重置，直接用原始价位相除会被重置跳变污染。"""
    board_ret = board_ret.fillna(0.0)
    k = board_ret.shape[1]
    mom60 = np.expm1(np.log1p(board_ret).rolling(60).sum())   # 60日复利动量（与课15同款）

    def target_fn(t, _panel):
        if t < 60:
            return np.zeros(k)                            # 60日动量不可得：全部现金（计息）
        m = mom60.iloc[t]                                 # 只用 ≤t 数据，无前视
        if m.isna().any():
            return np.zeros(k)
        top3 = m.nlargest(3).index
        w = pd.Series(0.0, index=board_ret.columns)
        w[top3] = 1.0 / 3.0
        return w.values

    nav, wu, wa, cost, turn = monthly_engine(board_ret, target_fn)
    return nav.pct_change().fillna(0.0), nav, wa


def strat_s4_vol_target(ret_sh, ret_mt):
    """S4 波动率目标择时：上证+茅台各半仓，按 10% 年化目标波动缩放；日频调仓。
    规则：w_i(t) = 0.5 × min(1, 10% / σ_i(20日年化，用至昨日))；超目标波动就减仓，最多半仓。"""
    df = pd.concat([ret_sh.rename('上证'), ret_mt.rename('茅台')], axis=1).dropna()
    vol = df.rolling(20).std() * np.sqrt(TRADING_DAYS)     # 20日已实现波动（年化）
    ratio = (TARGET_VOL / vol.shift(1)).clip(upper=1.0)    # 用昨日波动定今日仓位（无前视）
    w = (0.5 * ratio).fillna(0.0)
    gross = (w * df).sum(axis=1) + (1 - w.sum(axis=1)) * RF_DAILY
    cost = COST * w.diff().abs().sum(axis=1).fillna(0.0)
    return (gross - cost).rename('S4')


# ================================================================ 第2步 对齐
def build_panel():
    """加载数据 → 跑四条子策略 → 各自日净收益 → 取共同区间对齐成面板。"""
    # ---- 数据
    px_idx_ohlc = load_ohlc('index_000001_上证指数.csv',
                            dict(start='2010-01-04', end='2026-09-17', s0=3200.0, mu=0.04, sigma=0.20),
                            '上证指数')
    px_mt_ohlc = load_ohlc('stock_600519_贵州茅台.csv',
                           dict(start='2015-01-05', end='2026-09-17', s0=180.0, mu=0.12, sigma=0.32),
                           '贵州茅台')
    boards = ['银行', '保险', '证券', '房地产开发', '汽车整车',
              '半导体', '医疗器械', '白酒', '煤炭行业', '电力行业']
    px_boards = {}
    for b in boards:
        px_boards[b] = load_ohlc(f'board_{b}.csv',
                                 dict(start='2021-12-29', end='2026-09-17', s0=1000.0,
                                      mu=0.03, sigma=0.25), f'行业-{b}')
    ret_idx, g1 = clean_glitch(px_idx_ohlc['close'])
    ret_mt, g2 = clean_glitch(px_mt_ohlc['close'])
    ret_idx = ret_idx.fillna(0.0)                  # 上证清洗后日收益
    ret_mt = ret_mt.fillna(0.0)                    # 茅台清洗后日收益
    board_ret = {}
    nb = 0
    for b in boards:
        r, g = clean_glitch(px_boards[b]['close'])
        board_ret[b] = r
        nb += g
    print(f'毛刺清洗（单日|收益|>30% 收益率置0，价位不动）：上证 {g1} 个、茅台 {g2} 个、'
          f'10行业合计 {nb} 个')

    # S3 用 10 个行业的"共同重叠期"（先对齐日期、再清洗）
    close_panel = pd.DataFrame({b: px_boards[b]['close'] for b in boards}).dropna()
    board_ret = pd.DataFrame({b: board_ret[b].reindex(close_panel.index) for b in boards})
    board_ret = board_ret.mask(board_ret.abs() > 0.30, 0.0).fillna(0.0)   # 对齐后再清一遍
    print(f'10行业共同重叠期：{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}，'
          f'共 {len(close_panel)} 个交易日')

    # ---- 四条子策略（各自独立子账户的日净收益）
    s1 = strat_s1_dual_ma(px_idx_ohlc, ret_idx).rename('S1_双均线')
    s2 = strat_s2_turtle(px_mt_ohlc, ret_mt).rename('S2_海龟')
    s3, _, _ = strat_s3_rotation(board_ret)
    s3 = s3.rename('S3_行业轮动')
    s4 = strat_s4_vol_target(ret_idx, ret_mt)

    strategies = {'S1_双均线': s1, 'S2_海龟': s2, 'S3_行业轮动': s3, 'S4_波动率目标': s4}
    print('\n== 各子策略自身起始日与样本量（对齐前的"原生历史"）==')
    for k, v in strategies.items():
        print(f'  {k}: {v.index[0].date()} ~ {v.index[-1].date()}，{len(v)} 日')

    panel = pd.concat(strategies, axis=1).dropna()          # 取四条策略共同区间
    panel.columns = list(strategies.keys())
    print(f'\n对齐后共同区间：{panel.index[0].date()} ~ {panel.index[-1].date()}，'
          f'共 {len(panel)} 个交易日')
    for k, v in strategies.items():
        lost = (panel.index[0] - v.index[0]).days / 365.25
        print(f'  {k}: 为对齐舍弃了约 {lost:.1f} 年原生历史'
              if lost > 0.08 else f'  {k}: 自身起点≈共同起点，几乎无损失')
    print('  → 谁损失了历史样本：S3（行业轮动）受限于板块数据 2021-12 才开始，'
          '把 S1/S2/S4 的更早历史全部"拖没"了——这是面板对齐的代价，第3步的全历史列可见差异。')
    return panel, strategies


# ================================================================ 第3步 相关性与单策略成绩单
def section3(panel, strategies):
    print('\n' + '=' * 72)
    print('第3步 策略相关性与单策略成绩单（共同区间）')
    print('=' * 72)
    corr = panel.corr()
    print('相关系数矩阵：')
    print(corr.round(3).to_string())
    off = corr.values[np.triu_indices(4, k=1)]
    avg_corr = float(np.mean(off))
    print(f'平均两两相关（6对）：{avg_corr:.3f}')

    rows = []
    for k in panel.columns:
        st = perf_stats(panel[k])
        st_full = perf_stats(strategies[k])
        others = [c for c in panel.columns if c != k]
        st['与其他策略平均相关'] = float(np.mean([panel[k].corr(panel[o]) for o in others]))
        st['全历史年化'] = st_full['年化收益']
        st['全历史夏普'] = st_full['夏普']
        st['全历史最大回撤'] = st_full['最大回撤']
        st['自身起始日'] = strategies[k].index[0].date()
        rows.append(pd.Series(st, name=k))
    metrics = pd.DataFrame(rows)
    print('\n单策略成绩单（共同区间 vs 全历史）：')
    print(metrics.round(3).to_string())

    # 本课灵魂：等权组合的夏普能否超过最佳单策略？（公式直觉：SR_p ≈ SR·√(N/(1+(N-1)ρ))，
    # 前提是各策略"正期望且夏普大体相当"）
    ew_daily = panel.mean(axis=1)                          # 粗验证用（正式等权组合见第4步引擎版）
    st_ew = perf_stats(ew_daily)
    best = metrics['夏普'].idxmax()
    sr_best, sr_ew = float(metrics.loc[best, '夏普']), float(st_ew['夏普'])
    boost_pred = np.sqrt(4 / (1 + 3 * avg_corr))           # 同夏普、等权、两两相关ρ 的理论提升
    print(f'\n[验证] 等权组合（简单日平均）夏普 {sr_ew:.3f} vs 最佳单策略 {best} 夏普 {sr_best:.3f}')
    if sr_best > 0 and sr_ew > sr_best:
        print(f'[公式直觉] ρ={avg_corr:.3f} 时理论提升 √(4/(1+3ρ))={boost_pred:.2f} 倍，'
              f'实际 {sr_ew / sr_best:.2f} 倍 → 验证成功')
    else:
        n_neg = int((metrics['夏普'] < 0).sum())
        print(f'[证伪] 组合夏普未超过最佳单策略，如实归因：'
              f'①四条策略中有 {n_neg} 条夏普为负——分散化放大的只是"正期望"，'
              f'负期望策略合在一起只会更均匀地亏，公式前提不成立；'
              f'②"同夏普"假设不成立（四策略夏普极差 {metrics["夏普"].max() - metrics["夏普"].min():.2f}，'
              f'最差者会拖累组合）；③ρ={avg_corr:.3f} 不算很低（A股策略同涨同跌，相关性天然偏高）。'
              f'理论公式在"低相关+正期望"前提下预测可提升 {boost_pred:.2f} 倍——本样本不满足前提。')

    # 相关性热力图
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4), corr.columns, rotation=20, ha='right')
    ax.set_yticks(range(4), corr.columns)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(corr.values[i, j]) > 0.55 else 'black', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label='相关系数')
    ax.set_title(f'图02 四策略日收益相关系数热力图（共同区间，平均两两相关 {avg_corr:.2f}）')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_策略相关性热力图.png'), dpi=150)
    plt.close(fig)
    return metrics, corr, avg_corr


# ================================================================ 第4步 四种配钱方式
def make_target_fns(panel):
    """生成四种方案的月末目标权重函数（均只用 ≤t 数据；估计窗口不足252日时退化为等权）。"""
    names = list(panel.columns)
    n = len(names)

    def ew(_t, _p):
        return np.full(n, 1.0 / n)

    def inv_vol(t, p):
        hist = p.iloc[max(0, t - EST_WINDOW + 1):t + 1]
        if len(hist) < EST_WINDOW:
            return np.full(n, 1.0 / n)
        vol = hist.std() * np.sqrt(TRADING_DAYS)
        w = 1.0 / vol
        return (w / w.sum()).values

    def inv_var(t, p):                                     # 风险平价的"逆方差代理"简化版
        hist = p.iloc[max(0, t - EST_WINDOW + 1):t + 1]
        if len(hist) < EST_WINDOW:
            return np.full(n, 1.0 / n)
        var = hist.var()
        w = 1.0 / var
        return (w / w.sum()).values

    def min_var(t, p):
        hist = p.iloc[max(0, t - EST_WINDOW + 1):t + 1]
        if len(hist) < EST_WINDOW:
            return np.full(n, 1.0 / n)
        sigma = hist.cov().values
        try:
            import cvxpy as cp
            wv = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(wv, cp.psd_wrap(sigma))),
                              [cp.sum(wv) == 1, wv >= 0])
            for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
                try:
                    prob.solve(solver=solver)
                    if wv.value is not None and np.all(np.isfinite(wv.value)):
                        return np.clip(wv.value, 0, 1) / np.clip(wv.value, 0, 1).sum()
                except Exception:
                    continue
        except Exception:
            pass
        return np.full(n, 1.0 / n)                         # 求解失败兜底：等权

    return {'等权': ew, '逆波动率': inv_vol, '最小方差': min_var, '风险平价代理': inv_var}


def section4(panel):
    print('\n' + '=' * 72)
    print('第4步 组合方式对比：等权 / 逆波动率 / 最小方差(cvxpy) / 风险平价代理，月度再平衡')
    print('=' * 72)
    fns = make_target_fns(panel)
    results = {}
    for name, fn in fns.items():
        nav, wu, wa, cost, turn = monthly_engine(panel, fn)
        results[name] = dict(nav=nav, wu=wu, wa=wa, cost=cost, turn=turn)

    rows = []
    years = len(panel) / TRADING_DAYS
    for name, r in results.items():
        st = perf_stats(r['nav'].pct_change().fillna(0.0))
        st['年化成本拖累'] = r['cost'].mean() * TRADING_DAYS
        st['年均换手'] = r['turn'].sum() / years
        rows.append(pd.Series(st, name=name))
    pm = pd.DataFrame(rows)
    print('组合层成绩单（月度再平衡、含子策略间调仓成本单边万3；'
          'IV/MinVar/RP 前252日估计不足时按等权过渡）：')
    print(pm.round(4).to_string())

    me_pos = np.where(month_end_mask(panel.index))[0]
    ok = me_pos[me_pos >= EST_WINDOW - 1]
    first_est_date = panel.index[ok[0]].date() if len(ok) else None
    print(f'估计方案（逆波动率/最小方差/风险平价）开始使用自身权重的首个月末：{first_est_date}'
          f'（此前估计窗口不足252日，按等权过渡）')
    if first_est_date is not None:
        for name in results:
            if name == '等权':
                continue
            w_est = results[name]['wa'][results[name]['wa'].index >= pd.Timestamp(first_est_date)]
            mean_w = w_est.mean()
            print('  ' + name + ' 估计期平均权重: '
                  + '，'.join(f'{c}={v:.0%}' for c, v in mean_w.items()))

    best_scheme = pm['夏普'].idxmax()
    best_single = pm.drop(index=['等权', '逆波动率', '最小方差', '风险平价代理'])
    print(f'夏普最高的组合方案：{best_scheme}（{pm.loc[best_scheme, "夏普"]:.3f}）')

    # ---- 图03 组合净值对比（叠加最佳单策略虚线参照）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1.4]})
    for i, name in enumerate(results):
        ax1.plot(results[name]['nav'], label=f'{name}（夏普{pm.loc[name, "夏普"]:.2f}）',
                 color=COL_S[i], lw=1.6)
    single_sharpes = {c: perf_stats(panel[c])['夏普'] for c in panel.columns}
    best_single_name = max(single_sharpes, key=single_sharpes.get)
    ax1.plot((1 + panel[best_single_name]).cumprod(), ls='--', color='gray', lw=1.0,
             label=f'最佳单策略参照：{best_single_name}（夏普{single_sharpes[best_single_name]:.2f}）')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylabel('净值（起点=1）')
    ax1.set_title(f'图03 四种配钱方式净值对比（月度再平衡、含子策略间调仓成本万3/单边，'
                  f'{panel.index[0].date()} ~ {panel.index[-1].date()}）')
    ax1.grid(alpha=0.3)
    for i, name in enumerate(results):
        nav = results[name]['nav']
        dd = nav / nav.cummax() - 1
        ax2.plot(dd, color=COL_S[i], lw=1.0)
    ax2.set_ylabel('回撤')
    ax2.set_xlabel('日期')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_组合方式净值对比.png'), dpi=150)
    plt.close(fig)

    # ---- 图04 权重演变（2×2 堆叠面积）
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (name, color) in zip(axes.ravel(), [(k, COL_S[i]) for i, k in enumerate(results)]):
        wa = results[name]['wa']
        ax.stackplot(wa.index, [wa[c].values for c in panel.columns],
                     labels=panel.columns, colors=COL_S, alpha=0.85)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc='lower left', fontsize=8, ncol=2)
    fig.suptitle('图04 四种配钱方式的子策略权重演变（月末再平衡；估计方案前252日按等权过渡）')
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, '04_组合权重演变.png'), dpi=150)
    plt.close(fig)
    return results, pm, best_scheme


# ================================================================ 第5步 回撤归因与危机相关性
def avg_pairwise_corr(win):
    c = win.corr().values
    k = c.shape[0]
    iu = np.triu_indices(k, k=1)
    return float(np.mean(c[iu]))


def section5(panel, results, pm):
    print('\n' + '=' * 72)
    print('第5步 回撤拼图：等权组合最大回撤期的子策略贡献分解 + 危机中相关性检查')
    print('=' * 72)
    nav_ew = results['等权']['nav']
    dd = nav_ew / nav_ew.cummax() - 1
    trough = dd.idxmin()
    peak = nav_ew.loc[:trough].idxmax()
    print(f'等权组合最大回撤：{dd.min():.2%}，区间 {peak.date()}（峰）→ {trough.date()}（谷）')

    wu = results['等权']['wu']
    win = panel.loc[peak:trough]
    wwin = wu.loc[peak:trough]
    contrib = (wwin * win).sum()                            # 各子策略贡献（日收益简单加总）
    cost_sum = results['等权']['cost'].loc[peak:trough].sum()
    nav_port = results['等权']['nav'].loc[peak:trough]
    port_cum = nav_port.iloc[-1] / nav_port.iloc[0] - 1
    print('回撤期内各子策略贡献（占账户收益的点数，日收益加总近似）：')
    for k, v in contrib.items():
        print(f'  {k}: {v:+.2%}')
    print(f'  调仓成本合计: {-cost_sum:+.2%}，组合区间净收益（复利）: {port_cum:+.2%}')
    biggest = contrib.idxmin()
    print(f'→ 回撤期主要"肇事者"：{biggest}（贡献 {contrib.min():+.2%}）')

    corr_win = avg_pairwise_corr(win)
    print(f'相关性检查：回撤窗口内平均两两相关 {corr_win:.3f} vs 全样本 {AVG_CORR:.3f} '
          f'（{"上升——危机中分散化打折" if corr_win > AVG_CORR else "未上升"}）')

    # 最深60个交易日的"危机段"：等权组合滚动60日对数收益最低的窗口
    r_ew = results['等权']['nav'].pct_change().fillna(0.0)
    roll60 = np.log1p(r_ew).rolling(60).sum()
    worst60 = {}
    if roll60.notna().any():
        t_end = roll60.idxmin()
        i = panel.index.get_loc(t_end)
        w0, w1 = panel.index[i - 59], t_end
        win60 = panel.loc[w0:w1]
        contrib60 = (results['等权']['wu'].loc[w0:w1] * win60).sum()
        corr60 = avg_pairwise_corr(win60)
        cum60 = float(np.expm1(roll60.min()))
        print(f'最深60日危机段：{w0.date()} ~ {w1.date()}，组合区间收益 {cum60:+.2%}，'
              f'平均两两相关 {corr60:.3f} vs 全样本 {AVG_CORR:.3f}')
        for k, v in contrib60.items():
            print(f'    {k} 贡献 {v:+.2%}')
        worst60 = dict(start=w0, end=w1, cum=cum60, corr=corr60, contrib=contrib60)

    # 2024年"924行情"普涨普跌事件窗
    s0, s1 = pd.Timestamp(STRESS_WIN[0]), pd.Timestamp(STRESS_WIN[1])
    ev = panel.loc[(panel.index >= s0) & (panel.index <= s1)]
    if len(ev) > 5:
        corr_ev = avg_pairwise_corr(ev)
        print(f'2024-09~10 事件窗（{ev.index[0].date()} ~ {ev.index[-1].date()}，{len(ev)}日）'
              f'平均两两相关 {corr_ev:.3f} vs 全样本 {AVG_CORR:.3f}')
        for c in panel.columns:
            cum = (1 + ev[c]).prod() - 1
            print(f'    {c} 事件窗累计 {cum:+.2%}')
    else:
        corr_ev = np.nan
        print('2024-09 事件窗：面板中无足够数据，跳过')

    # ---- 图05
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5),
                                   gridspec_kw={'height_ratios': [2.2, 1.3]})
    for i, c in enumerate(panel.columns):
        rebased = (1 + panel[c].loc[peak:trough]).cumprod()
        ax1.plot(rebased, color=COL_S[i], lw=1.4, label=c)
    ax1.plot(nav_port / nav_port.iloc[0], color='black', lw=2.2, label='等权组合')
    ax1.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax1.legend(fontsize=9)
    ax1.set_ylabel('窗口内净值（峰=1）')
    ax1.set_title(f'图05 等权组合最大回撤期拆解（{peak.date()} → {trough.date()}，'
                  f'组合回撤 {dd.min():.1%}）：谁在跌？相关性是否上升？')
    ax1.grid(alpha=0.3)

    bars = list(contrib.values) + [-cost_sum]
    labels = list(contrib.index) + ['调仓成本']
    colors = COL_S + ['gray']
    bars_arr = np.array(bars)
    ax2.bar(range(len(bars)), bars_arr, color=colors, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(range(len(bars)), labels, rotation=15, ha='right', fontsize=9)
    for x, v in enumerate(bars):
        ax2.text(x, v + (0.001 if v >= 0 else -0.003), f'{v:+.1%}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax2.set_ylabel('对账户收益的贡献')
    ax2.set_title('各子策略在回撤窗口的收益贡献（正=托底，负=拖累）')
    ax2.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_最大回撤期贡献分解.png'), dpi=150)
    plt.close(fig)
    return dict(peak=peak, trough=trough, mdd=dd.min(), contrib=contrib,
                corr_win=corr_win, corr_event=corr_ev, port_cum=port_cum, worst60=worst60)


# ================================================================ 第6步 图01 与落盘
def section6(panel, strategies, results, sm, pm, s5):
    # ---- 图01 四策略净值与回撤（共同区间）
    navs = (1 + panel).cumprod()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1.4]})
    for i, c in enumerate(panel.columns):
        st = sm.loc[c]
        ax1.plot(navs[c], color=COL_S[i], lw=1.5,
                 label=f'{c}（夏普{st["夏普"]:.2f}，回撤{st["最大回撤"]:.0%}）')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylabel('净值（起点=1）')
    ax1.set_title(f'图01 四条子策略净值与回撤（共同区间 {panel.index[0].date()} ~ '
                  f'{panel.index[-1].date()}，各子账户独立、含内部成本与2%现金利息）')
    ax1.grid(alpha=0.3)
    for i, c in enumerate(panel.columns):
        dd = navs[c] / navs[c].cummax() - 1
        ax2.plot(dd, color=COL_S[i], lw=1.0)
    ax2.set_ylabel('回撤')
    ax2.set_xlabel('日期')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_四策略净值与回撤.png'), dpi=150)
    plt.close(fig)

    # ---- CSV（utf-8-sig）
    sm_out = sm[['自身起始日', '年化收益', '年化波动', '夏普', '最大回撤',
                 '与其他策略平均相关', '全历史年化', '全历史夏普', '全历史最大回撤']]
    sm_out.round(4).to_csv(os.path.join(OUT_DIR, 'strategy_metrics.csv'), encoding='utf-8-sig')
    pm_out = pm[['年化收益', '年化波动', '夏普', '最大回撤', '年化成本拖累', '年均换手']]
    pm_out.round(4).to_csv(os.path.join(OUT_DIR, 'portfolio_metrics.csv'), encoding='utf-8-sig')
    print(f'\n已写出 CSV：outputs/strategy_metrics.csv、outputs/portfolio_metrics.csv')


# ================================================================ main
def main():
    print('=' * 72)
    print('34_策略组合：把四条策略装进一个账户（毕业设计）')
    print('=' * 72)
    panel, strategies = build_panel()

    global AVG_CORR
    sm, corr, avg_corr = section3(panel, strategies)
    AVG_CORR = avg_corr

    results, pm, best_scheme = section4(panel)
    s5 = section5(panel, results, pm)
    section6(panel, strategies, results, sm, pm, s5)

    print('\n' + '=' * 72)
    print('总结（全部为共同区间实际运行数字）')
    print('=' * 72)
    best_single = sm['夏普'].idxmax()
    ew_sh = pm.loc['等权', '夏普']
    print(f'1) 单策略夏普：' + '，'.join(f'{k}={v:.3f}' for k, v in sm['夏普'].items())
          + f'；最佳单策略 = {best_single}')
    print(f'2) 平均两两相关 = {AVG_CORR:.3f}；等权组合夏普 = {ew_sh:.3f} '
          f'（{"超过" if ew_sh > sm.loc[best_single, "夏普"] else "未超过"}最佳单策略 '
          f'{sm.loc[best_single, "夏普"]:.3f}），夏普最高的配钱方式 = {best_scheme}'
          f'（{pm.loc[best_scheme, "夏普"]:.3f}）')
    drag = pm['年化成本拖累']
    print(f'3) 年化成本拖累：' + '，'.join(f'{k}={v:.3%}' for k, v in drag.items()))
    print(f'4) 回撤归因：等权组合最大回撤 {s5["mdd"]:.2%}（{s5["peak"].date()}→'
          f'{s5["trough"].date()}），主拖累 = '
          + '，'.join(f'{k} {v:+.1%}' for k, v in s5['contrib'].items())
          + f'；窗口内平均两两相关 {s5["corr_win"]:.3f} vs 全样本 {AVG_CORR:.3f}')
    w60 = s5.get('worst60', {})
    if w60:
        print(f'   最深60日危机段 {w60["start"].date()}~{w60["end"].date()}: 组合 {w60["cum"]:+.2%}，'
              f'平均两两相关 {w60["corr"]:.3f}（全样本 {AVG_CORR:.3f}）')
    print(f'5) 数据来源：{"合成数据兜底" if SYNTH_FLAG else "data/cache 真实缓存"}；'
          f'成本口径：单边万3；现金利息：2%年化')
    print('完成。图表见 outputs/')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()          # 按工作区规范：绝不以非0退出；异常时打印堆栈供排查
        print('【脚本异常】以上为堆栈信息，请检查数据与依赖后重试。')
