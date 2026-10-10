# -*- coding: utf-8 -*-
"""
04_网格交易：用近10年上证指数真实数据，验证"网格赚波动钱、亏趋势钱"
=====================================================================
本课只做一件事：把"高抛低吸"机械化成一张价格网格，跑真实回测，
逐年对照网格与买入持有的收益，看看这句行话到底对不对。

运行（在本目录下）：
    /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py

产出（outputs/）：
    01_网格买卖点示意图.png  近2年买卖点 + 格线
    02_净值对比.png          网格 vs 买入持有 净值与回撤
    03_分年度收益对比.png     逐年收益柱状对比（核心证据）
    04_合成情景对比.png       震荡市/单边市 两条合成路径的网格表现
    metrics.csv              主回测+参数敏感性的绩效指标
    yearly_returns.csv       分年度收益表

数据：../data/cache/index_000001_上证指数.csv（真实日K，2010起）；
     若缓存缺失则用 np.random.default_rng(42) 生成合成数据兜底（图中会标注），绝不失败。
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境出图
import matplotlib.pyplot as plt

# 中文字体与负号显示（按课程规范设置）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ======================================================================
# 0. 参数区（全部可调，改这里即可）
# ======================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, '..', 'data', 'cache', 'index_000001_上证指数.csv'))
OUT_DIR = os.path.join(HERE, 'outputs')

CAPITAL = 1_000_000.0   # 初始资金（元）
N_GRID = 20             # 主线参数：网格数（敏感性实验另取 10 / 40）
COST_RATE = 3e-4        # 单边交易成本 万3（买入、卖出各收一次，即"双边成本万3"）
RF_ANNUAL = 0.02        # 无风险利率（夏普比率用 rf=2%，与本系列课01-03口径一致）
YEARS_BACK = 10         # 回测窗口：近10年
SENS_N = [10, 20, 40]   # 参数敏感性：格数三档

# ======================================================================
# 1. 数据加载：真实缓存优先，缺失则合成兜底
# ======================================================================
def load_close():
    """返回 (收盘价序列, 是否合成数据)。"""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', parse_dates=['date'])
        close = df.set_index('date')['close'].sort_index()
        start = close.index[-1] - pd.DateOffset(years=YEARS_BACK)
        return close.loc[close.index >= start], False
    # ---- 合成兜底：带正弦震荡的几何随机游走（rng(42)，图和README均标注"合成数据"）----
    rng = np.random.default_rng(42)
    n = 2400
    idx = pd.bdate_range('2016-09-19', periods=n)
    eps = rng.standard_normal(n)
    logp = np.log(3000.0) + np.cumsum(5e-5 + 0.012 * eps) + 0.10 * np.sin(2 * np.pi * np.arange(n) / 500)
    return pd.Series(np.exp(logp), index=idx, name='close'), True

# ======================================================================
# 2. 网格核心：格线数组 / 所在格编号 / 网格回测引擎
# ======================================================================
def make_grid_lines(p_low, p_high, n_grid):
    """等比网格：格线在对数价格上等距 → 每一格的涨跌幅相同。
    第 k 条格线 L_k = p_low * (p_high/p_low)^(k/n)，k=0..n（共 n+1 条线、n 个格子）。"""
    ratios = np.linspace(0.0, 1.0, n_grid + 1)
    return p_low * (p_high / p_low) ** ratios

def cell_of(price, log_low, log_step, n_grid):
    """价格所在的格编号 k∈[0, n-1]（0=最下面一格）。价格击穿区间后被夹在边界格。"""
    k = np.floor((np.log(price) - log_low) / log_step)
    return int(np.clip(k, 0, n_grid - 1))

def run_grid(close, n_grid=N_GRID, cost=COST_RATE, capital=CAPITAL):
    """
    网格回测（日频、收盘价撮合）：
    - 区间 [p_low, p_high] 取回测期内最低/最高收盘价，等比切 n_grid 格；
    - 每格资金 lot = capital/(n_grid+1)；目标持仓格数 = n_grid - 所在格编号：
      所在格编号 k=0 是最下面一格（价格最低）→ 持有 n 格仓位（≤ n/(n+1)，永不满仓）；
      k=n_grid-1 是最上面一格（价格最高）→ 只持有 1 格（永不满空）。
      即"价格越低仓位越重、越高仓位越轻"的低多高少。
    - 相邻两天"目标持仓格数"每变动 1 格 → 成交 1 格金额（价格下穿格线买入、上穿格线卖出），
      跳空穿过多根格线则按格数补足多笔成交；
    - 买卖均按成交额计 cost 单边成本。
    返回 (逐日DataFrame, meta字典)。
    """
    p = close.to_numpy(float)
    p_low, p_high = float(p.min()), float(p.max())
    if p_high <= p_low:                      # 极端兜底：价格完全不动
        p_high = p_low * 1.01
    lines = make_grid_lines(p_low, p_high, n_grid)
    log_low = np.log(p_low)
    log_step = np.log(p_high / p_low) / n_grid
    lot = capital / (n_grid + 1)             # 一格金额

    cash = capital                           # 现金
    units = 0.0                              # 持有指数"份额"，总资产 = cash + units*price
    lots_prev = None                         # 当前持仓格数（状态变量）
    trades = 0                               # 成交笔数（每格1笔，建仓记1笔）
    clamped = 0                              # 卖出保护触发次数（正常应为0）
    rows = []

    for i, (dt, price) in enumerate(close.items()):
        if i == 0:
            # 期初建仓：按期初价所在格确定目标格数（期初价在区间中点时≈一半现金一半底仓）
            k = cell_of(price, log_low, log_step, n_grid)
            lots = n_grid - k
            dl = 0
            units += lots * lot / price
            cash -= lots * lot * (1 + cost)
            trades += 1
        else:
            k = cell_of(price, log_low, log_step, n_grid)
            lots = n_grid - k                # 目标持仓格数：低多高少
            dl = lots - lots_prev
            if dl > 0:                       # 价格下穿格线 → 买入 dl 格金额
                units += dl * lot / price
                cash -= dl * lot * (1 + cost)
                trades += int(dl)
            elif dl < 0:                     # 价格上穿格线 → 卖出 -dl 格金额
                want = (-dl) * lot / price   # 想卖的份额
                sell = min(want, units)      # 安全夹：不许卖穿仓（正常不会触发）
                clamped += int(sell < want - 1e-12)
                units -= sell
                cash += sell * price * (1 - cost)
                trades += int(-dl)
        lots_prev = lots
        pos = units * price                  # 持仓市值
        rows.append((dt, cash + pos, cash, pos, pos / (cash + pos), k, dl, trades))

    w = pd.DataFrame(rows, columns=['date', 'wealth', 'cash', 'pos_value',
                                    'pos_ratio', 'cell', 'dk', 'trades']).set_index('date')
    meta = {'lines': lines, 'lot': lot, 'p_low': p_low, 'p_high': p_high,
            'grid_step': np.exp(log_step) - 1.0, 'trades': trades, 'clamped': clamped}
    return w, meta

def run_buy_hold(close, cost=COST_RATE, capital=CAPITAL):
    """买入持有基准：期初全仓买入并持有（扣一次买入成本）。"""
    p = close.to_numpy(float)
    units = capital / (1 + cost) / p[0]
    return pd.Series(units * p, index=close.index, name='buy_hold')

# ======================================================================
# 3. 绩效统计与分年度收益表
# ======================================================================
def perf_stats(wealth, trades=None, pos_ratio=None):
    """年化收益 / 年化波动 / 最大回撤 / 夏普 / 交易笔数 / 现金利用率 / 期末净值。

    夏普 = (年化收益 - rf) / 年化波动，rf = RF_ANNUAL = 2%（年化），
    与本系列课01-03口径一致（几何年化收益减无风险利率后除以年化波动）。
    """
    ret = wealth.pct_change().dropna()
    years = max((wealth.index[-1] - wealth.index[0]).days / 365.25, 1e-9)
    cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
    vol = float(ret.std() * np.sqrt(252))
    mdd = float(-(wealth / wealth.cummax() - 1.0).min())      # 最大回撤（正数，亏幅）
    sharpe = float((cagr - RF_ANNUAL) / vol) if vol > 0 else np.nan
    return {'年化收益': float(cagr), '年化波动': vol, '最大回撤': mdd, '夏普': sharpe,
            '交易笔数': int(trades) if trades is not None else 0,
            '现金利用率': float(pos_ratio.mean()) if pos_ratio is not None else 1.0,
            '期末净值': float(wealth.iloc[-1] / wealth.iloc[0]),
            '区间总收益': float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)}

def yearly_table(wg, wbh):
    """分年度收益表：网格 vs 买入持有（首年以回测起点为基期，首尾为部分年）。"""
    vy = pd.DataFrame({'网格': wg, '买入持有': wbh}).dropna()
    g = vy.groupby(vy.index.year)
    last, n_days = g.last(), g.size()
    ret = last / last.shift(1) - 1.0
    ret.iloc[0] = last.iloc[0] / vy.iloc[0] - 1.0             # 首年基期=回测起点
    out = pd.DataFrame({'网格收益': ret['网格'], '买入持有收益': ret['买入持有'],
                        '交易日数': n_days})
    out['超额(网格-买入持有)'] = out['网格收益'] - out['买入持有收益']
    out['备注'] = ''
    out.index.name = '年份'
    out.iloc[0, out.columns.get_loc('备注')] = f"部分年（自{vy.index[0]:%Y-%m-%d}起）"
    out.iloc[-1, out.columns.get_loc('备注')] = f"部分年（截至{vy.index[-1]:%Y-%m-%d}）"
    return out

# ======================================================================
# 4. 绘图
# ======================================================================
def plot_trades(res, meta, close, tag, path):
    """图01：近2年收盘价 + 格线 + 买卖点。"""
    w = res.loc[res.index >= res.index[-1] - pd.DateOffset(years=2)]
    px = close.loc[w.index]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(px.index, px.values, color='#1f77b4', lw=1.2, label='上证指数收盘价')
    lo, hi = px.min() * 0.99, px.max() * 1.01
    for L in meta['lines']:                                   # 画落在可视范围内的格线
        if lo <= L <= hi:
            ax.axhline(L, color='gray', lw=0.6, ls=':', alpha=0.8)
    buy = w[w['dk'] > 0]
    sell = w[w['dk'] < 0]
    ax.scatter(buy.index, px.loc[buy.index], marker='^', color='red', s=42,
               zorder=5, label=f'买入（下穿格线，{len(buy)}天）')
    ax.scatter(sell.index, px.loc[sell.index], marker='v', color='green', s=42,
               zorder=5, label=f'卖出（上穿格线，{len(sell)}天）')
    ax.set_title(f'近2年上证指数网格买卖点示意（N={N_GRID}格等比网格，格距{meta["grid_step"]*100:.2f}%，{tag}）')
    ax.set_ylabel('点位')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_nav(wg, wbh, tag, path):
    """图02：净值对比（上）+ 动态回撤（下）。"""
    nav_g, nav_b = wg / CAPITAL, wbh / CAPITAL
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1]})
    ax1.plot(nav_g.index, nav_g.values, color='#d62728', lw=1.4, label='网格策略')
    ax1.plot(nav_b.index, nav_b.values, color='#7f7f7f', lw=1.2, ls='--', label='买入持有')
    ax1.axhline(1.0, color='k', lw=0.6, alpha=0.5)
    ax1.annotate(f'期末净值：网格 {nav_g.iloc[-1]:.3f} vs 买入持有 {nav_b.iloc[-1]:.3f}',
                 xy=(0.36, 0.94), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    ax1.set_title(f'近10年上证指数：网格 vs 买入持有（{tag}，双边成本万3）')
    ax1.set_ylabel('净值（期初=1）')
    ax1.legend(loc='upper left')
    dd_g = nav_g / nav_g.cummax() - 1
    dd_b = nav_b / nav_b.cummax() - 1
    ax2.fill_between(dd_g.index, dd_g.values * 100, 0, color='#d62728', alpha=0.35, label='网格回撤')
    ax2.plot(dd_b.index, dd_b.values * 100, color='#7f7f7f', lw=1.0, ls='--', label='买入持有回撤')
    ax2.set_ylabel('动态回撤 %')
    ax2.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_yearly(yt, path):
    """图03：分年度收益对比柱状图（核心证据）。"""
    labels = [f"{y}*" if str(r).strip() else str(y) for y, r in yt['备注'].items()]
    x = np.arange(len(yt))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, yt['网格收益'] * 100, width=0.4, color='#d62728', label='网格')
    ax.bar(x + 0.2, yt['买入持有收益'] * 100, width=0.4, color='#7f7f7f', label='买入持有')
    for xi, v in zip(x - 0.2, yt['网格收益'] * 100):
        ax.annotate(f'{v:+.1f}', (xi, v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=7.5, color='#d62728')
    for xi, v in zip(x + 0.2, yt['买入持有收益'] * 100):
        ax.annotate(f'{v:+.1f}', (xi, v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=7.5, color='#555555')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_title('分年度收益：网格 vs 买入持有（*为部分年；震荡/下跌年网格占优、单边牛市网格跑输）')
    ax.set_ylabel('年度收益 %')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_scenarios(scn, path):
    """图04：合成情景对比（震荡 vs 单边）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, (wg, wbh)) in zip(axes, scn.items()):
        ax.plot(wg.index, wg.values / CAPITAL, color='#d62728', lw=1.4, label='网格')
        ax.plot(wbh.index, wbh.values / CAPITAL, color='#7f7f7f', lw=1.2, ls='--', label='买入持有')
        ax.axhline(1.0, color='k', lw=0.6, alpha=0.5)
        ax.set_title(name)
        ax.set_ylabel('净值（期初=1）')
        ax.annotate(f'网格 {wg.iloc[-1]/CAPITAL:.3f}\n买入持有 {wbh.iloc[-1]/CAPITAL:.3f}',
                    xy=(0.98, 0.04), xycoords='axes fraction', ha='right', va='bottom', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('合成情景实验：网格的"适用行情"（合成数据，np.random.default_rng(42)，非真实行情）')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ======================================================================
# 5. 情景实验：两条合成路径（rng(42)）
# ======================================================================
def make_scenarios():
    """合成两条路径（各约3年、750个交易日）：
    A 均值回归震荡：OU过程围绕3000点反复拉回；
    B 单边上涨：恒定正漂移+噪声，一路向右上。"""
    rng = np.random.default_rng(42)
    n = 750
    eps = rng.standard_normal((2, n))
    idx = pd.bdate_range('2024-01-02', periods=n)
    # 情景A：x_{t+1} = 0.93*x_t + 0.018*eps → 围绕0均值回归（对数价格）
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.93 * x[t - 1] + 0.018 * eps[0, t]
    path_a = pd.Series(3000.0 * np.exp(x), index=idx)
    # 情景B：每日对数漂移 +0.0008 → 3年约+80%
    logb = np.log(3000.0) + np.cumsum(0.0008 + 0.011 * eps[1])
    path_b = pd.Series(np.exp(logb), index=idx)
    return {'情景A：均值回归震荡（合成）': path_a, '情景B：单边上涨（合成）': path_b}

# ======================================================================
# 6. 主流程
# ======================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    close, is_synth = load_close()
    tag = '合成数据rng(42)兜底' if is_synth else '真实数据'

    print('=' * 72)
    print(f'04_网格交易 | 数据：{"合成数据(default_rng(42)兜底)" if is_synth else "真实上证指数日K"}'
          f' | 窗口 {close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}（{len(close)}个交易日）')
    print('=' * 72)

    # ---- 主回测：N=20 网格 vs 买入持有 --------------------------------
    res, meta = run_grid(close, n_grid=N_GRID)
    if meta['clamped']:
        print(f'[提示] 卖出保护触发 {meta["clamped"]} 次（不影响主结论）')
    wbh = run_buy_hold(close)
    k0 = int(res['cell'].iloc[0])
    print(f'\n网格区间 [{meta["p_low"]:.0f}, {meta["p_high"]:.0f}]，'
          f'等比{N_GRID}格，每格涨幅 {meta["grid_step"]*100:.2f}%，'
          f'每格资金 {meta["lot"]:,.0f} 元')
    print(f'期初建仓：期初价位于自下往上第{k0 + 1}/{N_GRID}格 → 持有{N_GRID - k0}格、'
          f'仓位 {(N_GRID - k0) / (N_GRID + 1) * 100:.1f}%'
          f'（期初价在区间中点时恰为50%，偏低则多于一半——网格"低多高少"）')

    rows = []
    st_g = perf_stats(res['wealth'], meta['trades'], res['pos_ratio'])
    st_b = perf_stats(wbh, 1, pd.Series(1.0, index=wbh.index))
    rows.append({'策略': f'网格(N={N_GRID})', '格数N': N_GRID, **st_g})
    rows.append({'策略': '买入持有', '格数N': np.nan, **st_b})

    print('\n[主回测] 近10年绩效对比')
    print(pd.DataFrame(rows[:2]).set_index('策略').to_string(float_format=lambda v: f'{v:.4f}'))

    # ---- 参数敏感性：格数 10 / 20 / 40 --------------------------------
    print('\n[参数敏感性] 格数 N = 10 / 20 / 40')
    sens_rows = []
    for n in SENS_N:
        w_n, m_n = run_grid(close, n_grid=n)
        st_n = perf_stats(w_n['wealth'], m_n['trades'], w_n['pos_ratio'])
        sens_rows.append({'策略': f'网格(N={n})', '格数N': n, **st_n})
        print(f"  N={n:>2}: 年化 {st_n['年化收益']*100:+6.2f}% | 最大回撤 {st_n['最大回撤']*100:5.2f}%"
              f" | 交易 {st_n['交易笔数']:>4} 笔 | 现金利用率 {st_n['现金利用率']*100:5.1f}%")
    rows += sens_rows

    # ---- 分年度收益表（核心证据）--------------------------------------
    yt = yearly_table(res['wealth'], wbh)
    print('\n[分年度收益] 网格 vs 买入持有（核心证据：震荡/熊年网格占优，单边牛年网格跑输）')
    show = yt.copy()
    for c in ['网格收益', '买入持有收益', '超额(网格-买入持有)']:
        show[c] = (show[c] * 100).map(lambda v: f'{v:+.2f}%')
    print(show.to_string())
    n_win = int((yt['超额(网格-买入持有)'] > 0).sum())
    print(f'网格超额为正的年份：{n_win}/{len(yt)}；'
          f'赢在：{list(yt.index[yt["超额(网格-买入持有)"] > 0])}；'
          f'输在：{list(yt.index[yt["超额(网格-买入持有)"] <= 0])}')

    # ---- 情景实验（合成数据）------------------------------------------
    print('\n[情景实验] 合成路径（np.random.default_rng(42)，非真实行情）')
    scn_paths = make_scenarios()
    scn_nav, scn_rows = {}, []
    for name, path in scn_paths.items():
        w_s, m_s = run_grid(path)
        b_s = run_buy_hold(path)
        scn_nav[name] = (w_s['wealth'], b_s)
        st = perf_stats(w_s['wealth'], m_s['trades'], w_s['pos_ratio'])
        sb = perf_stats(b_s, 1, pd.Series(1.0, index=b_s.index))
        scn_rows.append({'情景': name, '网格年化': st['年化收益'], '买入持有年化': sb['年化收益'],
                         '网格回撤': st['最大回撤'], '买入持有回撤': sb['最大回撤'],
                         '网格交易笔数': st['交易笔数']})
        print(f"  {name}: 网格年化 {st['年化收益']*100:+6.2f}% vs 买入持有 {sb['年化收益']*100:+6.2f}%"
              f" | 回撤 {st['最大回撤']*100:.2f}% vs {sb['最大回撤']*100:.2f}%"
              f" | 交易 {st['交易笔数']} 笔")

    # ---- 落盘 ---------------------------------------------------------
    metrics = pd.DataFrame(rows)[['策略', '格数N', '年化收益', '年化波动', '最大回撤',
                                  '夏普', '交易笔数', '现金利用率', '期末净值', '区间总收益']]
    metrics.round(6).to_csv(os.path.join(OUT_DIR, 'metrics.csv'), encoding='utf-8-sig', index=False)
    yt.round(6).to_csv(os.path.join(OUT_DIR, 'yearly_returns.csv'), encoding='utf-8-sig')

    plot_trades(res, meta, close, tag, os.path.join(OUT_DIR, '01_网格买卖点示意图.png'))
    plot_nav(res['wealth'], wbh, tag, os.path.join(OUT_DIR, '02_净值对比.png'))
    plot_yearly(yt, os.path.join(OUT_DIR, '03_分年度收益对比.png'))
    plot_scenarios(scn_nav, os.path.join(OUT_DIR, '04_合成情景对比.png'))

    print('\n完成：metrics.csv / yearly_returns.csv 与 4 张图已写入 outputs/')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
