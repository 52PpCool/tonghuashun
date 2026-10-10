#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
03 海龟交易法则 —— 把交易写成机器可执行的规则（教学版，事件循环回测）
================================================================================

本课复刻 Richard Dennis 与 William Eckhardt 在 1983 年"海龟交易员"实验中
教授的系统一（System 1，短周期版）：

    入场：收盘价突破"前 20 日最高价"（20 日唐奇安通道上轨）
    离场：收盘价跌破"前 10 日最低价"（10 日通道下轨），或触发 2N 止损
    仓位：波动率倒数仓位 —— 每个单位只让账户承担 1% 的风险
    加仓：价格每上涨 0.5N 加一个单位，最多 4 个单位；加仓后止损整体上移
    N 值：ATR(20)，即"真实波动幅度"的 20 日均值

与海龟原版的三处刻意简化（README 有详细说明）：
    1) 收盘价产生信号、次日开盘价执行，替代原版"盘中触线立即成交"；
       这同时规避了前视偏差，并天然满足 A 股 T+1（当天买的当天不卖）；
    2) 简化掉了"跳过第一次突破"过滤器；
    3) A 股现货不能加杠杆：单个单位超出可用现金时，按现金上限截断
       （代码中标注"现金上限"）。

数据：data/cache/stock_600519_贵州茅台.csv（2015 年起，前复权日K）
运行：
    cd /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/03_海龟交易法则
    /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
输出：outputs/ 下 3 张图 + metrics.csv + trades.csv
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面后端：只写 PNG 文件，不弹窗口（服务器/批处理必备）
import matplotlib.pyplot as plt

# 中文显示：先 Noto Sans CJK SC，退化到文泉驿微米黑；负号用 ASCII 显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 参数区：所有"旋钮"集中在这里，改完重跑即可做敏感性分析
# =====================================================================
HERE      = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / 'data' / 'cache' / 'stock_600519_贵州茅台.csv'
OUT_DIR   = HERE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

INIT_CASH  = 1_000_000   # 初始资金（元）
RISK_PCT   = 0.01        # 每个单位的风险预算：账户权益的 1%（海龟原版设定）
ENTRY_WIN  = 20          # 入场通道：前 20 日最高价（唐奇安上轨）
EXIT_WIN   = 10          # 离场通道：前 10 日最低价（唐奇安下轨）
N_WINDOW   = 20          # N 值 = ATR(20)
ADD_STEP   = 0.5         # 价格每上涨 0.5N 加仓一个单位
MAX_UNITS  = 4           # 最多加到 4 个单位
STOP_MULT  = 2.0         # 止损距离 = 2N（最新一个单位的入场价 - 2N）
COMMISSION = 2.5e-4      # 佣金：万 2.5，双边收取
STAMP_TAX  = 5.0e-4      # 印花税：千 0.5（2023-08 下调后的税率），仅卖出收取
SLIPPAGE   = 5.0e-4      # 滑点：万 5，双边（买入成交价上浮 / 卖出下浮）
BOARD_LOT  = 100         # A 股一手 = 100 股，买入须为 100 股整数倍
RF_ANNUAL  = 0.02        # 无风险利率（夏普比率用 rf=2%）
PLOT_YEARS = 3           # 图 01 展示最近 3 年的买卖点

# =====================================================================
# 2. 数据与指标
# =====================================================================
def load_data(path: Path) -> pd.DataFrame:
    """读取前复权日K缓存（utf-8-sig 带 BOM；列：date,open,high,low,close,volume）。"""
    df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df[['open', 'high', 'low', 'close']]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加海龟三大核心指标（全部只用"截至昨天"的数据，杜绝未来函数）：

    N       = ATR(20)：真实波幅 TR 的 20 日简单均值。
              TR = max(当日高-低, |当日高-昨收|, |当日低-昨收|)。
              （海龟原版用 Wilder 指数平滑，此处为教学清晰用滚动均值，差异很小）
    don_up  = 前 20 日最高价（不含今日）——突破它入场
    don_low = 前 10 日最低价（不含今日）——跌破它离场
    两条通道都 .shift(1)：若不含 shift，今日价格会把自己顶进通道，形成前视偏差。
    """
    out = df.copy()
    prev_close = out['close'].shift(1)
    tr = pd.concat([
        out['high'] - out['low'],
        (out['high'] - prev_close).abs(),
        (out['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)                      # 真实波幅 TR
    out['N'] = tr.rolling(N_WINDOW).mean()      # N 值 = ATR(20)
    out['don_up'] = out['high'].rolling(ENTRY_WIN).max().shift(1)
    out['don_low'] = out['low'].rolling(EXIT_WIN).min().shift(1)
    return out


# ---- 交易成本（参数化，双边）----------------------------------------
def buy_fill_price(p):  return p * (1 + SLIPPAGE)              # 买入实际成交价：上浮滑点
def sell_fill_price(p): return p * (1 - SLIPPAGE)              # 卖出实际成交价：下浮滑点
def buy_fee(amount):    return amount * COMMISSION             # 买入：仅佣金
def sell_fee(amount):   return amount * (COMMISSION + STAMP_TAX)  # 卖出：佣金 + 印花税


# =====================================================================
# 3. 状态机 —— 海龟交易员的"大脑"
#
#    状态转移图：
#
#      [FLAT 空仓] --收盘突破20日上轨--> [LONG 持仓·第1单位]
#         ^                                  |
#         |                        价格较最新入场价再涨 0.5N
#         |                                  v
#         +----- 卖出全部 ------------ [LONG 第2/3/4单位·加仓]
#
#    离场信号（任一触发，次日开盘全部卖出）：
#      ① 收盘价 < 止损价（最新入场价 - 2N）   → "2N止损"
#      ② 收盘价 < 前 10 日最低价（趋势失效）  → "10日通道下破"
#
#    时序安排（防前视偏差 + 满足 T+1）：
#      每天开盘 OPEN：执行昨天收盘生成的挂单 pending
#      每天收盘 CLOSE：用当日数据评估信号 → 写入 pending（次日才执行）
#      于是"今天买的，最早明天开盘才可能卖"，T+1 天然满足。
# =====================================================================
STATE_FLAT = 'FLAT'   # 空仓：等待 20 日上轨突破
STATE_LONG = 'LONG'   # 持仓：管理金字塔加仓与 2N 止损


class TurtleTrader:
    """一只"海龟"：只交易单一标的（本课为 600519 贵州茅台）。"""

    def __init__(self):
        self.state = STATE_FLAT        # 当前状态机状态
        self.cash = INIT_CASH          # 现金（元）
        self.shares = 0                # 当前持股总数
        self.units = 0                 # 已建立的单位数（1~4）
        self.last_entry_price = 0.0    # 最新一个单位的入场价（加仓/止损的锚点）
        self.n0 = 0.0                  # 本次交易开仓时的 N 值（0.5N 加仓间距、2N 止损都用它）
        self.stop_price = 0.0          # 当前止损价（加仓后整体上移）
        self.pending = None            # 次日开盘要执行的订单（信号与执行分离的关键）
        self.trades = []               # 已完结交易列表
        self._open_trade = None        # 正在进行的交易
        self.equity_curve = {}         # 日期 -> 每日收盘按市值计的总权益

    # ---------- 账户 ----------
    def equity(self, price):
        """按某价格给持仓估市值：总权益 = 现金 + 持股市值。"""
        return self.cash + self.shares * price

    # ---------- 仓位：波动率倒数 ----------
    def _unit_shares(self, equity_val, fill_px):
        """一个"单位"应买多少股（海龟仓位公式的股票版）：

            Unit = 账户权益的 1% / (N × 每股波动价值)

        股票的"每股波动价值"就是 N 本身（股价每波动 1N，每股盈亏 N 元），所以：

            股数 = 0.01 × 权益 / N

        直觉：N 大 = 这只票最近波动猛 = 少买；N 小 = 波动温和 = 多买。
        效果：任何一个单位，价格反向走 1N，账户大约亏 1% ——
        每笔交易给组合贡献的"风险"几乎相同，这就是"波动率倒数仓位"。

        A 股现货不能加杠杆：若一个单位超出可用现金，按现金能买到的上限截断
        （海龟原版是期货保证金交易，1 个单位名义价值可以远超账户，此处必须封顶）。
        """
        budget = RISK_PCT * equity_val            # 本单位的风险预算（元）
        raw = budget / self.n0                    # 理论股数 = 预算 / N
        lots = int(raw // BOARD_LOT) * BOARD_LOT  # 向下取整到 100 股
        # 现金上限：把佣金滑点算进去，最多买到现金允许的手数
        per_cost = fill_px * (1 + COMMISSION)
        affordable = int(self.cash / per_cost // BOARD_LOT) * BOARD_LOT
        return max(min(lots, affordable), 0)      # 0 股 = 现金不足，放弃该单位

    # ---------- 执行：买入（开盘） ----------
    def _do_buy(self, dt, open_px, kind):
        """执行开仓/加仓单。kind: 'OPEN'=第1单位, 'ADD'=加仓单位。"""
        px = buy_fill_price(open_px)              # 买入实际成交价（含滑点）
        sh = self._unit_shares(self.equity(open_px), px)
        if sh <= 0:                               # 现金上限触发：放弃该单位
            return
        amount = sh * px
        fee = buy_fee(amount)
        self.cash -= amount + fee
        self.shares += sh
        self.units += 1
        self.last_entry_price = px
        # 止损整体上移：最新入场价 - 2N（每个单位共用同一条止损线）
        self.stop_price = px - STOP_MULT * self.n0
        # 记录止损线演变（按笔存储，供图03绘制阶梯线）
        self._open_trade['stop_history'].append((dt, self.stop_price))
        self._open_trade['fills'].append(
            {'date': dt, 'price': px, 'shares': sh, 'fee': fee, 'kind': kind})

    # ---------- 执行：卖出全部（开盘） ----------
    def _do_sell_all(self, dt, open_px, reason):
        """离场：次日开盘全部卖出（含滑点、佣金、印花税），状态机回到 FLAT。"""
        px = sell_fill_price(open_px)
        amount = self.shares * px
        fee = sell_fee(amount)                    # 卖出：佣金 + 印花税
        self.cash += amount - fee
        t = self._open_trade
        t['exit_date'] = dt
        t['exit_price'] = px
        t['exit_reason'] = reason
        t['invest'] = sum(f['price'] * f['shares'] + f['fee'] for f in t['fills'])
        t['proceeds'] = amount - fee
        t['pnl'] = t['proceeds'] - t['invest']
        t['units'] = self.units
        t['total_shares'] = self.shares
        self.trades.append(t)
        # 状态机复位：LONG -> FLAT
        self.state = STATE_FLAT
        self.shares = 0
        self.units = 0
        self.stop_price = 0.0
        self.last_entry_price = 0.0
        self._open_trade = None

    # ---------- 开盘：执行昨日信号 ----------
    def execute_orders(self, dt, bar):
        """事件循环第①步：开盘执行昨天收盘写入的挂单。"""
        if self.pending is None:
            return
        order, self.pending = self.pending, None
        if order['type'] == 'SELL_ALL':
            if self.state == STATE_LONG:          # 防御：空仓时不应有卖单
                self._do_sell_all(dt, bar['open'], order['reason'])
        else:                                     # BUY_OPEN / BUY_ADD
            if self.state == STATE_FLAT:          # 空仓 → 第 1 单位，开一笔新交易
                self.state = STATE_LONG
                self._open_trade = {'fills': [], 'n0': self.n0, 'stop_history': []}
                self._do_buy(dt, bar['open'], 'OPEN')
            else:                                 # 持仓 → 加仓单位
                self._do_buy(dt, bar['open'], 'ADD')

    # ---------- 收盘：评估信号，生成明日订单 ----------
    def evaluate_signals(self, dt, bar):
        """事件循环第③步：收盘评估当日信号 → 写入 pending（次日开盘执行）。"""
        c = bar['close']
        if self.state == STATE_LONG:
            # ① 止损优先：收盘跌破止损价（最新入场价 - 2N）→ 全部离场
            if c < self.stop_price:
                self.pending = {'type': 'SELL_ALL', 'reason': '2N止损'}
            # ② 趋势失效：收盘跌破前 10 日最低价 → 全部离场
            elif c < bar['don_low']:
                self.pending = {'type': 'SELL_ALL', 'reason': '10日通道下破'}
            # ③ 金字塔加仓：收盘较最新入场价再涨 0.5N 且未满 4 单位 → 加 1 单位
            elif (self.units < MAX_UNITS
                  and c >= self.last_entry_price + ADD_STEP * self.n0):
                self.pending = {'type': 'BUY_ADD'}
        else:
            # 空仓：收盘突破前 20 日最高价 → 开第 1 个单位
            if c > bar['don_up']:
                self.n0 = bar['N']                # 记录信号日的 N：定仓位、定间距、定止损
                self.pending = {'type': 'BUY_OPEN'}


# =====================================================================
# 4. 回测主循环（事件循环：开盘执行 → 收盘估值 → 收盘出信号）
# =====================================================================
def run_backtest(df: pd.DataFrame, start_idx: int) -> TurtleTrader:
    trader = TurtleTrader()
    for i in range(start_idx, len(df)):
        dt, bar = df.index[i], df.iloc[i]
        trader.execute_orders(dt, bar)                    # ① 开盘：执行昨日信号
        trader.equity_curve[dt] = trader.equity(bar['close'])  # ② 收盘：按市价记权益
        trader.evaluate_signals(dt, bar)                  # ③ 收盘：评估信号 → 明日订单
    return trader


def buy_hold_equity(df: pd.DataFrame, start_idx: int) -> pd.Series:
    """买入持有基准：期初全仓买入（含佣金滑点），持有到期末（按收盘价估值，不换手）。"""
    px = buy_fill_price(df['close'].iloc[start_idx])
    lots = int(INIT_CASH / (px * (1 + COMMISSION)) // BOARD_LOT) * BOARD_LOT
    cash_left = INIT_CASH - lots * px - buy_fee(lots * px)
    eq = cash_left + lots * df['close'].iloc[start_idx:]
    eq.name = '买入持有'
    return eq


# =====================================================================
# 5. 绩效统计
# =====================================================================
def perf_stats(equity: pd.Series) -> dict:
    """由每日权益曲线计算：总收益 / 年化收益 / 年化波动 / 夏普(rf=2%) / 最大回撤。"""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    daily = equity.pct_change().dropna()
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    return {'总收益': total_ret, '年化收益': ann_ret, '年化波动': ann_vol,
            '夏普(rf=2%)': sharpe, '最大回撤': max_dd}


def trade_stats(trades: list) -> dict:
    """交易层面的统计：交易次数 / 胜率 / 盈亏比（平均盈利 ÷ 平均亏损的绝对值）。"""
    n = len(trades)
    if n == 0:
        return {'交易次数': 0, '胜率': np.nan, '盈亏比': np.nan}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    plr = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    return {'交易次数': n, '胜率': len(wins) / n, '盈亏比': plr}


def trades_to_frame(trades: list) -> pd.DataFrame:
    """把交易列表整理成 trades.csv：每笔交易一行（入场/加仓/离场/盈亏）。"""
    rows = []
    for i, t in enumerate(trades, 1):
        fills, adds = t['fills'], t['fills'][1:]
        rows.append({
            'trade_id': i,
            'entry_date': str(fills[0]['date'].date()),
            'entry_price': round(fills[0]['price'], 2),   # 含滑点的成交价
            'N_入场': round(t['n0'], 2),
            'units': t['units'],
            '加仓日期': '|'.join(str(f['date'].date()) for f in adds),
            '加仓价': '|'.join(f"{f['price']:.2f}" for f in adds),
            '加仓股数': '|'.join(str(f['shares']) for f in adds),
            'total_shares': t['total_shares'],
            'avg_cost': round(t['invest'] / t['total_shares'], 2),  # 含费成本
            '最终止损价': round(t['fills'][-1]['price'] - STOP_MULT * t['n0'], 2),
            'exit_date': str(t['exit_date'].date()),
            'exit_price': round(t['exit_price'], 2),
            'exit_reason': t['exit_reason'],
            '投入成本(含费)': round(t['invest'], 2),
            '卖出净得(扣费)': round(t['proceeds'], 2),
            'pnl': round(t['pnl'], 2),
            'ret_pct': round(t['pnl'] / t['invest'] * 100, 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 6. 绘图
# =====================================================================
def _collect_markers(trades):
    """从交易记录里提取买卖点：买单 = 每个单位的成交，卖单 = 离场。"""
    buys, sells = [], []
    for t in trades:
        for k, f in enumerate(t['fills'], 1):
            buys.append((f['date'], f['price'], k))
        sells.append((t['exit_date'], t['exit_price'], t['exit_reason']))
    return buys, sells


def plot01(df, trades):
    """图01：最近 3 年的价格 + 唐奇安通道 + 买卖点。"""
    start = df.index[-1] - pd.DateOffset(years=PLOT_YEARS)
    win = df.loc[start:]
    buys, sells = _collect_markers(trades)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(win.index, win['close'], lw=1.2, color='#1f77b4', label='收盘价（前复权）')
    ax.plot(win.index, win['don_up'], lw=1.0, ls='--', color='#ff7f0e',
            label=f'{ENTRY_WIN}日唐奇安上轨（入场参考）')
    ax.plot(win.index, win['don_low'], lw=1.0, ls=':', color='#2ca02c',
            label=f'{EXIT_WIN}日通道下轨（离场参考）')
    for d, p, k in buys:
        if d >= win.index[0]:
            ax.scatter(d, p, marker='^', s=90, color='red', zorder=5)
            ax.annotate(f'买U{k}', (d, p), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=8, color='darkred')
    for d, p, reason in sells:
        if d >= win.index[0]:
            ax.scatter(d, p, marker='v', s=90, color='green', zorder=5)
            ax.annotate('卖', (d, p), textcoords='offset points',
                        xytext=(0, -14), ha='center', fontsize=8, color='darkgreen')
    ax.set_title(f'海龟系统一买卖点（贵州茅台600519，近{PLOT_YEARS}年，前复权）')
    ax.set_ylabel('价格（元）')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '01_价格与唐奇安通道买卖点.png', dpi=150)
    plt.close(fig)


def plot02(eq_strat, eq_bh):
    """图02：净值对比（上）+ 回撤对比（下）。"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(eq_strat.index, eq_strat.values / INIT_CASH, lw=1.5, color='#d62728',
             label='海龟策略（系统一）')
    ax1.plot(eq_bh.index, eq_bh.values / INIT_CASH, lw=1.2, ls='--', color='#7f7f7f',
             label='买入持有')
    ax1.axhline(1.0, color='black', lw=0.8, alpha=0.5)
    ax1.set_title('净值对比：海龟策略 vs 买入持有（初始资金=100万，含佣金/印花税/滑点）')
    ax1.set_ylabel('净值（期初=1）')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    for eq, name, color in ((eq_strat, '海龟策略', '#d62728'), (eq_bh, '买入持有', '#7f7f7f')):
        dd = (eq / eq.cummax() - 1) * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.35, color=color, label=name)
    ax2.set_title('回撤（水下深度，%）')
    ax2.set_ylabel('回撤 %')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '02_净值对比.png', dpi=150)
    plt.close(fig)


def plot03(df, trade):
    """图03：代表性交易的细节 —— 价格、通道、加仓点、2N止损线的整体上移。"""
    idx = df.index
    i0 = max(idx.get_loc(trade['fills'][0]['date']) - 15, 0)
    i1 = min(idx.get_loc(trade['exit_date']) + 15, len(idx) - 1)
    win = df.iloc[i0:i1 + 1]
    fig, ax = plt.subplots(figsize=(13, 7))
    # 持股期底色
    ax.axvspan(trade['fills'][0]['date'], trade['exit_date'],
               color='gold', alpha=0.10, label='持股区间')
    ax.plot(win.index, win['close'], lw=1.4, color='#1f77b4', label='收盘价（前复权）')
    ax.plot(win.index, win['don_up'], lw=1.0, ls='--', color='#ff7f0e',
            label='20日唐奇安上轨')
    ax.plot(win.index, win['don_low'], lw=1.0, ls=':', color='#2ca02c',
            label='10日通道下轨')
    # 加仓点（第1单位也是"加仓序列"的起点）
    for k, f in enumerate(trade['fills'], 1):
        ax.scatter(f['date'], f['price'], marker='^', s=110, color='red', zorder=5)
        ax.annotate(f'第{k}单位 {f["shares"]}股\n@{f["price"]:.0f}',
                    (f['date'], f['price']), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=9, color='darkred')
    # 2N 止损线演变：阶梯线，每加仓一次整体上移一格
    xs = [d for d, _ in trade['stop_history']] + [trade['exit_date']]
    ys = [s for _, s in trade['stop_history']] + [trade['stop_history'][-1][1]]
    ax.step(xs, ys, where='post', lw=1.8, ls='--', color='red',
            label='2N止损线（加仓后整体上移）')
    # 离场点
    ax.scatter(trade['exit_date'], trade['exit_price'], marker='v', s=110,
               color='green', zorder=5)
    ax.annotate(f'离场：{trade["exit_reason"]}\n@{trade["exit_price"]:.0f}',
                (trade['exit_date'], trade['exit_price']), textcoords='offset points',
                xytext=(0, -30), ha='center', fontsize=9, color='darkgreen')
    ax.set_title(
        f'代表性交易细节：{trade["fills"][0]["date"].date()} 入场，'
        f'{trade["units"]}个单位，{trade["exit_date"].date()} '
        f'"{trade["exit_reason"]}"离场，盈亏 {trade["pnl"]:+,.0f} 元'
        f'（{trade["pnl"] / trade["invest"] * 100:+.1f}%）')
    ax.set_ylabel('价格（元）')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '03_单笔交易加仓与止损细节.png', dpi=150)
    plt.close(fig)


# =====================================================================
# 7. 主流程
# =====================================================================
def main():
    print('=' * 68)
    print('03 海龟交易法则（系统一）—— 贵州茅台 600519，前复权日K 回测')
    print('=' * 68)

    df = add_indicators(load_data(DATA_PATH))
    print(f'数据区间：{df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日')

    # 有效指标从第 ~21 天开始；信号次日执行，故再 +1
    valid = df.dropna(subset=['N', 'don_up', 'don_low'])
    start_idx = df.index.get_loc(valid.index[0]) + 1
    print(f'回测起点：{df.index[start_idx].date()}（指标预热 20 日后）\n')

    # ---- 跑策略与基准 ----
    trader = run_backtest(df, start_idx)
    eq_strat = pd.Series(trader.equity_curve).sort_index()
    eq_bh = buy_hold_equity(df, start_idx)

    # ---- 绩效表 ----
    metrics = pd.DataFrame([
        {'组合': '海龟策略(系统一)', **perf_stats(eq_strat), **trade_stats(trader.trades)},
        {'组合': '买入持有', **perf_stats(eq_bh),
         '交易次数': np.nan, '胜率': np.nan, '盈亏比': np.nan},
    ])
    metrics.to_csv(OUT_DIR / 'metrics.csv', index=False, encoding='utf-8-sig')

    # ---- 交易明细 ----
    trades_df = trades_to_frame(trader.trades)
    trades_df.to_csv(OUT_DIR / 'trades.csv', index=False, encoding='utf-8-sig')

    # ---- 三张图 ----
    plot01(df, trader.trades)
    plot02(eq_strat, eq_bh)
    rep = None
    if trader.trades:
        rep = max(trader.trades, key=lambda t: (t['units'], t['pnl']))
        plot03(df, rep)

    # ---- 控制台报告 ----
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(metrics.to_string(index=False))
    print(f"\n期末权益：策略 {eq_strat.iloc[-1]:,.0f} 元 | 买入持有 {eq_bh.iloc[-1]:,.0f} 元"
          f"（初始 {INIT_CASH:,} 元）")
    print(f"完成交易：{len(trader.trades)} 笔")
    if trader.trades:
        reasons = pd.Series([t['exit_reason'] for t in trader.trades]).value_counts()
        print('离场原因分布：', dict(reasons))
        wins = [t for t in trader.trades if t['pnl'] > 0]
        print(f"盈利笔数 {len(wins)}，亏损笔数 {len(trader.trades) - len(wins)}")
        if rep is not None:
            print(f"\n代表性交易（加仓单位最多）已画入图03：")
            print(f"  入场 {rep['fills'][0]['date'].date()} @ {rep['fills'][0]['price']:.2f}"
                  f"，共 {rep['units']} 个单位 / {rep['total_shares']} 股")
            print(f"  离场 {rep['exit_date'].date()} @ {rep['exit_price']:.2f}"
                  f"（{rep['exit_reason']}），盈亏 {rep['pnl']:+,.0f} 元"
                  f"（{rep['pnl'] / rep['invest'] * 100:+.2f}%）")
    print(f"\n全部输出已写入：{OUT_DIR}")
    print('完成。')


if __name__ == '__main__':
    main()
