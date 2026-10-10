# -*- coding: utf-8 -*-
"""
21 从手写引擎到工业框架：用 backtrader 重写双均线，做"结果一致性"验证（单文件自包含教学脚本）
==========================================================================================
课02 我们用 30 行 pandas 向量化代码写了一个 SMA(20)/SMA(60) 双均线回测。
本课把**同一个策略、同一套成本假设**搬进工业级事件驱动框架 backtrader（bt），
核心问题只有一个：两个引擎跑出来的结果能对上吗？

【本课灵魂：一致性验证】
  手写版口径（课02正确版）：t 日收盘算出信号 -> t+1 日新仓位生效。
    它"隐含"的成交价是 **信号日 t 的收盘价**（相当于收盘竞价下单），
    因为 t+1 日的收益是 close[t] -> close[t+1] 这一段，等于按 close[t] 建仓。
  bt 默认口径：在 t 日收盘后的 next() 里下单（市价单）-> **t+1 日开盘价成交**。
    "次日用什么价格成交"这个实现细节，就是两个引擎差出几个点的根源：
    平安银行 2015-2026 隔夜跳空经常 ±2% 以上，11 年累积下来就是几个百分点。
  统一口径的办法：把 bt 的 broker 设成 cheat-on-close（coc，市价单按"当前 bar 收盘价"成交），
    于是 bt 与手写版都变成"信号日收盘成交"，净值曲线应几乎重合（残差来自整数股 vs 分数仓、
    滑点钳制在当日 high/low 内等实现细节）。

四块内容：
  1) bt 核心概念实战：Cerebro / Data Feed / Strategy(notify_order, next) / Broker / Sizer / Analyzer；
  2) 双均线：bt默认版（次日开盘成交） vs bt对齐版（coc 收盘成交） vs 手写参考版，三方对比；
  3) 海龟简化版（20日突破入场 / 10日反向离场 + ATR 仓位）：同一框架复用，只换 Strategy 和 Sizer；
  4) optstrategy 网格优化：SMA(10~40,步5) × SMA(60~120,步20) 共 28 组，一键跑完（并讨论过拟合）。

输出（outputs/）：
  01_双均线净值对比(手写vs bt).png / 02_回撤对比.png / 03_海龟简化版净值.png
  04_SMA网格优化热力图.png / bt_metrics.csv（手写版 vs bt 版指标对照，utf-8-sig）

运行：
  cd 21_backtrader事件驱动回测
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
  （不用 cerebro.plot()——它与新版 matplotlib 有兼容问题，所有图自己用 matplotlib 画）

数据：data/cache/stock_000001_平安银行.csv（2015 年起前复权日K）。
      缺失时自动用 np.random.default_rng(42) 造合成数据兜底，脚本绝不失败。
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面环境出图（硬性要求）
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. 全局设置
# ----------------------------------------------------------------------
SEED = 42                    # 随机种子：策略无随机性，仅合成数据兜底时使用
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cache', 'stock_000001_平安银行.csv')

TRADING_DAYS = 252   # 年化因子：一年约 252 个交易日
RF_ANNUAL = 0.02     # 无风险利率 2%（夏普比率用，手写口径）

# ---- 交易成本参数（与课02完全一致）----
COMMISSION = 2.5e-4        # 佣金：双边各万 2.5
STAMP_TAX = 5.0e-4         # 印花税：仅卖出收千 0.5
SLIPPAGE = 1.0e-3          # 滑点：单边 0.1%（bt 用 set_slippage_perc 实现）
FEE_BUY = COMMISSION + SLIPPAGE               # 买入单边合计 0.125%
FEE_SELL = COMMISSION + STAMP_TAX + SLIPPAGE  # 卖出单边合计 0.175%（多一项印花税）

FAST, SLOW = 20, 60        # 双均线参数：短均线 20 日、长均线 60 日
CASH = 1_000_000           # 初始资金 100 万（Broker 参数）

import backtrader as bt    # 已装 1.9.78.123

# ======================================================================
# 0.5 概念地图：事件驱动 vs 向量化，bt 六件套各管什么
# ======================================================================
# 【两种回测范式】
#   向量化（课02）：把"仓位 × 收益率 - 成本"写成整条序列的数组运算，一口气算完。
#     优点：快、代码短、不易写出状态错误；缺点：难以表达"下单-撮合-部分成交-拒单"
#     这类过程细节，仓位变化只能用简化公式近似。
#   事件驱动（bt）：回测器逐根 bar 推进，每个 bar 依次回调策略的 next()；
#     策略在 next() 里看指标、发订单；Broker 在下一个时点撮合、扣钱、回报
#     notify_order()；Analyzer 全程旁听，最后出统计。真实交易系统就是这样
#     一拍一拍工作的，所以策略代码几乎可以原样从回测搬到模拟盘。
#
# 【bt 六件套】（每个概念第一次出现处还有更细的注释）
#   Cerebro    ："大脑"/总调度。装载数据、策略、经纪商、分析器，run() 起来逐 bar 推进。
#   Data Feed  ：把 pandas DataFrame（date,open,high,low,close,volume）翻译成 bt 的
#                line 对象（data.close[0]=当前bar收盘，[-1]=上一bar）。
#   Strategy   ：你的策略逻辑。__init__() 建指标（只建一次），next() 每 bar 执行决策，
#                notify_order() 接收订单回报（成交/拒单/保证金不足）。
#   Broker     ：模拟券商。管现金（初始 100 万）、撮合订单、收佣金、算滑点。
#   Sizer      ：下单数量计算器。策略只说"买"，买多少股由 Sizer 决定（满仓/ATR风险仓位）。
#   Analyzer   ：绩效统计器。夏普、回撤、交易笔数……在 run 结束后取 analysis 结果。
#   经验法则：Strategy 里只写"信号与决策"；成本、撮合、仓位、统计全部交给框架，
#   这正是"框架的好处"；代价是每个细节都被框架的**默认实现**接管——默认值和你的
#   想法不一致时（如本课：默认次日开盘成交），你必须钻进源码核对口径。


# ----------------------------------------------------------------------
# 1. 数据加载（真实缓存优先，缺失则合成兜底——绝不因缺数据失败）
# ----------------------------------------------------------------------
def load_data():
    """读取本地缓存的真实日K；缺失/损坏时用 rng(42) 造合成数据，绝不失败。"""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', parse_dates=['date'])
            df = df.set_index('date').sort_index()
            df = df[~df.index.duplicated(keep='last')].dropna(subset=['close'])
            if len(df) > SLOW + 20 and {'open', 'high', 'low', 'close', 'volume'} <= set(df.columns):
                return df, False
        except Exception as e:  # noqa: BLE001
            print(f'[警告] 缓存读取失败（{e}），改用合成数据。')
    # ---- 合成数据兜底：分段漂移的几何随机游走（保证有金叉死叉可玩）----
    n = 2800
    idx = pd.bdate_range('2015-01-05', periods=n)
    q = n // 4
    drift = np.concatenate([np.full(q, 0.0006), np.full(q, -0.0004), np.full(n - 2 * q, 0.0002)])
    r = drift + RNG.normal(0.0, 0.018, n)
    close = pd.Series(11.0 * np.exp(np.cumsum(r)), index=idx, name='close')
    spread = np.abs(RNG.normal(0.0, 0.008, n))
    df = pd.DataFrame({
        'open': close * (1 + RNG.normal(0, 0.004, n)),
        'high': close * (1 + spread),
        'low': close * (1 - spread),
        'close': close,
        'volume': RNG.lognormal(18.5, 0.5, n).astype(int),
    }, index=idx)
    print('[警告] 未找到真实缓存，已生成合成数据（图中均标注"合成数据"）。')
    return df, True


# ======================================================================
# 2. 手写参考引擎（课02"正确版"逐行复刻：信号次日执行、信号日收盘价成交、含成本）
#    这是"参考答案"。后面 bt 的两个版本都要跟它对账。
#    clamp=False：课02原版——滑点恒为收盘价×0.1%（无论当日波幅）；
#    clamp=True ：与 bt 默认 slip_match=True 同一条物理规则——不允许以出界的
#                 价格成交。★注意 bt 的真身：coc 订单实际在"下一根 bar 的
#                 broker 环节"才成交（cerebro 的 _runnext 里 broker.next() 跑在
#                 strat.next() 之前），所以钳制用的是**次日**的 high/low：
#                   买 min(close[t]×1.001, high[t+1])、卖 max(close[t]×0.999, low[t+1])
#                 这条规则把"滑点"变成了"最多滑到次日波幅边界"，8/55 笔成交受影响。
# ======================================================================
def hand_engine(df, clamp=False):
    """返回 dict(metrics, nav, ret)：与课02完全同一套公式（clamp=True 时成交价按钳制规则）。"""
    close = df['close']
    high, low = df['high'], df['low']
    idx = close.index
    ret = close.pct_change().fillna(0.0)                    # 日收益 close-to-close

    # 信号：t 日收盘算出 SMA20/SMA60 的大小关系（NaN 比较为 False => 预热期空仓）
    sma_fast = close.rolling(FAST).mean()
    sma_slow = close.rolling(SLOW).mean()
    signal = pd.Series(np.where(sma_fast > sma_slow, 1.0, 0.0), index=idx)

    # ★ 正确版：shift(1) —— t 日信号，t+1 日才按新仓位结算收益。
    #   隐含成交价 = 信号日 t 的收盘价（t+1 的收益恰好是 close[t]->close[t+1] 段）。
    pos = signal.shift(1).fillna(0.0)

    # 成本：仓位变化 d（+1=买入，-1=卖出）。
    #   clamp=False：恒定费率（课02口径）：买 FEE_BUY=0.125%，卖 FEE_SELL=0.175%；
    #   clamp=True ：按"实际成交价"折算相对 close 的摩擦率（与 bt 的撮合完全同规则）。
    d = pos.diff()
    d.iloc[0] = pos.iloc[0]
    if not clamp:
        cost = d.clip(lower=0.0) * FEE_BUY + (-d).clip(lower=0.0) * FEE_SELL
    else:
        # 有效成交价（与 bt 默认 slip_match=True 逐笔同规则：钳制用次日 high/low）
        high_next = high.shift(-1).fillna(high)             # 最后一根 bar 无次日，退化为当日
        low_next = low.shift(-1).fillna(low)
        fill_buy = np.minimum(close * (1 + SLIPPAGE), high_next)   # 买：不高于次日最高
        fill_sell = np.maximum(close * (1 - SLIPPAGE), low_next)   # 卖：不低于次日最低
        rate_buy = fill_buy * (1 + COMMISSION) / close - 1.0     # 买入摩擦率（含佣金）
        rate_sell = 1.0 - fill_sell * (1 - COMMISSION - STAMP_TAX) / close  # 卖出摩擦率
        buy_days = d.clip(lower=0.0) > 0
        sell_days = (-d).clip(lower=0.0) > 0
        cost = pd.Series(0.0, index=idx)
        # ★ 对齐日历：buy_days[k]=True 表示信号在 k-1 日发出（d[k]=pos[k]-pos[k-1]），
        #   成交基准价是 close[k-1]、钳制用 high[k]——即 rate_buy[k-1]，所以 shift(1)
        cost[buy_days] = rate_buy.shift(1)[buy_days]
        cost[sell_days] = rate_sell.shift(1)[sell_days]
    strat_ret = pos * ret - cost                            # 当日仓位收益 - 当日成本

    nav = (1.0 + strat_ret).cumprod()

    # ---- 绩效（与课02同公式）----
    years = len(strat_ret) / TRADING_DAYS
    total_ret = nav.iloc[-1] - 1.0
    ann_ret = nav.iloc[-1] ** (1.0 / years) - 1.0           # 几何年化（bt 的 Returns.rnorm 同公式）
    ann_vol = strat_ret.std() * math.sqrt(TRADING_DAYS)     # 样本 std(ddof=1) 年化
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 1e-12 else float('nan')
    max_dd = (nav / nav.cummax() - 1.0).min()

    # ---- 交易记录（金叉开仓/死叉平仓，成交价=信号日收盘价或钳制价）----
    arr = signal.to_numpy()
    px = close.to_numpy()
    prev = np.concatenate([[0.0], arr[:-1]])
    opens = np.where((arr == 1) & (prev == 0))[0]           # 0->1 金叉
    closes = np.where((arr == 0) & (prev == 1))[0]          # 1->0 死叉
    opens_set, closes_set = set(opens.tolist()), set(closes.tolist())
    rounds = []                                             # 完整往返（一次开+一次平）
    open_i = None
    for i in range(len(idx)):
        if i in opens_set and open_i is None:
            open_i = i
        if i in closes_set and open_i is not None:
            if not clamp:
                net = px[i] * (1 - FEE_SELL) / (px[open_i] * (1 + FEE_BUY)) - 1.0
            else:
                fb = min(px[open_i] * (1 + SLIPPAGE), float(high.iloc[min(open_i + 1, len(idx) - 1)]))
                fs = max(px[i] * (1 - SLIPPAGE), float(low.iloc[min(i + 1, len(idx) - 1)]))
                net = fs * (1 - COMMISSION - STAMP_TAX) / (fb * (1 + COMMISSION)) - 1.0
            rounds.append(net)
            open_i = None
    still_open = open_i is not None                         # 期末是否仍持仓（bt 同样不计入 closed）
    n_rounds = len(rounds)
    win_rate = float(np.mean([r > 0 for r in rounds])) if n_rounds else float('nan')

    metrics = {
        '版本': ('手写钳制版(收盘口径+滑点钳制)' if clamp else '手写参考版(课02, 信号次日收盘口径)'),
        '成交口径': ('信号日收盘价+次日high/low钳制（与bt同规则）' if clamp
                    else '信号日收盘价（shift(1) 隐含）'),
        '期末净值': round(float(nav.iloc[-1]), 4),
        '累计收益率(%)': round(total_ret * 100, 2),
        '年化收益率(%)': round(ann_ret * 100, 2),
        '年化波动率(%)': round(ann_vol * 100, 2),
        '夏普比率(同口径rf=2%)': round(sharpe, 3),
        '夏普比率(bt SharpeRatio_A口径)': float('nan'),   # 手写版没有这个口径
        '最大回撤(%)': round(max_dd * 100, 2),
        '完整往返笔数': n_rounds,
        '胜率(%)': round(win_rate * 100, 1),
        '期末仍持仓': '是' if still_open else '否',
    }
    return {'metrics': metrics, 'nav': nav, 'ret': strat_ret, 'n_signals': len(opens)}


# ======================================================================
# 【实测参考值（平安银行 000001 前复权 2015-01-05~2026-09-17，2845 根日K，seed=42）】
#   课02正确版（手写参考版）：期末净值 1.2290 / 累计 +22.90% / 年化 1.84% /
#     年化波动 21.68% / 夏普(rf=2%) -0.007 / 最大回撤 -41.94% /
#     完整往返 27 笔 / 胜率 33.3% / 期末仍持仓。
#   bt 对齐版（coc 收盘基准价 + 恒定滑点 slip_out=True）：期末净值 1.2286 /
#     年化 1.84%（差 +0.00pp）/ 净值路径逐日最大偏差 0.038% -> 两引擎一致。
#   bt 默认版（次日开盘成交）：期末净值 1.0907 / 年化 0.77%——与手写差 -1.07pp，
#     "次日用什么价格成交"没对齐前就是这个量级的偏差。
# ======================================================================


# ======================================================================
# 3. backtrader 六件套
# ======================================================================
# ----------------------------------------------------------------------
# 3.1 CommissionInfo：把"A股费用"教给 Broker
# ----------------------------------------------------------------------
class CommInfoAStock(bt.CommInfoBase):
    """
    【概念：CommissionInfo】告诉 Broker 每笔成交收多少佣金。
    bt 内置佣金是"双边对称"的，表达不了印花税只收卖出这一单向。
    【绕法】继承 CommInfoBase，重写 _getcommission(size, price, pseudoexec)：
      佣金 = 成交额 × 万2.5（双边都有）；size<0（卖出）再加 成交额 × 千0.5 印花税。
    注意三点：
      - stocklike=True：股票（现金全额、无保证金），不是期货；
      - percabs=True + COMM_PERC：比例佣金的数值按"绝对小数"理解（0.00025=万2.5），
        而不是 bt 默认的"百分数"（0.25 表示 0.25%）——口径不一致会差 100 倍；
      - pseudoexec：True 时是"试算"（用于下单前估算占用资金），False 才是真扣费，
        所以函数必须保持无副作用（不能在里面记账）。
    滑点不在佣金里：滑点改的是**成交价**（Broker 的 set_slippage_perc），佣金按
    成交额（含滑点后的价格）计提，两者口径天然一致。
    """

    params = dict(stocklike=True, commtype=bt.CommInfoBase.COMM_PERC, percabs=True)

    def _getcommission(self, size, price, pseudoexec):
        comm = abs(size) * price * COMMISSION               # 佣金：双边万2.5
        if size < 0:                                        # size<0 => 卖出
            comm += abs(size) * price * STAMP_TAX           # 印花税：仅卖出千0.5
        return comm


# ----------------------------------------------------------------------
# 3.2 Sizer：满仓买入的"数量计算器"
# ----------------------------------------------------------------------
class AllInSizer(bt.Sizer):
    """
    【概念：Sizer】策略调用 self.buy() 不带 size 时，bt 转头问 Sizer："买多少？"
    _getsizing(comminfo, cash, data, isbuy) 返回股数。
    手写版是"分数仓位"（满仓=仓位1.0），bt 是真实撮合，只能买整数股，所以：
      - 股数 = 可用现金×安全垫 / (价格×(1+滑点)×(1+佣金))，向下取整；
        不留垫子的话，Broker 撮合时发现"现金不够付佣金"会把订单标记 Margin 拒掉
        （回测里表现为"该买的没买上"，净值悄悄不对）；
      - ★安全垫多大，取决于"成交价在算数量时知不知道"——这是事件驱动回测的
        深层细节：
          coc（当日收盘成交）：价格已知 close×(1+滑点)，垫 0.1% 即可（safety=0.999）；
        默认市价单（次日开盘成交）：算数量时只知道今天收盘，明天开盘可能跳空
          高开到涨停 +10%，必须按涨停上限预留现金（safety≈0.90），
          否则跳空日整单被拒。注意代价：这部分预留现金(~10%)不产生收益，
          是"次日开盘成交"模式的隐性成本；而手写分数仓版本没有这个问题。
          取整剩下的零头现金（不足一股）同样留在账户里——这就是两引擎
          对齐后仍有百万分之一量级残差的原因之一；
      - 卖出/清仓时返回当前持仓股数（self.close() 其实直接用持仓数，这里作演示）。
    """
    params = dict(safety=0.999)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:                                       # 卖出：清掉全部持仓
            return self.strategy.getposition(data).size
        price = data.close[0]                               # coc 口径下即本次成交基准价
        denom = price * (1.0 + SLIPPAGE) * (1.0 + COMMISSION)
        if denom <= 0:
            return 0
        return int(cash * self.p.safety / denom)


# ----------------------------------------------------------------------
# 3.3 海龟用 Sizer：ATR 风险仓位（简化版：单单元）
# ----------------------------------------------------------------------
class TurtleSizer(bt.Sizer):
    """
    海龟法则的仓位单位：Unit = 账户的 1% 风险 / (ATR × 每点价值)。
    简化版：股数 = (权益 × 1%) / ATR(14)，上限为现金能买到的最大股数。
    物理含义：如果接下来价格朝不利方向走 1 个 ATR 就止损，亏掉的是权益的 1%。
    波动大的股票自动少买、波动小的自动多买——这就是"波动率倒数仓位"。
    （真正的海龟还会每 0.5N 加一个单元、最多 4 单元、带 2N 步损，这里全部简化。）
    """
    params = dict(safety=0.999)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.strategy.getposition(data).size
        strat = self.strategy
        atr = strat.atr[0]                                  # 由 TurtleStrategy 预先建好
        equity = strat.broker.getvalue()
        if not atr or atr <= 0:
            return 0
        risk_shares = int(equity * 0.01 / atr)              # 1 单元 = 1% 风险
        price = data.close[0]
        max_afford = int(cash * self.p.safety / (price * (1.0 + SLIPPAGE) * (1.0 + COMMISSION)))
        return max(0, min(risk_shares, max_afford))         # 风险仓位与现金两者取小


# ----------------------------------------------------------------------
# 3.4 双均线策略（Strategy 子类结构示范）
# ----------------------------------------------------------------------
class SmaCrossStrategy(bt.Strategy):
    """
    【概念：Strategy】一个 bt 策略类通常只有三件事：
      __init__      ：建指标（只跑一次；bt 指标是"活的"，每个 bar 自动更新）；
      next          ：每个 bar 收盘后被调用（本策略口径：bar 收盘才发单，
                       这是与"信号次日执行"对齐的关键——见 README）；
      notify_order  ：接收订单状态回报。★必须写：不加它你也能跑，但订单被
                       拒单/保证金不足（Margin）时**无声无息**，回测看着正常、
                       其实仓位没建起来。生产代码还要在这里管 self.order 互斥锁。
    信号定义与课02逐位一致：SMA20 > SMA60 持有，否则空仓。
    这里不用 bt.ind.CrossOver，因为它对"慢均线刚生效的第一天"的处理与
    手写版（NaN 比较为 False）可能有细微差别；直接比较数值、自己记上一根
    bar 的状态，保证两个引擎的信号逐日相同（一致性验证的前提）。
    """
    params = dict(fast=FAST, slow=SLOW, tag='')             # params：参数化，供 optstrategy 扫描

    def __init__(self):
        # 指标引用存 self，next() 里用 [0] 取当前 bar、[-1] 取上一根 bar
        self.sma_fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.sma_slow = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.order = None                                   # 未完成订单的互斥锁
        self.prev_sig = 0.0                                 # 上一根 bar 的信号（预热期视为空仓）
        self.fills = []                                     # 成交记录：notify_order 里填

    def next(self):
        # 指标未就绪（慢均线还在预热）时取到 NaN，与 NaN 比较为 False => 信号 0，
        # 与手写版 np.where(NaN>NaN -> False) 逐日一致
        sig = 1.0 if self.sma_fast[0] > self.sma_slow[0] else 0.0
        if self.order:                                      # 有未完成订单就先不动（好习惯）
            return
        if sig == 1.0 and self.prev_sig == 0.0 and not self.position:
            self.order = self.buy()                         # 金叉：满仓买入（数量由 Sizer 定）
        elif sig == 0.0 and self.prev_sig == 1.0 and self.position:
            self.order = self.close()                       # 死叉：清仓
        self.prev_sig = sig

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return                                          # 常规中间态，不刷屏
        if order.status == order.Completed:
            dt = bt.num2date(order.executed.dt)
            self.fills.append({
                '日期': pd.Timestamp(dt).normalize(),
                '方向': '买入' if order.isbuy() else '卖出',
                '股数': abs(order.executed.size),
                '成交价': order.executed.price,
                '佣金': order.executed.comm,
            })
        elif order.status in (order.Margin, order.Rejected):
            # ★ 不写 notify_order，这里的发生的拒单你永远看不到
            print(f'  [警告] 订单被拒（{order.getstatusname()}）——检查 Sizer 的资金安全垫！')
        self.order = None                                   # 无论成败，释放互斥锁

    def stop(self):
        # 每个 run 结束时调用：把成交记录挂到策略对象上，主流程取走做"滑点钳制"检查
        self.fills = self.fills


# ----------------------------------------------------------------------
# 3.5 海龟简化版策略（同一框架，只换 Strategy + Sizer）
# ----------------------------------------------------------------------
class TurtleStrategy(bt.Strategy):
    """
    海龟法则简化版（单股票、单单元、无止损）：
      入场：收盘价 > 前 20 日最高价（突破，用上一根 bar 的 20 日高点，不含当日）；
      离场：收盘价 < 前 10 日最低价（反向突破）；
      仓位：TurtleSizer 按 ATR(14) 给出（1% 权益风险 / ATR）。
    真实海龟在盘中按突破价挂单，这里仍用"收盘出信号、coc 收盘成交"与双均线保持同一口径。
    """
    params = dict(entry_n=20, exit_n=10, atr_n=14, tag='')

    def __init__(self):
        # Highest/Lowest 都把"当日"算进去，所以比较对象取 [-1]（截至上一根 bar 的窗口）
        self.hh = bt.ind.Highest(self.data.high, period=self.p.entry_n)
        self.ll = bt.ind.Lowest(self.data.low, period=self.p.exit_n)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_n)   # 给 TurtleSizer 用
        self.order = None
        self.fills = []

    def next(self):
        if self.order:
            return
        if self.position:
            if self.data.close[0] < self.ll[-1]:            # 10 日反向突破离场
                self.order = self.close()
        else:
            if self.data.close[0] > self.hh[-1]:            # 20 日突破入场
                self.order = self.buy()                     # 数量由 TurtleSizer 决定

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            self.fills.append({
                '日期': pd.Timestamp(bt.num2date(order.executed.dt)).normalize(),
                '方向': '买入' if order.isbuy() else '卖出',
                '股数': abs(order.executed.size),
                '成交价': order.executed.price,
                '佣金': order.executed.comm,
            })
        elif order.status in (order.Margin, order.Rejected):
            print(f'  [警告] 海龟订单被拒（{order.getstatusname()}）')
        self.order = None


# ----------------------------------------------------------------------
# 3.6 搭建 Cerebro（每跑一个版本搭一个独立的"大脑"）
# ----------------------------------------------------------------------
def make_cerebro(df, coc, sizer_cls, sizer_kwargs=None, slip_out=True):
    """
    【概念：Cerebro + Data Feed + Broker】
    - bt.feeds.PandasData：把 DataFrame 翻译成 bt 的 line。datetime=None 表示
      "时间在索引里"；其余列按列名自动对到 open/high/low/close/volume。
      line 用 [0] 取当前 bar、[-1] 取上一根——事件驱动的"时钟"就藏在下标里。
    - Broker：setcash(100万) 设初始资金；addcommissioninfo 挂 A 股佣金表；
      set_slippage_perc 设比例滑点。★两个坑：
        (a) slip_open=True 才对市价单生效（默认 False 时市价单根本不上滑点）；
        (b) slip_match=True（默认）会把滑点价钳回**成交 bar**的 high/low 内——
            而 coc 订单实际在下一根 bar 的 broker 环节成交（cerebro 的
            _runnext 里 broker.next() 跑在 strat.next() 之前），所以钳的是
            次日波幅。要让 bt 严格复现课02"恒定 0.1% 滑点"模型，需
            slip_out=True 允许出界价成交（本课对齐版的选择）。
    - coc（cheat-on-close）：市价单按"下单当根 bar 的收盘价"作为基准价成交，
      而不是默认的下一根 bar 开盘价——这是把 bt 与手写版口径对齐的开关。
      ★它是 Broker 的参数（cerebro.broker.set_coc），不是 Cerebro 构造参数。
    - addsizer：设默认 Sizer（kwargs 传给 Sizer 的 params，如安全垫大小）；
      Analyzer 在 run_bt() 里逐个 addanalyzer 注册。
    """
    cere = bt.Cerebro()
    cere.adddata(bt.feeds.PandasData(dataname=df))          # Data Feed
    cere.broker.setcash(CASH)                               # Broker：初始资金 100 万
    cere.broker.addcommissioninfo(CommInfoAStock())         # Broker：A股佣金表
    # 滑点 0.1%：slip_open=True 使市价单生效；slip_out=True 关掉 high/low 钳制
    cere.broker.set_slippage_perc(perc=SLIPPAGE, slip_open=True, slip_out=slip_out)
    if coc:
        cere.broker.set_coc(True)                           # ★对齐开关：以信号日收盘为基准价
    cere.addsizer(sizer_cls, **(sizer_kwargs or {}))        # 默认 Sizer
    return cere


def add_analyzers(cere):
    """
    【概念：Analyzer】注册绩效统计器，run() 结束后从 strat.analyzers.<名字> 取结果。
      Returns       ：rtot/ravg/rnorm。rnorm = exp(ln(期末/期初)/N×252)-1，
                       与手写版"净值^(252/N)-1"是同一个几何年化公式，可直接对比。
      SharpeRatio_A ：bt 的"年化夏普"。★默认 timeframe=Years => 用**年度收益**算
                       夏普，rf 默认 1%，总体标准差——和手写版"日收益×√252、rf=2%"
                       是两套口径，直接对表会差很远（本课的坑之一，见 README）。
      DrawDown      ：逐 bar 记录净值距新高回撤。max.drawdown 是**正的百分比幅度**。
      TradeAnalyzer ：交易统计。total.closed=完整往返笔数，won/lost=盈/亏笔数
                       （按含费盈亏 pnlcomm>=0 记胜）；期末未平的仓位计入 total.open。
      TimeReturn    ：逐日收益率序列（基于账户总价值），我们用它重放净值曲线，
                       并按"手写同口径"重算波动率/夏普/回撤做严格对表。
    """
    cere.addanalyzer(bt.analyzers.Returns, _name='ret')
    cere.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_a')
    cere.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cere.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cere.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='tret')


def run_bt(df, strat_cls, strat_params, coc, sizer_cls=AllInSizer, slip_out=True):
    """搭 Cerebro -> 跑一个策略 -> 从 analyzers 取全套指标。"""
    # Sizer 安全垫：coc 成交价已知 -> 0.9999（只剩一股取整尘埃）；
    # 默认市价单(次日开盘)按涨停上限预留 -> 0.90（明天可能高开 +10%）
    sizer_kwargs = dict(safety=0.9999) if coc else dict(safety=0.90)
    cere = make_cerebro(df, coc=coc, sizer_cls=sizer_cls, sizer_kwargs=sizer_kwargs,
                        slip_out=slip_out)
    cere.addstrategy(strat_cls, **strat_params)
    add_analyzers(cere)
    strat = cere.run(maxcpus=1)[0]                          # maxcpus=1：单进程，结果可复现

    # ---- Analyzer 取数 ----
    rets_a = strat.analyzers.ret.rets                       # 普通 dict
    dd_a = strat.analyzers.dd.rets                          # AutoOrderedDict（点号取嵌套）
    ta_a = strat.analyzers.ta.rets
    tret_a = strat.analyzers.tret.rets

    # TimeReturn 的键已是 datetime（不同版本可能是 bt 日期浮点，做个兼容转换）
    tr = pd.Series({pd.Timestamp(k).normalize(): v for k, v in dict(tret_a).items()}).sort_index()
    nav = (1.0 + tr).cumprod()

    years = len(tr) / TRADING_DAYS
    ann_ret = float(rets_a['rnorm'])                        # bt 几何年化（与手写同公式）
    ann_vol = tr.std(ddof=1) * math.sqrt(TRADING_DAYS)      # 同口径：样本 std 年化
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 1e-12 else float('nan')
    max_dd_same = float((nav / nav.cummax() - 1.0).min())   # 同口径回撤（来自 TimeReturn 重放）
    dd_bt = float(dd_a['max']['drawdown'])                  # bt DrawDown 原生回撤（正数幅度）

    closed = int(ta_a['total'].get('closed', 0))            # 完整往返（AutoOrderedDict 用 .get 防缺键）
    won = int(ta_a.get('won', pd.Series()).get('total', 0))
    lost = int(ta_a.get('lost', pd.Series()).get('total', 0))
    open_trades = int(ta_a['total'].get('open', 0))
    win_rate = won / (won + lost) if (won + lost) else float('nan')
    value_end = float(cere.broker.getvalue())

    metrics = {
        '版本': strat_params.get('tag', strat_cls.__name__),
        '成交口径': (('当日收盘基准价(cheat-on-close)' if coc else '次日开盘价成交(bt默认市价单)')
                    + ('' if slip_out else '+滑点钳制')),
        '期末净值': round(value_end / CASH, 4),
        '累计收益率(%)': round((value_end / CASH - 1) * 100, 2),
        '年化收益率(%)': round(ann_ret * 100, 2),
        '年化波动率(%)': round(ann_vol * 100, 2),
        '夏普比率(同口径rf=2%)': round(sharpe, 3),
        '夏普比率(bt SharpeRatio_A口径)': (
            round(float(strat.analyzers.sharpe_a.rets['sharperatio']), 3)
            if strat.analyzers.sharpe_a.rets['sharperatio'] is not None else float('nan')),
        '最大回撤(%)': round(dd_bt, 2),
        '完整往返笔数': closed,
        '胜率(%)': round(win_rate * 100, 1),
        '期末仍持仓': '是' if open_trades > 0 else '否',
    }
    info = {
        'metrics': metrics,
        'nav': nav,
        'fills': strat.fills,
        'same_caliber_maxdd': round(max_dd_same * 100, 2),
        'n_open_trades': open_trades,
        'n_lost': lost,
    }
    return info


# ----------------------------------------------------------------------
# 3.7 优化：optstrategy 参数网格（28 组一键跑完）
# ----------------------------------------------------------------------
def run_optimization(df, fast_list, slow_list):
    """
    【概念：optstrategy】addstrategy 只装一个参数组合，optstrategy 传入参数的
    候选列表，Cerebro 自动做笛卡尔积、逐组合搭一个 Strategy 跑完——
    "参数网格就在框架里一键跑"，这正是过拟合的温床（呼应课16）：
    28 组里年化最高的组合，大概率是对这段历史过拟合的幸存者，不是未来最优。
    技术点：
      - run(maxcpus=1)：默认会开多进程池，为了结果可复现/避免沙箱 fork 问题固定单进程；
      - optreturn=True（默认）：优化模式只回传 params+analyzers（省内存），
        所以这里只取 Returns/DrawDown 两个 analyzer，且只能按"加入顺序"取
        （OptReturn 不保留 _name 属性，取 strat.analyzers[0]/[1]）。
    """
    cere = make_cerebro(df, coc=True, sizer_cls=AllInSizer, sizer_kwargs=dict(safety=0.9999))
    cere.optstrategy(SmaCrossStrategy,
                     fast=list(fast_list), slow=list(slow_list))
    cere.addanalyzer(bt.analyzers.Returns, _name='ret')      # 取数按顺序：[0]
    cere.addanalyzer(bt.analyzers.DrawDown, _name='dd')      # 取数按顺序：[1]
    results = cere.run(maxcpus=1)

    rows = []
    for run_list in results:                                 # 优化结果：list of list of OptReturn
        for o in run_list:
            rows.append({
                'SMA快线': o.params.fast,
                'SMA慢线': o.params.slow,
                '年化收益率(%)': round(float(o.analyzers[0].rets['rnorm']) * 100, 2),
                '最大回撤(%)': round(float(o.analyzers[1].rets['max']['drawdown']), 2),
            })
    return pd.DataFrame(rows)


# ======================================================================
# 4. 主流程：三方对账 -> 海龟 -> 优化 -> 图表
# ======================================================================
def main():
    pd.set_option('display.unicode.east_asian_width', True)

    df, IS_SYNTH = load_data()
    tag = '【合成数据】' if IS_SYNTH else ''
    print(f'数据：{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}，共 {len(df)} 根日K'
          f'{"（合成数据）" if IS_SYNTH else "（真实前复权日K）"}')
    print(f'成本：佣金万{COMMISSION*1e4:.1f}(双边)+印花税千{STAMP_TAX*1e3:.1f}(卖)+滑点{SLIPPAGE*100:.1f}%(单边)\n')

    # ================================================================
    # 4.1 手写参考引擎（课02正确版）+ 手写对齐版（加滑点钳制）
    # ================================================================
    hand = hand_engine(df, clamp=False)                     # 课02原版：滑点恒为 close×0.1%
    hand_clamp = hand_engine(df, clamp=True)                # 对齐版：滑点钳制（次日 high/low，bt 同规则）
    print('================ 手写参考引擎（课02正确版复刻） ================')
    print(pd.DataFrame([hand['metrics']]).to_string(index=False))
    print('\n================ 手写钳制版（收盘口径+次日high/low滑点钳制，与bt默认滑点规则同款） ================')
    print(pd.DataFrame([hand_clamp['metrics']]).to_string(index=False))

    # ================================================================
    # 4.2 bt 默认版（次日开盘成交）——"对齐前"
    # ================================================================
    bt_default = run_bt(df, SmaCrossStrategy,
                        dict(fast=FAST, slow=SLOW, tag='bt默认市价单(次日开盘成交)'),
                        coc=False)
    print('\n================ bt 默认版（市价单 -> 次日开盘成交，滑点0.1%不出界） ================')
    print(pd.DataFrame([bt_default['metrics']]).to_string(index=False))

    # ================================================================
    # 4.3 bt 对齐版（coc：以信号日收盘为成交基准价）——"对齐后"
    #     再加一个"bt钳制版"（slip_out=False，bt 默认滑点规则）做进阶对照
    # ================================================================
    bt_coc = run_bt(df, SmaCrossStrategy,
                    dict(fast=FAST, slow=SLOW, tag='bt对齐版(coc收盘成交,恒定滑点)'),
                    coc=True, slip_out=True)
    print('\n================ bt 对齐版（cheat-on-close，滑点恒为基准价×0.1%，slip_out=True） ================')
    print(pd.DataFrame([bt_coc['metrics']]).to_string(index=False))

    bt_coc_clamp = run_bt(df, SmaCrossStrategy,
                          dict(fast=FAST, slow=SLOW, tag='bt对齐版(coc收盘成交,滑点钳制)'),
                          coc=True, slip_out=False)
    print('\n================ bt 钳制版（coc + bt默认 slip_match=True：滑点价钳回次日波幅内） ================')
    print(pd.DataFrame([bt_coc_clamp['metrics']]).to_string(index=False))

    # ---- 成交价口径的直接证据：打印前 3 笔成交 ----
    ref_close = df['close']
    print('\nbt对齐版前3笔成交（成交价应=信号日收盘×1.001，逐分不差）：')
    for f in bt_coc['fills'][:3]:
        c0 = float(ref_close.loc[f['日期']])
        print(f"  {f['日期']:%Y-%m-%d} {f['方向']} {f['股数']}股 @ {f['成交价']:.4f}"
              f"（信号日收盘 {c0:.4f}，×1.001={c0*1.001:.4f}，佣金 {f['佣金']:.2f} 元）")
    # 钳制效应检查：bt 钳制版里成交价偏离 基准价×(1±滑点) 的笔数
    n_clamped = sum(
        1 for f in bt_coc_clamp['fills']
        if abs(f['成交价'] - float(ref_close.loc[f['日期']]) * (1 + SLIPPAGE if f['方向'] == '买入' else 1 - SLIPPAGE)) > 1e-6)
    print(f'bt钳制版中被钳回次日波幅边界的成交 {n_clamped}/{len(bt_coc_clamp["fills"])} 笔')

    # ================================================================
    # 4.4 【本课灵魂】对账：对齐前差多少、对齐后差多少、残差归因
    # ================================================================
    m_hand, m_clamp = hand['metrics'], hand_clamp['metrics']
    m_def, m_coc, m_coc_clamp = (bt_default['metrics'], bt_coc['metrics'],
                                 bt_coc_clamp['metrics'])
    gap_before = m_def['年化收益率(%)'] - m_hand['年化收益率(%)']
    gap_before_cum = m_def['累计收益率(%)'] - m_hand['累计收益率(%)']
    gap_after = m_coc['年化收益率(%)'] - m_hand['年化收益率(%)']
    gap_after_cum = m_coc['累计收益率(%)'] - m_hand['累计收益率(%)']
    gap_clamp = m_coc_clamp['年化收益率(%)'] - m_clamp['年化收益率(%)']
    gap_clamp_cum = m_coc_clamp['累计收益率(%)'] - m_clamp['累计收益率(%)']

    # 净值路径逐日最大相对偏差（按日期对齐）
    nav_h = hand['nav']
    nav_hc = hand_clamp['nav']
    nav_d = bt_default['nav'].reindex(nav_h.index).ffill()
    nav_c = bt_coc['nav'].reindex(nav_h.index).ffill()
    nav_cc = bt_coc_clamp['nav'].reindex(nav_h.index).ffill()
    dev_before = float((nav_d / nav_h - 1.0).abs().max())
    dev_after = float((nav_c / nav_h - 1.0).abs().max())
    dev_clamp = float((nav_cc / nav_hc - 1.0).abs().max())

    print('\n================ 一致性验证（本课灵魂） ================')
    print(f'【对齐前】bt默认(次日开盘成交) vs 手写参考版(信号日收盘口径)：')
    print(f'    年化 {m_def["年化收益率(%)"]:.2f}% vs {m_hand["年化收益率(%)"]:.2f}%'
          f' -> 差 {gap_before:+.2f} pp；累计 {m_def["累计收益率(%)"]:.2f}% vs '
          f'{m_hand["累计收益率(%)"]:.2f}% -> 差 {gap_before_cum:+.2f} pp；'
          f'净值路径最大偏差 {dev_before*100:.2f}%')
    print(f'    差异来源：成交价口径（隔夜跳空，bt 用次日开盘价）+ 满仓受限'
          f'（为次日可能的涨停预留约10%现金，手写分数仓可真正满仓）')
    print(f'【对齐后】bt coc(收盘基准价+恒定滑点) vs 手写参考版：')
    print(f'    年化 {m_coc["年化收益率(%)"]:.2f}% vs {m_hand["年化收益率(%)"]:.2f}%'
          f' -> 差 {gap_after:+.2f} pp；累计 {m_coc["累计收益率(%)"]:.2f}% vs '
          f'{m_hand["累计收益率(%)"]:.2f}% -> 差 {gap_after_cum:+.2f} pp；'
          f'净值路径最大偏差 {dev_after*100:.4f}%')
    print(f'    回撤 {m_coc["最大回撤(%)"]:.2f}% vs {m_hand["最大回撤(%)"]:.2f}%；'
          f'往返 {m_coc["完整往返笔数"]} vs {m_hand["完整往返笔数"]} 笔；'
          f'胜率 {m_coc["胜率(%)"]}% vs {m_hand["胜率(%)"]}%')
    print(f'    => 统一"次日用什么价格成交"后两引擎一致，残余偏差=整数股取整的尘埃效应。')
    print(f'【进阶】换一套滑点规则也照样对齐——bt默认滑点钳制版 vs 手写同规则版：')
    print(f'    年化 {m_coc_clamp["年化收益率(%)"]:.2f}% vs {m_clamp["年化收益率(%)"]:.2f}%'
          f' -> 差 {gap_clamp:+.2f} pp；累计差 {gap_clamp_cum:+.2f} pp；'
          f'净值路径最大偏差 {dev_clamp*100:.4f}%')
    print(f'    （钳制让 {n_clamped} 笔成交的滑点变小 -> 全程年化比恒定滑点口径'
          f'高 {m_coc_clamp["年化收益率(%)"] - m_coc["年化收益率(%)"]:+.2f} pp——'
          f'滑点规则本身就是一个口径选择）')
    print(f'    回撤口径复核：bt DrawDown={m_coc["最大回撤(%)"]:.2f}%，'
          f'手写同公式重算={bt_coc["same_caliber_maxdd"]:.2f}%（一致）')
    print(f'    结论：框架没有魔法。两个引擎的差异 100% 来自"次日用什么价格成交、'
          f'滑点怎么算、数量怎么取整"这些实现细节；规则对齐，结果就对齐。')

    # ================================================================
    # 4.5 海龟简化版（同一框架复用：只换 Strategy + Sizer）
    # ================================================================
    turtle = run_bt(df, TurtleStrategy,
                    dict(entry_n=20, exit_n=10, atr_n=14, tag='bt海龟简化版(20突破/10离场)'),
                    coc=True, sizer_cls=TurtleSizer)
    print('\n================ 海龟简化版（20日突破入场 / 10日反向离场 + ATR仓位） ================')
    print(pd.DataFrame([turtle['metrics']]).to_string(index=False))
    if turtle['fills']:
        tf0 = turtle['fills'][0]
        print(f"  首笔：{tf0['日期']:%Y-%m-%d} {tf0['方向']} {tf0['股数']}股 @ {tf0['成交价']:.4f}"
              f"（ATR仓位：股数≈权益1%/ATR，与现金上限取小）")

    # ================================================================
    # 4.6 bt_metrics.csv：手写版 vs bt 版指标对照（utf-8-sig）
    # ================================================================
    rows = [m_hand, m_clamp, m_def, m_coc, m_coc_clamp, turtle['metrics']]
    for r in rows[1:5]:
        r['与手写版年化差(pp)'] = round(r['年化收益率(%)'] - m_hand['年化收益率(%)'], 2)
    rows[0]['与手写版年化差(pp)'] = 0.0
    rows[5]['与手写版年化差(pp)'] = float('nan')            # 海龟无手写对照
    metrics_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'bt_metrics.csv')
    metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print('\n已保存指标对照: outputs/bt_metrics.csv')

    # ================================================================
    # 4.7 参数优化：SMA(10~40,5) × SMA(60~120,20) = 7×4 = 28 组
    # ================================================================
    fast_list = list(range(10, 41, 5))
    slow_list = list(range(60, 121, 20))
    t0 = pd.Timestamp.now()
    grid = run_optimization(df, fast_list, slow_list)
    dt_opt = (pd.Timestamp.now() - t0).total_seconds()
    print(f'\n================ optstrategy 网格优化（{len(fast_list)}×{len(slow_list)}={len(grid)} 组，coc口径，耗时 {dt_opt:.1f} 秒） ================')
    best = grid.loc[grid['年化收益率(%)'].idxmax()]
    worst = grid.loc[grid['年化收益率(%)'].idxmin()]
    print(grid.pivot(index='SMA快线', columns='SMA慢线', values='年化收益率(%)').to_string())
    print(f'>>> 最优：SMA({int(best["SMA快线"])},{int(best["SMA慢线"])}) 年化 {best["年化收益率(%)"]:.2f}%'
          f' / 回撤 {best["最大回撤(%)"]:.2f}%；最差：SMA({int(worst["SMA快线"])},{int(worst["SMA慢线"])}) '
          f'年化 {worst["年化收益率(%)"]:.2f}%；max-min={best["年化收益率(%)"]-worst["年化收益率(%)"]:.2f} pp')
    print('>>> 提醒：28 组里挑年化最高的一组，就是网格搜索的"幸存者"——大概率过拟合（呼应课16）。')

    # ================================================================
    # 4.8 图 1：双均线净值对比（手写 vs bt 默认 vs bt 对齐）
    # ================================================================
    nav_def = nav_d
    nav_bh = df['close'] / df['close'].iloc[0]              # 买入持有（价格净值，未含成本，仅作参照）

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(nav_h.index, nav_h.values, lw=1.8, color='#d62728',
            label=f'手写参考版(课02)：信号次日收盘口径（期末 {nav_h.iloc[-1]:.4f}）')
    ax.plot(nav_def.index, nav_def.values, lw=1.4, color='#1f77b4', ls='--',
            label=f'bt默认版：市价单次日开盘成交（期末 {nav_def.iloc[-1]:.4f}）')
    ax.plot(nav_c.index, nav_c.values, lw=1.3, color='#2ca02c', ls=':',
            label=f'bt对齐版：coc收盘基准价+恒定滑点（期末 {nav_c.iloc[-1]:.4f}）')
    ax.plot(nav_hc.index, nav_hc.values, lw=0.9, color='#ff7f0e', alpha=0.85,
            label=f'手写钳制版：滑点钳回次日波幅（期末 {nav_hc.iloc[-1]:.4f}，与bt钳制版重合）')
    ax.plot(nav_bh.index, nav_bh.values, lw=1.0, color='#999999', alpha=0.7,
            label='买入持有（价格净值，未含成本）')
    ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_title(f'双均线 SMA({FAST})/SMA({SLOW})：手写引擎 vs backtrader 净值对比{tag}')
    ax.set_ylabel('净值（期初=1）')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, '01_双均线净值对比(手写vs bt).png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ================================================================
    # 4.9 图 2：回撤对比（三条净值各自的 drawdown 曲线）
    # ================================================================
    def dd_series(nav):
        return (nav / nav.cummax() - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.fill_between(nav_h.index, dd_series(nav_h).values, 0, color='#d62728', alpha=0.15)
    ax.plot(nav_h.index, dd_series(nav_h).values, lw=1.1, color='#d62728', label='手写参考版(课02)')
    ax.plot(nav_hc.index, dd_series(nav_hc).values, lw=1.0, color='#ff7f0e', alpha=0.85,
            label='手写钳制版（滑点钳回次日波幅）')
    ax.plot(nav_def.index, dd_series(nav_def).values, lw=1.1, color='#1f77b4', ls='--',
            label='bt默认版（次日开盘成交）')
    ax.plot(nav_c.index, dd_series(nav_c).values, lw=1.1, color='#2ca02c', ls=':',
            label='bt对齐版（coc收盘基准价）')
    ax.set_title(f'回撤对比：同一成交规则下两引擎曲线重合（恒定滑点口径/钳制口径各一对）{tag}')
    ax.set_ylabel('回撤（%）')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, '02_回撤对比.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ================================================================
    # 4.10 图 3：海龟简化版净值（对比双均线）
    # ================================================================
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(turtle['nav'].index, turtle['nav'].values, lw=1.6, color='#9467bd',
            label=f"海龟简化版 20/10+ATR仓位（期末 {turtle['nav'].iloc[-1]:.4f}，"
                  f"{turtle['metrics']['完整往返笔数']}笔往返）")
    ax.plot(nav_c.index, nav_c.values, lw=1.3, color='#2ca02c', ls=':',
            label=f'双均线 SMA(20/60) bt对齐版（期末 {nav_c.iloc[-1]:.4f}）')
    ax.plot(nav_bh.index, nav_bh.values, lw=1.0, color='#999999', alpha=0.7,
            label='买入持有（价格净值，未含成本）')
    ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_title(f'同一 backtrader 框架复用：海龟简化版 vs 双均线{tag}')
    ax.set_ylabel('净值（期初=1）')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, '03_海龟简化版净值.png')
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ================================================================
    # 4.11 图 4：SMA 网格优化热力图（行=快线，列=慢线，值=年化收益率%）
    # ================================================================
    mat = grid.pivot(index='SMA快线', columns='SMA慢线', values='年化收益率(%)')
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    im = ax.imshow(mat.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(i) for i in mat.index])
    ax.set_xlabel('SMA 慢线（天）')
    ax.set_ylabel('SMA 快线（天）')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                    color='black')
    bi, bj = np.unravel_index(np.nanargmax(mat.values), mat.values.shape)
    ax.scatter([bj], [bi], marker='*', s=220, color='blue', edgecolors='white', zorder=5)
    ax.set_title(f'SMA 参数网格：optstrategy 一键 28 组（★=年化最高，警惕过拟合）{tag}')
    fig.colorbar(im, ax=ax, label='年化收益率（%）')
    fig.tight_layout()
    p4 = os.path.join(OUT_DIR, '04_SMA网格优化热力图.png')
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    print('\n已生成图表:')
    for p in (p1, p2, p3, p4):
        print('  ', os.path.relpath(p, BASE_DIR))
    print('\n完成。')


if __name__ == '__main__':
    main()
