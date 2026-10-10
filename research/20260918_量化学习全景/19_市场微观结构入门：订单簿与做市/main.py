# -*- coding: utf-8 -*-
"""
第19课 市场微观结构入门：订单簿与做市
=====================================================================
【合成数据 / 模拟市场】本课所有"行情"均由限价订单簿模拟器合成（随机种子42），
不是真实逐笔数据。仅用真实日频数据做参数校准：
    data/cache/stock_000001_平安银行.csv（日线）→ 日波动率 / 参考价 / 成交量。

这节课打开量化机构高频组的黑箱：
  1. 限价订单簿（十档）怎么运转：市价单吃深度、限价单挂深度、价格"跳档"；
  2. 做市商怎么赚钱：买卖两侧各挂一档，赚买卖价差；
  3. 做市商怕什么：库存风险（单侧被连续成交）+ 逆向选择（知情交易者总在
     对自己有利的时候成交）；
  4. 用模拟回答：知情交易者占比 p 多低时，做市才赚钱？

三个关键建模假设（避免模拟假象，README 详述）：
  a) 做市商挂单并入订单簿深度统一记账（FIFO 队列模型，无"幽灵挂单"）；
  b) 噪声交易者对价差有弹性：价差 > 2 档时市价单流失（知情交易者无所谓，
     他们赚的是信息租金，照吃）；
  c) 价差越宽，公众限价单越倾向挂进价差内部（快速收敛，中间价不长期失真）。

运行：python main.py   （总耗时 < 2 分钟）
产出：outputs/ 下 5 张图 + market_maker_results.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境绘图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)
DATA_CSV = os.path.join(BASE, '..', 'data', 'cache', 'stock_000001_平安银行.csv')


# =====================================================================
# 第 1 部分：真实数据校准参数（读不到就用经验值兜底，脚本绝不失败）
# =====================================================================
def load_calibration():
    """从平安银行日线读取：日波动率（全样本与近250日）、最新收盘价、近250日均成交量/成交额。"""
    cal = dict(sigma_full=0.019, sigma_day=0.0097, p0=11.61,
               avg_volume=9.77e7, avg_amount=1.071e9,
               source='经验值兜底（真实CSV读取失败）')
    try:
        df = pd.read_csv(DATA_CSV)
        ret = np.log(df['close']).diff().dropna()
        cal['sigma_full'] = float(ret.std())                   # 2015年以来全样本日波动率（校准目标）
        cal['sigma_day'] = float(ret.tail(250).std())          # 近250日日波动率（当前状态，仅报告）
        cal['p0'] = float(df['close'].iloc[-1])                # 参考价格
        cal['avg_volume'] = float(df['volume'].tail(250).mean())
        cal['avg_amount'] = float((df['volume'].tail(250) * df['close'].tail(250)).mean())
        cal['source'] = '真实数据 data/cache/stock_000001_平安银行.csv'
    except Exception as e:  # noqa: BLE001 —— 兜底不报错
        print(f'  [警告] 读取真实数据失败（{e}），改用经验值。')
    return cal


# =====================================================================
# 第 2 部分：模拟器参数（合成市场；波动率校准见第 4 部分）
# =====================================================================
P = dict(
    seed=SEED,
    tick=0.01,           # A股最小报价单位 0.01 元
    n_levels=10,         # 买十档 / 卖十档
    t_day=14400.0,       # 一个交易日=4小时=14400秒（忽略午休，简化）
    p0=11.61,            # 初始参考价（校准覆盖）
    # ---- 订单到达强度（单位：笔/天，泊松过程=指数间隔时间）----
    lam_m=2200.0,        # 市价单到达率（校准后覆盖：由日波动率目标反推）
    lim_ratio=2.3,       # 限价单率 = 2.3 × 市价单率（比例固定 -> 订单簿深度不随 λm 漂移）
    can_ratio=1.2,       # 撤单率   = 1.2 × 市价单率
    lam_m0=2200.0,       # 校准试跑用的初始市价单率
    # ---- 订单大小分布（对数正态：中位数 / sigma）----
    mkt_med=250.0, mkt_sig=1.1,     # 市价单：中位数250股，右偏厚尾
    lim_med=400.0, lim_sig=0.8,     # 限价单
    dep_med0=1000.0, dep_grow=0.15, dep_sig=0.5,   # 初始每档深度：越深越厚
    dep_far=1300.0,                                # 远端新补档位深度
    # ---- 知情交易者（信息事件）----
    burst_k=5,           # 每次信息事件 = 同方向连续 5 笔大单（瞬间连发）
    burst_size_mult=1.8, # 知情单比普通单大 1.8 倍 -> burst 累计啃穿约2~3档、推价约2~3档
    # ---- 交易者行为 ----
    elastic_max=2,       # 噪声交易者价差弹性：价差>2档时市价单流失（知情者不受限）
    # ---- 做市商 ----
    mm_qty=300.0,        # 每侧挂单量（股）
    mm_dt=5.0,           # 每 5 秒决策一次（重挂/撤单；瞬间撤单=无延迟简化）
    mm_improve=True,     # A/B开关：价差>=3档时做市商是否向内改价1档
    narrow_p=0.55,       # 价差>=3档时公众限价单挂进价差内部的概率
    mm_qpos=0.15,        # 排队位置：排在档位公众深度的15%之后（1.0=队尾=最慢；
                         #   越接近0=排队越靠前=越快，这正是高频军备竞赛的意义）
    q_max=3000.0,        # 库存上限（股）；超过则撤掉"越买越多"的那一侧
    grid_dt=10.0,        # 行情采样间隔（秒）
    snap_t=9000.0,       # 订单簿快照时刻（秒）
)


# =====================================================================
# 第 3 部分：限价订单簿 + 做市商的"事件驱动"单日模拟（撮合核心）
# =====================================================================
def simulate_day(P, p_informed, sim_id, record=False):
    """
    跑一个模拟交易日。p_informed = 知情订单占全部市价单的笔数比例。
    事件循环：下一次事件时间 ~ 指数分布（泊松过程），事件类型按强度加权抽取：
        市价单 / 限价单 / 撤单 / 信息事件(知情 bursts)
    价格是"事件驱动"推进的：最优档位深度被吃完 -> 价格跳一档。
    记账约定：bid/ask 字典存"每档总深度"（公众单+做市商单）；
              做市商挂单以 mm['*_t']/mm['*_q'] 为唯一事实源。
    """
    rng = np.random.default_rng([P['seed'], sim_id])   # 每次模拟独立且可复现
    TICK, T_DAY = P['tick'], P['t_day']

    # ---------- 初始订单簿：买十档 / 卖十档（key=价格档位(整数tick), value=总深度(股)）----------
    p0t = int(round(P['p0'] / TICK))
    bid = {p0t - 1 - i: 0.0 for i in range(P['n_levels'])}   # 1160,1159,...,1151
    ask = {p0t + 0 + i: 0.0 for i in range(P['n_levels'])}   # 1161,1162,...,1170
    for i in range(P['n_levels']):
        med = P['dep_med0'] * (1 + P['dep_grow'] * i)
        bid[p0t - 1 - i] = float(rng.lognormal(np.log(med), P['dep_sig']))
        ask[p0t + i] = float(rng.lognormal(np.log(med), P['dep_sig']))

    def refill_far(side):
        """某侧最优档被吃完（价格跳档）后，在远端补一个新档位，保持十档。"""
        if side == 'ask':
            ask[max(ask) + 1] = float(rng.lognormal(np.log(P['dep_far']), P['dep_sig']))
        else:
            bid[min(bid) - 1] = float(rng.lognormal(np.log(P['dep_far']), P['dep_sig']))

    # ---------- 做市商状态 ----------
    mm = dict(cash=0.0, inv=0.0,                      # 现金 / 库存（股，多头为正）
              bid_t=None, bid_q=0.0, ask_t=None, ask_q=0.0,   # 挂单价格档位与剩余量
              bid_pulled=False, ask_pulled=False)     # 库存超限后的单侧撤单状态
    spread_income = 0.0     # 价差收入 = Σ 每笔成交相对当时中间价的"边"(edge)
    n_fills = 0
    max_abs_inv = 0.0
    fills = []              # 展示用

    # ---------- 事件强度（笔/秒）。限价/撤单率按市价单率等比缩放，深度平稳 ----------
    # 口径：lam_m 是"全部市价单流"（噪声+知情合计）。知情订单笔数占比恰为 p：
    #   知情率 = p×λm；噪声率 = (1−p)×λm  =>  两者合计 = λm。
    # （旧写法 lam_i = p×λm 且噪声率不缩放，实际知情占比只有 p/(1+p)，x 轴会被低估——已修正。）
    lam_m = (1.0 - p_informed) * P['lam_m'] / T_DAY    # 噪声市价单率
    lam_l = P['lim_ratio'] * P['lam_m'] / T_DAY        # 限价/撤单率锚定总市价单流，深度平稳
    lam_c = P['can_ratio'] * P['lam_m'] / T_DAY
    # 知情订单"笔数"占比 = p  =>  信息事件发生率 = p * 市价单率 / 每次burst笔数
    lam_i = p_informed * P['lam_m'] / (P['burst_k'] * T_DAY)

    # ---------- 记录容器 ----------
    grid_t, grid_mid, grid_spread = [], [], []      # 10秒网格行情
    mm_t, mm_inv, mm_inc, mm_pnl = [], [], [], []   # 做市商5秒状态
    bursts = []                                     # 信息事件(时间,方向)
    snap = None
    stats = dict(trades=0, traded_vol=0.0, inf_trades=0, inf_vol=0.0, jumps=0,
                 lost=0)                            # lost=因价差过宽而流失的噪声单
    next_grid, next_mm = 0.0, 0.0

    def consume(side, qty, t, informed):
        """撮合引擎：市价单吃单边深度。吃到最优档吃完 -> 价格跳档。
        档位队列模型：[公众单(做市商之前)] [做市商的单] [公众单(做市商之后)]。"""
        nonlocal spread_income, n_fills, max_abs_inv
        levels = ask if side == 'ask' else bid
        if informed:
            stats['inf_trades'] += 1
        while qty > 1e-9 and len(levels) > 0:
            p = min(levels) if side == 'ask' else max(levels)   # 当前对手方最优档
            D = levels[p]
            mid_now = (max(bid) + min(ask)) * TICK / 2.0
            mmq = mm['ask_q'] if (side == 'ask' and mm['ask_t'] == p) else \
                  (mm['bid_q'] if (side == 'bid' and mm['bid_t'] == p) else 0.0)
            pa, mf, pb = 0.0, 0.0, 0.0
            if mmq > 1e-9:                                  # 做市商在此档：按队列位置成交
                public = D - mmq
                ahead = P['mm_qpos'] * public               # 排在做市商前面的公众深度
                pa = min(qty, ahead); qty -= pa
                mf = min(qty, mmq); qty -= mf               # 做市商被成交
                pb = min(qty, public - ahead); qty -= pb
                levels[p] -= pa + mf + pb
                if side == 'ask':
                    mm['ask_q'] -= mf
                else:
                    mm['bid_q'] -= mf
                if mf > 0:
                    price = p * TICK
                    if side == 'ask':                       # 卖在中间价上方 => 赚价差
                        edge = (price - mid_now) * mf
                        mm['cash'] += price * mf; mm['inv'] -= mf
                    else:                                   # 买在中间价下方 => 赚价差
                        edge = (mid_now - price) * mf
                        mm['cash'] -= price * mf; mm['inv'] += mf
                    spread_income += edge; n_fills += 1
                    fills.append((t, 1 if side == 'bid' else -1, price, mf, edge))
                    max_abs_inv = max(max_abs_inv, abs(mm['inv']))
                if mm['ask_q'] <= 1e-9:                     # 挂单被吃完 -> 同步状态
                    mm['ask_t'] = None; mm['ask_q'] = 0.0
                if mm['bid_q'] <= 1e-9:
                    mm['bid_t'] = None; mm['bid_q'] = 0.0
            else:                                           # 档位里没有做市商
                pa = min(qty, D); qty -= pa
                levels[p] -= pa
            if informed:
                stats['inf_vol'] += pa + mf + pb            # 知情订单消耗的量
            if levels[p] <= 1e-9:                           # 最优档被吃完
                del levels[p]
                refill_far(side)                            # 远端补档，保持十档
                stats['jumps'] += 1                         # 价格跳一档

    def market_order(t):
        """噪声市价单：价差太宽就流失（弹性）；否则吃掉对手方最优档。"""
        spread_t = min(ask) - max(bid)
        if spread_t > P['elastic_max']:
            stats['lost'] += 1
            return
        side = 'ask' if rng.random() < 0.5 else 'bid'       # 买市价单吃卖盘，反之亦然
        size = float(rng.lognormal(np.log(P['mkt_med']), P['mkt_sig']))
        consume(side, size, t, informed=False)
        stats['trades'] += 1; stats['traded_vol'] += size

    def info_burst(t):
        """信息事件：知情交易者知道方向，同方向连续 K 笔大单瞬间连发（无视价差）。
        做市商在 burst 内来不及反应（撤单有延迟）-> 挂单被'选择性成交'。"""
        d = 1 if rng.random() < 0.5 else -1
        side = 'ask' if d > 0 else 'bid'
        bursts.append((t, d))
        for _ in range(P['burst_k']):
            size = float(rng.lognormal(np.log(P['mkt_med'] * P['burst_size_mult']),
                                       P['mkt_sig']))
            consume(side, size, t, informed=True)
            stats['trades'] += 1; stats['traded_vol'] += size

    def limit_order():
        """限价单：55%挂最优档；价差越宽，越多比例挂进价差内部收窄价差
        （价差>=3档时内部概率0.55，快速让中间价回归公允）；其余随机挂一档。"""
        buy = rng.random() < 0.5
        size = float(rng.lognormal(np.log(P['lim_med']), P['lim_sig']))
        bb, ba = max(bid), min(ask)
        p_in = P['narrow_p'] if ba - bb >= 3 else 0.25      # 内部改价概率（随价差宽度提高）
        r = rng.random()
        if buy:
            if r < 0.55:
                p = bb
            elif r < 0.55 + p_in and ba - bb >= 2:
                p = bb + 1
            else:
                keys = list(bid); p = keys[int(rng.random() * len(keys))]
            bid[p] = bid.get(p, 0.0) + size
        else:
            if r < 0.55:
                p = ba
            elif r < 0.55 + p_in and ba - bb >= 2:
                p = ba - 1
            else:
                keys = list(ask); p = keys[int(rng.random() * len(keys))]
            ask[p] = ask.get(p, 0.0) + size

    def cancel_order():
        """撤单：按"公众深度"加权随机选一档，撤掉该档公众深度的 20%~60%（不动做市商单）。"""
        levels = bid if rng.random() < 0.5 else ask
        keys = list(levels)
        if not keys:
            return
        w = np.array([max(levels[k] - (mm['ask_q'] if (levels is ask and mm['ask_t'] == k) else
                                       (mm['bid_q'] if (levels is bid and mm['bid_t'] == k) else 0.0)),
                          0.0) for k in keys])
        tot = w.sum()
        if tot <= 0:
            return
        u, acc = rng.random() * tot, 0.0
        for k, wk in zip(keys, w):
            acc += wk
            if u <= acc:
                levels[k] -= wk * rng.uniform(0.2, 0.6)     # 部分撤单
                break

    def mm_decision(t):
        """做市商状态机（每5秒）：撤旧单 -> 库存超上限则单侧停报（滞回恢复）-> 重挂两侧。
        报价锚定：买价不超过 对手方最优卖价 - mm_max_spread；卖价不低于 对手方最优买价
        + mm_max_spread（价差被撕开时做市商把报价收回来，信息因此快速进入中间价）。
        简化：撤单/重挂瞬间完成；真实世界有延迟与排队位置问题（见README'常见的坑'）。"""
        # 1) 撤掉旧挂单（从订单簿深度中扣除）
        for s in ('bid', 'ask'):
            p = mm[f'{s}_t']
            if p is not None:
                book = bid if s == 'bid' else ask
                book[p] -= mm[f'{s}_q']
                if book[p] <= 1e-9:
                    del book[p]
                    refill_far(s)
                mm[f'{s}_t'] = None; mm[f'{s}_q'] = 0.0
        # 2) 库存上限状态机（带滞回：超限停报，消化到一半才恢复）
        if mm['inv'] >= P['q_max']:
            mm['bid_pulled'] = True                    # 库存太多 -> 停止接买单
        elif mm['inv'] <= P['q_max'] / 2:
            mm['bid_pulled'] = False
        if mm['inv'] <= -P['q_max']:
            mm['ask_pulled'] = True
        elif mm['inv'] >= -P['q_max'] / 2:
            mm['ask_pulled'] = False
        # 3) 重挂两侧：价差<=2档时挂在最优档；价差>=3档时向内改价1档
        #    （帮市场快速收窄价差、把成交edge限制在1档以内；不做跨侧锚定——
        #     锚到过期对手价会变成接盘侠）
        if not mm['bid_pulled']:
            bb, ba = max(bid), min(ask)
            p = bb + 1 if (ba - bb >= 3 and P['mm_improve']) else bb
            bid[p] = bid.get(p, 0.0) + P['mm_qty']
            mm['bid_t'] = p; mm['bid_q'] = P['mm_qty']
        if not mm['ask_pulled']:
            bb, ba = max(bid), min(ask)
            p = ba - 1 if (ba - bb >= 3 and P['mm_improve']) else ba
            ask[p] = ask.get(p, 0.0) + P['mm_qty']
            mm['ask_t'] = p; mm['ask_q'] = P['mm_qty']
        if record:
            mid = (max(bid) + min(ask)) * TICK / 2.0
            mm_t.append(t); mm_inv.append(mm['inv'])
            mm_inc.append(spread_income)
            mm_pnl.append(mm['cash'] + mm['inv'] * mid)   # 累计盯市盈亏

    mm_decision(0.0)

    # ---------- 主事件循环 ----------
    t = 0.0
    while t < T_DAY:
        tot = lam_m + lam_l + lam_c + lam_i
        t += rng.exponential(1.0 / tot)               # 泊松过程：下一次事件间隔
        if t >= T_DAY:
            break
        u = rng.random() * tot                        # 按强度加权选事件类型
        if u < lam_m:
            market_order(t)
        elif u < lam_m + lam_l:
            limit_order()
        elif u < lam_m + lam_l + lam_c:
            cancel_order()
        else:
            info_burst(t)
        # --- 定时任务（网格行情采样 / 做市商决策 / 快照）---
        while t >= next_grid and next_grid < T_DAY:
            bb, ba = max(bid), min(ask)
            grid_t.append(next_grid); grid_mid.append((bb + ba) * TICK / 2.0)
            grid_spread.append(ba - bb); next_grid += P['grid_dt']
        while t >= next_mm and next_mm < T_DAY:
            mm_decision(next_mm); next_mm += P['mm_dt']
        if record and snap is None and t >= P['snap_t']:
            nb = sorted(bid.items(), key=lambda x: -x[0])[:P['n_levels']]
            na = sorted(ask.items(), key=lambda x: x[0])[:P['n_levels']]
            snap = (nb, na)

    # ---------- 日终结算 ----------
    mid_T = (max(bid) + min(ask)) * TICK / 2.0
    total_pnl = mm['cash'] + mm['inv'] * mid_T        # 盯市总盈亏（库存按中间价估值）
    adverse_pnl = total_pnl - spread_income           # 库存/逆向选择损益 = 总 - 价差收入
    out = dict(
        total_pnl=total_pnl, spread_income=spread_income, adverse_pnl=adverse_pnl,
        n_fills=n_fills, max_abs_inv=max_abs_inv, n_bursts=len(bursts),
        final_inv=mm['inv'], mid_T=mid_T,
        daily_return=mid_T / grid_mid[0] - 1.0 if grid_mid else 0.0,
    )
    if record:  # 展示日的完整轨迹
        mids = np.array(grid_mid)
        rets = np.diff(mids) / mids[:-1] * 1e4        # 10秒收益（bp）
        out.update(
            grid_t=np.array(grid_t), grid_mid=mids, grid_spread=np.array(grid_spread),
            rets=rets, lag1=float(np.corrcoef(rets[:-1], rets[1:])[0, 1]),
            mm_t=np.array(mm_t), mm_inv=np.array(mm_inv),
            mm_inc=np.array(mm_inc), mm_pnl=np.array(mm_pnl),
            bursts=bursts, snap=snap, fills=fills, stats=stats,
        )
    return out


# =====================================================================
# 第 4 部分：波动率两步校准（用真实日波动率定标模拟器的市价单强度）
# =====================================================================
def calibrate(cal, verbose=True):
    """第1步：先用初始 lam_m 试跑若干天，测模拟日波动率；
    第2步：日波动率 ~ sqrt(lam_m)，按 (目标/实测)^2 缩放 lam_m 后固定。
    校准目标取全样本日波动率（代表市场的典型活跃状态）。"""
    target = cal['sigma_full']
    P2 = dict(P); P2['p0'] = cal['p0']; P2['lam_m'] = P['lam_m0']
    pilot = [simulate_day(P2, 0.0, 100 + i)['daily_return'] for i in range(10)]
    sig_pilot = float(np.std(pilot, ddof=1))
    lam = float(np.clip(P['lam_m0'] * (target / max(sig_pilot, 1e-6)) ** 2, 1000, 12000))
    P['lam_m'] = lam
    P['p0'] = cal['p0']
    if verbose:
        print(f'  校准目标(平安银行2015年以来全样本日波动率) = {target*100:.2f}%'
              f'（近250日为 {cal["sigma_day"]*100:.2f}%，当前处于低波动状态）')
        print(f'  试跑10天实测日波动率 = {sig_pilot*100:.2f}%')
        print(f'  校准后市价单强度 lam_m = {lam:.0f} 笔/天')
    return sig_pilot


# =====================================================================
# 第 5 部分：主流程 —— 展示日 + p扫描 + 图表 + CSV + 延伸对照
# =====================================================================
def main():
    t_start = time.time()
    print('=' * 68)
    print('第19课 市场微观结构入门：订单簿与做市（合成数据/模拟市场，种子42）')
    print('=' * 68)

    # ---- 1. 真实数据校准 ----
    print('\n[1] 读取真实数据校准参数')
    cal = load_calibration()
    print(f'  数据来源: {cal["source"]}')
    print(f'  参考价={cal["p0"]:.2f}元  近250日日波动率={cal["sigma_day"]*100:.2f}%')
    print(f'  近250日均成交量={cal["avg_volume"]/1e4:.0f}万股  日均成交额={cal["avg_amount"]/1e8:.2f}亿元')
    calibrate(cal)

    # ---- 2. 展示日模拟（p=0.2，记录全部轨迹）----
    print('\n[2] 展示日模拟（知情交易者比例 p=0.20）')
    show = simulate_day(P, 0.20, 200, record=True)
    st = show['stats']
    tick_bp = P['tick'] / show['grid_mid'].mean() * 1e4
    spread_bp = show['grid_spread'] * P['tick'] / show['grid_mid'].mean() * 1e4
    print(f'  价格跳档次数={st["jumps"]}  噪声市价单成交={st["trades"]-st["inf_trades"]}笔'
          f'（因价差>2档流失 {st["lost"]} 笔）')
    print(f'  知情成交 {st["inf_trades"]} 笔 / {st["inf_vol"]/1e4:.1f}万股，'
          f'占总成交量 {st["inf_vol"]/st["traded_vol"]*100:.1f}%')
    print(f'  日收益={show["daily_return"]*100:+.2f}%  10秒收益滞后1阶自相关={show["lag1"]:+.3f}')
    print(f'  平均价差={spread_bp.mean():.1f}bp（中位数 {np.median(spread_bp):.1f}bp，1档={tick_bp:.1f}bp）')
    print(f'  做市商：成交 {show["n_fills"]} 笔，价差收入 {show["spread_income"]:+.0f} 元，'
          f'库存/逆向选择损益 {show["adverse_pnl"]:+.0f} 元，总盈亏 {show["total_pnl"]:+.0f} 元')
    print(f'  库存峰值={show["max_abs_inv"]:.0f}股  日终库存={show["final_inv"]:+.0f}股  信息事件={show["n_bursts"]}次')

    # ---- 3. p 扫描：知情交易者比例 vs 做市商盈亏 ----
    print('\n[3] 做市盈亏 vs 知情交易者比例 p（每个p模拟10天）')
    p_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
    rows = []
    for pi, p in enumerate(p_grid):
        for d in range(10):
            r = simulate_day(P, p, 1000 + pi * 100 + d)
            rows.append(dict(p=p, day=d + 1, 总盈亏=r['total_pnl'], 价差收入=r['spread_income'],
                             库存逆向选择损益=r['adverse_pnl'], 成交笔数=r['n_fills'],
                             库存峰值=r['max_abs_inv'], 信息事件数=r['n_bursts'],
                             日收益=r['daily_return']))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, 'market_maker_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'  已写出 {csv_path}')

    g = df.groupby('p')
    summary = pd.DataFrame({
        '平均总盈亏': g['总盈亏'].mean(), '盈亏标准差': g['总盈亏'].std(),
        '平均价差收入': g['价差收入'].mean(), '平均逆向选择损益': g['库存逆向选择损益'].mean(),
        '平均成交笔数': g['成交笔数'].mean(), '平均库存峰值': g['库存峰值'].mean()})
    print('\n  p     平均总盈亏    价差收入   逆向选择   成交笔数  库存峰值')
    for p, r in summary.iterrows():
        print(f'  {p:4.2f}  {r["平均总盈亏"]:10.0f}  {r["平均价差收入"]:9.0f}  '
              f'{r["平均逆向选择损益"]:9.0f}  {r["平均成交笔数"]:7.0f}  {r["平均库存峰值"]:7.0f}')

    # 盈亏平衡点：平均总盈亏穿零处（线性插值）
    ps, ys = summary.index.values, summary['平均总盈亏'].values
    p_star = None
    for i in range(len(ps) - 1):
        if ys[i] > 0 and ys[i + 1] <= 0:
            p_star = ps[i] + (0 - ys[i]) / (ys[i + 1] - ys[i]) * (ps[i + 1] - ps[i])
            break
    if p_star is None:
        p_star = float(ps[-1]) if ys[-1] > 0 else 0.0
    print(f'\n  >>> 做市盈亏平衡的知情交易者比例 p* ≈ {p_star:.3f}（低于此比例做市才赚钱）')

    # ---- 4. 延伸对照：点差×成交量 vs 真实成交额 ----
    print('\n[4] 延伸对照：散户支付的点差成本 vs 平安银行真实成交额')
    tick_bp_real = P['tick'] / cal['p0'] * 1e4
    spread_cost = P['tick'] * cal['avg_volume']                 # 每股成交约付1个点差
    ratio = spread_cost / cal['avg_amount']
    print(f'  真实：日均成交 {cal["avg_volume"]/1e4:.0f} 万股，成交额 {cal["avg_amount"]/1e8:.2f} 亿元')
    print(f'  若每股交易付1个点差({P["tick"]}元={tick_bp_real:.1f}bp)：'
          f'全市场每天付给流动性提供者 ≈ {spread_cost/1e4:.0f} 万元')
    print(f'  占成交额比例 ≈ {ratio*1e4:.1f} bp —— 这就是"散户支付了价差"的量级')
    print(f'  （本模拟一天成交 {st["traded_vol"]/1e4:.0f} 万股，仅为真实的 '
          f'{st["traded_vol"]/cal["avg_volume"]*100:.1f}%，模拟盈亏绝对值只用于讲机制）')

    # =================================================================
    # 第 6 部分：绘图
    # =================================================================
    # ---- 图01：订单簿十档深度快照 ----
    nb, na = show['snap']
    bp = [k * P['tick'] for k, _ in nb]; bd = [v for _, v in nb]
    ap = [k * P['tick'] for k, _ in na]; ad = [v for _, v in na]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    h = 0.9 * P['tick']                       # 条高=0.9个最小报价单位（barh高度是数据单位！）
    ax.barh(bp, [-d for d in bd], height=h, color='#d62728', alpha=0.85, label='买盘(挂单深度)')
    ax.barh(ap, ad, height=h, color='#2ca02c', alpha=0.85, label='卖盘(挂单深度)')
    for y, d in zip(bp, bd):
        ax.text(-d - 30, y, f'{d:.0f}', va='center', ha='right', fontsize=8, color='#d62728')
    for y, d in zip(ap, ad):
        ax.text(d + 30, y, f'{d:.0f}', va='center', ha='left', fontsize=8, color='#2ca02c')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(sorted(set(bp + ap)))
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('挂单量（股，买盘向左 / 卖盘向右）')
    ax.set_ylabel('价格（元）')
    ax.set_title(f'图01 某时刻(t={P["snap_t"]:.0f}s)的十档订单簿深度快照（合成数据/模拟市场）')
    ax.legend(loc='lower right')
    ax.margins(y=0.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, '01_订单簿深度图.png'), dpi=150); plt.close()

    # ---- 图02：最优买卖价时间序列 ----
    gt = show['grid_t'] / 60.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(gt, show['grid_mid'] + show['grid_spread'] / 2 * P['tick'], lw=0.8,
            color='#2ca02c', label='最优卖价 ask1')
    ax.plot(gt, show['grid_mid'] - show['grid_spread'] / 2 * P['tick'], lw=0.8,
            color='#d62728', label='最优买价 bid1')
    ax.plot(gt, show['grid_mid'], lw=1.0, color='#1f77b4', label='中间价 mid')
    for tb, d in show['bursts']:
        ax.axvline(tb / 60.0, color='gray', alpha=0.35, lw=0.7,
                   ymin=0.45 if d > 0 else 0.0, ymax=1.0 if d > 0 else 0.55)
    ax.set_xlabel('模拟时间（分钟，4小时交易日）')
    ax.set_ylabel('价格（元）')
    ax.set_title('图02 模拟期间最优买卖价时间序列（灰线=信息事件：知情大单方向，合成数据）')
    ax.legend(loc='upper left', ncol=3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, '02_bid_ask时间序列.png'), dpi=150); plt.close()

    # ---- 图03：做市商库存轨迹 + 累计盈亏分解 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    mt = show['mm_t'] / 60.0
    ax1.plot(mt, show['mm_inv'], lw=0.9, color='#1f77b4')
    ax1.axhline(P['q_max'], color='#d62728', ls='--', lw=1, label=f'库存上限 +{P["q_max"]:.0f}股（触发买单撤单）')
    ax1.axhline(-P['q_max'], color='#2ca02c', ls='--', lw=1, label=f'库存下限 -{P["q_max"]:.0f}股（触发卖单撤单）')
    ax1.axhline(0, color='k', lw=0.6)
    ax1.set_ylabel('做市商库存（股）')
    ax1.set_title('图03 做市商库存轨迹与累计盈亏（展示日 p=0.20，合成数据）')
    ax1.legend(loc='upper right', fontsize=9)
    ax2.plot(mt, show['mm_inc'], lw=1.2, color='#2ca02c', label='累计价差收入（成交边）')
    ax2.plot(mt, show['mm_pnl'], lw=1.2, color='#1f77b4', label='累计盯市总盈亏')
    ax2.fill_between(mt, show['mm_inc'], show['mm_pnl'],
                     where=show['mm_pnl'] < show['mm_inc'], color='#d62728', alpha=0.25,
                     label='库存/逆向选择损耗')
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_xlabel('模拟时间（分钟）'); ax2.set_ylabel('累计盈亏（元）')
    ax2.legend(loc='upper left', fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, '03_做市商库存轨迹.png'), dpi=150); plt.close()

    # ---- 图04：做市盈亏 vs 知情交易者比例 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.errorbar(ps, ys, yerr=summary['盈亏标准差'].values, fmt='o-', color='#1f77b4',
                 ecolor='#9ecae1', capsize=3, label='日均总盈亏 ± 标准差')
    ax1.axhline(0, color='k', lw=0.8)
    if 0 < p_star < ps[-1]:
        ax1.axvline(p_star, color='#d62728', ls='--', lw=1.2,
                    label=f'盈亏平衡 p*≈{p_star:.2f}')
    ax1.set_xlabel('知情交易者订单占比 p'); ax1.set_ylabel('做市商日均盈亏（元）')
    ax1.set_title('做市盈亏 vs 知情交易者比例')
    ax1.legend(fontsize=9)
    w = 0.038
    ax2.bar(ps - w / 2, summary['平均价差收入'], w, color='#2ca02c', label='价差收入')
    ax2.bar(ps + w / 2, summary['平均逆向选择损益'], w, color='#d62728', label='库存/逆向选择损益')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xlabel('知情交易者订单占比 p'); ax2.set_ylabel('元/天')
    ax2.set_title('盈亏分解：价差收入 vs 逆向选择损失')
    ax2.legend(fontsize=9)
    fig.suptitle('图04 做市商盈亏 vs 知情交易者比例 p（每点10个模拟日，合成数据）', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, '04_做市盈亏vs知情交易者比例.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # ---- 图05：模拟收益与价差分布 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    nz = show['rets'][show['rets'] != 0.0]           # 排除"10秒内价格没动"的零收益样本
    n_zero = int((show['rets'] == 0.0).sum())
    ax1.hist(nz, bins=40, color='#1f77b4', alpha=0.85)
    ax1.set_xlabel(f'10秒中间价收益（bp，剔除0收益样本{n_zero}个）'); ax1.set_ylabel('频数')
    ax1.set_title(f'日内10秒收益分布（滞后1阶自相关 {show["lag1"]:+.2f}）')
    vals, cnts = np.unique(show['grid_spread'], return_counts=True)
    ax2.bar([f'{v:.0f}档' for v in vals], cnts, color='#ff7f0e', alpha=0.85)
    for x, c in zip(range(len(vals)), cnts):
        ax2.text(x, c, str(c), ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel(f'买卖价差（档；1档={tick_bp:.1f}bp）'); ax2.set_ylabel('10秒采样频数')
    ax2.set_title('买卖价差分布（价差小、多数时间=1个最小报价单位）')
    fig.suptitle('图05 模拟市场的统计特征：收益分布与价差分布（合成数据，对照真实微观结构）', y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, '05_模拟收益与价差分布.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # ---- 汇总 ----
    print('\n[5] 产出文件（outputs/）')
    for f in sorted(os.listdir(OUT)):
        print('   ', f)
    print(f'\n完成，总耗时 {time.time()-t_start:.1f} 秒。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
