#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
broker.py —— 第40课：纸面撮合台

关键设计：**不重写撮合逻辑**。本类直接调用 quantlab.engine.fill_order —— 与第39课
回测引擎逐字相同的函数（同一佣金/印花税/滑点/一字板/T+1 规则）。回测与纸面交易
共享同一撮合台，是消除"回测一套规则、实盘另一套规则"这类系统性偏差的工程手段。

与回测引擎的差异只在"时钟来源"：回测器自己推进日历；这里每次调用由运行器
（main.py）喂入"今天的行情"，状态（现金/持仓/挂单）由 SQLite 持久化，进程重启可恢复。
"""
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantlab.engine import Account, BrokerConfig, Order, fill_order  # noqa: E402


class LiveBroker:
    """纸面撮合台。用法（每日一次）：
        broker.on_open(d, bars, prev_closes)      # 1) 执行此前挂单（今日开盘价）
        ... 收盘后计算信号 ...
        broker.submit(d, code, side, shares)      # 2) 提交新订单（明日开盘成交）
        broker.mark(d, last_closes)               # 3) 估值
    """

    def __init__(self, cfg: BrokerConfig, acct: Account = None):
        self.cfg = cfg
        self.acct = acct or Account(cfg.cash)
        self.orders: list[Order] = []             # 本次进程生命周期内创建/恢复的订单
        self.new_fills: list[tuple[Order, dict]] = []  # (订单, 成交记录) 待入库
        self.new_rejects: list[Order] = []        # 状态发生变化的拒单，待入库

    # ---- 下单 ----
    def submit(self, d, code, side, shares) -> Order:
        assert side in ("BUY", "SELL") and shares > 0
        o = Order(pd.Timestamp(d), code, side, int(shares), False, created_i=0)
        self.orders.append(o)
        return o

    def has_live(self, code, side) -> bool:
        """同代码同方向是否已有未完成订单（避免重复挂单堆积）。
        注意：REJECTED 的订单不算 live——次日重挂是预期行为（追单），但拒单状态
        必须先持久化，否则重启后 stale PENDING 会和新单一起成交（本课实战bug）。"""
        return any(o.code == code and o.side == side and o.status == "PENDING"
                   for o in self.orders)

    # ---- 每日撮合 ----
    def on_open(self, d, bars: dict, prev_closes: dict):
        """今日开盘：撮合所有 PENDING 挂单（停牌/无行情的代码跳过=挂单冻结）。"""
        t = pd.Timestamp(d)
        for o in self.orders:
            if o.status != "PENDING" or o.created >= t:
                continue                           # 今日新挂的单不今日成交（next_open 纪律）
            bar = bars.get(o.code)
            if bar is None:
                continue
            fill_order(o, self.acct, self.cfg, 0,
                       ref_price=float(bar["open"]), high=float(bar["high"]),
                       low=float(bar["low"]), prev_close=prev_closes.get(o.code),
                       t_now=t, at_close=False)
            if o.status == "FILLED":
                self.new_fills.append((o, self.acct.trades[-1]))
            elif o.status == "REJECTED":
                self.new_rejects.append(o)

    # ---- 估值 ----
    def mark(self, last_closes: dict) -> tuple[float, float]:
        """返回 (现金, 总市值)。持仓用最近已知收盘价估值（停牌取停牌前）。"""
        mv = sum(sh * last_closes.get(c, 0.0) for c, sh in self.acct.position.items() if sh)
        return self.acct.cash, mv

    def positions(self) -> dict:
        return {c: sh for c, sh in self.acct.position.items() if sh}
