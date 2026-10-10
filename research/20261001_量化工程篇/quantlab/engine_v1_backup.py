#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quantlab.engine —— 第39课：自写迷你事件驱动回测引擎（A股约束版）

设计立场：课02 用 30 行 pandas 向量化回测（整条序列一次算完），课21 用工业框架
backtrader。本引擎是第三条路——**自己把"回测"这两个字拆开重装**：
一个逐 bar 推进时钟的事件循环 + 一台遵守 A 股规则的撮合台。

为什么值得造这个轮子：
  回测收益 ≠ 可实现收益，差距全部藏在撮合规则里。向量化和框架都会替你"悄悄处理"
  这些细节，本引擎把每条规则写成显式开关，让它们可以被逐个关掉做消融实验：
    1. T+1        当日买入的股票当日不可卖出（True/False）
    2. 涨跌停拒单  开盘/收盘封死在涨停价：买单拒（买不进）；跌停：卖单拒（卖不出）
    3. 停牌       当日无行情 bar：挂单冻结、持仓冻结，复牌后挂单按复牌日重估
    4. 手数       买入必须为 100 股整数倍（向下取整）
    5. 成本       佣金双边万2.5（最低5元）+ 印花税卖出千0.5 + 滑点单边0.1%（与课02/21同口径）
    6. 成交时点   exec_model='close'（信号日收盘成交，即 bt 的 cheat-on-close，
                  对齐课02口径）| 'next_open'（次日开盘成交，bt 默认市价单口径）

诚实声明（必须读）：
  - 价格输入是前复权日K。涨跌停板的真实价格 = round(不复权前收 × (1±10%), 2)，
    前复权在除权日两侧对"前收×1.1"的比值几乎不失真（复权是乘常数），但逐日四舍五入
    仍有分位级噪声，故封板判定用 pct×0.998 的容差近似。少数极端除权日附近可能误判。
  - 单一"恒定滑点"模型是粗近似，真实冲击成本与成交量相关（课30 TWAP/VWAP 的动机）。
  - 本引擎只支持市价单与整手多头，不含融资融券/拆分/分红现金流再投资（前复权价已
    含分红除权的价格效应）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252      # 与课02一致
RF_ANNUAL = 0.02        # 与课02一致（夏普用无风险利率）
AT_LIMIT_TOL = 0.985   # 封板涨幅容差：|涨跌幅| ≥ pct×0.985 即视为封板（吸收低价股前复权舍入噪声）


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def default_limit_pct(code: str, d: pd.Timestamp) -> float:
    """A股涨跌停幅度：主板10%；创业板(300/301)与科创板(688)自 2020-08-24 起为20%。
    （ST 5% 无法从代码推断，本引擎不处理——案例股均非 ST。）"""
    if d >= pd.Timestamp("2020-08-24") and (code.startswith(("300", "301", "688"))):
        return 0.20
    return 0.10


@dataclass
class BrokerConfig:
    """撮合台的所有"物理定律"。默认值 = 课02/课21 的成本口径，可直接对账。"""
    cash: float = 1_000_000.0
    commission: float = 2.5e-4        # 佣金：双边万2.5
    min_commission: float = 5.0       # 单笔最低5元
    stamp_tax: float = 5.0e-4         # 印花税：卖出千0.5（2023-08后税率）
    slippage: float = 1.0e-3          # 恒定滑点：单边0.1%（买入上浮/卖出下浮）
    slip_clamp: bool = False          # True: 成交价钳制在当日 high/low 内（bt slip_match 行为）
    board_lot: int = 100              # 一手100股
    t_plus_1: bool = True             # 当日买入不可当日卖出
    limit_rules: bool = True          # 涨跌停拒单
    limit_pct_fn: Callable[[str, pd.Timestamp], float] = default_limit_pct


# ---------------------------------------------------------------------------
# 订单与记录
# ---------------------------------------------------------------------------
class Order:
    _seq = 0

    def __init__(self, date, code, side, shares, use_all, created_i, pct=None):
        Order._seq += 1
        self.id = Order._seq
        self.created = date          # 下单日（信号日）
        self.created_i = created_i
        self.code = code
        self.side = side             # 'BUY' / 'SELL'
        self.shares = shares         # 目标股数；use_all=True 时在成交时按资金/持仓现算
        self.use_all = use_all       # buy: 全仓买入; sell: 清仓
        self.pct = pct               # 按市值/持仓比例下单（成交时现算数量）
        self.status = "PENDING"      # PENDING / FILLED / REJECTED
        self.reason = ""             # REJECTED 原因：LIMIT_UP_BUY / LIMIT_DOWN_SELL / T_PLUS_1 / NO_CASH
        self.fill_date = None
        self.fill_price = np.nan
        self.fee = np.nan

    def __repr__(self):
        return (f"Order#{self.id} {self.created.date()} {self.side} {self.code} "
                f"{self.shares}股 -> {self.status}{'('+self.reason+')' if self.reason else ''}")


@dataclass
class Result:
    """一次回测的全部产出：净值、订单簿、成交簿、指标。"""
    name: str
    equity: pd.Series            # 按日收盘估值
    orders: pd.DataFrame
    trades: pd.DataFrame
    config: BrokerConfig
    exec_model: str

    def metrics(self) -> dict:
        eq = self.equity
        ret = eq.pct_change().dropna()
        years = len(ret) / TRADING_DAYS
        ann = eq.iloc[-1] / eq.iloc[0] - 1.0
        ann_ret = (1.0 + ann) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        ann_vol = float(ret.std()) * np.sqrt(TRADING_DAYS) if len(ret) > 1 else np.nan
        sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol and ann_vol > 1e-12 else np.nan
        mdd = float((eq / eq.cummax() - 1.0).min())
        if len(self.orders):
            filled = self.orders[self.orders["status"] == "FILLED"]
            rejected = self.orders[self.orders["status"] == "REJECTED"]
        else:
            filled = rejected = self.orders
        return {
            "name": self.name, "exec_model": self.exec_model,
            "final_nav": float(eq.iloc[-1] / eq.iloc[0]),
            "total_ret_pct": ann * 100,
            "ann_ret_pct": ann_ret * 100, "ann_vol_pct": ann_vol * 100,
            "sharpe": sharpe, "max_dd_pct": mdd * 100,
            "n_filled": len(filled), "n_rejected": len(rejected),
            "n_buy": int((filled["side"] == "BUY").sum()),
            "n_sell": int((filled["side"] == "SELL").sum()),
        }


# ---------------------------------------------------------------------------
# 策略接口与上下文
# ---------------------------------------------------------------------------
class Strategy:
    """子类实现 on_bar(ctx)。每天收盘后（当天 OHLC 已可见）被调用一次。
    ctx.buy()/ctx.sell() 下的是市价单：按 exec_model 在当日收盘或次日开盘撮合。
    可选钩子 on_fill(tr)：每笔成交后回调（tr 为成交记录 dict），用于跟踪
    实际成交价驱动的状态（如海龟的止损锚点、纸面交易器的持仓台账）。"""

    def on_start(self, ctx) -> None:  # 可选钩子
        pass

    def on_fill(self, tr: dict) -> None:  # 可选钩子
        pass

    def on_bar(self, ctx) -> None:
        raise NotImplementedError


class Context:
    """策略能看到的世界：当天的行情、自己的账户、下单 API。
    故意不给未来数据——history(n) 只到今天为止。"""

    def __init__(self, bt: "Backtester", codes, i):
        self._bt = bt
        self._codes = codes
        self.i = i
        self.date = bt.calendar[i]

    # ---- 行情 ----
    @property
    def codes(self):
        return list(self._codes)

    def bar(self, code):
        """当日该标的的 (open,high,low,close)；停牌返回 None。"""
        return self._bt.bar_of(code, self.i)

    def history(self, code, n) -> pd.DataFrame:
        """截至今天（含）的最近 n 根bar（停牌日无行）。"""
        return self._bt.history_of(code, self.i, n)

    def prev_close(self, code):
        return self._bt.prev_close_of(code, self.i)

    # ---- 账户 ----
    @property
    def cash(self):
        return self._bt.cash

    def position(self, code):
        return self._bt.position.get(code, 0)

    def last_close(self, code):
        """用于估值的最近收盘价（停牌取停牌前收盘）。"""
        return self._bt.last_close_of(code, self.i)

    @property
    def equity(self):
        return self._bt.mark_equity(self.i)

    def last_buy_date(self, code):
        return self._bt.last_buy_date.get(code)

    # ---- 下单（市价单） ----
    def buy(self, code, pct=None, shares=None):
        """pct: 按当前总市值的比例买入（会在成交时按可用资金/手数折算）；
        shares: 指定股数。二者都不给 = 全仓。"""
        self._bt.submit(self.i, code, "BUY", shares, all_=(pct is None and shares is None), pct=pct)

    def sell(self, code, pct=None, shares=None):
        """pct: 卖出当前持仓的比例；不给参数 = 清仓。"""
        self._bt.submit(self.i, code, "SELL", shares, all_=(pct is None and shares is None), pct=pct)


# ---------------------------------------------------------------------------
# 引擎本体
# ---------------------------------------------------------------------------
class Backtester:
    """多标的单账户事件驱动引擎。data: {code: DataFrame(open/high/low/close, DatetimeIndex)}。
    日历 = 所有标的交易日的并集；某标的当日缺行 = 停牌（挂单/持仓冻结）。"""

    def __init__(self, data: dict, config: BrokerConfig = None,
                 exec_model: str = "close", name: str = "bt"):
        assert exec_model in ("close", "next_open")
        self.config = config or BrokerConfig()
        self.exec_model = exec_model
        self.name = name
        self.data = {c: d.dropna(subset=["close"]) for c, d in data.items()}
        self.calendar = sorted(set().union(*[set(d.index) for d in self.data.values()]))
        self.codes = list(self.data.keys())

    # ---- 行情原语 ----
    def bar_of(self, code, i):
        df = self.data[code]
        d = self.calendar[i]
        return df.loc[d] if d in df.index else None

    def history_of(self, code, i, n):
        df = self.data[code]
        d = self.calendar[i]
        return df.loc[:d].tail(n)

    def prev_close_of(self, code, i):
        df = self.data[code]
        d = self.calendar[i]
        s = df.loc[:d, "close"]
        return float(s.iloc[-2]) if len(s) >= 2 else np.nan   # 不含今天

    def last_close_of(self, code, i):
        df = self.data[code]
        d = self.calendar[i]
        s = df.loc[:d, "close"]
        return float(s.iloc[-1]) if len(s) else np.nan

    # ---- 账户原语 ----
    def mark_equity(self, i):
        total = self.cash
        for c in self.codes:
            if self.position.get(c, 0):
                total += self.position[c] * self.last_close_of(c, i)
        return total

    # ---- 下单 ----
    def submit(self, i, code, side, shares, all_=False, pct=None):
        assert side in ("BUY", "SELL")
        # 同代码同方向的旧挂单自动撤单（模拟"撤单重挂"，避免停牌/拒单期间订单堆积）
        for od in self.orders_list:
            if od.status == "PENDING" and od.code == code and od.side == side:
                od.status, od.reason = "CANCELLED", "REPLACED"
        if pct is not None:
            self._pct_pending.append((code, side, pct))
            return
        self.orders_list.append(Order(self.calendar[i], code, side,
                                      int(shares) if shares else 0, all_, i))

    # ---- 撮合原语（课40 纸面交易运行器直接复用） ----
    def try_fill(self, order: Order, i, ref_price, high=None, low=None,
                 prev_close=None, t_now=None, at_close=False):
        """对一张 PENDING 订单按 ref_price(滑点前基础价) 尝试撮合。
        at_close=True 表示按当日收盘撮合（启用"收盘封板"判定）；
        next_open 模式只按一字板判定拒单。成交则改写账户，拒单则落 reason。"""
        cfg = self.config
        code, side = order.code, order.side

        # 1) 涨跌停封板检查（先于定价：封板时价格再漂亮也轮不到你）
        #    判定用"一字板"（high==low，即开=高=低=收）为主——对前复权舍入免疫；
        #    收盘撮合时另判"收盘封板"（收==高/低 且涨幅达阈值）。容差 0.985 吸收
        #    新股低价段的复权舍入噪声。
        if cfg.limit_rules and prev_close is not None and not math.isnan(prev_close):
            pct = cfg.limit_pct_fn(code, t_now)
            chg = ref_price / prev_close - 1.0
            flat = (high is not None and low is not None and high == low)
            sealed_up = flat or (at_close and high is not None and ref_price == high)
            sealed_dn = flat or (at_close and low is not None and ref_price == low)
            if side == "BUY" and sealed_up and chg >= pct * AT_LIMIT_TOL:
                order.status, order.reason = "REJECTED", "LIMIT_UP_BUY"
                return
            if side == "SELL" and sealed_dn and chg <= -pct * AT_LIMIT_TOL:
                order.status, order.reason = "REJECTED", "LIMIT_DOWN_SELL"
                return

        # 2) 定价（恒定滑点 + 可选钳制）
        if side == "BUY":
            px = ref_price * (1 + cfg.slippage)
        else:
            px = ref_price * (1 - cfg.slippage)
        if cfg.slip_clamp and high is not None and low is not None:
            px = min(max(px, low), high)

        # 3) 数量确定（手数取整）
        pos = self.position.get(code, 0)
        if side == "BUY":
            budget = self.cash / (1 + cfg.commission) if order.use_all else \
                (order.shares * px if order.shares else self.mark_equity(i) * (order.pct or 1.0))
            shares = int(budget / px / cfg.board_lot) * cfg.board_lot
            if shares <= 0:
                order.status, order.reason = "REJECTED", "NO_CASH"
                return
        else:
            if order.use_all:
                shares = pos
            elif order.shares:
                shares = min(int(order.shares), pos)
            else:
                shares = int(pos * (order.pct or 1.0) / cfg.board_lot) * cfg.board_lot
            if shares <= 0:
                order.status, order.reason = "REJECTED", "NO_POSITION"
                return
            # 4) T+1：当日买入的股票当日不可卖
            if cfg.t_plus_1 and self.last_buy_date.get(code) == t_now:
                order.status, order.reason = "REJECTED", "T_PLUS_1"
                return

        # 5) 成交并记账
        notional = shares * px
        fee = max(cfg.commission * notional, cfg.min_commission)
        if side == "BUY":
            fee += 0.0
            if notional + fee > self.cash + 1e-6:   # 资金不够整手 → 缩到现金允许的最大整手
                shares = int(self.cash / (1 + cfg.commission) / px / cfg.board_lot) * cfg.board_lot
                if shares <= 0:
                    order.status, order.reason = "REJECTED", "NO_CASH"
                    return
                notional = shares * px
                fee = max(cfg.commission * notional, cfg.min_commission)
            self.cash -= notional + fee
            self.position[code] = pos + shares
            self.last_buy_date[code] = t_now
        else:
            fee += cfg.stamp_tax * notional
            self.cash += notional - fee
            self.position[code] = pos - shares
            if self.position[code] == 0:
                self.last_buy_date.pop(code, None)
        order.status = "FILLED"
        order.fill_date, order.fill_price, order.fee = t_now, px, fee
        self.trades_list.append(dict(date=t_now, code=code, side=side, shares=shares,
                                     price=round(px, 4), fee=round(fee, 2),
                                     cash_after=round(self.cash, 2)))

    def _try_fill_and_notify(self, od, i, ref_price, high, low, prev_close, t_now, at_close=False):
        """撮合 + 成交回调（策略的 on_fill 钩子）。"""
        self.try_fill(od, i, ref_price, high, low, prev_close, t_now, at_close=at_close)
        if od.status == "FILLED" and self.trades_list:
            self._strategy.on_fill(self.trades_list[-1])

    # ---- 主循环 ----
    def run(self, strategy: Strategy) -> Result:
        cfg = self.config
        self._strategy = strategy
        self.cash = cfg.cash
        self.position = {}
        self.last_buy_date = {}
        self.orders_list = []
        self.trades_list = []
        self._pct_pending = []          # pct 语义的订单：转成正式 Order 后提交
        equity_rows = []

        strategy.on_start(Context(self, self.codes, 0))

        for i, t in enumerate(self.calendar):
            # (1) 次日开盘模式：先撮合昨日及更早的挂单（开盘价）
            if self.exec_model == "next_open":
                for od in [o for o in self.orders_list if o.status == "PENDING" and o.created_i < i]:
                    bar = self.bar_of(od.code, i)
                    if bar is None:
                        continue        # 停牌：挂单继续冻结
                    self._try_fill_and_notify(od, i, float(bar["open"]), float(bar["high"]),
                                          float(bar["low"]), self.prev_close_of(od.code, i), t,
                                          at_close=False)

            # (2) 策略决策（今天收盘后；今天的行情可见）
            self._pct_pending = []
            ctx = Context(self, self.codes, i)
            strategy.on_bar(ctx)
            for code, side, pct in self._pct_pending:
                self.orders_list.append(Order(t, code, side, 0, False, i, pct=pct))

            # (3) 收盘模式：所有未成交挂单（含停牌遗留）都在今日收盘重估——
            #     复牌首日按复牌日收盘成交，正是"复牌连板"现象的来源
            if self.exec_model == "close":
                for od in [o for o in self.orders_list if o.status == "PENDING" and o.created_i <= i]:
                    bar = self.bar_of(od.code, i)
                    if bar is None:
                        continue
                    self._try_fill_and_notify(od, i, float(bar["close"]), float(bar["high"]),
                                          float(bar["low"]), self.prev_close_of(od.code, i), t,
                                          at_close=True)

            equity_rows.append((t, self.mark_equity(i)))

        orders_df = pd.DataFrame([{
            "id": o.id, "created": o.created, "code": o.code, "side": o.side,
            "shares_req": "ALL" if o.use_all else
                          (f"{o.pct:.0%}市值" if getattr(o, "pct", None) else o.shares),
            "status": o.status, "reason": o.reason, "fill_date": o.fill_date,
            "fill_price": o.fill_price, "fee": o.fee,
        } for o in self.orders_list])
        trades_df = pd.DataFrame(self.trades_list)
        return Result(self.name, pd.Series(dict(equity_rows), name="equity"),
                      orders_df, trades_df, cfg, self.exec_model)


# ---------------------------------------------------------------------------
# 简要自检：python engine.py 直接跑一个"两根bar"的玩具用例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    df = pd.DataFrame({"open": [10.0, 10.5, 11.0, 10.8],
                       "high": [10.6, 11.0, 11.05, 11.0],
                       "low": [9.9, 10.4, 10.9, 10.7],
                       "close": [10.5, 11.0, 11.0, 10.9]}, index=idx)
    class BuyOnce(Strategy):
        def on_bar(self, ctx):
            if ctx.i == 0:
                ctx.buy("TEST")
    bt = Backtester({"TEST": df}, exec_model="close")
    r = bt.run(BuyOnce())
    print(r.orders[["created", "side", "status", "reason", "fill_price", "fee"]].to_string(index=False))
    print("期末净值:", round(r.metrics()["final_nav"], 4))
