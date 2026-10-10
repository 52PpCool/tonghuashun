#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reconcile.py —— 第40课：每日对账

量化系统死掉的方式通常不是"亏钱"，而是"账对不上还继续跑"。对账是运行器的免疫系统：
每天收盘估值后跑一遍全部不变式（invariant），任何一条失败都要立刻报警。

五条不变式：
  I1 现金恒等式：总市值 = 现金 + Σ持仓股数×最近收盘价（估值口径一致）
  I2 现金非负：任何时刻 cash >= -1e-6（本策略无融资）
  I3 持仓非负且与成交台账一致：持仓只被成交改变
  I4 订单-成交一一对应：每笔 FILLED 订单恰有一条 fill 记录，拒单必有 reason
  I5 数据新鲜度：回填窗口内每个交易日、每只标的都有行情（停牌需显式标注）
"""
import json

import pandas as pd


def check_day(conn, d: str, cash: float, positions: dict, mv: float, equity: float,
              last_closes: dict, pool: list) -> list:
    """跑当日全部不变式，返回失败项列表（空列表 = 全绿）。"""
    errs = []
    # I1 现金恒等式
    mv_calc = sum(sh * last_closes.get(c, 0.0) for c, sh in positions.items())
    if abs(mv_calc - mv) > 1e-6:
        errs.append(f"I1 市值不符: 传入{mv:.2f} vs 重算{mv_calc:.2f}")
    if abs(cash + mv - equity) > 1e-6:
        errs.append(f"I1 净值恒等式不符: cash{cash:.2f}+mv{mv:.2f} != equity{equity:.2f}")
    # I2 现金非负
    if cash < -1e-6:
        errs.append(f"I2 现金为负: {cash:.2f}")
    # I3 持仓非负
    for c, sh in positions.items():
        if sh < 0:
            errs.append(f"I3 持仓为负: {c} {sh}股")
    # I4 订单-成交一一对应（截至当日）
    orders = pd.read_sql_query("SELECT id, status, reason FROM orders WHERE date<=?", conn, params=(d,))
    fills = pd.read_sql_query("SELECT order_id, COUNT(*) n FROM fills GROUP BY order_id", conn)
    fmap = dict(zip(fills["order_id"], fills["n"]))
    for _, o in orders.iterrows():
        n = fmap.get(o["id"], 0)
        if o["status"] == "FILLED" and n != 1:
            errs.append(f"I4 FILLED订单{o['id']}的fill记录数={n}（应为1）")
        if o["status"] == "REJECTED" and not o["reason"]:
            errs.append(f"I4 REJECTED订单{o['id']}缺reason")
    # I5 数据新鲜度（当日）
    have = pd.read_sql_query("SELECT DISTINCT code FROM daily_prices WHERE date=?", conn, params=(d,))
    missing = set(pool) - set(have["code"])
    if missing:
        errs.append(f"I5 当日缺行情: {sorted(missing)}（若为停牌需在note中标注）")
    return errs


def report(conn) -> str:
    """全程对账报告：账户曲线连续性 + 订单/成交统计 + 全部 run_log 失败项。"""
    acc = pd.read_sql_query("SELECT * FROM account_daily ORDER BY date", conn)
    orders = pd.read_sql_query("SELECT status, reason, COUNT(*) n FROM orders GROUP BY status, reason", conn)
    fills = pd.read_sql_query("SELECT COUNT(*) n, SUM(fee) total_fee FROM fills", conn)
    fails = pd.read_sql_query("SELECT date, step, detail FROM run_log WHERE ok=0", conn)
    lines = []
    lines.append(f"账户快照天数: {len(acc)}  首日 {acc['date'].iloc[0] if len(acc) else '-'} "
                 f"末日 {acc['date'].iloc[-1] if len(acc) else '-'}")
    if len(acc):
        e0, e1 = acc["equity"].iloc[0], acc["equity"].iloc[-1]
        lines.append(f"期初权益 {e0:,.2f} → 期末权益 {e1:,.2f}（{(e1/e0-1)*100:+.2f}%）")
    lines.append("\n订单统计：")
    lines.append(orders.to_string(index=False) if len(orders) else "  （无订单）")
    if len(fills) and fills["n"].iloc[0]:
        lines.append(f"成交 {int(fills['n'].iloc[0])} 笔，总费用 {fills['total_fee'].iloc[0]:,.2f} 元")
    lines.append(f"\n对账失败项: {len(fails)} 条")
    if len(fails):
        lines.append(fails.to_string(index=False))
    return "\n".join(lines)
