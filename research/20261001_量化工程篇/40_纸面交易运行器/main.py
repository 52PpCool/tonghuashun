#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py —— 第40课：纸面交易运行器（每日盘后自动跑的"活"策略）

一个真实量化系统的每日循环（本程序逐条实现）：
    1. 恢复状态（现金/持仓/挂单，从 SQLite）——进程随时可挂、可重启
    2. 开盘撮合昨日挂单（复用第39课引擎的 fill_order，同一套 A 股规则）
    3. 收盘后计算信号（SMA20/60，只用截至今天的数据——防前视）
    4. 按目标权重提交订单（等权 1/5，明日开盘成交）
    5. 估值、写账户快照
    6. 对账（五条不变式，任何失败立即记录 ok=0）

用法：
    python main.py --init                      # 建库+从本地留档导入历史行情（离线）
    python main.py --live-refresh              # 联网刷新行情（新浪qfq全量重拉，复权陷阱见 datafeed.py）
    python main.py --backfill 2026-06-01:2026-09-30   # 逐日回放（模拟盘后任务的3个月历史）
    python main.py --date 2026-09-30           # 单日运行（幂等：重跑不产生重复订单）
    python main.py --report                    # 对账报告 + 导出曲线

合规声明：纯纸面交易（paper trading），不连接任何真实券商接口，不构成投资建议。
"""
import argparse
import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
sys.path.insert(0, ARCH)
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

from quantlab.engine import Account, BrokerConfig, Order
from quantlab.data import load_cache, load_profile_price, QUANT_ROOT
from quantlab.datafeed import fetch_sina_daily
import store as st
import signalcalc as sigmod
from broker import LiveBroker
import reconcile as rec

# 股票池：5 只等权（全部有本地留档，--init 离线可建库）
POOL = {
    "000001": ("平安银行", "SZ", "cache"),
    "600519": ("贵州茅台", "SH", "cache"),
    "000063": ("中兴通讯", "SZ", "cache"),
    "002371": ("北方华创", "SZ", "profile37"),
    "300308": ("中际旭创", "SZ", "profile37"),
}
WEIGHT = 1.0 / len(POOL)          # 目标权重：等权
CFG = BrokerConfig(cash=1_000_000.0)   # 成本口径与课02/39一致


# ---------------------------------------------------------------------------
def cmd_init(conn):
    """建库 + 从本地留档导入行情（不联网）。"""
    for code, (name, exch, src) in POOL.items():
        conn.execute("INSERT OR REPLACE INTO instruments VALUES (?,?,?,?)",
                     (code, name, exch, f"导入来源: {src}"))
    conn.commit()
    for code, (name, _, src) in POOL.items():
        if src == "cache":
            df = load_cache(code)
        else:
            df = load_profile_price(code)
        st.upsert_prices(conn, code, df, source=f"import:{src}")
        di = df["date"] if "date" in df.columns else df.index
        print(f"[init] {code} {name}: 导入 {len(df)} 根bar ({di[0].date()} ~ {di[-1].date()})")
    st.log(conn, "-", "init", f"导入 {len(POOL)} 只标的历史行情", ok=1)


def cmd_refresh(conn, start="2015-01-01"):
    """联网刷新（新浪 qfq 全量重拉 + 整表覆盖，规避前复权增量错位，见 datafeed.py docstring）。"""
    for code in POOL:
        try:
            df = fetch_sina_daily(code, start=start)
            st.upsert_prices(conn, code, df, source="sina_qfq_full")
            print(f"[refresh] {code}: {len(df)} 根bar ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
        except RuntimeError as e:
            print(f"[refresh] {code} 失败（保留库内旧数据）: {e}")
            st.log(conn, "-", "refresh", f"{code} 失败: {e}", ok=0)


# ---------------------------------------------------------------------------
def run_day(conn, d: str, closes: dict) -> list:
    """单日循环：恢复状态 → 开盘撮合 → 收盘信号 → 提交订单 → 估值 → 对账。
    完全从 DB 恢复状态，因此 --backfill 的每一天与 --date 单日跑是同一条代码路径。"""
    st.clear_day(conn, d)
    signals = {c: sigmod.sma_signal(closes[c]) for c in closes}
    cash, positions, pend = st.load_state(conn, d)
    acct = Account(cash if cash is not None else CFG.cash)
    if cash is None:
        acct.position = {}
    else:
        acct.position = dict(positions)
    broker = LiveBroker(CFG, acct)
    for _, r in pend.iterrows():                 # 恢复未成交挂单（保留原订单号）
        o = Order(pd.Timestamp(r["date"]), r["code"], r["side"], r["shares"], False, 0)
        o.id, o.status = int(r["id"]), "PENDING"
        broker.orders.append(o)

    # 1) 开盘撮合昨日挂单
    bars = st.load_bars(conn, d)
    prev_rows = pd.read_sql_query(
        "SELECT code, close, MAX(date) FROM daily_prices WHERE date<? GROUP BY code",
        conn, params=(d,))
    prev_closes = dict(zip(prev_rows["code"], prev_rows["close"]))
    broker.on_open(d, bars, prev_closes)
    for o, tr in broker.new_fills:
        st.update_order(conn, o)
        st.save_fill(conn, o.id, tr)
    for o in broker.new_rejects:                 # 拒单也必须即时持久化！否则重启后
        st.update_order(conn, o)                 # stale PENDING 会和新单重复成交（实战bug）
    if broker.new_fills or broker.new_rejects:
        detail = "; ".join(f"{o.side} {o.code} @{o.fill_price:.2f}" for o, _ in broker.new_fills) \
            + "".join(f"；拒单 {o.code}({o.reason})" for o in broker.new_rejects)
        st.log(conn, d, "order", f"开盘撮合 {len(broker.new_fills)}笔成交/{len(broker.new_rejects)}笔拒单: {detail}", ok=1)

    # 2) 收盘信号（只用截至今天的数据；当日缺行情=停牌，信号沿用前值）
    sig_today = {}
    for code in POOL:
        sser = signals[code]
        if pd.Timestamp(d) in sser.index:
            s = int(sser.loc[d])
        else:                                    # 停牌日：沿用最近一个有效信号
            prior = sser.loc[:pd.Timestamp(d)]
            s = int(prior.iloc[-1]) if len(prior) else 0
        sig_today[code] = s
        close = closes[code]
        f = float(close.rolling(sigmod.FAST).mean().loc[:pd.Timestamp(d)].iloc[-1]) \
            if len(close.loc[:pd.Timestamp(d)]) >= sigmod.FAST else float("nan")
        sl = float(close.rolling(sigmod.SLOW).mean().loc[:pd.Timestamp(d)].iloc[-1]) \
            if len(close.loc[:pd.Timestamp(d)]) >= sigmod.SLOW else float("nan")
        st.save_signal(conn, code, d, s, f, sl)
    st.log(conn, d, "signal", " ".join(f"{c}={s}" for c, s in sig_today.items()), ok=1)

    # 3) 提交订单（幂等：该日已有订单则跳过——重跑不重复下单）
    exist = pd.read_sql_query("SELECT COUNT(*) n FROM orders WHERE date=?", conn, params=(d,))
    if exist["n"].iloc[0] == 0:
        last_closes_d = st.last_known_closes(conn, d)
        cash_now, mv_now = broker.mark(last_closes_d)
        equity_now = cash_now + mv_now
        for code in POOL:
            s, pos = sig_today[code], broker.acct.position.get(code, 0)
            if s == 1 and pos == 0 and not broker.has_live(code, "BUY"):
                px = last_closes_d.get(code)
                if px is None:
                    continue
                sh = int(equity_now * WEIGHT / (px * (1 + CFG.slippage)) // CFG.board_lot) * CFG.board_lot
                if sh > 0:
                    o = broker.submit(d, code, "BUY", sh)
                    o.id = st.save_order(conn, o)
            elif s == 0 and pos > 0 and not broker.has_live(code, "SELL"):
                o = broker.submit(d, code, "SELL", pos)
                o.id = st.save_order(conn, o)
        st.log(conn, d, "submit", f"提交 {len([o for o in broker.orders if o.created == pd.Timestamp(d)])} 笔订单", ok=1)
    else:
        st.log(conn, d, "submit", "该日已有订单记录，跳过提交（幂等）", ok=1)

    # 4) 估值 + 快照
    last_closes = st.last_known_closes(conn, d)
    cash_f, mv = broker.mark(last_closes)
    equity = cash_f + mv
    st.save_account(conn, d, cash_f, mv, equity, broker.positions())

    # 5) 对账
    errs = rec.check_day(conn, d, cash_f, broker.positions(), mv, equity, last_closes, list(POOL))
    st.log(conn, d, "day_done", "对账全绿" if not errs else "对账失败: " + "; ".join(errs),
           ok=1 if not errs else 0)
    return errs


def cmd_backfill(conn, window: str):
    start, end = window.split(":")
    cal = st.trading_calendar(conn, start, end)
    print(f"[backfill] {start} ~ {end}：{len(cal)} 个交易日（5只池并集日历）")
    closes = {c: st.load_close_series(conn, c) for c in POOL}
    n_err = 0
    for d in cal:
        errs = run_day(conn, d, closes)
        n_err += len(errs)
    print(f"[backfill] 完成，对账失败项 {n_err} 条")


def cmd_report(conn):
    rep = rec.report(conn)
    print(rep)
    with open(os.path.join(OUT, "对账报告.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    acc = pd.read_sql_query("SELECT * FROM account_daily ORDER BY date", conn)
    if len(acc):
        acc.to_csv(os.path.join(OUT, "account_daily.csv"), index=False, encoding="utf-8-sig")
        orders = pd.read_sql_query("SELECT * FROM orders ORDER BY id", conn)
        orders.to_csv(os.path.join(OUT, "orders.csv"), index=False, encoding="utf-8-sig")
        fills = pd.read_sql_query(
            "SELECT f.*, o.date AS order_date FROM fills f JOIN orders o ON f.order_id=o.id ORDER BY f.id", conn)
        fills.to_csv(os.path.join(OUT, "fills.csv"), index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(pd.to_datetime(acc["date"]), acc["equity"] / 1e6, lw=1.4)
        ax.set_title("纸面交易运行器：账户净值（回填演示，SMA20/60 等权5股，成本同课02）")
        ax.set_ylabel("净值（初始100万）"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "回填_净值曲线.png"), dpi=150)
        print(f"\n[report] 曲线/订单/成交流水已导出 {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="纸面交易运行器（纯模拟，不接实盘）")
    ap.add_argument("--init", action="store_true", help="建库并从本地留档导入历史行情")
    ap.add_argument("--live-refresh", action="store_true", help="联网刷新行情（新浪源qfq全量重拉）")
    ap.add_argument("--backfill", type=str, help="逐日回放，如 2026-06-01:2026-09-30")
    ap.add_argument("--date", type=str, help="单日运行（幂等）")
    ap.add_argument("--report", action="store_true", help="对账报告与导出")
    args = ap.parse_args()

    conn = st.connect()
    if args.init:
        cmd_init(conn)
    if args.live_refresh:
        cmd_refresh(conn)
    if args.backfill:
        cmd_backfill(conn, args.backfill)
    if args.date:
        closes = {c: st.load_close_series(conn, c) for c in POOL}
        errs = run_day(conn, args.date, closes)
        print(f"[date] {args.date} 对账失败项 {len(errs)}")
    if args.report:
        cmd_report(conn)
    if not any([args.init, args.live_refresh, args.backfill, args.date, args.report]):
        ap.print_help()
