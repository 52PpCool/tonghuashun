#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
store.py —— 第40课：纸面交易运行器的状态持久化（SQLite 七张表）

为什么选 SQLite：单文件零依赖、事务保证、断点可恢复——"运行中的策略"最需要的是
**状态机持久化**：今天进程挂了，明天重启必须能从"最后已知状态"继续，且同一天
重跑不产生重复记录（幂等）。DuckDB（第41课）擅长分析查询，SQLite 擅长事务状态，
两者在工程上各司其职。

七张表：
  instruments       标的主档（代码/名称/交易所）
  daily_prices      日K行情（code+date 唯一，记录来源与抓取时间）
  signals           每日信号（SMA20/60）
  orders            订单簿（状态机：PENDING→FILLED/REJECTED/CANCELLED）
  fills             成交流水（与 orders 一对一）
  account_daily     每日账户快照（现金/市值/净值/持仓JSON）
  run_log           运行日志（每步一行，ok=0 表示该日对账有失败项）
"""
import json
import os
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "data", "paper.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    code TEXT PRIMARY KEY, name TEXT, exchange TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS daily_prices (
    code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT, fetched_at TEXT, PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS signals (
    code TEXT, date TEXT, signal INTEGER, sma_fast REAL, sma_slow REAL,
    PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, date TEXT, code TEXT, side TEXT, shares INTEGER,
    status TEXT, reason TEXT, fill_date TEXT, fill_price REAL, fee REAL);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, date TEXT, code TEXT,
    side TEXT, shares INTEGER, price REAL, fee REAL, cash_after REAL);
CREATE TABLE IF NOT EXISTS account_daily (
    date TEXT PRIMARY KEY, cash REAL, market_value REAL, equity REAL, positions TEXT);
CREATE TABLE IF NOT EXISTS run_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, step TEXT, detail TEXT, ok INTEGER);
"""


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_prices(conn, code: str, df: pd.DataFrame, source: str):
    """行情入库：按 (code,date) 幂等 upsert。df 的日期可以是 'date' 列或索引。"""
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    d = df["date"] if "date" in df.columns else df.index
    rows = [(code, pd.Timestamp(t).strftime("%Y-%m-%d"), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r.get("volume", 0.0)), source, now)
            for t, (_, r) in zip(d, df.iterrows())]
    conn.executemany("INSERT OR REPLACE INTO daily_prices VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def load_close_series(conn, code: str) -> pd.Series:
    df = pd.read_sql_query("SELECT date, close FROM daily_prices WHERE code=? ORDER BY date",
                           conn, params=(code,))
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]), name=code)


def load_bars(conn, d: str) -> dict:
    """某日的全部行情 {code: row dict}。"""
    rows = pd.read_sql_query("SELECT * FROM daily_prices WHERE date=?", conn, params=(d,))
    return {r["code"]: r for _, r in rows.iterrows()}


def trading_calendar(conn, start: str, end: str) -> list:
    rows = pd.read_sql_query(
        "SELECT DISTINCT date FROM daily_prices WHERE date BETWEEN ? AND ? ORDER BY date",
        conn, params=(start, end))
    return rows["date"].tolist()


def last_known_closes(conn, d: str) -> dict:
    """截至 d 各代码最近一个收盘价（停牌估值用）。"""
    rows = pd.read_sql_query(
        """SELECT code, close, MAX(date) FROM daily_prices WHERE date<=? GROUP BY code""",
        conn, params=(d,))
    return dict(zip(rows["code"], rows["close"]))


def save_order(conn, o) -> int:
    cur = conn.execute(
        "INSERT INTO orders (date, code, side, shares, status, reason, fill_date, fill_price, fee) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (o.created.strftime("%Y-%m-%d"), o.code, o.side, int(o.shares or 0),
         o.status, o.reason,
         o.fill_date.strftime("%Y-%m-%d") if o.fill_date is not None else None,
         None if o.fill_price != o.fill_price else float(o.fill_price),
         None if o.fee != o.fee else float(o.fee)))
    conn.commit()
    return cur.lastrowid


def update_order(conn, o):
    conn.execute("UPDATE orders SET status=?, reason=?, fill_date=?, fill_price=?, fee=? WHERE id=?",
                 (o.status, o.reason,
                  o.fill_date.strftime("%Y-%m-%d") if o.fill_date is not None else None,
                  None if o.fill_price != o.fill_price else float(o.fill_price),
                  None if o.fee != o.fee else float(o.fee), o.id))
    conn.commit()


def save_fill(conn, order_id: int, tr: dict):
    conn.execute("INSERT INTO fills (order_id, date, code, side, shares, price, fee, cash_after) "
                 "VALUES (?,?,?,?,?,?,?,?)",
                 (order_id, tr["date"].strftime("%Y-%m-%d"), tr["code"], tr["side"],
                  int(tr["shares"]), float(tr["price"]), float(tr["fee"]), float(tr["cash_after"])))
    conn.commit()


def save_account(conn, d: str, cash: float, mv: float, equity: float, positions: dict):
    conn.execute("INSERT OR REPLACE INTO account_daily VALUES (?,?,?,?,?)",
                 (d, cash, mv, equity, json.dumps(positions, ensure_ascii=False)))
    conn.commit()


def save_signal(conn, code: str, d: str, sig: int, f: float, s: float):
    conn.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?)",
                 (code, d, int(sig), float(f), float(s)))
    conn.commit()


def log(conn, d: str, step: str, detail: str, ok: int = 1):
    conn.execute("INSERT INTO run_log (date, step, detail, ok) VALUES (?,?,?,?)", (d, step, detail, ok))
    conn.commit()


def load_state(conn, before_date: str):
    """重启恢复：从 DB 读出 before_date 之前的最后已知账户状态与未成交挂单。
    返回 (cash, positions, pending_rows)。"""
    acc = pd.read_sql_query("SELECT * FROM account_daily WHERE date<? ORDER BY date DESC LIMIT 1",
                            conn, params=(before_date,))
    cash, positions = None, {}
    if len(acc):
        cash = float(acc["cash"].iloc[0])
        positions = json.loads(acc["positions"].iloc[0])
        positions = {k: int(v) for k, v in positions.items() if int(v) != 0}
    pend = pd.read_sql_query(
        "SELECT * FROM orders WHERE status='PENDING' AND date<?", conn, params=(before_date,))
    return cash, positions, pend


def clear_day(conn, d: str):
    """幂等：重跑某日时清掉该日的信号/账户/日志（订单/成交保留——真实系统不可撤销，
    本运行器以"订单只增不改"为原则，重跑日只重算信号与快照）。"""
    for sql in ("DELETE FROM signals WHERE date=?", "DELETE FROM account_daily WHERE date=?",
                "DELETE FROM run_log WHERE date=? AND step!='order'"):
        conn.execute(sql, (d,))
    conn.commit()
