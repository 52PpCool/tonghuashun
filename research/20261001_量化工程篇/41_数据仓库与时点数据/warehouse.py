#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
warehouse.py —— 第41课：本地数据仓库（DuckDB + Parquet 思维）

为什么选 DuckDB：分析型查询（整表扫描、窗口函数、按月聚合）是 SQLite 的弱项、
DuckDB 的强项；单文件嵌入式、可直接读 Parquet/CSV/pandas。分工：
  SQLite（第40课）= 事务状态（订单/持仓，频繁小写入）
  DuckDB（本课）  = 分析仓库（全市场行情/清单，批量写入+复杂查询）

四张表：
  instrument        时点标的主档：上市日 + 退市日（构成"任意历史时点的可选股票池"）
  daily_price       日K（前复权，来自缓存/新浪）
  raw_price         不复权日K（新浪，与前复权**同日拉取**保证同一复权基）
  adj_factor        复权因子 = raw_close / qfq_close（分红除权日跳变）
  industry_snapshot 行业快照（课37/38的137只候选，同花顺一级行业，**仅当前时点**）

诚实边界：iFinD 无"行业分类历史变更"工具，akshare 亦无免费时点行业——
38课遗留的"前视分类偏差"本课只能**量化暴露**、无法还清（见 main.py 实验三）。
"""
import os
import sys
from datetime import date, datetime

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
sys.path.insert(0, ARCH)
DB_PATH = os.path.join(HERE, "warehouse", "quant.duckdb")

from quantlab.data import load_cache, load_profile_price, CACHE_DIR
from quantlab.datafeed import fetch_sina_daily


def connect(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS instrument (
        code VARCHAR PRIMARY KEY, name VARCHAR, exchange VARCHAR,
        list_date DATE, delist_date DATE, source VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS daily_price (
        code VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
        close DOUBLE, volume DOUBLE, PRIMARY KEY (code, date))""")
    con.execute("""CREATE TABLE IF NOT EXISTS raw_price (
        code VARCHAR, date DATE, close DOUBLE, volume DOUBLE, PRIMARY KEY (code, date))""")
    con.execute("""CREATE TABLE IF NOT EXISTS adj_factor (
        code VARCHAR, date DATE, factor DOUBLE, PRIMARY KEY (code, date))""")
    con.execute("""CREATE TABLE IF NOT EXISTS industry_snapshot (
        code VARCHAR PRIMARY KEY, name VARCHAR, ths_industry VARCHAR, as_of VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS industry_alt (
        code VARCHAR PRIMARY KEY, csrc_industry VARCHAR, as_of VARCHAR)""")
    return con


def build_instruments(con) -> dict:
    """上市/退市清单 → instrument 表。返回统计。"""
    import akshare as ak
    rows = []
    # ---- 在市股票：上交所主板 + 科创板，深交所（主板+创业板在同一接口）----
    for symbol, exch in [("主板A股", "SH"), ("科创板", "SH")]:
        df = ak.stock_info_sh_name_code(symbol=symbol)
        for _, r in df.iterrows():
            rows.append((str(r["证券代码"]).zfill(6), r["证券简称"], exch,
                         pd.Timestamp(r["上市日期"]).date(), None, f"sh_name_code:{symbol}"))
    df = ak.stock_info_sz_name_code(symbol="A股列表")
    alt = []
    for _, r in df.iterrows():
        rows.append((str(r["A股代码"]).zfill(6), r["A股简称"], "SZ",
                     pd.Timestamp(r["A股上市日期"]).date(), None, "sz_name_code"))
        if "所属行业" in df.columns and pd.notna(r.get("所属行业")):
            alt.append((str(r["A股代码"]).zfill(6), str(r["所属行业"]),
                        date.today().isoformat()))
    n_alive = len(rows)
    # ---- 退市股票：沪/深退市清单（沪市清单为"暂停上市日期"，作为终止交易日的代理）----
    df = ak.stock_info_sh_delist()
    cols = list(df.columns)
    end_col = "终止上市日期" if "终止上市日期" in cols else "暂停上市日期"
    for _, r in df.iterrows():
        rows.append((str(r["公司代码"]).zfill(6), r["公司简称"], "SH",
                     pd.Timestamp(r["上市日期"]).date(),
                     pd.Timestamp(r[end_col]).date(), f"sh_delist:{end_col}"))
    df = ak.stock_info_sz_delist()
    for _, r in df.iterrows():
        rows.append((str(r["证券代码"]).zfill(6), r["证券简称"], "SZ",
                     pd.Timestamp(r["上市日期"]).date(),
                     pd.Timestamp(r["终止上市日期"]).date(), "sz_delist"))
    n_delist = sum(1 for r in rows if r[4] is not None)
    con.executemany("INSERT OR REPLACE INTO instrument VALUES (?,?,?,?,?,?)", rows)
    if alt:
        con.executemany("INSERT OR REPLACE INTO industry_alt VALUES (?,?,?)", alt)
    con.commit()
    return {"在市": n_alive, "退市": n_delist, "沪退市口径": end_col,
            "深市行业快照": len(alt)}


def ingest_cache_stocks(con) -> int:
    """缓存里的 30 只个股日K（前复权）→ daily_price。"""
    n = 0
    for f in sorted(os.listdir(CACHE_DIR)):
        if not f.startswith("stock_"):
            continue
        code = f.split("_")[1]
        df = load_cache(f[6:])          # 用文件名剩余部分精确定位
        d = df.reset_index().rename(columns={"date": "date"})
        rows = [(code, pd.Timestamp(r["date"]).date(), float(r["open"]), float(r["high"]),
                 float(r["low"]), float(r["close"]), float(r.get("volume", 0.0) or 0.0))
                for _, r in d.iterrows()]
        con.executemany("INSERT OR REPLACE INTO daily_price VALUES (?,?,?,?,?,?,?)", rows)
        n += 1
    con.commit()
    return n


def build_adj_factor(con, codes: list, start="2015-01-01", log=print) -> int:
    """同一时刻拉取 不复权 + 前复权 两套收盘价 → 复权因子表。
    【关键】两者必须同一天拉取：前复权基随每次除权整体重算，跨取数日拼接因子必错位
    （第40课教训的镜像）。"""
    n = 0
    for code in codes:
        try:
            raw = fetch_sina_daily(code, start=start, adjust="")
            qfq = fetch_sina_daily(code, start=start, adjust="qfq")
        except RuntimeError as e:
            log(f"[factor] {code} 失败: {e}")
            continue
        m = raw[["date", "close"]].merge(qfq[["date", "close"]], on="date", suffixes=("_raw", "_qfq"))
        m = m[(m["close_qfq"] > 0)]
        m["factor"] = m["close_raw"] / m["close_qfq"]
        con.executemany("INSERT OR REPLACE INTO raw_price VALUES (?,?,?,?)",
                        [(code, pd.Timestamp(r["date"]).date(), float(r["close"]),
                          float(r["volume"])) for _, r in raw.iterrows()])
        con.executemany("INSERT OR REPLACE INTO adj_factor VALUES (?,?,?)",
                        [(code, pd.Timestamp(r["date"]).date(), float(r["factor"]))
                         for _, r in m.iterrows()])
        n += 1
        con.commit()
    return n


def load_industry_snapshot(con) -> int:
    """课37/38 的行业快照（137+28 只候选，同花顺一级，时点=2026-09 查询日）。"""
    base = os.path.join(os.path.dirname(ARCH), "20261001_十倍股画像", "data")
    as_of = date(2026, 9, 30).isoformat()
    rows = {}
    p1 = pd.read_csv(os.path.join(base, "profile_basic.csv"), dtype={"code": str, "list_date": str})
    for _, r in p1.iterrows():
        rows[r["code"]] = (r["code"], r["name"], r["ths_industry"], as_of)
    p2 = pd.read_csv(os.path.join(os.path.dirname(ARCH), "20261001_伪十倍股对照画像",
                                  "data", "profile_control.csv"), dtype={"code": str})
    for _, r in p2.iterrows():
        rows.setdefault(r["code"], (r["code"], r["name"], r["ths_industry"], as_of))
    con.executemany("INSERT OR REPLACE INTO industry_snapshot VALUES (?,?,?,?)", list(rows.values()))
    con.commit()
    return len(rows)
