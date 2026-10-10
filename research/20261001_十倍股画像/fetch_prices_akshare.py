#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_prices_akshare.py —— 十倍股画像 第二步：拉取 88 只十倍股的日K（前复权）

用途：计算「从窗口起点（或上市日）到首次达到10倍」用了多少个交易日/自然日，
     以及「十倍途中」的最大回撤（拿得住吗）。

数据源：akshare stock_zh_a_daily（新浪源，qfq 前复权）。
  2026-10-01 留痕：东财源 stock_zh_a_hist 在首轮批量时对全部 88 只 ConnectionError
  （疑似限频封禁，见 data/fetch_prices_log.txt 首版），故切换新浪源重跑成功。
窗口：2016-08-15 ~ 2026-08-31（研究窗口为 2016-09-01 起，起点略提前以拿到基准价）。
限速：每只间隔 1.0s + 失败重试2次（指数退避）；幂等：data/prices/{code}.csv 已存在则跳过。
留痕：每只成败与行数写入 data/fetch_prices_log.txt；原始 CSV 全部落盘不删除。

用法：python fetch_prices_akshare.py [--fresh]
"""
import json
import os
import sys
import time

import akshare as ak
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
TENX_CSV = "/home/ppku/zcodespace/tonghuashun/research/20260905_最近10年A股十倍股/数据/results_verified.csv"
PRICE_DIR = os.path.join(DIR, "data", "prices")
LOG = os.path.join(DIR, "data", "fetch_prices_log.txt")
START, END = "20160815", "20260831"
FRESH = "--fresh" in sys.argv

os.makedirs(PRICE_DIR, exist_ok=True)


def load_tenx():
    df = pd.read_csv(TENX_CSV)
    tenx = df[df["isTenX(≥1000%为1)"] == 1]
    out = []
    for _, r in tenx.iterrows():
        code = str(r["股票代码"]).split(".")[0]
        out.append({"code": code, "symbol": r["股票代码"], "name": r["股票简称"],
                    "verified_gain": float(r["核验区间涨幅%(前复权)"])})
    return out


def main():
    stocks = load_tenx()
    print(f"[info] 待取 {len(stocks)} 只")
    log, fail = [], []
    for i, s in enumerate(stocks, 1):
        fpath = os.path.join(PRICE_DIR, f"{s['code']}.csv")
        if not FRESH and os.path.exists(fpath) and os.path.getsize(fpath) > 100:
            log.append(f"[skip] {s['symbol']} {s['name']} 已存在")
            continue
        try:
            symbol = ("sh" if s["symbol"].endswith(".SH") else "sz") + s["code"]
            df = None
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_daily(symbol=symbol, start_date=START,
                                             end_date=END, adjust="qfq")
                    break
                except Exception:  # noqa: BLE001 —— 重试退避
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            if df is None or df.empty:
                raise RuntimeError("empty df")
            df.to_csv(fpath, index=False, encoding="utf-8-sig")
            log.append(f"[ok]   {s['symbol']} {s['name']} rows={len(df)} "
                       f"first={df['date'].iloc[0]} last={df['date'].iloc[-1]}")
            print(log[-1])
        except Exception as e:  # noqa: BLE001 —— 失败留痕继续跑
            fail.append(s["symbol"])
            log.append(f"[FAIL] {s['symbol']} {s['name']} {type(e).__name__}: {str(e)[:120]}")
            print(log[-1])
        time.sleep(1.0)
        if i % 20 == 0:
            print(f"[progress] {i}/{len(stocks)}")

    with open(LOG, "a", encoding="utf-8") as f:  # append：多轮尝试都留痕（含东财源全失败的首轮）
        f.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"source={'sina' if 'stock_zh_a_daily' else 'em'} fresh={FRESH} =====\n")
        f.write("\n".join(log) + "\n")
    print(f"[done] 成功 {len(stocks) - len(fail)}/{len(stocks)}，失败 {len(fail)}: {fail}")
    with open(os.path.join(DIR, "data", "fetch_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total": len(stocks), "failed": fail,
                   "window": [START, END], "adjust": "qfq"}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
