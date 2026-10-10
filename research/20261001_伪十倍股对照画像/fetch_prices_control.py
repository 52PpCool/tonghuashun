#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_prices_control.py —— 对照画像 第三步：拉 28 只对照组日K（新浪源，前复权）
与 20261001_十倍股画像/fetch_prices_akshare.py 同一管道（东财源已封禁，直接用新浪源）。
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


def load_control():
    df = pd.read_csv(TENX_CSV)
    ctl = df[(df["核验区间涨幅%(前复权)"] >= 500) & (df["核验区间涨幅%(前复权)"] < 1000)
             & (df["isTenX(≥1000%为1)"] == 0)]
    return [{"code": str(r["股票代码"]).split(".")[0], "symbol": r["股票代码"],
             "name": r["股票简称"], "verified_gain": float(r["核验区间涨幅%(前复权)"])}
            for _, r in ctl.iterrows()]


def main():
    stocks = load_control()
    print(f"[info] 对照组 {len(stocks)} 只")
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
                    df = ak.stock_zh_a_daily(symbol=symbol, start_date=START, end_date=END, adjust="qfq")
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            if df is None or df.empty:
                raise RuntimeError("empty df")
            df.to_csv(fpath, index=False, encoding="utf-8-sig")
            log.append(f"[ok]   {s['symbol']} {s['name']} rows={len(df)} first={df['date'].iloc[0]}")
            print(log[-1])
        except Exception as e:  # noqa: BLE001
            fail.append(s["symbol"])
            log.append(f"[FAIL] {s['symbol']} {s['name']} {type(e).__name__}: {str(e)[:120]}")
            print(log[-1])
        time.sleep(1.0)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} source=sina fresh={FRESH} =====\n")
        f.write("\n".join(log) + "\n")
    json.dump({"total": len(stocks), "failed": fail, "window": [START, END], "adjust": "qfq"},
              open(os.path.join(DIR, "data", "fetch_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[done] 成功 {len(stocks) - len(fail)}/{len(stocks)}，失败 {len(fail)}: {fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
