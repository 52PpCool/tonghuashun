# -*- coding: utf-8 -*-
"""
共享数据层：一次性下载并缓存全部真实数据（akshare / yfinance / Kenneth French）。
所有点子的脚本只读本目录的 CSV 缓存，保证离线可复现。

用法:
    python fetch_data.py            # 只下载缺失的
    python fetch_data.py --force    # 强制全部重新下载

数据源与降级策略:
    - 个股日K(前复权): akshare(东财) → 失败时 yfinance(雅虎, 自动前复权)
    - 指数日K:          akshare index_zh_a_hist → yfinance
    - 行业板块日K:      akshare 东财行业板块 → 失败时用对应行业ETF(yfinance)替代
    - Fama-French因子:  Kenneth French 官网
注意: 东财接口有访问频率限制, 脚本已内置 2~5 秒随机间隔与重试。
"""
import os
import sys
import time
import io
import zipfile
import urllib.request
import random

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
os.makedirs(CACHE, exist_ok=True)

START_STOCK = "20150101"   # 个股起点
START_UNIVERSE = "20180101"  # 股票池/板块起点
END = "20260918"
END_YF = "2026-09-18"

STOCKS = {
    "000001": "平安银行", "600519": "贵州茅台", "601318": "中国平安",
    "600036": "招商银行", "601398": "工商银行", "601939": "建设银行",
    "601166": "兴业银行", "600276": "恒瑞医药", "300760": "迈瑞医疗",
    "000333": "美的集团", "000651": "格力电器", "600887": "伊利股份",
    "601888": "中国中免", "600900": "长江电力", "601012": "隆基绿能",
    "002594": "比亚迪",   "601899": "紫金矿业", "600030": "中信证券",
    "601988": "中国银行", "600028": "中国石化", "601857": "中国石油",
    "000002": "万科A",    "600585": "海螺水泥", "002415": "海康威视",
    "000063": "中兴通讯", "603288": "海天味业", "600809": "山西汾酒",
    "300059": "东方财富", "600104": "上汽集团", "000858": "五粮液",
}
SOLO_STOCKS = ["000001", "600519"]          # 讲解用单股, 从2015年起
PAIR_STOCKS = ["601398", "601939", "600036", "601166", "000001"]

INDEXES = {"000300": "沪深300", "000001": "上证指数"}

# 行业数据: 优先东财板块名, 失败则用上海上市的对应行业ETF(雅虎源)
BOARDS = [
    ("银行",     "512800.SS", "银行ETF"),
    ("半导体",   "512480.SS", "半导体ETF"),
    ("白酒",     "512690.SS", "酒ETF"),
    ("医疗器械", "512170.SS", "医疗ETF"),
    ("证券",     "512000.SS", "券商ETF"),
    ("保险",     "512070.SS", "证券保险ETF"),
    ("房地产开发", "512200.SS", "房地产ETF"),
    ("汽车整车", "516110.SS", "汽车ETF"),
    ("电力行业", "159611.SZ", "电力ETF"),
    ("煤炭行业", "515220.SS", "煤炭ETF"),
]

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def _retry(fn, n=2, wait=8):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            print(f"    尝试{i+1}失败: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(wait)
    return None


def _std_cols(df):
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume",
                            "成交额": "amount", "换手率": "turnover"})
    cols = [c for c in ["date", "open", "high", "low", "close", "volume",
                        "amount", "turnover"] if c in df.columns]
    return df[cols]


def fetch_stock_ak(code, start):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=END, adjust="qfq")
    return _std_cols(df)


def fetch_index_ak(code, start):
    import akshare as ak
    df = ak.index_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=END)
    return _std_cols(df)


def fetch_board_ak(name, start):
    import akshare as ak
    df = ak.stock_board_industry_hist_em(symbol=name, start_date=start,
                                         end_date=END, adjust="qfq")
    return _std_cols(df)


def fetch_yf(yf_symbol, start):
    import yfinance as yf
    df = yf.Ticker(yf_symbol).history(start=start, end=END_YF, auto_adjust=True)
    if len(df) == 0:
        raise RuntimeError("yfinance 空数据")
    df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                                          "Close": "close", "High": "high",
                                          "Low": "low", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]]


def yf_symbol_of(code):
    return code + (".SS" if code.startswith(("6", "5")) else ".SZ")


def save(df, fname):
    if df is None or len(df) == 0:
        print(f"  !! {fname} 全部来源失败, 跳过")
        return False
    df.to_csv(os.path.join(CACHE, fname), index=False, encoding="utf-8-sig")
    print(f"  OK {fname:36s} {len(df):6d} 行  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    return True


def _yf_date(ymd):
    """'20180101' -> '2018-01-01'"""
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


def get_with_fallback(fname, ak_fn, yf_sym, start, yf_first=False):
    if os.path.exists(os.path.join(CACHE, fname)):
        return False
    df, src = None, None
    if yf_first and yf_sym:
        time.sleep(1)
        df = _retry(lambda: fetch_yf(yf_sym, _yf_date(start)))
        src = "yfinance"
    if df is None:
        time.sleep(random.uniform(2, 4))
        df = _retry(ak_fn)
        src = "akshare"
    if df is None and yf_sym:
        time.sleep(1)
        df = _retry(lambda: fetch_yf(yf_sym, _yf_date(start)))
        src = "yfinance"
    ok = save(df, fname)
    if ok:
        print(f"     (来源: {src})")
    return ok


def download_ff3():
    out = os.path.join(CACHE, "ff3_monthly.csv")
    if os.path.exists(out):
        return
    try:
        req = urllib.request.Request(FF3_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        text = zf.read(zf.namelist()[0]).decode("latin-1")
        rows = []
        for ln in text.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 6:
                try:
                    rows.append([parts[0]] + [float(p) for p in parts[1:]])
                except ValueError:
                    continue
        df = pd.DataFrame(rows, columns=["yyyymm", "Mkt-RF", "SMB", "HML", "RF"])
        df = df[df["yyyymm"] < "210000"]
        df.to_csv(out, index=False)
        print(f"  OK ff3_monthly.csv (Kenneth French) {len(df)} 行")
    except Exception as e:
        print(f"  !! ff3 下载失败: {e}")


def main():
    print("== 1. 个股日K ==")
    for code, name in STOCKS.items():
        start = START_STOCK if code in SOLO_STOCKS else START_UNIVERSE
        get_with_fallback(f"stock_{code}_{name}.csv",
                          lambda c=code, s=start: fetch_stock_ak(c, s),
                          yf_symbol_of(code), start, yf_first=True)

    print("== 2. 指数日K ==")
    idx_yf = {"000300": "510300.SS", "000001": "000001.SS"}  # 000300 雅虎无数据, 用510300ETF代替
    for code, name in INDEXES.items():
        get_with_fallback(f"index_{code}_{name}.csv",
                          lambda c=code: fetch_index_ak(c, "20100101"),
                          idx_yf[code], "20100101", yf_first=True)

    print("== 3. 行业数据(板块, 失败用ETF替代) ==")
    for bname, etf, etf_name in BOARDS:
        get_with_fallback(f"board_{bname}.csv",
                          lambda n=bname: fetch_board_ak(n, START_UNIVERSE),
                          etf, START_UNIVERSE, yf_first=True)

    print("== 4. Fama-French 三因子(月) ==")
    download_ff3()

    n = len(os.listdir(CACHE))
    print(f"\n缓存目录现有 {n} 个文件 → {CACHE}")


if __name__ == "__main__":
    main()
