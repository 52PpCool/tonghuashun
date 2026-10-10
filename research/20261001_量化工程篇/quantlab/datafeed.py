#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quantlab.datafeed —— 网络行情数据源（第40课纸面交易运行器 / 第42课补数共用）

数据源：akshare 新浪接口 stock_zh_a_daily（东财 stock_zh_a_hist 在本机被 IP 限频封禁，
见课37教训）。限速 1s/只 + 3 次指数退避重试。

【增量更新的复权陷阱】（本模块最重要的教学点）：
  前复权(qfq)价格序列会在每次分红除权后**整体重算**——历史部分全部平移。
  所以"只拉尾部增量"拼到旧的前复权历史上，除权日之前的新旧两段会错位！
  正确做法二选一：
    a) 前复权：每次全量重拉、整表替换（5 只股票约 10 秒，本运行器采用）；
    b) 原始价(adjust="")：永不重算，才允许真正的尾部增量拼接（第41课建仓用）。
"""
import time

import pandas as pd

LAST_ERR = None


def fetch_sina_daily(code6: str, start: str = "2015-01-01", end: str | None = None,
                     adjust: str = "qfq", retries: int = 3, sleep: float = 1.0) -> pd.DataFrame:
    """拉取单只股票日K（新浪源）。code6 为 6 位代码。返回列 date/open/high/low/close/volume。"""
    import akshare as ak
    symbol = ("sh" if code6.startswith(("6", "9")) else "sz") + code6
    global LAST_ERR
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, start_date=start.replace("-", ""),
                                     end_date=(end or "2099-12-31").replace("-", ""),
                                     adjust=adjust)
            if df is None or not len(df):
                raise ValueError(f"{code6} 空返回")
            df = df.rename(columns={"day": "date"})[["date", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            time.sleep(sleep)                      # 限速：新浪免费源 1s/只
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:                     # noqa: BLE001 —— 网络抓取需宽捕获
            LAST_ERR = str(e)
            time.sleep(2 ** (attempt + 1))          # 指数退避：2s/4s/8s
    raise RuntimeError(f"{code6} 拉取失败（{retries}次）：{LAST_ERR}")
