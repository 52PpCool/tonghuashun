#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
signal.py —— 第40课：策略信号（SMA20/60 择时）

时序纪律（不可违反，代码里用"收盘后才能看到今日bar"来保证）：
  信号在 T 日收盘后计算 → 提交订单 → T+1 日开盘撮合。
  任何"用 T 日收盘价算信号又在 T 日成交"的写法都是前视偏差（课02 的核心教训）。
"""
import pandas as pd

FAST, SLOW = 20, 60


def sma_signal(close: pd.Series) -> pd.Series:
    """与课02/39 完全同款：SMA(fast)>SMA(slow) → 1，否则 0（预热期 NaN→0 空仓）。"""
    return (close.rolling(FAST).mean() > close.rolling(SLOW).mean()).astype(float)


FAST_SMA = None  # （fast/slow 常数在上方 FAST/SLOW）
