#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quantlab.data —— 数据加载小工具（第39-42课共用）

缓存来源：research/20260918_量化学习全景/data/cache/（30只股票+10板块+指数，前复权，utf-8-sig）
以及 research/20261001_十倍股画像/data/prices/（88只新浪源日K，前复权）。
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.dirname(HERE)                       # 20261001_量化工程篇/
QUANT_ROOT = os.path.join(os.path.dirname(ARCHIVE), "20260918_量化学习全景")
CACHE_DIR = os.path.join(QUANT_ROOT, "data", "cache")
PROFILE_PRICES = os.path.join(os.path.dirname(ARCHIVE), "20261001_十倍股画像", "data", "prices")


def load_price_csv(path: str) -> pd.DataFrame:
    """读一张日K CSV（缓存列：date,open,high,low,close,volume；新浪9列也能读，多余列丢弃）。
    返回仅含 open/high/low/close 的 DataFrame，Index 为 DatetimeIndex('date')，按日期升序。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    return df[cols].astype(float)


def load_cache(fragment: str) -> pd.DataFrame:
    """按文件名片段在 data/cache/ 里找日K，如 load_cache('000001_平安银行')。"""
    hits = [f for f in os.listdir(CACHE_DIR) if fragment in f]
    if not hits:
        raise FileNotFoundError(f"cache 中找不到 {fragment}: {CACHE_DIR}")
    return load_price_csv(os.path.join(CACHE_DIR, hits[0]))


def load_profile_price(code6: str) -> pd.DataFrame:
    """读课37留档的新浪源日K（code6 为6位代码，如 '601882'）。"""
    return load_price_csv(os.path.join(PROFILE_PRICES, f"{code6}.csv"))
